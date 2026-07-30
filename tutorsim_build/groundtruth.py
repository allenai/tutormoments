#!/usr/bin/env python3
"""Build ground-truth files from raw human annotations via LLM enrichment.

Reads teacher annotations from a JSONL file and, for each conversation:
  - reuses strategy_label for moments unchanged from an existing ground-truth file
    (matched by annotator_id + turn_start + turn_end + annotation_type + result text),
  - classifies new/changed moments via the Anthropic batch API,
  - writes the merged result to <out_dir>/<conv_id>.json.

Only scaffolding and rapport records are processed; caption records are skipped.

Output format -- one JSON file per conversation:
  {
    "conversation_id": "<uuid>",
    "num_turns": <int>,           # max turn_end seen across all moments
    "key_moments": [ {<moment>}, ... ]
  }
where each moment carries: turn_start, turn_end, annotation_type, annotator_id,
situation, action, result, strategy_label, and (scaffolding only) situation_label,
situation_label_agg, action_decomposed, result_decomposed, overscaffold_decomposed,
action_direction_agg, student_outcome_agg, plus optional cut_turn / moment_id.

Decomposition (action, result, and -- for scaffolding -- over-scaffolding) and the
action/result aggregation are cached by a content hash of the source text, so a plain
run only processes moments not already done. Editing a prompt does NOT invalidate the
cache (it keys on text, not the prompt) -- use the refresh_* flags to force a re-run.

The LLM-call infrastructure (ModelClient/run_batch/build_batch_entry) and the
decompose/structure helpers are reused from tutorsim.client / tutorsim.scoring; the
situation/effectiveness helpers from tutorsim_build.classify. Model/params come from the
`groundtruth` config block (tutorsim.config.get_groundtruth_phase_config).
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tutorsim.config import get_groundtruth_phase_config, get_labeller_config
from tutorsim.scoring import (
    DEFAULT_RESULT_LABEL,
)
from tutorsim.scoring import (
    JUNK_TEXTS as DECOMPOSE_JUNK_TEXTS,
)
from tutorsim.scoring import (
    build_overscaffold_prompt as _build_overscaffold_prompt,
)
from tutorsim.scoring import (
    format_facet_list as _format_facet_list,
)
from tutorsim.scoring import (
    load_decompose_prompt as _load_decompose_prompt,
)
from tutorsim.scoring import (
    load_structure_prompt as _load_structure_prompt,
)
from tutorsim.scoring import (
    parse_action_label as _parse_action_label,
)
from tutorsim.scoring import (
    parse_decomposed as _parse_decomposed,
)
from tutorsim.scoring import (
    parse_result_label as _parse_result_label,
)
from tutorsim_build.classify import (
    JUNK_TEXTS,
    VALID_LABELS,
    _parse_situation_label,
    load_labeller_prompt,
    load_labeller_templates,
    load_situation_prompt,
    pick_template,
)

# tutorsim_build/groundtruth.py -> tutorsim_build -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Default annotations input: the published HuggingFace release artifact (gitignored;
# downloaded locally / from the release). Override with build_ground_truth(input_path=...).
ANNOTATIONS_JSONL = (
    DATA_DIR / "huggingface-release-20260630" / "tutoring_provider_a_annotations.jsonl"
)

# situate and decompose shared the same junk-text set as classify; alias for clarity.
SIT_JUNK_TEXTS = JUNK_TEXTS


def default_out_dir(labeller: str = "hybrid") -> Path:
    """Default per-conversation output directory for a given labeller version."""
    return DATA_DIR / f"ground_truth_{labeller}"


def compute_iou(range_a, range_b):
    """Compute IoU between two (start, end) turn ranges.

    Used only by the scaffolding-cluster grouping below.
    """
    set_a = set(range(range_a[0], range_a[1] + 1))
    set_b = set(range(range_b[0], range_b[1] + 1))
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


def _phase_cfg() -> dict:
    """Ground-truth phase config (model/poll_interval/thinking/...)."""
    return get_groundtruth_phase_config()


def _use_thinking(cfg: dict) -> bool:
    """Normalize the config `thinking` value to the bool run_batch expects.

    Mirrors tutorsim.scoring.score: a string like "adaptive"/"enabled" (or True)
    enables thinking; anything else (e.g. "disabled", False) disables it. Passing
    the raw config string straight through would make ANY non-empty string truthy.
    """
    return cfg.get("thinking", False) in ("adaptive", "enabled", True)


def load_from_jsonl(path, annotation_types=("scaffolding", "rapport")):
    """Load annotations from a tutoring_provider_a_annotations.jsonl file.

    Returns list of (conv_id, conv_data) sorted by conv_id, where conv_data is:
      {"annotations": [...], "num_turns": <int>}

    Only records whose annotation_type is in `annotation_types` are kept (caption
    records are always skipped). Pass annotation_types=("scaffolding",) for a
    scaffolding-only build; conversations left with no matching records are omitted
    entirely. turn_number_start/end are mapped to turn_start/end. annotator_id and
    annotation_type are promoted from the record level to each moment. num_turns is
    the max turn_end seen across all kept moments for that conversation.
    """
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("annotation_type") not in annotation_types:
                continue
            conv_id = record["transcript_id"]
            annotator_id = record.get("annotator_id", "")
            annotation_type = record["annotation_type"]
            for ta in record.get("turn_annotations", []):
                if (
                    ta.get("turn_number_start") is None
                    or ta.get("turn_number_end") is None
                ):
                    continue
                entry = {
                    "annotator_id": annotator_id,
                    "turn_start": ta["turn_number_start"],
                    "turn_end": ta["turn_number_end"],
                    "annotation_type": annotation_type,
                    "situation": ta.get("situation", ""),
                    "action": ta.get("action", ""),
                    "result": ta.get("result", ""),
                    "_timestamp": ta.get("annotation_timestamp", ""),
                }
                if "cut_turn" in ta:
                    entry["cut_turn"] = ta["cut_turn"]
                if "moment_id" in ta:
                    entry["moment_id"] = ta["moment_id"]
                groups[conv_id].append(entry)

    result = []
    for conv_id, annotations in sorted(groups.items()):
        annotations.sort(key=lambda a: a["_timestamp"])
        for a in annotations:
            del a["_timestamp"]
        num_turns = max(
            (a["turn_end"] for a in annotations if a["turn_end"] is not None), default=0
        )
        result.append((conv_id, {"annotations": annotations, "num_turns": num_turns}))
    return result


def moment_key(m):
    """Stable key identifying an annotation moment across runs."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("result", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def situation_key(m):
    """Stable key for a situation label -- same as moment_key but hashes situation text."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("situation", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def load_existing_labels(gt_dir):
    """Return {conv_id: {moment_key: strategy_label}} from ground truth in gt_dir."""
    existing = {}
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return existing
    for f in gt_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        existing[f.stem] = {
            moment_key(m): m.get("strategy_label")
            for m in d.get("key_moments", [])
            if m.get("strategy_label")
        }
    return existing


def load_existing_situation_labels(gt_dir):
    """Return {conv_id: {situation_key: situation_label}} from ground truth in gt_dir."""
    existing = {}
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return existing
    for f in gt_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        existing[f.stem] = {
            situation_key(m): m["situation_label"]
            for m in d.get("key_moments", [])
            if m.get("annotation_type") == "scaffolding" and m.get("situation_label")
        }
    return existing


def action_decompose_key(m):
    """Stable key for caching action decomposition by content hash."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("action", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def result_decompose_key(m):
    """Stable key for caching result decomposition by content hash."""
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5((m.get("result", "") or "").encode("utf-8")).hexdigest()[:12],
    )


def overscaffold_decompose_key(m):
    """Stable key for caching over-scaffolding decomposition by content hash.

    The over-scaffold prompt reads situation + action + result, so the cache key
    hashes all three (joined) -- a change to any of them re-decomposes.
    """
    blob = "\x1f".join(
        [
            m.get("situation", "") or "",
            m.get("action", "") or "",
            m.get("result", "") or "",
        ]
    )
    return (
        m.get("annotator_id", ""),
        m.get("turn_start"),
        m.get("turn_end"),
        m.get("annotation_type", ""),
        hashlib.md5(blob.encode("utf-8")).hexdigest()[:12],
    )


def load_existing_overscaffold_decompositions(gt_dir):
    """Return {conv_id: {overscaffold_decompose_key: facets}} for scaffolding
    moments that already carry overscaffold_decomposed in ground truth.

    Kept separate from load_existing_decompositions so --refresh-overscaffold and
    --refresh-decomp invalidate independently.
    """
    existing = {}
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return existing
    for f in gt_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        cache = {
            overscaffold_decompose_key(m): m["overscaffold_decomposed"]
            for m in d.get("key_moments", [])
            if m.get("annotation_type") == "scaffolding"
            and "overscaffold_decomposed" in m
        }
        if cache:
            existing[f.stem] = cache
    return existing


def load_existing_decompositions(gt_dir):
    """Return {conv_id: {"action": {action_decompose_key: facets}, "result": {result_decompose_key: facets}}}."""
    existing = {}
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return existing
    for f in gt_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        action_cache = {}
        result_cache = {}
        for m in d.get("key_moments", []):
            if "action_decomposed" in m:
                action_cache[action_decompose_key(m)] = m["action_decomposed"]
            if "result_decomposed" in m:
                result_cache[result_decompose_key(m)] = m["result_decomposed"]
        if action_cache or result_cache:
            existing[f.stem] = {"action": action_cache, "result": result_cache}
    return existing


def load_existing_action_result_agg(gt_dir):
    """Return {conv_id: {cluster_signature: cached_entry}} for cache reuse, where
    cached_entry is {"unified_action", "unified_result", "action_direction_agg",
    "student_outcome_agg"} and cluster_signature = frozenset(moment_key(m) for m
    in the cluster).

    Re-derives clusters from each existing ground truth file's scaffolding moments
    via _scaffolding_clusters -- clustering is a pure function of turn ranges and
    annotator_id, so it reproduces identically run-to-run when the underlying
    moments are unchanged. Reconstructing the unified facet lists this way (rather
    than storing a hash) lets us detect drift: if re-decomposition changed a
    moment's action_decomposed/result_decomposed, the freshly unified list will
    differ from the cached one and the cluster gets reclassified.
    """
    existing = {}
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return existing
    for f in gt_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
        cache = {}
        for cluster_indices, cluster_moments in _scaffolding_clusters(
            d.get("key_moments", [])
        ):
            agg_a = next(
                (
                    m["action_direction_agg"]
                    for m in cluster_moments
                    if "action_direction_agg" in m
                ),
                None,
            )
            agg_r = next(
                (
                    m["student_outcome_agg"]
                    for m in cluster_moments
                    if "student_outcome_agg" in m
                ),
                None,
            )
            if agg_a is None and agg_r is None:
                continue
            unified_action, unified_result = _unify_facets(cluster_moments)
            sig = frozenset(moment_key(m) for m in cluster_moments)
            cache[sig] = {
                "unified_action": unified_action,
                "unified_result": unified_result,
                "action_direction_agg": agg_a,
                "student_outcome_agg": agg_r,
            }
        if cache:
            existing[f.stem] = cache
    return existing


def _invalidate_agg_cache(cache, field):
    """Return a copy of the agg cache with the targeted agg label(s) cleared so
    plan_action_result_agg reclassifies them while still reusing the other field.

    field:
      - "action": clear action_direction_agg (reclassify action, reuse result)
      - "result": clear student_outcome_agg (reclassify result, reuse action)
      - "both":   drop the whole cache (reclassify both)

    Clearing means setting the cached label to None, which makes
    plan_action_result_agg's `... is not None` reuse predicate fail for that
    field. The untouched field's cached label and unified-facet list remain, so
    it reuses normally when its facets are unchanged. The input cache is not
    mutated.
    """
    if field == "both":
        return {}
    key = "action_direction_agg" if field == "action" else "student_outcome_agg"
    return {
        conv_id: {sig: {**entry, key: None} for sig, entry in clusters.items()}
        for conv_id, clusters in cache.items()
    }


def _invalidate_decomp_cache(cache, field):
    """Return a copy of the decomposition cache with the targeted field's cached
    facets cleared so the main planning loop re-decomposes them instead of reusing.

    field:
      - "action": clear cached action facets (re-decompose action, reuse result)
      - "result": clear cached result facets (re-decompose result, reuse action)
      - "both":   drop the whole cache (re-decompose both)

    Re-decomposition cascades to the action/result aggregation automatically: when
    new facets differ from the cached unified-facet list, plan_action_result_agg
    already detects the drift and reclassifies; identical facets reuse the agg
    label. The input cache is not mutated.
    """
    if field == "both":
        return {}
    clear = "action" if field == "action" else "result"
    return {conv_id: {**sub, clear: {}} for conv_id, sub in cache.items()}


def decompose_batch(items):
    """Batch decompose action / result / over-scaffolding into atomic facets.

    items: list of {key, field ("action"|"result"|"overscaffold"), ...}:
      - action: {text}
      - result: {text} plus situation+action
      - overscaffold: {situation, action, result} (no "text"; the prompt reads
        all three, and junk-skipping is handled by _build_overscaffold_prompt)
    Returns {key: [facets]}.
    """
    if not items:
        return {}
    from tutorsim.client import ModelClient, build_batch_entry, run_batch

    cfg = _phase_cfg()
    client = ModelClient(cfg["model"])

    action_template = _load_decompose_prompt("decompose_action.md")
    result_template = _load_decompose_prompt("decompose_result.md")
    overscaffold_template = _load_decompose_prompt("decompose_overscaffold.md")

    entries = []
    results = {}
    for it in items:
        if it["field"] == "overscaffold":
            prompt = _build_overscaffold_prompt(
                it.get("situation", ""),
                it.get("action", ""),
                it.get("result", ""),
                overscaffold_template,
            )
            if prompt is None:  # both action and result are junk -- nothing to analyze
                results[it["key"]] = []
                continue
            entries.append(
                build_batch_entry(key=it["key"], prompt_text=prompt, json_mode=True)
            )
            continue
        text = (it["text"] or "").strip()
        if text.lower() in DECOMPOSE_JUNK_TEXTS:
            results[it["key"]] = []
            continue
        if it["field"] == "action":
            prompt = action_template.replace("{action}", text)
        else:
            prompt = (
                result_template.replace("{situation}", it.get("situation", ""))
                .replace("{action}", it.get("action", ""))
                .replace("{result}", text)
            )
        entries.append(
            build_batch_entry(key=it["key"], prompt_text=prompt, json_mode=True)
        )

    if not entries:
        return results

    print(
        f"  Submitting {len(entries)} decomposition requests to batch API "
        f"(model={cfg['model']})..."
    )
    raw = run_batch(
        client,
        entries,
        json_mode=True,
        display_name="decomposition",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=_use_thinking(cfg),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    for key, result in raw.items():
        if "error" in result or not result.get("text"):
            print(f"  WARNING: error for {key}: {result.get('error', 'no text')}")
            results[key] = []
            continue
        facets, had_error = _parse_decomposed(result["text"])
        if had_error:
            print(
                f"  WARNING: could not parse decomposition for {key}: {result['text'][:100]!r}"
            )
        results[key] = facets

    return results


def action_direction_classify_batch(items):
    """Batch classify unified action facet lists into scaffolding/rigor direction.

    Mirrors the structure pass's action classification (same prompt + parser), but
    runs once per scaffolding cluster on the cluster's unified action_decomposed
    facets rather than once per individual annotation.

    items: list of {key, facets}. Returns {key: action_label}.
    """
    if not items:
        return {}
    from tutorsim.client import ModelClient, build_batch_entry, run_batch

    cfg = _phase_cfg()
    client = ModelClient(cfg["model"])
    template = _load_structure_prompt("classify_action.md")

    entries = [
        build_batch_entry(
            key=it["key"],
            prompt_text=template.replace(
                "{action_list}", _format_facet_list(it["facets"])
            ),
            json_mode=True,
        )
        for it in items
    ]

    print(
        f"  Submitting {len(entries)} action direction classifications to batch API "
        f"(model={cfg['model']})..."
    )
    results = run_batch(
        client,
        entries,
        json_mode=True,
        display_name="action_direction_agg_classification",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=_use_thinking(cfg),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    labels = {}
    for key, result in results.items():
        if "error" in result or not result.get("text"):
            print(f"  WARNING: error for {key}: {result.get('error', 'no text')}")
            labels[key] = "unclear"
            continue
        label, had_error = _parse_action_label(result["text"])
        if had_error:
            print(
                f"  WARNING: could not parse action direction label for {key}: {result['text'][:100]!r}"
            )
        labels[key] = label

    return labels


def student_outcome_classify_batch(items):
    """Batch classify unified result facet lists into a student-outcome verdict.

    Mirrors the structure pass's result classification (same prompt + parser), but
    runs once per scaffolding cluster on the cluster's unified result_decomposed
    facets rather than once per individual annotation.

    items: list of {key, facets}. Returns {key: result_label str}.
    """
    if not items:
        return {}
    from tutorsim.client import ModelClient, build_batch_entry, run_batch

    cfg = _phase_cfg()
    client = ModelClient(cfg["model"])
    template = _load_structure_prompt("classify_student_result.md")

    entries = [
        build_batch_entry(
            key=it["key"],
            prompt_text=template.replace(
                "{student_list}", _format_facet_list(it["facets"])
            ),
            json_mode=False,
        )
        for it in items
    ]

    print(
        f"  Submitting {len(entries)} student outcome classifications to batch API "
        f"(model={cfg['model']})..."
    )
    results = run_batch(
        client,
        entries,
        json_mode=False,
        display_name="student_outcome_agg_classification",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=_use_thinking(cfg),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    labels = {}
    for key, result in results.items():
        if "error" in result or not result.get("text"):
            print(f"  WARNING: error for {key}: {result.get('error', 'no text')}")
            labels[key] = "unclear"
            continue
        result_label, had_error = _parse_result_label(result["text"])
        if had_error:
            print(
                f"  WARNING: could not parse student outcome label for {key}: {result['text'][:100]!r}"
            )
        labels[key] = result_label

    return labels


def situation_classify_batch(items):
    """Batch classify situation appropriateness for scaffolding moments.

    items: list of {key, situation}. Returns {key: {"scaffolding": ..., "rigor": ...}}.
    """
    if not items:
        return {}
    from tutorsim.client import ModelClient, build_batch_entry, run_batch

    cfg = _phase_cfg()
    client = ModelClient(cfg["model"])
    prompt_template = load_situation_prompt()

    entries = []
    labels = {}
    for it in items:
        situation = it["situation"]
        if (situation or "").strip().lower() in SIT_JUNK_TEXTS:
            labels[it["key"]] = {"scaffolding": "unclear", "rigor": "unclear"}
            continue
        prompt = prompt_template.replace("{situation}", situation)
        entries.append(
            build_batch_entry(key=it["key"], prompt_text=prompt, json_mode=True)
        )

    if not entries:
        return labels

    print(
        f"  Submitting {len(entries)} situation classifications to batch API "
        f"(model={cfg['model']})..."
    )
    results = run_batch(
        client,
        entries,
        json_mode=True,
        display_name="situation_classification",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=_use_thinking(cfg),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    for key, result in results.items():
        if "error" in result:
            print(f"  WARNING: error for {key}: {result['error']}")
            labels[key] = {"scaffolding": "unclear", "rigor": "unclear"}
            continue
        sit_label, had_error = _parse_situation_label(result["text"])
        if had_error:
            print(
                f"  WARNING: could not parse situation label for {key}: {result['text'][:100]!r}"
            )
        labels[key] = sit_label

    return labels


def classify_batch(items, labeller="hybrid"):
    """Run batch effectiveness classification. `items` is list of dicts with keys:
    key, annotation_type, situation, action, result_text. Returns {key: label}.

    labeller="hybrid" routes per annotation_type using the groundtruth.labeller
    dict in config. Any other value loads classify_{labeller}.txt as a single
    shared template (legacy behavior).
    """
    if not items:
        return {}
    from tutorsim.client import ModelClient, build_batch_entry, run_batch

    cfg = _phase_cfg()
    client = ModelClient(cfg["model"])

    if labeller == "hybrid":
        templates = load_labeller_templates(get_labeller_config())
    else:
        templates = {None: load_labeller_prompt(f"classify_{labeller}")}

    entries = []
    labels = {}
    for it in items:
        text = it["result_text"]
        stripped = (text or "").strip().lower()
        if stripped in JUNK_TEXTS:
            labels[it["key"]] = "unclear"
            continue
        annotation_type = it.get("annotation_type", "unknown")
        template = pick_template(templates, annotation_type)
        prompt = (
            template.replace("{annotation_type}", annotation_type)
            .replace("{situation}", it.get("situation", ""))
            .replace("{action}", it.get("action", ""))
            .replace("{result_text}", text)
        )
        entries.append(
            build_batch_entry(
                key=it["key"],
                prompt_text=prompt,
                json_mode=False,
            )
        )

    if not entries:
        return labels

    print(
        f"  Submitting {len(entries)} classifications to Anthropic batch API "
        f"(model={cfg['model']})..."
    )
    results = run_batch(
        client,
        entries,
        json_mode=False,
        display_name="effectiveness_classification_refresh",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=_use_thinking(cfg),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
    )

    for key, result in results.items():
        if "error" in result:
            print(f"  WARNING: error for {key}: {result['error']}")
            labels[key] = "unclear"
            continue
        label = result["text"].strip().lower().rstrip(".")
        labels[key] = label if label in VALID_LABELS else "unclear"

    return labels


def _normalize_sit(val):
    """Normalize unclear/None -> no_mention."""
    if val in ("unclear", None, ""):
        return "no_mention"
    return val  # "yes", "no", or "no_mention"


def _cluster_by_iou(moments, threshold=1.0):
    """Group moment indices into IoU-based connected-component clusters.

    Same-annotator pairs are not directly linked (but may be transitively grouped).
    Returns a list of lists of indices into `moments`.
    """
    n = len(moments)
    if n == 0:
        return []
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if moments[i].get("annotator_id") == moments[j].get("annotator_id"):
                continue
            iou = compute_iou(
                (moments[i]["turn_start"], moments[i]["turn_end"]),
                (moments[j]["turn_start"], moments[j]["turn_end"]),
            )
            if iou >= threshold:
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

    clusters = defaultdict(list)
    for i in range(n):
        clusters[_find(i)].append(i)
    return list(clusters.values())


def _scaffolding_clusters(moments, threshold=1.0):
    """Cluster scaffolding moments in `moments` by IoU >= threshold.

    Shared by situation_label_agg, action_direction_agg/student_outcome_agg
    computation, and cache reconstruction so all three agree on cluster membership.

    Returns a list of (cluster_indices, cluster_moments), where cluster_indices
    are indices into `moments` and cluster_moments are the corresponding dicts
    (in the same order).
    """
    scaf_idxs = [
        i for i, m in enumerate(moments) if m.get("annotation_type") == "scaffolding"
    ]
    if not scaf_idxs:
        return []
    scaf_moments = [moments[i] for i in scaf_idxs]
    clusters = _cluster_by_iou(scaf_moments, threshold=threshold)
    return [
        ([scaf_idxs[ci] for ci in cluster], [scaf_moments[ci] for ci in cluster])
        for cluster in clusters
    ]


def _unify_facets(cluster_moments):
    """Concatenate action_decomposed / result_decomposed across a cluster's moments.

    Keeps one contribution per annotator (first occurrence), mirroring the
    situation_label_agg dedup so an annotator with multiple moments transitively
    grouped into the same cluster isn't double-counted.

    Returns (unified_action_facets, unified_result_facets).
    """
    seen_ann = set()
    unified_action = []
    unified_result = []
    for m in cluster_moments:
        ann_id = m.get("annotator_id", "")
        if ann_id in seen_ann:
            continue
        seen_ann.add(ann_id)
        unified_action.extend(m.get("action_decomposed") or [])
        unified_result.extend(m.get("result_decomposed") or [])
    return unified_action, unified_result


def _majority_vote_tuple(tuples):
    """Return the majority (scaf, rigor) tuple, or None on a tie."""
    if not tuples:
        return None
    counts = Counter(tuples).most_common(2)
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]
    return None  # tie


_TUPLE_TO_AGG = {
    ("yes", "yes"): "both",
    ("yes", "no"): "scaffolding",
    ("no", "yes"): "rigor",
    ("no", "no"): "neither",
}


def compute_situation_label_agg(moments):
    """Return {idx: agg_label} for each scaffolding moment in `moments`.

    Groups overlapping scaffolding moments (IoU == 1.0, i.e. exact match) into clusters,
    majority-votes the (scaffolding, rigor) tuple (remapping no_mention/unclear -> no),
    and maps the winner to 'both'/'scaffolding'/'rigor'/'neither', 'mixed' for ties,
    or 'unknown' when every annotator in the cluster had both slots as no_mention.
    """
    result = {}
    for cluster_indices, cluster_moments in _scaffolding_clusters(moments):
        seen_ann = set()
        vote_tuples = []
        for m in cluster_moments:
            ann_id = m.get("annotator_id", "")
            if ann_id in seen_ann:
                continue
            seen_ann.add(ann_id)
            sl = m.get("situation_label") or {}
            scaf = _normalize_sit(sl.get("scaffolding"))
            rigor = _normalize_sit(sl.get("rigor"))
            if scaf == "no_mention" and rigor == "no_mention":
                continue  # annotator gave no signal -- exclude from vote
            scaf = "no" if scaf == "no_mention" else scaf
            rigor = "no" if rigor == "no_mention" else rigor
            vote_tuples.append((scaf, rigor))
        if not vote_tuples:
            label = "unknown"
        else:
            winner = _majority_vote_tuple(vote_tuples)
            label = "mixed" if winner is None else _TUPLE_TO_AGG.get(winner, "mixed")
        for idx in cluster_indices:
            result[idx] = label
    return result


def plan_action_result_agg(
    conv_id, moments, cached_agg, to_action_direction, to_student_outcome
):
    """Plan action_direction_agg / student_outcome_agg for one conversation's clusters.

    For each cluster of overlapping scaffolding moments (IoU >= 1.0), unifies the
    cluster's action_decomposed / result_decomposed facets (one contribution per
    annotator, via _unify_facets) and either reuses the cached agg label (same
    moment_key set and unchanged unified facets), queues the unified list for
    classification, or short-circuits to the documented default when the unified
    list is empty ("unknown" / DEFAULT_RESULT_LABEL).

    Returns a list of (cluster_indices, action_item, result_item) where each item
    is ("reuse", value) | ("default", value) | ("classify", batch_key).
    """
    cached_convo = cached_agg.get(conv_id, {})
    plan = []
    for cluster_indices, cluster_moments in _scaffolding_clusters(moments):
        unified_action, unified_result = _unify_facets(cluster_moments)
        sig = frozenset(moment_key(m) for m in cluster_moments)
        cached_entry = cached_convo.get(sig)
        tag = "_".join(str(i) for i in cluster_indices)

        if (
            cached_entry
            and cached_entry["action_direction_agg"] is not None
            and cached_entry["unified_action"] == unified_action
        ):
            action_item = ("reuse", cached_entry["action_direction_agg"])
        elif unified_action:
            akey = f"{conv_id}__{tag}__action_agg"
            to_action_direction.append({"key": akey, "facets": unified_action})
            action_item = ("classify", akey)
        else:
            action_item = ("default", "unknown")

        if (
            cached_entry
            and cached_entry["student_outcome_agg"] is not None
            and cached_entry["unified_result"] == unified_result
        ):
            result_item = ("reuse", cached_entry["student_outcome_agg"])
        elif unified_result:
            rkey = f"{conv_id}__{tag}__result_agg"
            to_student_outcome.append({"key": rkey, "facets": unified_result})
            result_item = ("classify", rkey)
        else:
            result_item = ("default", DEFAULT_RESULT_LABEL)

        plan.append((cluster_indices, action_item, result_item))
    return plan


def _merge_scaffolding_only(new_moments, existing_moments):
    """Combine freshly built scaffolding moments with non-scaffolding moments
    preserved from an existing ground truth file.

    Used by --scaffolding-only so a scaffolding-only rebuild doesn't drop rapport
    moments already on disk: the freshly built scaffolding moments replace ALL
    existing scaffolding moments (stale ones are dropped), while every existing
    moment whose annotation_type is not "scaffolding" is carried through unchanged.
    Returns new_moments followed by the preserved moments.
    """
    preserved = [
        m for m in existing_moments if m.get("annotation_type") != "scaffolding"
    ]
    return list(new_moments) + preserved


def build_moment(ann, label):
    moment = {
        "turn_start": ann.get("turn_start"),
        "turn_end": ann.get("turn_end"),
        "annotation_type": ann.get("annotation_type", ""),
        "annotator_id": ann.get("annotator_id", ""),
        "situation": ann.get("situation", ""),
        "action": ann.get("action", ""),
        "result": ann.get("result", ""),
        "strategy_label": label,
    }
    if "cut_turn" in ann:
        moment["cut_turn"] = ann["cut_turn"]
    if "moment_id" in ann:
        moment["moment_id"] = ann["moment_id"]
    return moment


def validate_ground_truth(
    ground_truth: dict, *, all_moments: tuple = (), scaffolding_only: tuple = ()
) -> None:
    """Assert required keys exist on ground-truth moments; raise if any are absent.

    A missing key here means the ground-truth input is corrupt for this run.
    Enforce it up front so it fails loudly rather than being silently defaulted
    deep in the pipeline where it quietly biases results.

    Args:
        ground_truth: {"conversations": {conv_id: {"key_moments": [...]}}}.
        all_moments: keys required on EVERY moment (e.g. strategy_label).
        scaffolding_only: keys required only on annotation_type == "scaffolding"
            moments (situation_label_agg / action_direction_agg / student_outcome_agg);
            rapport moments legitimately lack them.

    Raises:
        ValueError naming each missing key, how many moments lack it, and an
        example location.
    """
    missing_counts: dict = {}
    examples: dict = {}
    for conv_id, conv in ground_truth.get("conversations", {}).items():
        for m in conv.get("key_moments", []):
            required = list(all_moments)
            if m.get("annotation_type") == "scaffolding":
                required += list(scaffolding_only)
            for k in required:
                if k not in m:
                    missing_counts[k] = missing_counts.get(k, 0) + 1
                    examples.setdefault(k, (conv_id, m.get("turn_start")))
    if missing_counts:
        parts = [
            f"'{k}' missing on {n} moment(s) "
            f"(e.g. {examples[k][0]} turn {examples[k][1]})"
            for k, n in sorted(missing_counts.items())
        ]
        raise ValueError("Ground truth input invalid: " + "; ".join(parts))


def _consolidate_to_jsonl(out_dir: Path, jsonl_path: Path) -> int:
    """Write every <conv_id>.json in out_dir (sorted by conv_id) as one JSONL line.

    Emits the published-style consolidated ground-truth artifact in one shot so
    the build can produce <out_dir>.jsonl directly. Returns the line count.
    """
    out_dir = Path(out_dir)
    n = 0
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for f in sorted(out_dir.glob("*.json"), key=lambda p: p.stem):
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_ground_truth(
    *,
    input_path,
    out_dir,
    labeller="hybrid",
    dry_run=False,
    scaffolding_only=False,
    refresh_agg=None,
    refresh_decomp=None,
    refresh_overscaffold=False,
    consolidate=False,
):
    """Build ground-truth JSON from raw human annotations.

    Reads annotations from `input_path`, reuses cached enrichment from any existing
    per-conversation files in `out_dir`, classifies/decomposes the rest via the
    Anthropic batch API, and writes one <conv_id>.json per conversation into
    `out_dir`. When `consolidate` is set, also writes <out_dir>.jsonl.

    Returns a plan-summary dict (counts of moments reused vs classified, etc.) so
    callers/tests can inspect the plan without parsing stdout. On a dry run, no LLM
    calls are made and nothing is written; the summary still reflects the plan.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    annotation_types = (
        ("scaffolding",) if scaffolding_only else ("scaffolding", "rapport")
    )
    print(f"Loading annotations from {input_path}...")
    if scaffolding_only:
        print(
            "--scaffolding-only: skipping rapport records; existing rapport moments "
            "will be preserved on write"
        )
    conversations = load_from_jsonl(input_path, annotation_types=annotation_types)
    print(f"Loaded {len(conversations)} conversations")

    existing_labels = load_existing_labels(out_dir)
    print(f"Loaded existing strategy labels for {len(existing_labels)} conversations")
    existing_situation_labels = load_existing_situation_labels(out_dir)
    sit_cache_total = sum(len(v) for v in existing_situation_labels.values())
    print(
        f"Loaded existing situation labels for {len(existing_situation_labels)} conversations ({sit_cache_total} moments)"
    )
    existing_decompositions = load_existing_decompositions(out_dir)
    decomp_action_total = sum(
        len(v.get("action", {})) for v in existing_decompositions.values()
    )
    decomp_result_total = sum(
        len(v.get("result", {})) for v in existing_decompositions.values()
    )
    print(
        f"Loaded existing decompositions: {decomp_action_total} action, {decomp_result_total} result"
    )
    if refresh_decomp:
        existing_decompositions = _invalidate_decomp_cache(
            existing_decompositions, refresh_decomp
        )
        decomp_field_desc = {
            "both": "action and result",
            "action": "action only",
            "result": "result only",
        }[refresh_decomp]
        print(f"--refresh-decomp={refresh_decomp}: re-decomposing {decomp_field_desc}")
    existing_overscaffold = load_existing_overscaffold_decompositions(out_dir)
    overscaffold_cache_total = sum(len(v) for v in existing_overscaffold.values())
    print(
        f"Loaded existing over-scaffold decompositions for {len(existing_overscaffold)} conversations ({overscaffold_cache_total} moments)"
    )
    if refresh_overscaffold:
        existing_overscaffold = {}
        print(
            "--refresh-overscaffold: re-decomposing over-scaffolding for all scaffolding moments"
        )
    existing_action_result_agg = load_existing_action_result_agg(out_dir)
    if refresh_agg:
        existing_action_result_agg = _invalidate_agg_cache(
            existing_action_result_agg, refresh_agg
        )
        field_desc = {
            "both": "action and result",
            "action": "action only",
            "result": "result only",
        }[refresh_agg]
        print(
            f"--refresh-agg={refresh_agg}: reclassifying {field_desc} "
            f"aggregation(s) for scaffolding clusters"
        )
    else:
        agg_cache_total = sum(len(v) for v in existing_action_result_agg.values())
        print(
            f"Loaded existing action/result aggregations for {len(existing_action_result_agg)} conversations ({agg_cache_total} clusters)"
        )

    # First pass: build per-conv plan (reuse vs classify) for both strategy and situation labels
    conv_plans = []
    to_classify = []  # [{key, annotation_type, situation, action, result_text}]
    to_situation_classify = []  # [{key, situation}] -- scaffolding moments only
    to_decompose = []  # [{key, field, text}]
    situation_plans = {}  # {conv_id: [s_item, ...]} parallel to plan
    decompose_plans = {}  # {conv_id: [(action_item, result_item, overscaffold_item), ...]}
    skipped_dup_count = 0

    for conv_id, conv_data in conversations:
        annotations = conv_data.get("annotations", [])
        known = existing_labels.get(conv_id, {})
        known_s = existing_situation_labels.get(conv_id, {})

        known_decomp = existing_decompositions.get(conv_id, {})
        known_action_decomp = known_decomp.get("action", {})
        known_result_decomp = known_decomp.get("result", {})
        known_overscaffold_decomp = existing_overscaffold.get(conv_id, {})

        plan = []
        s_plan = []
        d_plan = []
        for idx, ann in enumerate(annotations):
            sit_text = (ann.get("situation") or "").strip()
            action_text = (ann.get("action") or "").strip()
            result_text = (ann.get("result") or "").strip()
            if sit_text and action_text and sit_text == action_text:
                skipped_dup_count += 1
                continue
            if action_text and result_text and action_text == result_text:
                skipped_dup_count += 1
                continue
            k = moment_key(ann)
            if k in known:
                plan.append(("reuse", ann, known[k]))
            else:
                ckey = f"{conv_id}__{idx}"
                to_classify.append(
                    {
                        "key": ckey,
                        "annotation_type": ann.get("annotation_type", "unknown"),
                        "situation": ann.get("situation", ""),
                        "action": ann.get("action", ""),
                        "result_text": ann.get("result", ""),
                    }
                )
                plan.append(("classify", ann, ckey))

            if ann.get("annotation_type") == "scaffolding":
                sk = situation_key(ann)
                if sk in known_s:
                    s_plan.append(("reuse", known_s[sk]))
                else:
                    skey = f"{conv_id}__{idx}__sit"
                    to_situation_classify.append(
                        {"key": skey, "situation": ann.get("situation", "")}
                    )
                    s_plan.append(("classify", skey))
            else:
                s_plan.append(None)

            ak = action_decompose_key(ann)
            if ak in known_action_decomp:
                action_item = ("reuse", known_action_decomp[ak])
            else:
                dkey = f"{conv_id}__{idx}__action"
                to_decompose.append(
                    {"key": dkey, "field": "action", "text": ann.get("action", "")}
                )
                action_item = ("classify", dkey)

            rk = result_decompose_key(ann)
            if rk in known_result_decomp:
                result_item = ("reuse", known_result_decomp[rk])
            else:
                dkey = f"{conv_id}__{idx}__result"
                to_decompose.append(
                    {
                        "key": dkey,
                        "field": "result",
                        "text": ann.get("result", ""),
                        "situation": ann.get("situation", ""),
                        "action": ann.get("action", ""),
                    }
                )
                result_item = ("classify", dkey)

            # Over-scaffolding decomposition: scaffolding moments only.
            if ann.get("annotation_type") == "scaffolding":
                ok = overscaffold_decompose_key(ann)
                if ok in known_overscaffold_decomp:
                    overscaffold_item = ("reuse", known_overscaffold_decomp[ok])
                else:
                    okey = f"{conv_id}__{idx}__overscaffold"
                    to_decompose.append(
                        {
                            "key": okey,
                            "field": "overscaffold",
                            "situation": ann.get("situation", ""),
                            "action": ann.get("action", ""),
                            "result": ann.get("result", ""),
                        }
                    )
                    overscaffold_item = ("classify", okey)
            else:
                overscaffold_item = None

            d_plan.append((action_item, result_item, overscaffold_item))

        conv_plans.append((conv_id, conv_data, plan))
        situation_plans[conv_id] = s_plan
        decompose_plans[conv_id] = d_plan

    total_moments = sum(len(p) for _, _, p in conv_plans)
    reused = sum(1 for _, _, p in conv_plans for kind, *_ in p if kind == "reuse")
    to_class = total_moments - reused
    new_convs = sum(1 for cid, _, _ in conv_plans if cid not in existing_labels)
    sit_reused = sum(
        1
        for sp in situation_plans.values()
        for item in sp
        if item and item[0] == "reuse"
    )
    decomp_action_reused = sum(
        1 for dp in decompose_plans.values() for a, _, _ in dp if a[0] == "reuse"
    )
    decomp_result_reused = sum(
        1 for dp in decompose_plans.values() for _, r, _ in dp if r[0] == "reuse"
    )
    decomp_overscaffold_reused = sum(
        1
        for dp in decompose_plans.values()
        for _, _, o in dp
        if o is not None and o[0] == "reuse"
    )
    new_overscaffold = sum(1 for it in to_decompose if it["field"] == "overscaffold")

    print(f"Plan: {len(conv_plans)} conversations, {total_moments} moments")
    print(f"  Skipped (situation==action or action==result): {skipped_dup_count}")
    print(f"  Reuse existing strategy labels:   {reused}")
    print(f"  Classify new strategy labels:     {to_class}")
    print(f"  Reuse existing situation labels:  {sit_reused}")
    print(f"  Classify new situation labels:    {len(to_situation_classify)}")
    print(f"  Reuse existing action decomps:    {decomp_action_reused}")
    print(f"  Reuse existing result decomps:    {decomp_result_reused}")
    print(f"  Reuse existing overscaffold decomps: {decomp_overscaffold_reused}")
    print(
        f"  New decompositions (action+result+overscaffold): {len(to_decompose)} "
        f"({new_overscaffold} overscaffold)"
    )
    print(f"  Brand new conversations:          {new_convs}")

    summary = {
        "conversations": len(conv_plans),
        "total_moments": total_moments,
        "skipped_dup": skipped_dup_count,
        "reuse_strategy": reused,
        "classify_strategy": to_class,
        "reuse_situation": sit_reused,
        "classify_situation": len(to_situation_classify),
        "reuse_action_decomp": decomp_action_reused,
        "reuse_result_decomp": decomp_result_reused,
        "reuse_overscaffold_decomp": decomp_overscaffold_reused,
        "new_decompositions": len(to_decompose),
        "new_overscaffold": new_overscaffold,
        "new_conversations": new_convs,
        "dry_run": dry_run,
    }

    # Second pass: batch classify strategy labels, situation labels, decompositions.
    # Dry runs skip LLM calls; anything that would be classified falls back to a
    # placeholder so the aggregation plan can still be assembled and previewed.
    if dry_run:
        new_labels, new_situation_labels, new_decompositions = {}, {}, {}
    else:
        new_labels = classify_batch(to_classify, labeller=labeller)
        new_situation_labels = situation_classify_batch(to_situation_classify)
        new_decompositions = decompose_batch(to_decompose)

    def _resolve_situation(s_item):
        if s_item is None:
            return None
        if s_item[0] == "reuse":
            return s_item[1]
        return new_situation_labels.get(
            s_item[1], {"scaffolding": "unclear", "rigor": "unclear"}
        )

    # Third pass: build moments (everything except action_direction_agg / student_outcome_agg)
    conv_moments = {}  # {conv_id: (conv_data, moments)}
    for conv_id, conv_data, plan in conv_plans:
        s_plan = situation_plans[conv_id]
        d_plan = decompose_plans[conv_id]
        moments = []
        for (kind, ann, val), s_item, (
            action_item,
            result_item,
            overscaffold_item,
        ) in zip(plan, s_plan, d_plan):
            label = val if kind == "reuse" else new_labels.get(val, "unclear")
            moment = build_moment(ann, label)
            sit = _resolve_situation(s_item)
            if sit is not None:
                moment["situation_label"] = sit
            moment["action_decomposed"] = (
                action_item[1]
                if action_item[0] == "reuse"
                else new_decompositions.get(action_item[1], [])
            )
            moment["result_decomposed"] = (
                result_item[1]
                if result_item[0] == "reuse"
                else new_decompositions.get(result_item[1], [])
            )
            if overscaffold_item is not None:  # scaffolding moments only
                moment["overscaffold_decomposed"] = (
                    overscaffold_item[1]
                    if overscaffold_item[0] == "reuse"
                    else new_decompositions.get(overscaffold_item[1], [])
                )
            moments.append(moment)
        agg = compute_situation_label_agg(moments)
        for idx, agg_label in agg.items():
            moments[idx]["situation_label_agg"] = agg_label
        conv_moments[conv_id] = (conv_data, moments)

    # Fourth pass: plan action_direction_agg / student_outcome_agg per scaffolding cluster
    to_action_direction = []  # [{key, facets}]
    to_student_outcome = []  # [{key, facets}]
    agg_plans = {
        conv_id: plan_action_result_agg(
            conv_id,
            moments,
            existing_action_result_agg,
            to_action_direction,
            to_student_outcome,
        )
        for conv_id, (_, moments) in conv_moments.items()
    }
    n_clusters = sum(len(ap) for ap in agg_plans.values())
    action_kinds = Counter(a[0] for ap in agg_plans.values() for _, a, _ in ap)
    result_kinds = Counter(r[0] for ap in agg_plans.values() for _, _, r in ap)
    print(f"\nAction/result aggregation plan: {n_clusters} scaffolding clusters")
    print(f"  Reuse cached action direction labels:   {action_kinds['reuse']}")
    print(f"  Default action direction (no facets):   {action_kinds['default']}")
    print(f"  Classify new action direction labels:   {action_kinds['classify']}")
    print(f"  Reuse cached student outcome labels:    {result_kinds['reuse']}")
    print(f"  Default student outcome (no facets):    {result_kinds['default']}")
    print(f"  Classify new student outcome labels:    {result_kinds['classify']}")
    summary.update(
        {
            "scaffolding_clusters": n_clusters,
            "reuse_action_agg": action_kinds["reuse"],
            "default_action_agg": action_kinds["default"],
            "classify_action_agg": action_kinds["classify"],
            "reuse_result_agg": result_kinds["reuse"],
            "default_result_agg": result_kinds["default"],
            "classify_result_agg": result_kinds["classify"],
        }
    )
    pending_ar = sum(1 for it in to_decompose if it["field"] in ("action", "result"))
    if dry_run and pending_ar:
        print(
            f"  NOTE: {pending_ar} action/result decompositions are pending -- "
            f"clusters touching them used placeholder (empty) facets above, so their "
            f"reuse/classify counts are estimates and may change once decomposition runs"
        )

    if dry_run:
        print("\nDry run -- exiting without classifying or writing.")
        return summary

    new_action_direction_labels = action_direction_classify_batch(to_action_direction)
    new_student_outcome_labels = student_outcome_classify_batch(to_student_outcome)

    # Fifth pass: assign action_direction_agg / student_outcome_agg and write files
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_written = 0
    for conv_id, (conv_data, moments) in conv_moments.items():
        for cluster_indices, action_item, result_item in agg_plans[conv_id]:
            action_label = (
                action_item[1]
                if action_item[0] != "classify"
                else new_action_direction_labels.get(action_item[1], "neither")
            )
            result_label = (
                result_item[1]
                if result_item[0] != "classify"
                else new_student_outcome_labels.get(
                    result_item[1], DEFAULT_RESULT_LABEL
                )
            )
            for idx in cluster_indices:
                moments[idx]["action_direction_agg"] = action_label
                moments[idx]["student_outcome_agg"] = result_label

        gt_path = out_dir / f"{conv_id}.json"
        num_turns = conv_data.get("num_turns", 0)
        if scaffolding_only and gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as fp:
                existing = json.load(fp)
            moments = _merge_scaffolding_only(moments, existing.get("key_moments", []))
            num_turns = max(
                [num_turns, existing.get("num_turns", 0)]
                + [m["turn_end"] for m in moments if m.get("turn_end") is not None]
            )

        out = {
            "conversation_id": conv_id,
            "num_turns": num_turns,
            "key_moments": moments,
        }
        with open(gt_path, "w", encoding="utf-8") as fp:
            json.dump(out, fp, indent=2, ensure_ascii=False)
        gt_written += 1

    print("\nDone!")
    print(f"  Ground truth files written: {gt_written}")
    print(f"  Total ground truth: {len(list(out_dir.glob('*.json')))}")
    summary["gt_written"] = gt_written

    if consolidate:
        jsonl_path = out_dir.parent / f"{out_dir.name}.jsonl"
        n = _consolidate_to_jsonl(out_dir, jsonl_path)
        print(f"  Consolidated {n} conversations -> {jsonl_path}")
        summary["consolidated"] = str(jsonl_path)
        summary["consolidated_count"] = n

    return summary
