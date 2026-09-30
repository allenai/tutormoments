"""Build the action-taxonomy release: facet-level A-M classifications.

Maintainer-only release construction (`tutormoments-build dataset
build-action-taxonomy`). Reads the released Hugging Face dataset at a pinned
revision, decomposes every scaffolding moment's `action_decomposed` list into
facets for two sources --

  human          ground_truth.jsonl (the human tutors)
  benchmark_520  benchmark_520.jsonl (the paper's 7 models x 2 prompts),
                 situation label joined from moments.jsonl `dimension`

-- filters and classifies them with the runtime's taxonomy
(`tutormoments.taxonomy`), and writes a new release config:

  action_taxonomy.jsonl          one row per facet (excluded facets included,
                                 with category null + excluded_reason)
  action_taxonomy.schema.json    the row schema (enforced before writing)
  action_taxonomy.manifest.json  inputs, classifier, counts, usage, hashes

This is a regenerated, clearly labelled version, not a recovery of the
paper's classified pools (100 moments per LM cell). The paper's LM cells are
recoverable from benchmark_520 by filtering `source_round` to s42, s43 and
topup. The frozen
analysis/working-paper-20260630/v1_action_taxonomy_distribution.csv and the
locked kl_divergence_table.tex stay the paper's canonical numbers.

Release decisions (Ryan, 2026-09-30): classifier from the runtime's pinned
`taxonomy` config block; human and LM statements classified in separate pools
(separate resume sidecars); `action_label` / `result_label` taken from the
release's own label fields on both sides (LM: annotation `action_label` /
`result_label`; human: `action_direction_agg` / `student_outcome_agg` -- the
same vocabularies). The runtime adapters are not reused for labels because
they read `ann.result` (the free-text result paragraph) as `result_label` and
find no label keys at all in ground_truth key moments; the runtime is left
unchanged so per-run classified.csv files stay as they are.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from tutormoments import taxonomy as tx
from tutormoments.moments import file_sha256, records_content_hash
from tutormoments.resources import resource_text
from tutormoments.usage import EMPTY_USAGE, add_usage

logger = logging.getLogger(__name__)

REPO_ID = "allenai/tutormoments-preview"
INPUT_FILES = ("ground_truth.jsonl", "benchmark_520.jsonl", "moments.jsonl")
# The dataset card at the same revision, updated into upload/ for the Hub PR.
CARD_FILENAME = "README.md"

RELEASE_NAME = "action_taxonomy"
RELEASE_FILENAME = "action_taxonomy.jsonl"
SCHEMA_FILENAME = "action_taxonomy.schema.json"
MANIFEST_FILENAME = "action_taxonomy.manifest.json"
DEFAULT_OUT = "data/action_taxonomy_release"

SOURCE_HUMAN = "human"
SOURCE_LM = "benchmark_520"
SOURCES = (SOURCE_HUMAN, SOURCE_LM)

# moments.id = "{set_name}:{scenario_id}"; the LM moment_id uses this form, as
# per-run classified.csv files do.
MOMENT_SET_PREFIX = "balanced_520:"

# A pinned input must be a full commit SHA: branch names float and the
# dataset has no tags.
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

# Dry-run cost estimate, calibrated on a real Opus 4.8 classifier usage log
# (2,140 statements, 43 calls: 2.81 prompt chars per input token, 14.0 output
# tokens per statement). An estimate for the go/no-go decision only; the
# manifest records actual usage.
_CHARS_PER_INPUT_TOKEN = 2.8
_OUTPUT_TOKENS_PER_STATEMENT = 14


@dataclass
class ReleaseFacet(tx.Facet):
    """A taxonomy Facet plus its position in the source ground_truth row.

    `moment_id` follows the runtime (`{conversation_id}__{ts}_{te}__{annotator}`),
    under which some ground_truth key-moment records share an id;
    `key_moment_index` keeps every human row unique and traceable. None for LM.
    """

    key_moment_index: Optional[int] = None


@dataclass
class Pool:
    kept: list[ReleaseFacet] = field(default_factory=list)
    excluded: list[tuple[ReleaseFacet, str]] = field(default_factory=list)

    def unique_statements(self) -> set[str]:
        return {f.statement.strip() for f in self.kept if f.statement.strip()}


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def validate_revision(revision: str) -> str:
    if not _REVISION_RE.match(revision or ""):
        raise ValueError(
            f"--revision must be a full 40-character commit SHA, got {revision!r}"
        )
    return revision


def download_inputs(revision: str) -> dict[str, Path]:
    """Fetch the inputs and dataset card at `revision` (cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download

    validate_revision(revision)
    return {
        name: Path(
            hf_hub_download(REPO_ID, name, repo_type="dataset", revision=revision)
        )
        for name in (*INPUT_FILES, CARD_FILENAME)
    }


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def human_facets(ground_truth_path: Path) -> Iterator[ReleaseFacet]:
    """Facets from ground_truth.jsonl.

    Mirrors `taxonomy.load_key_moments_jsonl` (same filtering, ids, turns and
    situation label) except for the label mapping: `action_label` /
    `result_label` come from `action_direction_agg` / `student_outcome_agg`.
    A test pins the equivalence on every other field.
    """
    for row in _iter_jsonl(ground_truth_path):
        conv_id = row["conversation_id"]
        for kmi, km in enumerate(row.get("key_moments") or []):
            if km.get("annotation_type") != "scaffolding":
                continue
            statements = km.get("action_decomposed") or []
            if not statements:
                continue
            annotator = km.get("annotator_id") or "unknown"
            ts, te = km["turn_start"], km["turn_end"]
            for i, stmt in enumerate(statements):
                if not isinstance(stmt, str):
                    continue
                yield ReleaseFacet(
                    moment_id=f"{conv_id}__{ts}_{te}__{annotator}",
                    transcript_id=conv_id,
                    turn_start=ts,
                    turn_end=te,
                    statement_index=i,
                    statement=stmt,
                    annotation_type="scaffolding",
                    situation_label=km.get("situation_label_agg") or "unknown",
                    action_label=km.get("action_direction_agg"),
                    result_label=km.get("student_outcome_agg"),
                    source=SOURCE_HUMAN,
                    key_moment_index=kmi,
                )


def moment_dimensions(moments_path: Path) -> dict[str, str]:
    """moments.id -> dimension (the gold situation: scaffolding | rigor)."""
    return {m["id"]: m["dimension"] for m in _iter_jsonl(moments_path)}


def lm_facets(
    benchmark_path: Path, dimensions: dict[str, str]
) -> Iterator[ReleaseFacet]:
    """Facets from benchmark_520.jsonl, one replay per row.

    Mirrors `taxonomy.facets_from_annotations` (moment_id, transcript_id,
    turns, situation from the moment's `dimension`) except that
    `result_label` is the annotation's `result_label`, not its free-text
    `result`. Every row must join to moments.jsonl and carry exactly one
    annotation; anything else is a malformed input and raises.
    """
    for row in _iter_jsonl(benchmark_path):
        moment_id = MOMENT_SET_PREFIX + row["scenario_id"]
        if moment_id not in dimensions:
            raise ValueError(
                f"benchmark_520 row {moment_id!r} has no moments.jsonl row"
            )
        anns = row["annotation"]["annotations"]
        if len(anns) != 1:
            raise ValueError(
                f"benchmark_520 row {row['cell']} {moment_id!r} has "
                f"{len(anns)} annotations; expected exactly 1"
            )
        ann = anns[0]
        if ann.get("annotation_type") != "scaffolding":
            continue
        for i, stmt in enumerate(ann.get("action_decomposed") or []):
            if not isinstance(stmt, str):
                continue
            yield ReleaseFacet(
                moment_id=moment_id,
                transcript_id=tx._transcript_from_scenario(moment_id),
                turn_start=ann.get("turn_start", 0),
                turn_end=ann.get("turn_end", 0),
                statement_index=i,
                statement=stmt,
                annotation_type="scaffolding",
                situation_label=dimensions[moment_id],
                action_label=ann.get("action_label"),
                result_label=ann.get("result_label"),
                model=row["tutor_model"],
                prompt=row["prompt_mode"],
                source=SOURCE_LM,
            )


def build_pools(inputs: dict[str, Path]) -> dict[str, Pool]:
    """Filter each source into kept + excluded (with reason)."""
    dims = moment_dimensions(inputs["moments.jsonl"])
    facets = {
        SOURCE_HUMAN: human_facets(inputs["ground_truth.jsonl"]),
        SOURCE_LM: lm_facets(inputs["benchmark_520.jsonl"], dims),
    }
    pools = {}
    for source in SOURCES:
        kept, excluded = tx.build_pool(facets[source])
        pools[source] = Pool(kept=kept, excluded=excluded)
    return pools


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def pool_dir(out_dir: Path, source: str) -> Path:
    """Per-source resume sidecars (assignments.jsonl, usage_log.jsonl)."""
    return Path(out_dir) / "classify" / source


def classify_pools(
    pools: dict[str, Pool],
    out_dir: Path,
    spec: Any,
    *,
    max_batches: Optional[int] = None,
    client: Any = None,
) -> dict[str, dict[str, str]]:
    """Classify each source's kept statements in its own pool.

    Resume-safe via each pool's assignments.jsonl. `max_batches` caps batches
    per pool this invocation; otherwise every pool runs to completion (the
    runtime's first-run probe stop is disabled).
    """
    out: dict[str, dict[str, str]] = {}
    for source in SOURCES:
        out[source] = tx.classify_pool(
            pools[source].kept,
            pool_dir(out_dir, source),
            max_batches=max_batches,
            first_run_probe=10**9,
            client=client,
            model=spec.model,
            thinking=spec.thinking,
            batch_size=spec.batch_size,
        )
    return out


def missing_statements(pool: Pool, assignments: dict[str, str]) -> set[str]:
    return {s for s in pool.unique_statements() if s not in assignments}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _row(f: ReleaseFacet, excluded_reason: Optional[str]) -> dict:
    cat = f.category
    return {
        "source": f.source,
        "tutor_model": f.model,
        "prompt_mode": f.prompt,
        "moment_id": f.moment_id,
        "key_moment_index": f.key_moment_index,
        "transcript_id": f.transcript_id,
        "turn_start": f.turn_start,
        "turn_end": f.turn_end,
        "statement_index": f.statement_index,
        "statement": f.statement,
        "situation_label": f.situation_label,
        "action_label": f.action_label,
        "result_label": f.result_label,
        "stance_prefixed": f.stance_prefixed,
        "category": cat,
        "category_name": tx.NAME_BY_LETTER[cat] if cat else None,
        "orientation": tx.ORIENTATION_BY_LETTER[cat] if cat else None,
        "excluded_reason": excluded_reason,
    }


def _row_sort_key(r: dict) -> tuple:
    kmi = r["key_moment_index"]
    return (
        SOURCES.index(r["source"]),
        r["tutor_model"] or "",
        r["prompt_mode"] or "",
        r["moment_id"],
        -1 if kmi is None else kmi,
        r["statement_index"],
    )


def assemble_rows(
    pools: dict[str, Pool], assignments: dict[str, dict[str, str]]
) -> list[dict]:
    """Every facet as a release row, in a stable order.

    Raises KeyError if any kept statement is unclassified: the release file is
    only ever written from complete pools.
    """
    rows = []
    for source in SOURCES:
        pool = pools[source]
        for f in tx.attach_categories(pool.kept, assignments[source]):
            rows.append(_row(f, None))
        for f, reason in pool.excluded:
            f.category = None
            rows.append(_row(f, reason))
    rows.sort(key=_row_sort_key)
    return rows


def _shipped_schema_text() -> str:
    from importlib.resources import files

    return (files("tutormoments_build") / SCHEMA_FILENAME).read_text(encoding="utf-8")


def _shipped_schema() -> dict:
    return json.loads(_shipped_schema_text())


def validate_rows(rows: Iterable[dict]) -> None:
    """jsonschema-validate rows against the shipped schema; raise on the first bad one."""
    import jsonschema

    validator = jsonschema.Draft202012Validator(_shipped_schema())
    for i, row in enumerate(rows):
        errors = sorted(validator.iter_errors(row), key=str)
        if errors:
            e = errors[0]
            path = "/".join(str(p) for p in e.absolute_path) or "<row>"
            raise ValueError(
                f"Row {i} ({row.get('moment_id', '?')}) violates {SCHEMA_FILENAME} "
                f"at {path}: {e.message}"
            )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pool_usage(out_dir: Path, source: str) -> dict:
    """Sum a pool's usage_log.jsonl (every call across every invocation)."""
    usage = dict(EMPTY_USAGE)
    path = pool_dir(out_dir, source) / "usage_log.jsonl"
    if path.exists():
        for rec in _iter_jsonl(path):
            add_usage(usage, rec)
    return usage


def _moment_key(model: Optional[str], prompt: Optional[str], moment_id: str) -> tuple:
    """A moment is a moment id within a cell: each LM replay counts once."""
    return (model, prompt, moment_id)


def _count_block(rows: list[dict]) -> dict:
    kept = [r for r in rows if r["category"] is not None]

    def moments(rs):
        return {
            _moment_key(r["tutor_model"], r["prompt_mode"], r["moment_id"]) for r in rs
        }

    return {
        "facets": len(rows),
        "kept": len(kept),
        "excluded": len(rows) - len(kept),
        "moments": len(moments(rows)),
        "moments_with_kept": len(moments(kept)),
        "unique_kept_statements": len({r["statement"].strip() for r in kept}),
    }


def compute_counts(rows: list[dict]) -> dict:
    """Counts per source, per LM cell and per situation, derived from the rows."""
    out: dict[str, Any] = {}
    for source in SOURCES:
        src = [r for r in rows if r["source"] == source]
        block = _count_block(src)
        block["excluded_by_reason"] = {
            reason: sum(1 for r in src if r["excluded_reason"] == reason)
            for reason in tx.STRIP_REASONS
        }
        block["by_situation"] = {
            sit: _count_block([r for r in src if r["situation_label"] == sit])
            for sit in sorted({r["situation_label"] for r in src})
        }
        block["by_category"] = {
            letter: sum(1 for r in src if r["category"] == letter)
            for letter in tx.CATEGORY_LETTERS
        }
        if source == SOURCE_LM:
            cells = sorted({(r["tutor_model"], r["prompt_mode"]) for r in src})
            block["by_cell"] = {
                f"{m}__{p}": _count_block(
                    [r for r in src if (r["tutor_model"], r["prompt_mode"]) == (m, p)]
                )
                for m, p in cells
            }
        out[source] = block
    return out


def write_release(
    rows: list[dict],
    out_dir: Path,
    *,
    inputs: dict[str, Path],
    revision: str,
    spec: Any,
    version: str,
    created: str,
) -> dict:
    """Validate and write the three release files into `out_dir/upload/`."""
    from tutormoments import costing
    from tutormoments.models import pricing_version

    validate_rows(rows)
    upload = Path(out_dir) / "upload"
    upload.mkdir(parents=True, exist_ok=True)

    jsonl_path = upload / RELEASE_FILENAME
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    usage = {s: pool_usage(out_dir, s) for s in SOURCES}
    costs = {s: costing.role_cost_usd(usage[s]) for s in SOURCES}
    total_cost = (
        sum(costs.values()) if all(c is not None for c in costs.values()) else None
    )

    manifest = {
        "name": RELEASE_NAME,
        "version": version,
        "scheme_version": tx.SCHEME_VERSION,
        "record_count": len(rows),
        "content_hash": records_content_hash(rows),
        "file_sha256": file_sha256(jsonl_path),
        "inputs": {
            "repo_id": REPO_ID,
            "revision": revision,
            "file_sha256": {name: file_sha256(inputs[name]) for name in INPUT_FILES},
        },
        "classifier": {
            "model": spec.model,
            "thinking": spec.thinking,
            "batch_size": spec.batch_size,
            "pools": "separate per source (human, benchmark_520)",
            "prompt_resource": tx.CLASSIFY_PROMPT_RESOURCE,
            "prompt_sha256": _sha256_text(resource_text(tx.CLASSIFY_PROMPT_RESOURCE)),
            "categories_sha256": _sha256_text(tx._categories_block()),
        },
        "counts": compute_counts(rows),
        "usage": usage,
        "cost_usd": {**costs, "total": total_cost},
        "pricing_version": pricing_version(),
        "created": created,
    }
    with (upload / MANIFEST_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    (upload / SCHEMA_FILENAME).write_text(_shipped_schema_text(), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------


def _replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise ValueError(
            f"dataset card anchor found {n} times (expected 1): {old[:60]!r}"
        )
    return text.replace(old, new)


def update_dataset_card(card: str, manifest: dict) -> str:
    """Add the action_taxonomy config, subsets row, schema mention, section
    and source attribution to the dataset card. Anchors are exact strings of
    the card at the pinned revision; a changed card raises rather than
    silently producing a half-edited one."""
    if "config_name: action_taxonomy" in card:
        raise ValueError("dataset card already has an action_taxonomy config")
    counts = manifest["counts"]
    human, lm = counts[SOURCE_HUMAN], counts[SOURCE_LM]
    rev = manifest["inputs"]["revision"]
    clf = manifest["classifier"]
    thinking = json.dumps(clf["thinking"], separators=(", ", ": "))

    card = _replace_once(
        card,
        "        path: benchmark_520.jsonl\n---\n",
        "        path: benchmark_520.jsonl\n"
        f"  - config_name: {RELEASE_NAME}\n"
        "    data_files:\n"
        "      - split: train\n"
        f"        path: {RELEASE_FILENAME}\n"
        "---\n",
    )
    card = _replace_once(
        card,
        "(520 moments × 7 models × 2 prompts) |\n",
        "(520 moments × 7 models × 2 prompts) |\n"
        f"| `{RELEASE_NAME}` | {manifest['record_count']:,} | one decomposed tutor "
        f"action classified A–M ({human['facets']:,} human / {lm['facets']:,} AI "
        "tutor; model-generated labels) |\n",
    )
    card = _replace_once(
        card,
        "`moments.schema.json`,\n`benchmark_520.schema.json`)",
        f"`moments.schema.json`,\n`benchmark_520.schema.json`, `{SCHEMA_FILENAME}`)",
    )
    section = f"""## {RELEASE_NAME}

The tutor actions of every scaffolding moment, classified into the 13-category
A–M action taxonomy (scheme `{manifest["scheme_version"]}`): one row per
`action_decomposed` item, for the human tutors (`ground_truth`,
{human["facets"]:,} rows over {human["moments"]:,} moments) and the AI tutors
(`benchmark_520`, {lm["facets"]:,} rows over {lm["moments"]:,} replays). The
categories are **model-generated**: `{clf["model"]}` with provider-native parameters
`{thinking}`, batches of {clf["batch_size"]}, human and AI statements
classified in separate pools. Facets removed by the pre-classification filter
(statements whose subject is not the tutor, bare stances, and descriptions of
what the tutor did *not* do) are kept as rows with `category: null` and an
`excluded_reason`; distributions use the classified rows only, as a macro mean
over moments. `moment_id` joins AI rows to `moments.id`; human rows join to
`ground_truth` by `transcript_id` (= `conversation_id`) and `key_moment_index`.

These labels were **regenerated in October 2026** from this dataset at
revision `{rev}`; they are not the classification run behind the paper. The
paper classified a smaller sample — 100 moments per AI cell — so this config
does not exactly reproduce the paper's action-distribution figure or its
KL-divergence table. The paper's AI cells are recoverable from
`benchmark_520` by filtering `source_round` to `s42`, `s43` and `topup`
(100 moments per cell, 50 scaffolding / 50 rigor). The paper's canonical
numbers are
`analysis/working-paper-20260630/v1_action_taxonomy_distribution.csv` and
`kl_divergence_table.tex` in the code repository. `{MANIFEST_FILENAME}`
records the input revision and file hashes, the classifier configuration and
prompt hash, counts, token usage and cost. One `benchmark_520` replay has no
rows because its scoring failed in the source data (`action_decomposed` is
empty).

"""
    card = _replace_once(
        card, "## Deidentification\n", section + "## Deidentification\n"
    )
    card = _replace_once(
        card,
        "and GPT 5.5 and 5.4 mini (provided by OpenAI). ",
        "and GPT 5.5 and 5.4 mini (provided by OpenAI). The `action_taxonomy` "
        f"categories were generated with `{clf['model']}` (provided by Anthropic). ",
    )
    return card


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def estimate_cost(pending: Iterable[str], batch_size: int, model: str) -> dict:
    """Rough tokens and dollars to classify `pending` (no API calls)."""
    from tutormoments.costing import cost_usd
    from tutormoments.models import get_pricing

    pending = list(pending)
    n_batches = (len(pending) + batch_size - 1) // batch_size
    empty_prompt = tx._load_classify_prompt().substitute(
        categories_block=tx._categories_block(), statements_block=""
    )
    chars = n_batches * len(empty_prompt) + sum(len(s) + 5 for s in pending)
    tokens = {
        "input_uncached": round(chars / _CHARS_PER_INPUT_TOKEN),
        "output": len(pending) * _OUTPUT_TOKENS_PER_STATEMENT,
    }
    rates = get_pricing(model)
    return {
        "batches": n_batches,
        **tokens,
        "cost_usd": cost_usd(tokens, rates) if rates else None,
    }


def dry_run_report(
    pools: dict[str, Pool], out_dir: Path, spec: Any, revision: str
) -> str:
    lines = [
        f"Action-taxonomy release dry run @ {REPO_ID}:{revision}",
        f"classifier: {spec.model} thinking={spec.thinking} batch_size={spec.batch_size}",
        "",
    ]
    total_cost = 0.0
    for source in SOURCES:
        pool = pools[source]
        assigned = tx._read_assignments(pool_dir(out_dir, source) / "assignments.jsonl")
        unique = pool.unique_statements()
        pending = sorted(unique - assigned.keys())
        est = estimate_cost(pending, spec.batch_size, spec.model)
        total_cost += est["cost_usd"] or 0.0
        n_facets = len(pool.kept) + len(pool.excluded)
        kept_moments = {_moment_key(f.model, f.prompt, f.moment_id) for f in pool.kept}
        moments = kept_moments | {
            _moment_key(f.model, f.prompt, f.moment_id) for f, _ in pool.excluded
        }
        reasons = Counter(r for _, r in pool.excluded)
        lines += [
            f"[{source}]",
            f"  facets            {n_facets}",
            f"  moments           {len(moments)} "
            f"({len(kept_moments)} with a kept facet)",
            f"  kept / excluded   {len(pool.kept)} / {len(pool.excluded)}",
            "  excluded by reason "
            + ", ".join(f"{r}={reasons.get(r, 0)}" for r in tx.STRIP_REASONS),
            f"  unique statements {len(unique)} "
            f"({len(unique) - len(pending)} already assigned, {len(pending)} pending)",
            f"  batches pending   {est['batches']}",
            f"  est. tokens       in~{est['input_uncached']:,} out~{est['output']:,}",
            f"  est. cost         ~${est['cost_usd']:.2f}"
            if est["cost_usd"] is not None
            else "  est. cost         (model unpriced)",
            "",
        ]
    lines.append(
        f"estimated total: ~${total_cost:.2f} (rough; manifest records actuals)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_build(
    inputs: dict[str, Path],
    revision: str,
    out_dir: Path,
    *,
    dry_run: bool = False,
    max_batches: Optional[int] = None,
    version: str = "0",
    created: str = "",
    spec: Any = None,
    client: Any = None,
) -> Optional[dict]:
    """Pools -> (dry-run report | classify -> release). Returns the manifest,
    or None when dry-running or classification is still incomplete."""
    validate_revision(revision)
    if spec is None:
        from tutormoments.config import taxonomy_spec

        spec = taxonomy_spec()
    out_dir = Path(out_dir)
    pools = build_pools(inputs)

    if dry_run:
        print(dry_run_report(pools, out_dir, spec, revision))
        return None

    assignments = classify_pools(
        pools, out_dir, spec, max_batches=max_batches, client=client
    )
    missing = {s: missing_statements(pools[s], assignments[s]) for s in SOURCES}
    if any(missing.values()):
        logger.info(
            "classification incomplete (%s); re-run to resume",
            ", ".join(f"{s}: {len(m)} unassigned" for s, m in missing.items()),
        )
        return None

    rows = assemble_rows(pools, assignments)
    manifest = write_release(
        rows,
        out_dir,
        inputs=inputs,
        revision=revision,
        spec=spec,
        version=version,
        created=created,
    )
    if CARD_FILENAME in inputs:
        card = Path(inputs[CARD_FILENAME]).read_text(encoding="utf-8")
        (out_dir / "upload" / CARD_FILENAME).write_text(
            update_dataset_card(card, manifest), encoding="utf-8"
        )
    logger.info(
        "Wrote %d rows to %s (file_sha256 %s)",
        manifest["record_count"],
        out_dir / "upload" / RELEASE_FILENAME,
        manifest["file_sha256"],
    )
    return manifest


def _cli_build_action_taxonomy(args) -> None:
    """Dispatched from tutormoments_build.cli."""
    revision = validate_revision(args.revision)
    run_build(
        download_inputs(revision),
        revision,
        Path(args.out),
        dry_run=args.dry_run,
        max_batches=args.max_batches,
        version=args.version,
        created=args.created,
    )
