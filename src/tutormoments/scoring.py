"""Annotation dataclass and synthetic-conversation builder for tutormoments scoring.

Builds a synthetic conversation + detection per (scenario, transcript) pair,
then runs the 3-pass scoring pipeline (annotate -> decompose -> structure).

Annotate-pass helpers:
  - _format_excerpt: transcript excerpt with context_window=0 (no before/after context)
  - _suggestion_text: situation_label_agg -> suggestion sentence map
  - _build_annotate_entries: builds batch entries (5 substitutions, key scheme)
  - _parse_and_merge: parses batch results, accumulates usage, applies fallback

No module-level SDK imports.
"""

import json
import logging
import re as _re
from dataclasses import asdict, dataclass, field
from typing import Any

from tutormoments.resources import resource_text
from tutormoments.usage import EMPTY_USAGE, add_usage, sum_usage

logger = logging.getLogger(__name__)

_SCORER_PROMPTS_DIR = "prompts/scorer"

# Valid annotation types. The benchmark only uses "scaffolding" and "rapport".
VALID_ANNOTATION_TYPES = {"scaffolding", "rapport"}


# ---------------------------------------------------------------------------
# Annotate-pass: excerpt builder
# Benchmark forces context_window=0 (context_before=0, context_after=0).
# ---------------------------------------------------------------------------


def _format_excerpt(
    conversation: dict,
    turn_start: int,
    turn_end: int,
    context_before: int = 0,
    context_after: int = 0,
) -> str:
    """Extract a transcript excerpt around a detected moment, with context.

    Outputs the detected range with >>> markers, surrounded by context turns.
    The benchmark always calls with context_before=0, context_after=0.
    """
    turns = conversation.get("turns", [])
    if not turns:
        return ""

    # Find actual min/max turn numbers
    all_turn_nums = [t["turn_number"] for t in turns]
    min_turn = min(all_turn_nums)
    max_turn = max(all_turn_nums)

    # Calculate excerpt boundaries
    excerpt_start = max(min_turn, turn_start - context_before)
    excerpt_end = min(max_turn, turn_end + context_after)

    lines = []

    # Header if not starting from the beginning
    if excerpt_start > min_turn:
        lines.append(f"[... turns 1-{excerpt_start - 1} omitted ...]")
        lines.append("")

    marker_start_emitted = False
    for turn in turns:
        n = turn["turn_number"]
        if n < excerpt_start or n > excerpt_end:
            continue

        is_enrichment = turn.get("is_enrichment", False)

        # Emit start marker before the first dialogue turn at turn_start
        if n == turn_start and not is_enrichment and not marker_start_emitted:
            lines.append(f">>> DETECTED MOMENT START (Turn {turn_start}) <<<")
            marker_start_emitted = True

        role = turn["role"]
        text = turn["text"]
        if is_enrichment:
            lines.append(text)
        else:
            marker = " <<<" if turn_start <= n <= turn_end else ""
            lines.append(f"Turn {n}. {role}: {text}{marker}")

        # Emit end marker after the last dialogue turn at turn_end
        if n == turn_end and not is_enrichment:
            lines.append(f">>> DETECTED MOMENT END (Turn {turn_end}) <<<")

    # Footer if not ending at last turn
    if excerpt_end < max_turn:
        lines.append("")
        lines.append(f"[... turns {excerpt_end + 1}-{max_turn} omitted ...]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annotate-pass: suggestion text
# ---------------------------------------------------------------------------

_SITUATION_LABEL_AGG_TO_SUGGESTION = {
    "scaffolding": "A team of teachers believe that this moment is appropriate for scaffolding.",
    "rigor": "A team of teachers believe that this moment is appropriate for pushing for rigor.",
    "mixed": "A team of teachers believe that this moment is appropriate for either rigor or scaffolding.",
    "both": "A team of teachers believe that this moment is appropriate for either rigor or scaffolding.",
    "neither": "A team of teachers believe that this moment is not appropriate for either rigor or scaffolding.",
}
_SUGGESTION_UNKNOWN = "It's unclear to a team of teachers whether this moment is appropriate for rigor or scaffolding."


def _suggestion_text(situation_label_agg: str | None) -> str:
    """Map situation_label_agg to the suggestion sentence injected into the prompt."""
    return _SITUATION_LABEL_AGG_TO_SUGGESTION.get(
        situation_label_agg or "", _SUGGESTION_UNKNOWN
    )


# ---------------------------------------------------------------------------
# Annotate-pass: build batch entries
# Benchmark behavior:
#   - context_window=0 always (no before/after turns)
#   - annotator_style always injected as "" (style calibration via prompts, not text)
#   - No screenshot support needed here
# ---------------------------------------------------------------------------


def _load_annotate_prompt(ann_type: str) -> str:
    """Load the scorer annotate prompt for the given annotation type."""
    return resource_text(f"{_SCORER_PROMPTS_DIR}/annotate/{ann_type}.md")


def _build_annotate_entries(
    conversation_dict: dict,
    detections_by_conv: dict,
) -> list[dict]:
    """Build batch entries for the annotate pass.

    The benchmark always uses context_window=0 (no surrounding context turns).
    annotator_style is always "" (style controlled via prompt file selection).

    Key scheme: f"{conv_id}__{ann_type}__{idx}" where idx is 0-based over
    the detections list for that conversation.

    The 5 substitutions (in order, matching source):
      {annotator_style} -> ""
      {suggestion}      -> _suggestion_text(det["situation_label_agg"])
      {excerpt}         -> _format_excerpt(conversation, turn_start, turn_end, 0, 0)
      {turn_start}      -> str(turn_start)
      {turn_end}        -> str(turn_end)
    Any other {..} placeholders (e.g. {brief_description}) are left literal.

    Args:
        conversation_dict: {conv_id: conversation} mapping.
        detections_by_conv: {conv_id: {"detections": [...], "usage": {...}}} mapping.

    Returns:
        List of batch entry dicts (provider-neutral format from build_batch_entry).
    """
    from .client import build_batch_entry

    prompt_cache: dict[str, str] = {}
    entries: list[dict] = []

    for conv_id, conv_data in detections_by_conv.items():
        conversation = conversation_dict.get(conv_id)
        if not conversation:
            logger.warning("No transcript found for %s, skipping", conv_id)
            continue

        for idx, det in enumerate(conv_data.get("detections", [])):
            ann_type = det.get("annotation_type", "scaffolding")
            if ann_type not in VALID_ANNOTATION_TYPES:
                ann_type = "scaffolding"

            turn_start = det.get("turn_start", 0)
            turn_end = det.get("turn_end", turn_start)

            if ann_type not in prompt_cache:
                prompt_cache[ann_type] = _load_annotate_prompt(ann_type)

            excerpt = _format_excerpt(conversation, turn_start, turn_end, 0, 0)

            prompt = prompt_cache[ann_type]
            prompt = prompt.replace("{annotator_style}", "")
            prompt = prompt.replace(
                "{suggestion}", _suggestion_text(det.get("situation_label_agg"))
            )
            prompt = prompt.replace("{excerpt}", excerpt)
            prompt = prompt.replace("{turn_start}", str(turn_start))
            prompt = prompt.replace("{turn_end}", str(turn_end))

            key = f"{conv_id}__{ann_type}__{idx}"
            entries.append(build_batch_entry(key, prompt))

    return entries


# ---------------------------------------------------------------------------
# Annotate-pass: parse and merge
# ---------------------------------------------------------------------------


def _parse_and_merge(
    raw_entries: dict,
    detections_by_conv: dict,
) -> dict[str, dict]:
    """Parse batch results and merge with detections into final annotations.

    - JSON parses each result; if a list, takes [0].
    - Attaches _usage from the raw entry.
    - Merges situation/action/result with "" defaults on missing keys.
    - Uses fallback action text "[Analysis unavailable -- batch failed for this moment]"
      when no parsed result exists for a key.
    - Accumulates usage from p1 (detections_by_conv) + p2 (parsed results).

    Args:
        raw_entries: {key: {"text": str, "usage": dict}} or {"error": str}
        detections_by_conv: {conv_id: {"detections": [...], "usage": {...}}}

    Returns:
        {conv_id: {"conversation_id", "annotations", "usage",
                   "pass1_detections", "pass2_analyzed"}}
    """
    # Parse raw results
    analyses: dict[str, dict] = {}
    errors: list[dict] = []

    for key, data in raw_entries.items():
        if "error" in data:
            errors.append({"key": key, "error": data["error"]})
            continue

        text = data.get("text", "")
        if not text:
            errors.append({"key": key, "error": "Empty response"})
            continue

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}
            parsed["_usage"] = data.get("usage", {})
            analyses[key] = parsed
        except json.JSONDecodeError as e:
            errors.append(
                {"key": key, "error": f"JSON parse error: {e}", "raw": text[:500]}
            )

    if errors:
        logger.warning("Parse errors: %d", len(errors))
        for err in errors[:5]:
            logger.warning("  %s: %s", err["key"], err["error"])

    # Merge into final results
    results: dict[str, dict] = {}

    for conv_id, conv_data in detections_by_conv.items():
        detections = conv_data.get("detections", [])
        total_usage = sum_usage(conv_data.get("usage", {}))

        annotations = []
        for idx, det in enumerate(detections):
            ann_type = det.get("annotation_type", "scaffolding")
            key = f"{conv_id}__{ann_type}__{idx}"

            if key in analyses:
                a = analyses[key]
                annotations.append(
                    {
                        "annotation_type": ann_type,
                        "turn_start": det.get("turn_start"),
                        "turn_end": det.get("turn_end"),
                        "situation": a.get("situation", ""),
                        "action": a.get("action", ""),
                        "result": a.get("result", ""),
                    }
                )

                add_usage(total_usage, a.get("_usage", {}))
            else:
                annotations.append(
                    {
                        "annotation_type": ann_type,
                        "turn_start": det.get("turn_start", 0),
                        "turn_end": det.get("turn_end", 0),
                        "situation": "",
                        "action": "[Analysis unavailable -- batch failed for this moment]",
                        "result": "",
                    }
                )

        results[conv_id] = {
            "conversation_id": conv_id,
            "annotations": annotations,
            "usage": total_usage,
            "pass1_detections": len(detections),
            "pass2_analyzed": sum(
                1
                for i, d in enumerate(detections)
                if f"{conv_id}__{d.get('annotation_type', 'scaffolding')}__{i}"
                in analyses
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Annotation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Annotation:
    """A scored annotation produced by the 3-pass annotator pipeline.

    Fields mirror the annotator pipeline output, plus identifiers for the
    tutormoments scenario that generated the conversation being annotated.
    """

    scenario_id: str
    annotation_type: str  # "scaffolding" | "rapport"
    turn_start: int
    turn_end: int
    situation: str
    action: str
    result: str
    action_decomposed: list  # action decomposition phrases
    overscaffold_decomposed: list  # over-scaffolding decomposition phrases
    action_label: str  # "scaffolding" | "rigor" | "both" | "neither" | "unclear"
    usage: dict = field(default_factory=lambda: dict(EMPTY_USAGE))

    def to_dict(self) -> dict[str, Any]:
        """Return all fields as a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic conversation builder
# ---------------------------------------------------------------------------


def _build_synthetic_conversation(
    scenario: Any,
    transcript: Any,
) -> tuple[dict, dict]:
    """Build a full conversation dict and detections from a scenario + transcript.

    Ports annotator_bridge.build_synthetic_conversation (~lines 23-56) and
    annotator_bridge.build_synthetic_detections (~lines 59-112), plus the
    conv_id -> scenario_id remap from prepare_bulk_entries (~lines 159-166).

    Args:
        scenario: tutormoments.moments.Moment object.
        transcript: tutormoments.conversation.Transcript object.

    Returns:
        (conversation_dict, detections_by_id) where both are keyed by
        scenario.id (the remapped scenario_id, not the raw conv_id).
    """
    # ---- Build turns: context prefix + generated turns ----
    turns = []

    # Prefix turns from scenario.context: [{turn_number, role, text}]
    # Role stored lowercase in new schema; annotator pipeline expects uppercase.
    for ctx_turn in scenario.context:
        turns.append(
            {
                "turn_number": ctx_turn["turn_number"],
                "role": ctx_turn["role"].upper(),
                "text": ctx_turn["text"],
                "type": "DIALOGUE",
                "timestamp": "",
            }
        )

    # Generated turns from transcript.generated_turns: already {turn_number, role, text}
    for gen_turn in transcript.generated_turns:
        turns.append(
            {
                "turn_number": gen_turn["turn_number"],
                "role": gen_turn["role"],
                "text": gen_turn["text"],
                "type": "DIALOGUE",
                "timestamp": "",
            }
        )

    # ---- Build conversation dict (remapped: conversation_id = scenario.id) ----
    # Original annotator_bridge used scenario.conv_id as conversation_id, then
    # prepare_bulk_entries remapped it to scenario.scenario_id. In the new schema
    # scenario.id is the scenario_id and scenario.provenance["conv_id"] is the
    # raw conv_id. We produce the post-remap form directly.
    conv = {
        "conversation_id": scenario.id,
        "turns": turns,
        "context": scenario.student.get("context", ""),
        "num_turns": len(turns),
    }
    conversation_dict = {scenario.id: conv}

    # ---- Build detections ----
    if not transcript.generated_turns:
        return conversation_dict, {}

    first_gen = transcript.generated_turns[0]["turn_number"]
    last_gen = transcript.generated_turns[-1]["turn_number"]

    # Benchmark always annotates the scaffolding lens for detected moments.
    ann_type = "scaffolding"

    cut_turn = scenario.provenance.get("cut_turn", "?")
    hint = scenario.rubric.get("hint", "")
    description = f"AI tutor continuation from cut at turn {cut_turn}: {hint}"

    # situation_label_agg from scenario.dimension (= rubric["gold"])
    situation_label_agg = scenario.dimension

    detections = [
        {
            "turn_start": first_gen,
            "turn_end": last_gen,
            "annotation_type": ann_type,
            "situation": description,
            "situation_label_agg": situation_label_agg,
        }
    ]

    detections_by_id = {
        scenario.id: {
            "detections": detections,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    }

    return conversation_dict, detections_by_id


# ---------------------------------------------------------------------------
# Decompose pass
#
# Two sub-passes:
#   - action decomposition (decompose_action.md, substitutes {action})
#   - overscaffold decomposition (decompose_overscaffold.md, scaffolding target ONLY,
#     substitutes {situation}/{action}/{result}; skipped when BOTH action AND result are junk)
# Result decomposition (decompose_result.md) is no longer run by the scorer
# (student-outcome labelling was dropped from the paper); the template and
# loader remain for tutormoments_build.groundtruth.
# ---------------------------------------------------------------------------

# Placeholder/test annotation text to skip rather than send to the model.
JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}

_DECOMPOSE_PROMPTS_DIR = f"{_SCORER_PROMPTS_DIR}/decompose"


def _load_decompose_prompt(filename: str) -> str:
    """Load a decompose prompt by filename from the scorer decompose directory."""
    return resource_text(f"{_DECOMPOSE_PROMPTS_DIR}/{filename}")


def _coerce_facets(parsed: object) -> list[str] | None:
    """Coerce a parsed JSON value into a list of facet strings.

    Returns the facet list, or None if the value is not a recognizable facet
    container (signals a parse failure to the caller).

    Handles two shapes:
      - a bare array (Gemini/Anthropic honor the prompt's requested format), and
      - an object, because OpenAI's response_format={"type": "json_object"}
        cannot emit a top-level array. The model wraps the facets either under a
        key whose value is the list (e.g. {"facets": [...]}) or, when it has no
        list to hand, crams them across the object's keys and values
        (e.g. {"facet a": "facet b", ...}).
    """
    if isinstance(parsed, list):
        return [str(s) for s in parsed]

    if isinstance(parsed, dict):
        # Prefer the wrapper shape: a list-valued key holds the facets (e.g.
        # {"facets": [...]} or {"spans": [...]}). An empty list value
        # ({"spans": []}) is a real empty result -- return [], do NOT fall through
        # to the cram path, or the key and "[]" come back as two bogus facets.
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if list_values:
            return [str(s) for v in list_values for s in v]
        # No list value anywhere: facets were crammed across keys and values.
        # Interleave to preserve each pair's order.
        facets: list[str] = []
        for k, v in parsed.items():
            facets.append(str(k))
            facets.append(str(v))
        return facets

    return None


def _parse_decomposed(text: str) -> "tuple[list[str], bool]":
    """Parse facet strings from model output.

    Returns (facets list, had_error). Accepts a bare JSON array or an object
    wrapper (see _coerce_facets), and falls back to regex array extraction if
    json.loads fails. Used by the ground-truth build pipeline, which needs
    the lenient regex fallback that the scorer's direct _coerce_facets call
    does not provide.
    """
    # Attempt 1: standard JSON parse
    try:
        facets = _coerce_facets(json.loads(text))
        if facets is not None:
            return facets, False
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: extract a bracketed array from surrounding text
    m = _re.search(r"\[.*\]", text, _re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(s) for s in parsed], False
        except (json.JSONDecodeError, TypeError):
            pass

    return [], True


def _build_overscaffold_prompt(
    situation: str, action: str, result_text: str, template: str
) -> "str | None":
    """Build the over-scaffolding decomposition prompt for one annotation.

    Returns None when both action and result are junk -- there is no described
    tutor behavior or outcome to analyze, so the caller skips the API call and
    writes an empty facet list.
    """
    if (
        action.strip().lower() in JUNK_TEXTS
        and result_text.strip().lower() in JUNK_TEXTS
    ):
        return None
    return (
        template.replace("{situation}", situation)
        .replace("{action}", action)
        .replace("{result}", result_text)
    )


def _build_decompose_entries(
    results: dict,
    pass_type: str,
) -> tuple[list[dict], list[tuple[str, int]]]:
    """Build batch entries for one decompose pass (action or overscaffold).

    Args:
        results: {conv_id: {"annotations": [...]}} mapping. Each annotation
            must have "annotation_type", "situation", "action", "result".
        pass_type: One of "action", "overscaffold".
            - "action": substitutes {action} in decompose_action.md
            - "overscaffold": substitutes {situation}/{action}/{result} in
              decompose_overscaffold.md; only built for annotation_type=="scaffolding";
              skipped (empty facets) when BOTH action AND result are junk.

    Returns:
        (entries, locations) where entries is a list of build_batch_entry dicts
        (json_mode=True) and locations is a list of (conv_id, idx) tuples
        parallel to entries.

    Junk detection: action/result stripped + lowercased in JUNK_TEXTS ->
        no entry built; caller should assign [] for that annotation field.
    """
    from .client import build_batch_entry

    if pass_type == "action":
        template = _load_decompose_prompt("decompose_action.md")
    elif pass_type == "overscaffold":
        template = _load_decompose_prompt("decompose_overscaffold.md")
    else:
        raise ValueError(f"Unknown pass_type: {pass_type!r}")

    entries: list[dict] = []
    locations: list[tuple[str, int]] = []

    for conv_id, conv_data in results.items():
        for idx, ann in enumerate(conv_data.get("annotations", [])):
            ann_type = ann.get("annotation_type", "scaffolding")
            situation = ann.get("situation", "")
            action = ann.get("action", "")
            result_text = ann.get("result", "")

            if pass_type == "action":
                if action.strip().lower() in JUNK_TEXTS:
                    continue
                prompt = template.replace("{action}", action)
                key = f"action__{conv_id}__{idx}"

            elif pass_type == "overscaffold":
                # Overscaffold is scaffolding-specific
                if ann_type != "scaffolding":
                    continue
                # Skip when both action and result are junk (no tutor behavior to analyze)
                if (
                    action.strip().lower() in JUNK_TEXTS
                    and result_text.strip().lower() in JUNK_TEXTS
                ):
                    continue
                prompt = (
                    template.replace("{situation}", situation)
                    .replace("{action}", action)
                    .replace("{result}", result_text)
                )
                key = f"overscaffold__{conv_id}__{idx}"

            entries.append(build_batch_entry(key, prompt, json_mode=True))
            locations.append((conv_id, idx))

    return entries, locations


# ---------------------------------------------------------------------------
# Structure pass
#
# Benchmark/in-memory path only:
#   - entry building: action (json_mode=True)
#   - parse helpers (action; result kept only for tutormoments_build.groundtruth)
#   - default labels for no-facet annotations
# Result classification (classify_student_result.md -> result_label) is no
# longer run by the scorer (student-outcome labelling was dropped from the
# paper); the template, parser, and DEFAULT_RESULT_LABEL remain because
# tutormoments_build.groundtruth reuses them for student_outcome_agg.
# ---------------------------------------------------------------------------

_YES_NO_TO_ACTION_LABEL = {
    ("yes", "yes"): "both",
    ("yes", "no"): "scaffolding",
    ("no", "yes"): "rigor",
    ("no", "no"): "neither",
}

RESULT_LABEL_MAP = {"a": "pos", "b": "neg"}

# Default label for annotations where no result facets exist (distinct from
# "unclear", which signals a parse failure when facets were sent to the model).
DEFAULT_RESULT_LABEL = "no_evidence"

# Default label for annotations where no action facets exist.
DEFAULT_ACTION_LABEL = "neither"

_STRUCTURE_PROMPTS_DIR = f"{_SCORER_PROMPTS_DIR}/structure"


def _load_structure_prompt(filename: str) -> str:
    """Load a structure prompt by filename from the scorer structure directory."""
    return resource_text(f"{_STRUCTURE_PROMPTS_DIR}/{filename}")


def _format_facet_list(facets: list) -> str:
    """Format facets as a bullet list, one per line, for {action_list}/{student_list}."""
    return "\n".join(f"- {facet}" for facet in facets)


def _parse_action_label(text: str) -> tuple:
    """Parse the action label from model output text.

    classify_action.md asks for JSON {"scaffolding": "yes"|"no", "rigor":
    "yes"|"no"} -- two independent per-dimension judgments. Tries json.loads
    first (a list-wrapped response like [{...}] is unwrapped), then falls back
    to regex field extraction for responses with extra surrounding text.
    The resulting (scaffolding, rigor) tuple is mapped to a single action_label
    via _YES_NO_TO_ACTION_LABEL.

    Returns (label, had_error). Falls back to "unclear" if either dimension
    is missing or isn't "yes"/"no".
    """

    def _coerce(val) -> str | None:
        v = str(val).strip().lower()
        return v if v in ("yes", "no") else None

    scaffolding = rigor = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if isinstance(parsed, dict):
            scaffolding = _coerce(parsed.get("scaffolding"))
            rigor = _coerce(parsed.get("rigor"))
    except (json.JSONDecodeError, AttributeError, TypeError, IndexError):
        pass

    if scaffolding is None or rigor is None:
        m_scaf = _re.search(
            r'["\']?scaffolding["\']?\s*:\s*["\']?(yes|no)["\']?', text, _re.IGNORECASE
        )
        m_rigor = _re.search(
            r'["\']?rigor["\']?\s*:\s*["\']?(yes|no)["\']?', text, _re.IGNORECASE
        )
        if scaffolding is None and m_scaf:
            scaffolding = m_scaf.group(1).lower()
        if rigor is None and m_rigor:
            rigor = m_rigor.group(1).lower()

    if scaffolding is not None and rigor is not None:
        return _YES_NO_TO_ACTION_LABEL[(scaffolding, rigor)], False

    return "unclear", True


def _parse_result_label(text: str) -> tuple:
    """Parse the student-outcome label (a single bare letter) from model output text.

    Tries an exact match first (the documented bare-letter format), then falls
    back to the first line with markdown emphasis stripped, then a first-word
    regex. Returns (label, had_error), where label is "pos" | "neg" | "unclear".
    """
    cleaned = text.strip().lower().rstrip(".")
    if not cleaned:
        return "unclear", True
    if cleaned in RESULT_LABEL_MAP:
        return RESULT_LABEL_MAP[cleaned], False

    first_line = _re.sub(r"[*_`]", "", cleaned.splitlines()[0]).strip().rstrip(".")
    if first_line in RESULT_LABEL_MAP:
        return RESULT_LABEL_MAP[first_line], False
    m = _re.match(r"(a|b)\b", first_line)
    if m:
        return RESULT_LABEL_MAP[m.group(1)], False

    return "unclear", True


def _build_structure_entries(
    results: dict,
    target: str = "scaffolding",
) -> tuple:
    """Build batch entries for the structure pass (action classification).

    Result classification entries are no longer built (student-outcome
    labelling was dropped).

    Action entries use json_mode=True (model must return
    {"scaffolding": "yes"|"no", "rigor": "yes"|"no"}).

    Args:
        results: {conv_id: {"annotations": [...]}} mapping. Each annotation
            must have an "action_decomposed" list.
        target: annotation type to process (default "scaffolding"). Only annotations
            with annotation_type == target are processed.

    Returns:
        (action_entries, skip_action) where:
          - action_entries: list of build_batch_entry dicts with json_mode=True
          - skip_action: list of (conv_id, idx) with no action facets
    """
    from .client import build_batch_entry

    action_template = _load_structure_prompt("classify_action.md")

    action_entries: list = []
    skip_action: list = []

    for conv_id, conv_data in results.items():
        for idx, ann in enumerate(conv_data.get("annotations", [])):
            if ann.get("annotation_type", target) != target:
                continue

            action_facets = ann.get("action_decomposed") or []

            if not action_facets:
                skip_action.append((conv_id, idx))
            else:
                key = f"action__{conv_id}__{idx}"
                prompt = action_template.replace(
                    "{action_list}", _format_facet_list(action_facets)
                )
                action_entries.append(build_batch_entry(key, prompt, json_mode=True))

    return action_entries, skip_action


# ---------------------------------------------------------------------------
# score_batch(pairs) / score(scenario, transcript) -> Annotation
#
# Orchestrates the 3-pass scoring pipeline, pooled across scenarios:
#   Pass 1 (annotate): build entries, run_batch, _parse_and_merge
#   Pass 2 (decompose): build action+overscaffold entries in one batch,
#                       assign facets back onto the annotation dict
#   Pass 3 (structure): build action entries in one batch,
#                       assign action_label back onto the annotation dict
#
# Scorer model/params from config.scorer_spec() (claude-opus-4-6, adaptive thinking).
# context_window=0 always (benchmark-only override; see annotate pass docstring).
# Usage accumulated across all 3 passes is attached as Annotation.usage.
# ---------------------------------------------------------------------------


def score_batch(pairs: "list[tuple[Any, Any]]") -> "dict[str, Annotation]":
    """Run the 3-pass scorer over many (scenario, transcript) pairs, pooled.

    All pairs share ONE run_batch call per pass (annotate -> decompose ->
    structure), so a cell's scoring costs ~3 batch queue-waits regardless of
    how many moments it contains — the same pooling the original pipeline
    used. Per-scenario usage is attributed from entry keys.

    Pairs whose transcript has no generated turns get the minimal Annotation
    without contributing batch entries. Per-entry failures inside a successful
    batch degrade per-moment via the existing parse fallbacks.

    Args:
        pairs: list of (Moment, Transcript) pairs.

    Returns:
        {scenario_id: Annotation} for every input pair.
    """
    from .client import ModelClient, run_batch
    from .config import scorer_spec

    if not pairs:
        return {}

    spec = scorer_spec()
    scorer_model = spec.model

    client = ModelClient(scorer_model)

    # -------------------------------------------------------------------------
    # Build synthetic conversations + detections, merged across all pairs
    # -------------------------------------------------------------------------
    conversations: dict = {}
    detections: dict = {}
    annotations_out: "dict[str, Annotation]" = {}

    for scenario, transcript in pairs:
        conv_dict, det_by_id = _build_synthetic_conversation(scenario, transcript)
        if not det_by_id:
            # No generated turns: minimal Annotation, no batch entries.
            annotations_out[scenario.id] = Annotation(
                scenario_id=scenario.id,
                annotation_type="scaffolding",
                turn_start=0,
                turn_end=0,
                situation="",
                action="[No generated turns in transcript]",
                result="",
                action_decomposed=[],
                overscaffold_decomposed=[],
                action_label=DEFAULT_ACTION_LABEL,
                usage=dict(EMPTY_USAGE),
            )
            continue
        conversations.update(conv_dict)
        detections.update(det_by_id)

    if not detections:
        return annotations_out

    usage_by_sid = {sid: dict(EMPTY_USAGE) for sid in detections}

    # -------------------------------------------------------------------------
    # Pass 1: Annotate (one pooled batch)
    # -------------------------------------------------------------------------
    annotate_entries = _build_annotate_entries(conversations, detections)
    logger.info(
        "Classification pass 1/3 (annotate): %d entries, scorer=%s",
        len(annotate_entries),
        scorer_model,
    )
    annotate_raw = run_batch(
        client,
        annotate_entries,
        display_name="scorer_annotate",
        poll_interval=60,
        thinking=spec.thinking,
    )

    parsed = _parse_and_merge(annotate_raw, dict(detections))

    for sid, conv_data in parsed.items():
        if sid in usage_by_sid:
            usage_by_sid[sid] = sum_usage(usage_by_sid[sid], conv_data.get("usage", {}))

    # Build the results shape expected by _build_decompose_entries /
    # _build_structure_entries: {sid: {"annotations": [ann_dict, ...]}}
    results: dict = {}
    for sid in detections:
        annotations_list = parsed.get(sid, {}).get("annotations", [])
        if not annotations_list:
            annotations_list = [
                {
                    "annotation_type": "scaffolding",
                    "turn_start": 0,
                    "turn_end": 0,
                    "situation": "",
                    "action": "[Analysis unavailable -- batch failed for this moment]",
                    "result": "",
                }
            ]
        results[sid] = {"annotations": annotations_list}

    # -------------------------------------------------------------------------
    # Pass 2: Decompose (action + overscaffold in one pooled batch)
    # The sub-passes are combined into a single run_batch call, then split
    # by key prefix to assign.
    # (Result decomposition was dropped along with student-outcome labelling.)
    # -------------------------------------------------------------------------
    action_decomp_entries, action_decomp_locs = _build_decompose_entries(
        results, "action"
    )
    overscaffold_entries, overscaffold_locs = _build_decompose_entries(
        results, "overscaffold"
    )

    decompose_entries = action_decomp_entries + overscaffold_entries

    if decompose_entries:
        logger.info(
            "Classification pass 2/3 (decompose): %d entries", len(decompose_entries)
        )
        decompose_raw = run_batch(
            client,
            decompose_entries,
            display_name="scorer_decompose",
            poll_interval=60,
            thinking=spec.thinking,
        )
    else:
        logger.info("Classification pass 2/3 (decompose): skipped, no entries")
        decompose_raw = {}

    def _assign_decomposed(locs, entries, field_name):
        for (conv_id, idx), entry in zip(locs, entries):
            key = entry["key"]
            raw = decompose_raw.get(key, {})
            if conv_id in usage_by_sid:
                usage_by_sid[conv_id] = sum_usage(
                    usage_by_sid[conv_id], raw.get("usage", {})
                )
            text = raw.get("text", "")
            try:
                parsed_val = json.loads(text) if text else []
            except json.JSONDecodeError:
                parsed_val = []
            facets = _coerce_facets(parsed_val)
            anns = results.get(conv_id, {}).get("annotations", [])
            if idx < len(anns):
                anns[idx][field_name] = facets if facets is not None else []

    _assign_decomposed(action_decomp_locs, action_decomp_entries, "action_decomposed")
    _assign_decomposed(
        overscaffold_locs, overscaffold_entries, "overscaffold_decomposed"
    )

    # Ensure all annotations have decomposed fields (default [] for skipped ones).
    for conv_id, conv_data in results.items():
        for ann in conv_data.get("annotations", []):
            ann.setdefault("action_decomposed", [])
            ann.setdefault("overscaffold_decomposed", [])

    # -------------------------------------------------------------------------
    # Pass 3: Structure (action classification in one pooled batch)
    # -------------------------------------------------------------------------
    action_struct_entries, skip_action = _build_structure_entries(
        results, target="scaffolding"
    )
    structure_entries = action_struct_entries

    if structure_entries:
        logger.info(
            "Classification pass 3/3 (structure): %d entries", len(structure_entries)
        )
        structure_raw = run_batch(
            client,
            structure_entries,
            display_name="scorer_structure",
            poll_interval=60,
            thinking=spec.thinking,
        )
    else:
        logger.info("Classification pass 3/3 (structure): skipped, no entries")
        structure_raw = {}

    # Apply default labels to skipped (no-facet) annotations.
    for conv_id, idx in skip_action:
        anns = results.get(conv_id, {}).get("annotations", [])
        if idx < len(anns):
            anns[idx]["action_label"] = DEFAULT_ACTION_LABEL

    def _assign_labels(entries, prefix, parse_fn, field_name):
        # Key scheme: "{prefix}__{conv_id}__{idx}" -- conv_id may itself contain
        # "__", so split from the RIGHT to reliably extract the numeric idx.
        for entry in entries:
            key = entry["key"]
            raw = structure_raw.get(key, {})
            without_prefix = key[len(prefix) :]
            last_sep = without_prefix.rfind("__")
            if last_sep == -1:
                continue
            conv_id = without_prefix[:last_sep]
            idx_str = without_prefix[last_sep + 2 :]
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            if conv_id in usage_by_sid:
                usage_by_sid[conv_id] = sum_usage(
                    usage_by_sid[conv_id], raw.get("usage", {})
                )
            label, _ = parse_fn(raw.get("text", ""))
            anns = results.get(conv_id, {}).get("annotations", [])
            if idx < len(anns):
                anns[idx][field_name] = label

    _assign_labels(
        action_struct_entries, "action__", _parse_action_label, "action_label"
    )

    # Ensure all annotations have label fields.
    for conv_id, conv_data in results.items():
        for ann in conv_data.get("annotations", []):
            ann.setdefault("action_label", DEFAULT_ACTION_LABEL)

    # -------------------------------------------------------------------------
    # Build one Annotation per scenario
    # -------------------------------------------------------------------------
    for sid in detections:
        a = results[sid]["annotations"][0]
        annotations_out[sid] = Annotation(
            scenario_id=sid,
            annotation_type=a.get("annotation_type", "scaffolding"),
            turn_start=a.get("turn_start", 0),
            turn_end=a.get("turn_end", 0),
            situation=a.get("situation", ""),
            action=a.get("action", ""),
            result=a.get("result", ""),
            action_decomposed=a.get("action_decomposed", []),
            overscaffold_decomposed=a.get("overscaffold_decomposed", []),
            action_label=a.get("action_label", DEFAULT_ACTION_LABEL),
            usage=usage_by_sid[sid],
        )

    return annotations_out


def score(scenario: Any, transcript: Any) -> "Annotation":
    """Run the 3-pass scorer over one scenario+transcript and return an Annotation.

    The N=1 case of score_batch (see score_batch for the pass structure).
    Prefer score_batch when scoring many moments: it pools all moments into
    one batch job per pass instead of paying the batch queue wait per moment.

    Args:
        scenario: tutormoments.moments.Moment object.
        transcript: tutormoments.conversation.Transcript object.

    Returns:
        Annotation with all fields populated.
    """
    return score_batch([(scenario, transcript)])[scenario.id]


# ---------------------------------------------------------------------------
# Public API reused by tutormoments_build.groundtruth
# ---------------------------------------------------------------------------
# These helpers parse/build the scorer's own prompt formats. Most are runtime
# code (called by score() above); parse_result_label and DEFAULT_RESULT_LABEL
# (plus the decompose_result.md / classify_student_result.md templates) are
# retained ONLY for the ground-truth build pipeline, which reuses them across
# the package boundary for student_outcome_agg.

load_decompose_prompt = _load_decompose_prompt
parse_decomposed = _parse_decomposed
build_overscaffold_prompt = _build_overscaffold_prompt
load_structure_prompt = _load_structure_prompt
format_facet_list = _format_facet_list
parse_action_label = _parse_action_label
parse_result_label = _parse_result_label
