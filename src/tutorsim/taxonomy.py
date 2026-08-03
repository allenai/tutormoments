"""Tutor action-facet taxonomy: classify decomposed actions into a 13-letter
scheme (A-M) and compute population-level macro/moment statistics.

This is the taxonomy *data-generation* layer, part of the tutorsim runtime.
Figure *rendering* lives in the analysis notebooks under
analysis/working-paper-*, which import these symbols
(`from tutorsim import taxonomy as tx`) and draw from the tables produced here.

Input contracts (use either; both are tutorsim-native, no new file format):

  1. Key-moments JSONL: one row per conversation, shape
       {conversation_id, num_turns, key_moments: [...]}.
     Each key_moment carries annotation_type, turn_start, turn_end,
     situation_label_agg, action_decomposed, .... This matches the
     ground-truth bundle distributed alongside the benchmark (the human
     reference distribution).

  2. Tutorsim run results: a directory of per-scenario score files
       results/<run_id>/scores/<scenario_id>.json, where each file is an
       Annotation produced by tutorsim.scoring.score. Pair with the
       scenarios.jsonl used for the run to recover the situation label,
       model, and prompt for each scenario. (During a run, the LM side is
       classified in-memory via `facets_from_annotations`.)

Pipeline (each stage idempotent and writable to/readable from disk):
  load_* / facets_from_annotations  -> Facet stream
  build_pool -> kept facets + excluded facets (with reason)
  classify_pool -> {statement -> category letter}, resume-safe via sidecar
  build_headline_tables -> 5 pandas DataFrames (macro/moment)

pandas is an optional extra (install with `pip install 'tutorsim[analysis]'`),
needed only by the headline-table stage. Importing this module never requires
it; it is checked lazily when the headline stage runs. matplotlib/seaborn are
not used here -- they belong to the analysis figures.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from string import Template
from typing import Any, Iterable, Iterator, Optional

from tutorsim.resources import resource_text

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Frozen taxonomy + classifier prompt
# ============================================================================
#
# Scheme: lm_extended_v1. A-L are the human-tutor scheme frozen from the
# anthropic-train induction; M is the additive category from LM-side induction
# over the L-bucket. Definitions are immutable here -- do not edit at runtime.

SCHEME_VERSION = "lm_extended_v1"

CATEGORIES: list[dict[str, Any]] = [
    {
        "letter": "A",
        "name": "Guiding/funneling questions toward an answer",
        "orientation": "scaffolding",
        "definition": (
            "The tutor poses leading questions or step-by-step sub-step prompts "
            "that funnel the student toward a specific answer or next move "
            "without demanding independent justification."
        ),
        "examples": [
            "The tutor asks guiding questions.",
            "The tutor asks a series of guiding questions that funnel the student toward performing the addition.",
            "The tutor asks guiding sub-step questions (multiply, subtract, bring down, repeat).",
            "The tutor asks a guiding question prompting the student to identify the correct operation.",
        ],
    },
    {
        "letter": "B",
        "name": "Breaking the problem into steps",
        "orientation": "scaffolding",
        "definition": (
            "The tutor decomposes the task into smaller steps, sub-steps, or "
            "component parts to make it more manageable."
        ),
        "examples": [
            "The tutor breaks the problem into steps.",
            "The tutor breaks the problem into sub-steps.",
            "The tutor breaks the task into sub-steps.",
            "For the borrowing-across-zeros step, the tutor breaks the problem into a simpler sub-problem.",
        ],
    },
    {
        "letter": "C",
        "name": "Explaining, modeling, or co-solving",
        "orientation": "scaffolding",
        "definition": (
            "The tutor directly explains a concept or procedure, models a worked "
            "example, narrates the reasoning, co-solves the problem doing much of "
            "the cognitive work, or corrects an error BY explaining why it is "
            "wrong or re-teaching the concept."
        ),
        "examples": [
            "The tutor explains the concept.",
            "The tutor models worked examples.",
            "The tutor co-solves with the student.",
            "The tutor narrates each step.",
            "The tutor explains the procedure.",
        ],
    },
    {
        "letter": "D",
        "name": "Alternative representations and analogies",
        "orientation": "scaffolding",
        "definition": (
            "The tutor re-presents the problem or concept in a different form -- "
            "a diagram, visual, number line, manipulative, real-world analogy, "
            "reframing, or restating/rephrasing the problem in different words -- "
            "to make it more accessible."
        ),
        "examples": [
            "The tutor scaffolds by introducing a different representation.",
            "The tutor introduces a real-world analogy.",
            "The tutor draws a diagram to represent the scenario.",
            "The tutor uses a visual representation.",
            "The tutor connects multiplication to repeated addition.",
        ],
    },
    {
        "letter": "E",
        "name": "Hints, reminders, and narrowing support",
        "orientation": "scaffolding",
        "definition": (
            "The tutor provides hints, reminders of prior work or rules, "
            "highlights key information, or narrows answer options to lower "
            "difficulty and nudge the student forward (without re-presenting the "
            "whole problem)."
        ),
        "examples": [
            "The tutor offers a hint.",
            "The tutor rephrases the problem using simpler language.",
            "The tutor reduces the choices to two.",
            "The tutor highlights the key detail the student overlooked.",
            "The tutor provides hints by underlining digits.",
        ],
    },
    {
        "letter": "F",
        "name": "Supplying answers, steps, or corrections",
        "orientation": "scaffolding",
        "definition": (
            "The tutor supplies the result directly -- gives away the answer, "
            "fills in a step, performs the computation, or tersely states the "
            "correct answer without teaching it. (A correction or answer "
            "delivered BY explaining/modeling the why or how goes to C, not F.)"
        ),
        "examples": [
            "The tutor gives away the answer.",
            "The tutor fills in the answer.",
            "The tutor corrects the student's errors directly.",
            "The tutor supplies the answer.",
            "The tutor states the final answer.",
        ],
    },
    {
        "letter": "G",
        "name": "Prompting for explanation, justification, or reasoning",
        "orientation": "rigor",
        "definition": (
            "The tutor asks the student to explain, justify, or reason about HOW "
            "they arrived at an answer or WHY it works, or to define/articulate "
            "concepts independently -- eliciting the reasoning itself. (For "
            "merely checking, verifying, or rating confidence in an answer "
            "without articulating the reasoning, use J.)"
        ),
        "examples": [
            "The tutor asks the student to explain how they arrived at their answer.",
            "The tutor pushes for rigor by asking the student to justify how they arrived at their answer.",
            "The tutor asks the student to define a key vocabulary term.",
            "The tutor asks the student to verify whether their answer is correct.",
        ],
    },
    {
        "letter": "H",
        "name": "Withdrawing support / independent work",
        "orientation": "rigor",
        "definition": (
            "The tutor steps back, withholds help, or hands the task over so the "
            "student attempts or completes the work independently."
        ),
        "examples": [
            "The tutor pushes for rigor by withdrawing support.",
            "The tutor withdraws support.",
            "The tutor has the student attempt the problem independently.",
            "The tutor allows the student to struggle productively through extended pauses.",
            "The tutor lets the student work independently.",
        ],
    },
    {
        "letter": "I",
        "name": "Increasing complexity or challenge",
        "orientation": "rigor",
        "definition": (
            "The tutor raises cognitive demand by introducing harder problems, "
            "larger numbers, more complex variations, or advancing to more "
            "demanding topics."
        ),
        "examples": [
            "The tutor pushes for rigor by increasing problem complexity.",
            "The tutor pushes for rigor by introducing a harder problem.",
            "The tutor moves the student to a more challenging set of larger numbers.",
            "The tutor announces that the next problem will be harder.",
        ],
    },
    {
        "letter": "J",
        "name": "Prompting self-assessment / reconsideration",
        "orientation": "rigor",
        "definition": (
            "The tutor prompts the student to evaluate their own answer or "
            "reasoning -- to rate their confidence, check or verify whether it is "
            "correct, reconsider it, or find and fix an error themselves -- "
            "rather than supplying the correction. (Evaluating the answer, not "
            "articulating the reasoning behind it.)"
        ),
        "examples": [
            "The tutor asks the student to rate their confidence in an answer.",
            "The tutor asks the student to re-check specific responses.",
            "The tutor prompts the student to reconsider when they guess incorrectly.",
            "The tutor asks the student to verify whether their answer is correct.",
        ],
    },
    {
        "letter": "K",
        "name": "Affirmations, check-ins, and confirming answers",
        "orientation": "neutral",
        "definition": (
            "The tutor offers praise, encouragement, or reassurance; gauges the "
            "student's comfort, understanding, or readiness for the task; or "
            "confirms whether an answer is correct."
        ),
        "examples": [
            "The tutor checks for understanding.",
            "The tutor confirms the correct answer.",
            "The tutor offers praise.",
            "The tutor reassures the student.",
            "The tutor affirms the student's correct answer.",
        ],
    },
    {
        "letter": "L",
        "name": "Transitioning to a new problem or topic",
        "orientation": "neutral",
        "definition": (
            "The tutor advances the session by moving on to the next problem, "
            "question, step, topic, or task -- including transitions made "
            "without adding challenge or probing understanding."
        ),
        "examples": [
            "The tutor moves on to a new problem.",
            "The tutor transitions to the next problem.",
            "The tutor moves on to the next problem without increasing the challenge.",
            "The tutor transitions to a new topic (three-digit multiplication).",
            "The tutor moves on to the next problem without asking the student to explain their reasoning.",
        ],
    },
    {
        "letter": "M",
        "name": "Other",
        "orientation": "neutral",
        "definition": (
            "Off-task or logistical moments, technical handling, non-actions, "
            "bare stance statements with no concrete move, mechanical "
            "navigation, and reading the problem aloud verbatim -- but NOT "
            "restating/rephrasing it in different words, which is category D."
        ),
        "examples": [
            "The tutor reads the problem aloud.",
            "The tutor presents the problem.",
            "The tutor moves to the whiteboard.",
            "The tutor takes no substantive pedagogical action.",
        ],
    },
]

CATEGORY_LETTERS: list[str] = [c["letter"] for c in CATEGORIES]
ORIENTATION_BY_LETTER: dict[str, str] = {
    c["letter"]: c["orientation"] for c in CATEGORIES
}
NAME_BY_LETTER: dict[str, str] = {c["letter"]: c["name"] for c in CATEGORIES}
DEFINITION_BY_LETTER: dict[str, str] = {
    c["letter"]: c["definition"] for c in CATEGORIES
}
LAST_LETTER = CATEGORY_LETTERS[-1]


CLASSIFY_PROMPT_RESOURCE = "prompts/taxonomy/classify_actions.md"


def _load_classify_prompt() -> Template:
    """Load the classifier prompt template from the packaged prompts dir."""
    return Template(resource_text(CLASSIFY_PROMPT_RESOURCE))


def _categories_block() -> str:
    """The frozen scheme rendered for prompt injection."""
    return "\n".join(
        f"{c['letter']}. {c['name']} -- {c['definition']}. "
        f"Examples: " + "; ".join(f'"{e}"' for e in c["examples"][:4]) + "."
        for c in CATEGORIES
    )


# ============================================================================
# 2. Exclusion filters
# ============================================================================
#
# Four rules. A statement is kept only if it (a) has "the tutor" as its
# subject, (b) isn't a bare stance, (c) isn't a stance combined with negation,
# and (d) isn't an explicit non-action ("the tutor does not / fails to /
# takes no / makes no / ..."). The reason code travels with stripped rows
# so the boundary is auditable.

_STANCE_PHRASE = re.compile(
    r"\b(over-?scaffold(s|ing)?|under-?scaffold(s|ing)?|scaffolds?|scaffolding|"
    r"provides? scaffolding|providing scaffolding|pushes? for rigor|"
    r"pushing for rigor|push for rigor|maintains? rigor|increases? rigor|"
    r"raises? rigor)\b"
)

# Tokens that may sit alongside a stance phrase without adding pedagogical
# content (copulas, adverbs, transitions, stance-blending verbs, articles).
_STANCE_FILLER = re.compile(
    r"\b(the tutor|the student|is|are|was|were|been|being|mostly|primarily|"
    r"largely|mainly|heavily|heavy|lightly|light|mildly|gently|softly|somewhat|"
    r"consistently|throughout|generally|overall|again|here|appropriately|"
    r"effectively|minimally|continues?|continuing|begins by|starts by|initially|"
    r"then|also|still|shifts? to|shifts? into|moves? to|moves? into|"
    r"transitions? to|switches? to|returns? to|rather than|instead of|more than|"
    r"and|but|while|by|a bit|very|quite|fairly|"
    r"a|an|quickly|slowly|briefly|often|sometimes|occasionally|preemptively|"
    r"sparingly|frequently|repeatedly|periodically|now|just|a little|"
    r"blends?|blending|combines?|combining|mix(?:es)?|mixing|alternates?|"
    r"alternating|balances?|balancing|with|small|moderate|strong|some|"
    r"continued|continue|"
    r"of|uses?|using|in|this|moment|multiple|two|several|various|ways|"
    r"manner|unnecessarily|bordering|on|preemptive|occurs|deep|deeper|"
    r"pivots? to|pivoting to|rigor|the"
    r")\b"
)

_NEGATION_RE = re.compile(
    r"\b(does not|do not|doesn['’]t|did not|didn['’]t|don['’]t|"
    r"is not|isn['’]t|are not|aren['’]t|was not|wasn['’]t|were not|weren['’]t|"
    r"neither|nor)\b"
)

# Non-actions: "the tutor [negation lead]". Catches "does not / never / fails
# to / makes no / takes no / takes no substantive action" etc. in one pass.
_NON_ACTION_RE = re.compile(
    r"^(?:[\w ,'\-]{0,80},\s*)?the tutor "
    r"(?:does not|did not|does no |do not|doesn['’]t|didn['’]t|don['’]t|never |"
    r"fails? to |failed to |makes? no |made no |takes? no |took no |"
    r"provides? no |provided no |offers? no |offered no |gives? no |gave no |"
    r"does little |did little )"
)

# Requires "the tutor" to be the subject (possibly after a short adverbial
# lead). Statements about the student, the interaction, the focus, "There is
# no X", "No attempt is made to Y", etc. all fail this guard and get stripped.
_TUTOR_ACTOR_RE = re.compile(r"^(?:[\w ,'\-]{0,80},\s*)?the tutor\b")


def _normalize(s: str) -> str:
    s = s.strip().lower().rstrip(".")
    return re.sub(r"\s+", " ", s)


def _is_pure_stance(norm: str) -> bool:
    if not _STANCE_PHRASE.search(norm):
        return False
    s = _STANCE_PHRASE.sub(" ", norm)
    s = _STANCE_FILLER.sub(" ", s)
    s = re.sub(r"[^a-z]", " ", s)
    return re.sub(r"\s+", " ", s).strip() == ""


def filter_statement(statement: str) -> tuple[str, str, bool]:
    """Classify a single action statement as 'keep' or 'strip'.

    Returns (bucket, reason, stance_prefixed).
      bucket: "keep" | "strip"
      reason: "" when kept, otherwise one of non_tutor_actor | pure_stance |
              stance_negation | non_action
      stance_prefixed: True iff a stance phrase appears (audit-only flag).
    """
    norm = _normalize(statement)
    stance = bool(_STANCE_PHRASE.search(norm))
    if not _TUTOR_ACTOR_RE.match(norm):
        return ("strip", "non_tutor_actor", stance)
    if _is_pure_stance(norm):
        return ("strip", "pure_stance", stance)
    if stance and _NEGATION_RE.search(norm):
        return ("strip", "stance_negation", stance)
    if _NON_ACTION_RE.match(norm):
        return ("strip", "non_action", stance)
    return ("keep", "", stance)


STRIP_REASONS: tuple[str, ...] = (
    "non_tutor_actor",
    "pure_stance",
    "stance_negation",
    "non_action",
)


# ============================================================================
# 3. Canonical facet + input adapters
# ============================================================================
#
# A Facet is one action statement with enough provenance to compute every
# downstream view. Adapters yield Facet instances from each input format.


@dataclass
class Facet:
    moment_id: str  # unique per (transcript, moment, annotator-or-scenario)
    transcript_id: str  # parent transcript / conversation id
    turn_start: int
    turn_end: int
    statement_index: int  # position within action_decomposed
    statement: str  # the action_decomposed string itself
    annotation_type: str  # always "scaffolding" after filtering
    situation_label: str  # GT label of the moment: scaffolding|rigor|...
    action_label: Optional[str] = None  # SAR's tutor-action call (LM only)
    result_label: Optional[str] = None
    model: Optional[str] = None  # LM only
    prompt: Optional[str] = None  # LM only
    source: str = ""  # "hf" | "tutorsim" | "canonical"
    # Filled in by build_pool / classify_pool:
    stance_prefixed: bool = False
    category: Optional[str] = None  # one of CATEGORY_LETTERS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Facet":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def load_key_moments_jsonl(jsonl_path: Path) -> Iterator[Facet]:
    """Stream Facets from a key-moments JSONL.

    Expected row shape (matches the benchmark's ground-truth bundle):
      {"conversation_id": str, "num_turns": int, "key_moments": [
          {"annotation_type": "scaffolding"|"rapport",
           "turn_start": int, "turn_end": int,
           "situation_label_agg": str,
           "action_decomposed": [str, ...],
           "annotator_id": str, ...},
          ...
      ]}

    We walk `key_moments`, keep `annotation_type == "scaffolding"`, and
    yield one Facet per item in `action_decomposed`. The moment is keyed by
    `{conversation_id}__{turn_start}_{turn_end}__{annotator_id}` so each
    annotator's pass on the same physical moment counts as its own moment.
    """
    jsonl_path = Path(jsonl_path)
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conv_id = row["conversation_id"]
            for km in row.get("key_moments") or []:
                if km.get("annotation_type") != "scaffolding":
                    continue
                statements = km.get("action_decomposed") or []
                if not statements:
                    continue
                annotator = km.get("annotator_id") or "unknown"
                ts, te = km["turn_start"], km["turn_end"]
                moment_id = f"{conv_id}__{ts}_{te}__{annotator}"
                situation = km.get("situation_label_agg") or "unknown"
                for i, stmt in enumerate(statements):
                    if not isinstance(stmt, str):
                        continue
                    yield Facet(
                        moment_id=moment_id,
                        transcript_id=conv_id,
                        turn_start=ts,
                        turn_end=te,
                        statement_index=i,
                        statement=stmt,
                        annotation_type="scaffolding",
                        situation_label=situation,
                        action_label=km.get("action_label"),
                        result_label=km.get("result_label"),
                        source="hf",
                    )


def load_tutorsim_results(results_dir: Path, scenarios_path: Path) -> Iterator[Facet]:
    """Stream Facets from a tutorsim run results directory.

    Args:
        results_dir: a `results/<run_id>/` directory (must contain `scores/`
            and `config.json`).
        scenarios_path: the `scenarios.jsonl` the run was scored against.
            Used to recover each scenario's GT `situation_label` and any
            scenario-level metadata.

    The run's tutor (model) and mode (prompt) are read from `config.json`.
    `scenario_id` is the moment_id.
    """
    results_dir = Path(results_dir)
    scores_dir = results_dir / "scores"
    config_path = results_dir / "config.json"
    if not scores_dir.is_dir():
        raise FileNotFoundError(f"missing scores directory: {scores_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")

    # config.json is written by `tutorsim.cli` with the run's tutor + mode.
    config = json.loads(config_path.read_text())
    model = config["tutor"]
    prompt = config["mode"]

    scenario_label = _index_scenario_labels(Path(scenarios_path))

    for score_path in sorted(scores_dir.glob("*.json")):
        ann = json.loads(score_path.read_text())
        scenario_id = ann.get("scenario_id") or score_path.stem
        if ann.get("annotation_type") != "scaffolding":
            continue
        statements = ann.get("action_decomposed") or []
        if not statements:
            continue
        situation = scenario_label.get(scenario_id, "unknown")
        ts = ann.get("turn_start", 0)
        te = ann.get("turn_end", 0)
        for i, stmt in enumerate(statements):
            if not isinstance(stmt, str):
                continue
            yield Facet(
                moment_id=scenario_id,
                transcript_id=_transcript_from_scenario(scenario_id),
                turn_start=ts,
                turn_end=te,
                statement_index=i,
                statement=stmt,
                annotation_type="scaffolding",
                situation_label=situation,
                action_label=ann.get("action_label"),
                # Scored Annotation persists the field as `result` (not
                # `result_label`); read the correct key.
                result_label=ann.get("result"),
                model=model,
                prompt=prompt,
                source="tutorsim",
            )


def facets_from_annotations(
    annotations: Iterable[Any],
    moments: Iterable[Any],
    *,
    model: str,
    mode: str,
) -> Iterator[Facet]:
    """Stream Facets from in-memory scored Annotations + their Moments.

    The in-run equivalent of `load_tutorsim_results` -- no disk round-trip and
    no `scenarios.jsonl`: the situation label comes from each Moment's
    `.dimension`. `annotations` and `moments` are parallel (annotation[i]
    scored moment[i]). Only scaffolding annotations produce facets. `model` and
    `mode` are recorded on each facet as the tutor/prompt that produced the
    action (the classifier model is separate; see `classify_pool`).
    """
    for ann, moment in zip(annotations, moments):
        if getattr(ann, "annotation_type", None) != "scaffolding":
            continue
        statements = getattr(ann, "action_decomposed", None) or []
        if not statements:
            continue
        scenario_id = getattr(ann, "scenario_id", None) or getattr(moment, "id", "")
        situation = getattr(moment, "dimension", None) or "unknown"
        ts = getattr(ann, "turn_start", 0)
        te = getattr(ann, "turn_end", 0)
        action_label = getattr(ann, "action_label", None)
        result_label = getattr(ann, "result", None)
        for i, stmt in enumerate(statements):
            if not isinstance(stmt, str):
                continue
            yield Facet(
                moment_id=scenario_id,
                transcript_id=_transcript_from_scenario(scenario_id),
                turn_start=ts,
                turn_end=te,
                statement_index=i,
                statement=stmt,
                annotation_type="scaffolding",
                situation_label=situation,
                action_label=action_label,
                result_label=result_label,
                model=model,
                prompt=mode,
                source="tutorsim",
            )


def load_canonical_jsonl(path: Path) -> Iterator[Facet]:
    """Stream Facets from a flat jsonl with one Facet dict per row.

    Power-user input: the row schema is exactly the Facet dataclass fields.
    Unknown fields are dropped; missing required fields raise.
    """
    path = Path(path)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Facet.from_dict(json.loads(line))


def _index_scenario_labels(scenarios_path: Path) -> dict[str, str]:
    """Map scenario_id -> situation label from scenarios.jsonl.

    `Scenario.dimension` is the situation_label_agg set at dataset build time
    (see `tutorsim.moments`). Any moment without it is treated as
    "unknown".
    """
    out: dict[str, str] = {}
    with scenarios_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            sid = s.get("id")
            if not sid:
                continue
            out[sid] = s.get("dimension") or "unknown"
    return out


_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _transcript_from_scenario(scenario_id: str) -> str:
    """Pull the parent transcript_id from a composite scenario_id.

    Tutorsim scenario IDs typically end with the source transcript UUID; we
    return the LAST UUID if any are present, else the scenario_id itself.
    """
    m = _UUID_RE.findall(scenario_id)
    return m[-1] if m else scenario_id


# ============================================================================
# 4. Pool building + CSV I/O
# ============================================================================
#
# Pool building is a pure function over Facets: apply filter_statement, set
# .stance_prefixed, return (kept, excluded). CSVs preserve the full Facet
# schema so any stage can be resumed from disk.

_FACET_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Facet))


def _atomic_write_csv(path: Path, header: Iterable[str], rows: Iterable[dict]) -> None:
    """Write a CSV to a sibling .tmp then `os.replace` into the final path.

    Crash-mid-write leaves the .tmp on disk and the original (or no) file
    intact, so a re-run reads either the last good output or restarts.
    """
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def build_pool(facets: Iterable[Facet]) -> tuple[list[Facet], list[tuple[Facet, str]]]:
    """Split a facet stream into (kept, excluded_with_reason)."""
    kept: list[Facet] = []
    excluded: list[tuple[Facet, str]] = []
    for f in facets:
        bucket, reason, stance = filter_statement(f.statement)
        f.stance_prefixed = stance
        if bucket == "keep":
            kept.append(f)
        else:
            excluded.append((f, reason))
    return kept, excluded


def write_pool_csv(
    kept: Iterable[Facet], excluded: Iterable[tuple[Facet, str]], out_dir: Path
) -> None:
    """Persist kept facets and excluded facets (with reason) under `out_dir`.

    Uses atomic write-and-rename so a crash mid-write never leaves a partial CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(out_dir / "pool.csv", _FACET_FIELDS, (f.to_dict() for f in kept))
    _atomic_write_csv(
        out_dir / "excluded.csv",
        _FACET_FIELDS + ("reason",),
        ({**f.to_dict(), "reason": r} for f, r in excluded),
    )


def read_pool_csv(pool_dir: Path) -> tuple[list[Facet], list[tuple[Facet, str]]]:
    """Load (kept, excluded_with_reason) from a directory produced by
    `write_pool_csv`."""
    pool_dir = Path(pool_dir)
    kept = list(_read_facets_csv(pool_dir / "pool.csv"))
    excluded: list[tuple[Facet, str]] = []
    with (pool_dir / "excluded.csv").open() as f:
        for row in csv.DictReader(f):
            reason = row.pop("reason", "")
            excluded.append((_coerce_facet_row(row), reason))
    return kept, excluded


def _read_facets_csv(path: Path) -> Iterator[Facet]:
    with path.open() as f:
        for row in csv.DictReader(f):
            yield _coerce_facet_row(row)


def _coerce_facet_row(row: dict[str, str]) -> Facet:
    """Convert CSV strings back to Facet field types."""
    coerced: dict[str, Any] = {}
    for name in _FACET_FIELDS:
        raw = row.get(name, "")
        if name in ("turn_start", "turn_end", "statement_index"):
            coerced[name] = int(raw) if raw not in ("", None) else 0
        elif name == "stance_prefixed":
            coerced[name] = str(raw).strip().lower() in ("1", "true", "yes")
        elif name in ("action_label", "result_label", "model", "prompt", "category"):
            coerced[name] = raw if raw not in ("", None) else None
        else:
            coerced[name] = raw
    return Facet(**coerced)


def pool_report(kept: list[Facet], excluded: list[tuple[Facet, str]]) -> str:
    """Plain-text summary of the pool: keep/strip totals and per-reason counts."""
    from collections import Counter

    n_total = len(kept) + len(excluded)
    n_keep = len(kept)
    stance_kept = sum(1 for f in kept if f.stance_prefixed)
    reason_counts = Counter(r for _, r in excluded)
    lines = [
        "Tutor action-statement pool",
        "=" * 60,
        f"total facets : {n_total}",
        f"  KEEP       : {n_keep} ({100 * n_keep / n_total:.1f}%)  "
        f"[{stance_kept} stance-prefixed]",
        f"  STRIP      : {len(excluded)} ({100 * len(excluded) / n_total:.1f}%)",
        "",
        "Strip reasons:",
    ]
    n_strip = len(excluded) or 1
    for reason in STRIP_REASONS:
        c = reason_counts.get(reason, 0)
        lines.append(f"  {reason:<18s} {c:5d}  ({100 * c / n_strip:.1f}% of strip)")
    return "\n".join(lines)


# ============================================================================
# 5. LLM classifier (resume + checkpoint)
# ============================================================================
#
# Classifies unique statements against the frozen A-M scheme using
# claude-opus-4-8 with structured-output JSON (the category letter is a
# schema enum, so the model can't hallucinate one). Statements are
# deduplicated, batched, and assignments are appended to a sidecar JSONL
# so a crash or ctrl-C loses at most the in-flight batch.
#
# Designed for two-step operation: a small first-run probe (default 25
# batches, ~$1) for sanity-checking the distribution, then a resume call
# (re-run without --max-batches) to finish.

CLASSIFIER_MODEL = "claude-opus-4-8"
CLASSIFIER_BATCH_SIZE = 50
CLASSIFIER_MAX_RETRIES = 4
CLASSIFIER_FIRST_RUN_PROBE = 25
_CLASSIFIER_PROGRESS_EVERY = 30


def classify_pool(
    kept: Iterable[Facet],
    out_dir: Path,
    *,
    max_batches: Optional[int] = None,
    first_run_probe: int = CLASSIFIER_FIRST_RUN_PROBE,
    client: Any = None,
    model: Optional[str] = None,
    thinking: Optional[bool] = None,
    batch_size: Optional[int] = None,
) -> dict[str, str]:
    """Classify every unique statement in `kept` into A-M.

    Resumable: assignments are appended to ``out_dir/assignments.jsonl`` as
    each batch completes. Re-running with the same ``out_dir`` skips
    already-classified statements and continues. Per-call usage is logged
    to ``out_dir/usage_log.jsonl``.

    Args:
        kept: facets to classify (only `.statement` is read; same statement
            across multiple facets is classified once).
        out_dir: directory for the sidecar JSONL files.
        max_batches: if set, stop after that many batches THIS RUN. Useful
            for sanity probes. If unset and no prior assignments exist, the
            first run stops after `first_run_probe` batches automatically;
            a subsequent re-run with no `max_batches` finishes the pool.
        client: optional pre-built client exposing `.generate(...)` (a
            `tutorsim.client.ModelClient` or a compatible test double). If
            None we build a ModelClient from the `taxonomy` config block
            (requires the provider API key).
        model / thinking / batch_size: override the `taxonomy` config block;
            each falls back to config (then a module default) when None.

    Returns:
        Mapping from statement -> category letter, covering every statement
        in `kept`. Returns the FULL mapping only after every statement is
        assigned; partial runs return what has been assigned so far.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = out_dir / "assignments.jsonl"
    usage_path = out_dir / "usage_log.jsonl"

    # Resolve model / thinking / batch_size from the taxonomy config block,
    # honoring explicit overrides. Config import is local so `import
    # tutorsim.taxonomy` stays lightweight (no client/provider-SDK import
    # unless we actually classify).
    spec: dict = {}
    if client is None or model is None or thinking is None or batch_size is None:
        try:
            from tutorsim.config import taxonomy_spec

            spec = taxonomy_spec()
        except Exception:
            spec = {}
    if model is None:
        model = spec.get("model", CLASSIFIER_MODEL)
    if thinking is None:
        thinking = bool(spec.get("thinking", False))
    if batch_size is None:
        batch_size = int(spec.get("batch_size", CLASSIFIER_BATCH_SIZE))

    statements = sorted(
        {f.statement.strip() for f in kept if f.statement and f.statement.strip()},
        key=lambda s: (-1, s),  # stable ordering; frequency sorting done below
    )
    # Replace with frequency-descending order so high-recurrence statements
    # get classified first (useful for early sanity checks).
    from collections import Counter

    counts = Counter(f.statement.strip() for f in kept if f.statement)
    statements = [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    assigned = _read_assignments(assignments_path)
    pending = [s for s in statements if s not in assigned]
    n_done = len(statements) - len(pending)
    logger.info(
        "classify_pool: %d unique statements (%d already assigned, %d pending)",
        len(statements),
        n_done,
        len(pending),
    )
    if not pending:
        return assigned

    total_batches = (len(pending) + batch_size - 1) // batch_size
    if max_batches is not None:
        stop_after = max_batches
    elif n_done == 0:
        stop_after = min(first_run_probe, total_batches)
        logger.info(
            "First run with no prior progress -> stopping after %d batches "
            "as a sanity probe. Re-run to continue.",
            stop_after,
        )
    else:
        stop_after = total_batches

    if client is None:
        from tutorsim.client import ModelClient

        client = ModelClient(model)

    ask = _make_classifier(client, thinking=bool(thinking))

    for bi in range(stop_after):
        start = bi * batch_size
        batch = pending[start : start + batch_size]
        if not batch:
            break
        in_flight = batch
        for attempt in range(CLASSIFIER_MAX_RETRIES):
            label = f"batch {start + 1}-{start + len(batch)}"
            if attempt:
                label += f" retry{attempt}({len(in_flight)})"
            got, usage = ask(in_flight, label)
            _append_jsonl(usage_path, usage)
            assigned.update(got)
            _append_assignments(assignments_path, got)
            in_flight = [s for s in in_flight if s not in assigned]
            if not in_flight:
                break
        if in_flight:
            logger.warning(
                "Forced %d statements to '%s' after %d retries",
                len(in_flight),
                LAST_LETTER,
                CLASSIFIER_MAX_RETRIES,
            )
            forced = {s: LAST_LETTER for s in in_flight}
            assigned.update(forced)
            _append_assignments(assignments_path, forced)
        if (bi + 1) % _CLASSIFIER_PROGRESS_EVERY == 0:
            logger.info("classify_pool: %d/%d batches this run", bi + 1, stop_after)

    return assigned


def _make_classifier(client: Any, *, thinking: bool = False):
    """Build a `(statements, label) -> ({stmt: letter}, usage_dict)` callable.

    `client` exposes `.generate(...)` (a `tutorsim.client.ModelClient` or a
    compatible test double). The structured-output schema pins `category` to
    an enum of A-M so the model cannot return any other letter.
    """
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "category": {"type": "string", "enum": CATEGORY_LETTERS},
                    },
                    "required": ["id", "category"],
                },
            }
        },
        "required": ["assignments"],
    }
    cat_block = _categories_block()
    template = _load_classify_prompt()

    def ask(items: list[str], label: str) -> tuple[dict[str, str], dict]:
        statements_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(items))
        prompt = template.substitute(
            categories_block=cat_block,
            statements_block=statements_block,
        )
        # Route through the shared ModelClient: provider routing, retry, and
        # usage bookkeeping are shared with the tutor/student/scorer paths, and
        # the model comes from config. output_schema preserves the enum-
        # constrained structured output the classifier relies on.
        resp = client.generate(
            prompt,
            json_mode=False,
            max_tokens=4000,
            thinking=thinking,
            output_schema=schema,
        )
        text = resp.text
        # Structured-output guarantees parseable JSON in the normal case, but the
        # response may still arrive truncated or with leading prose on failures.
        # Returning {} drops the batch through to the retry loop in classify_pool.
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "classifier returned unparseable JSON for %s; "
                "treating batch as empty so retry-on-incomplete fires",
                label,
            )
            result = {"assignments": []}
        out: dict[str, str] = {}
        for a in result.get("assignments", []):
            idx = a["id"] - 1
            if 0 <= idx < len(items):
                out[items[idx]] = a["category"]
        in_t = resp.usage.get("input_tokens", 0)
        out_t = resp.usage.get("output_tokens", 0)
        usage = {
            "label": label,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "total_tokens": in_t + out_t,
        }
        return out, usage

    return ask


def _read_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["stmt"]] = d["cat"]
    return out


def _append_assignments(path: Path, new: dict[str, str]) -> None:
    with path.open("a") as f:
        for stmt, cat in new.items():
            f.write(json.dumps({"stmt": stmt, "cat": cat}) + "\n")


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def attach_categories(
    facets: Iterable[Facet], assignments: dict[str, str]
) -> list[Facet]:
    """Set `facet.category` for every facet based on its statement.

    Raises KeyError on any statement missing from `assignments` so partial
    runs cannot silently drop data downstream.
    """
    out: list[Facet] = []
    for f in facets:
        stmt = f.statement.strip()
        if stmt not in assignments:
            raise KeyError(f"no category for statement: {stmt[:80]!r}")
        f.category = assignments[stmt]
        out.append(f)
    return out


def write_classified_csv(facets: Iterable[Facet], path: Path) -> None:
    """Persist a classified facet stream to a CSV (full Facet schema).

    Uses atomic write-and-rename so a crash mid-write never leaves a partial CSV.
    """
    _atomic_write_csv(Path(path), _FACET_FIELDS, (f.to_dict() for f in facets))


def read_classified_csv(path: Path) -> list[Facet]:
    """Load a classified facet stream produced by `write_classified_csv`."""
    return list(_read_facets_csv(Path(path)))


def read_paper_distribution(csv_path: Path, series: str = "human"):
    """Load one series from a precomputed action-distribution CSV.

    Reads the paper's frozen distribution values (e.g.
    ``analysis/working-paper-20260630/v1_action_taxonomy_distribution.csv``),
    which store, per category letter, ``{series}__macro_mean_pct`` plus
    ``__ci_low`` / ``__ci_high``. Use this to reuse the published **human
    reference** distribution instead of re-classifying the ground truth.

    Args:
        csv_path: path to the distribution CSV.
        series: column-group prefix -- ``"human"`` (default) for the human
            reference, or a ``"<model>__<mode>"`` key (e.g.
            ``"claude_opus_4_8__plain"``) for a specific model cell.

    Returns:
        A pandas DataFrame indexed by ``letter`` with columns ``name``,
        ``orientation``, ``mean_pct``, ``ci_low``, ``ci_high``. Requires the
        ``analysis`` extra (pandas).
    """
    _require_analysis_extras()
    pd = importlib.import_module("pandas")
    df = pd.read_csv(Path(csv_path))
    col = f"{series}__macro_mean_pct"
    if col not in df.columns:
        raise KeyError(
            f"series '{series}' not found in {csv_path} (expected column '{col}')"
        )
    return pd.DataFrame(
        {
            "letter": df["letter"],
            "name": df["name"],
            "orientation": df["orientation"],
            "mean_pct": df[col],
            "ci_low": df[f"{series}__ci_low"],
            "ci_high": df[f"{series}__ci_high"],
        }
    ).set_index("letter")


# ============================================================================
# 6. Headline analysis (macro/moment tables)
# ============================================================================
#
# Every percentage in the headline is a macro mean over moments: for each
# moment we compute the within-moment fraction of facets in each category,
# then average across moments. This controls for per-moment verbosity so a
# moment decomposed into more facets doesn't dominate.
#
# Pandas is required from here down. Importing this module without pandas
# is fine; only these functions raise if it's missing.


def _require_analysis_extras() -> None:
    """Raise ImportError with an install hint if pandas is missing.

    The headline tables use pandas only; matplotlib/seaborn are needed by the
    analysis figures (notebooks + benchmark_perf_cost.py), not by this module.
    """
    try:
        importlib.import_module("pandas")
    except ImportError:
        raise ImportError(
            "tutorsim taxonomy headline tables require pandas. "
            "Install with: pip install 'tutorsim[analysis]'"
        )


def facets_to_dataframe(facets: Iterable[Facet]):
    """Materialise a Facet stream as a pandas DataFrame.

    Adds an `orientation` column derived from `category` so headline
    functions don't have to re-derive it on every groupby.
    """
    _require_analysis_extras()
    pd = importlib.import_module("pandas")
    df = pd.DataFrame([f.to_dict() for f in facets])
    if "category" in df.columns:
        df["orientation"] = df["category"].map(ORIENTATION_BY_LETTER)
    return df


def _normal_ci(values, z: float = 1.96) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) for a sequence of per-moment fractions."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    import math

    se = math.sqrt(var / n)
    return (mean, max(0.0, mean - z * se), min(1.0, mean + z * se))


def _moment_fractions(df, value_col: str, moment_col: str = "moment_id"):
    """Per-moment within-moment fractions of each value in `value_col`.

    Returns a DataFrame indexed by moment_id with one column per value;
    rows sum to 1. Missing combinations are 0.
    """
    importlib.import_module("pandas")
    counts = df.groupby([moment_col, value_col]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1).replace(0, 1)
    return counts.div(totals, axis=0)


def macro_distribution(df, group_keys: tuple[str, ...] = ()):
    """Macro-mean % of each category, optionally per `group_keys` cell.

    For each cell: per-moment within-moment fractions for each letter,
    averaged across moments. Returns a long-format DataFrame with columns
    `*group_keys, letter, n_moments, mean_pct, ci_low, ci_high`.
    """
    _require_analysis_extras()
    pd = importlib.import_module("pandas")
    out_rows = []
    grouper = df.groupby(list(group_keys)) if group_keys else [((), df)]
    for key, gdf in grouper:
        if not isinstance(key, tuple):
            key = (key,)
        moment_frac = _moment_fractions(gdf, "category")
        moment_frac = moment_frac.reindex(columns=CATEGORY_LETTERS, fill_value=0.0)
        n_moments = len(moment_frac)
        for letter in CATEGORY_LETTERS:
            mean, lo, hi = _normal_ci(moment_frac[letter].tolist())
            out_rows.append(
                {
                    **{k: v for k, v in zip(group_keys, key)},
                    "letter": letter,
                    "n_moments": n_moments,
                    "mean_pct": mean * 100,
                    "ci_low": lo * 100,
                    "ci_high": hi * 100,
                }
            )
    return pd.DataFrame(out_rows)


def macro_orientation(df, group_keys: tuple[str, ...] = ()):
    """Macro-mean % of each orientation (scaffolding/rigor/neutral) per cell.

    Same shape as `macro_distribution` but with an `orientation` column
    instead of `letter`.
    """
    _require_analysis_extras()
    pd = importlib.import_module("pandas")
    out_rows = []
    orients = ["scaffolding", "rigor", "neutral"]
    grouper = df.groupby(list(group_keys)) if group_keys else [((), df)]
    for key, gdf in grouper:
        if not isinstance(key, tuple):
            key = (key,)
        moment_frac = _moment_fractions(gdf, "orientation")
        moment_frac = moment_frac.reindex(columns=orients, fill_value=0.0)
        n_moments = len(moment_frac)
        for o in orients:
            mean, lo, hi = _normal_ci(moment_frac[o].tolist())
            out_rows.append(
                {
                    **{k: v for k, v in zip(group_keys, key)},
                    "orientation": o,
                    "n_moments": n_moments,
                    "mean_pct": mean * 100,
                    "ci_low": lo * 100,
                    "ci_high": hi * 100,
                }
            )
    return pd.DataFrame(out_rows)


def macro_appropriateness(df, group_keys: tuple[str, ...] = ()):
    """Per-cell appropriateness scores (macro / moment).

    For each cell: in scaffolding moments, the macro % of facets that are
    scaffolding-oriented; in rigor moments, the macro % that are
    rigor-oriented. A well-calibrated tutor scores high on both.
    """
    _require_analysis_extras()
    pd = importlib.import_module("pandas")
    out_rows = []
    grouper = df.groupby(list(group_keys)) if group_keys else [((), df)]
    for key, gdf in grouper:
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        for sit, target in (("scaffolding", "scaffolding"), ("rigor", "rigor")):
            sub = gdf[gdf["situation_label"] == sit]
            if sub.empty:
                row[f"{sit}_n_moments"] = 0
                row[f"{sit}_approp_pct"] = 0.0
                row[f"{sit}_ci_low"] = 0.0
                row[f"{sit}_ci_high"] = 0.0
                continue
            moment_frac = _moment_fractions(sub, "orientation")
            vals = moment_frac.get(target, pd.Series([0.0] * len(moment_frac))).tolist()
            mean, lo, hi = _normal_ci(vals)
            row[f"{sit}_n_moments"] = len(moment_frac)
            row[f"{sit}_approp_pct"] = mean * 100
            row[f"{sit}_ci_low"] = lo * 100
            row[f"{sit}_ci_high"] = hi * 100
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def prompt_effect_deltas(lm_df, model_col: str = "model", prompt_col: str = "prompt"):
    """Per-model SR-minus-plain deltas on the orientation rollup.

    Returns a DataFrame indexed by model with columns
    `d_scaffolding, d_rigor, d_neutral` (percentage points).
    """
    _require_analysis_extras()
    pd = importlib.import_module("pandas")  # used to build the final DataFrame
    orient = macro_orientation(lm_df, group_keys=(model_col, prompt_col))
    pivot = orient.pivot_table(
        index=[model_col, "orientation"], columns=prompt_col, values="mean_pct"
    )
    out = {}
    prompts = pivot.columns.tolist()
    if not {"plain"}.issubset(prompts):
        raise ValueError(
            f"prompt_effect_deltas requires a 'plain' prompt; got {prompts}"
        )
    sr_col = next((p for p in prompts if p != "plain"), None)
    if sr_col is None:
        raise ValueError(
            "prompt_effect_deltas requires a second prompt alongside 'plain'"
        )
    for (model, orientation), row in pivot.iterrows():
        out.setdefault(model, {})[f"d_{orientation}"] = row[sr_col] - row["plain"]
    return pd.DataFrame(out).T.reset_index(names=model_col)


def js_divergence(p, q, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence in base 2; output in [0, 1]."""
    import math

    p = list(p)
    q = list(q)

    def kl(a, b):
        out = 0.0
        for ai, bi in zip(a, b):
            ai = max(ai, eps)
            bi = max(bi, eps)
            out += ai * math.log2(ai / bi)
        return out

    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_divergence_to_human(
    human_df, lm_df, group_keys: tuple[str, ...] = ("model", "prompt")
):
    """Per LM cell, JS divergence between its macro distribution and human's.

    Returns a DataFrame with `*group_keys, n_moments, js_divergence` sorted
    by divergence ascending (closest to human first).
    """
    _require_analysis_extras()
    pd = importlib.import_module("pandas")  # used at return
    h = macro_distribution(human_df)
    human_vec = (
        h.set_index("letter")["mean_pct"].reindex(CATEGORY_LETTERS, fill_value=0.0)
        / 100
    ).tolist()
    cells = macro_distribution(lm_df, group_keys=group_keys)
    rows = []
    for key, sub in cells.groupby(list(group_keys)):
        if not isinstance(key, tuple):
            key = (key,)
        vec = (
            sub.set_index("letter")["mean_pct"].reindex(
                CATEGORY_LETTERS, fill_value=0.0
            )
            / 100
        ).tolist()
        rows.append(
            {
                **{k: v for k, v in zip(group_keys, key)},
                "n_moments": int(sub["n_moments"].iloc[0]),
                "js_divergence": js_divergence(vec, human_vec),
            }
        )
    return pd.DataFrame(rows).sort_values("js_divergence").reset_index(drop=True)


def build_headline_tables(
    human_facets: Iterable[Facet], lm_facets: Iterable[Facet]
) -> dict[str, Any]:
    """Compute all five headline tables in one pass.

    Returns a dict keyed by table name:
      distribution, orientation_rollup, appropriateness, prompt_effect,
      js_divergence_to_human.
    """
    _require_analysis_extras()
    human_df = facets_to_dataframe(human_facets)
    lm_df = facets_to_dataframe(lm_facets)
    human_df["cell"] = "human"
    lm_df["cell"] = lm_df["model"].astype(str) + " / " + lm_df["prompt"].astype(str)
    combined = _pd_concat([human_df, lm_df])
    return {
        "distribution": macro_distribution(combined, group_keys=("cell",)),
        "orientation_rollup": macro_orientation(combined, group_keys=("cell",)),
        "appropriateness": macro_appropriateness(combined, group_keys=("cell",)),
        "prompt_effect": prompt_effect_deltas(lm_df),
        "js_divergence_to_human": js_divergence_to_human(human_df, lm_df),
    }


def _pd_concat(frames):
    pd = importlib.import_module("pandas")
    return pd.concat(frames, ignore_index=True)


def write_headline_csvs(tables: dict[str, Any], out_dir: Path) -> None:
    """Persist every table to `out_dir/<table_name>.csv`.

    Uses atomic write-and-rename so a crash mid-write never leaves a partial CSV.
    """
    _require_analysis_extras()
    import os

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        final = out_dir / f"{name}.csv"
        tmp = final.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, final)


def read_headline_csvs(in_dir: Path) -> dict[str, Any]:
    """Load every table written by `write_headline_csvs`."""
    _require_analysis_extras()
    pd = importlib.import_module("pandas")
    in_dir = Path(in_dir)
    out: dict[str, Any] = {}
    for path in sorted(in_dir.glob("*.csv")):
        out[path.stem] = pd.read_csv(path)
    return out


# ============================================================================
# 7. Pipeline orchestration
# ============================================================================
#
# Each `run_*` function is idempotent and writes its outputs to disk so the
# next stage can read them back. They are usable from Python and from the CLI.

InputSpec = dict[
    str, Any
]  # {"kind": "key_moments"|"tutorsim"|"canonical", "path": "...", "scenarios": "..."}


def _adapter_for(spec: InputSpec) -> Iterator[Facet]:
    kind = spec.get("kind")
    if kind == "key_moments":
        return load_key_moments_jsonl(Path(spec["path"]))
    if kind == "tutorsim":
        return load_tutorsim_results(Path(spec["path"]), Path(spec["scenarios"]))
    if kind == "canonical":
        return load_canonical_jsonl(Path(spec["path"]))
    raise ValueError(f"unknown input kind: {kind!r}")


def run_classify(input_spec: InputSpec, out_dir: Path, **classify_kwargs) -> Path:
    """Filter -> classify -> write a classified.csv to `out_dir`.

    Returns the path to the written classified.csv. Sidecar files
    (assignments.jsonl, usage_log.jsonl) live alongside.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    facets = list(_adapter_for(input_spec))
    logger.info("loaded %d facets from %s", len(facets), input_spec)
    kept, excluded = build_pool(facets)
    write_pool_csv(kept, excluded, out_dir)
    print(pool_report(kept, excluded))

    assignments = classify_pool(kept, out_dir, **classify_kwargs)
    # Only finalise the classified CSV when every kept statement has a label;
    # partial runs leave the sidecar in place for the next invocation.
    missing = [
        f.statement.strip() for f in kept if f.statement.strip() not in assignments
    ]
    if missing:
        logger.info(
            "%d statements still unassigned; run again to finish classification",
            len(missing),
        )
        return out_dir / "classified.csv"
    enriched = attach_categories(kept, assignments)
    classified_path = out_dir / "classified.csv"
    write_classified_csv(enriched, classified_path)
    return classified_path


def run_headline(
    human_classified: Path, lm_classified: Path, out_dir: Path
) -> dict[str, Any]:
    """Compute the 5 headline tables from two classified.csv files."""
    human = read_classified_csv(Path(human_classified))
    lm = read_classified_csv(Path(lm_classified))
    tables = build_headline_tables(human, lm)
    write_headline_csvs(tables, Path(out_dir))
    return tables


def run_all(
    human_input: InputSpec, lm_input: InputSpec, out_dir: Path, **classify_kwargs
) -> dict[str, Any]:
    """Run the whole pipeline end-to-end.

    Layout under `out_dir`:
      human/         classified.csv, pool.csv, excluded.csv, assignments.jsonl
      lm/            classified.csv, pool.csv, excluded.csv, assignments.jsonl
      headline/      distribution.csv ... js_divergence_to_human.csv

    Figures are produced by paper-facing notebooks under
    `analysis/working-paper-*` that consume the headline CSVs.
    """
    out_dir = Path(out_dir)
    human_classified = run_classify(human_input, out_dir / "human", **classify_kwargs)
    lm_classified = run_classify(lm_input, out_dir / "lm", **classify_kwargs)
    if not human_classified.exists() or not lm_classified.exists():
        logger.info("classification incomplete; rerun to finish before headline")
        return {}
    tables = run_headline(human_classified, lm_classified, out_dir / "headline")
    return {"headline": tables}


def classify_run(
    annotations: Iterable[Any],
    moments: Iterable[Any],
    out_dir: Path,
    *,
    model: str,
    mode: str,
    client: Any = None,
) -> dict[str, Any]:
    """Classify a completed run's tutor actions; write the taxonomy artifacts.

    This is the in-run entrypoint (called on every `tutorsim run`). `model`
    and `mode` are the tutor/prompt recorded on the facets; the classifier
    model comes from the `taxonomy` config block (see `classify_pool`).

    Writes into `out_dir`: pool.csv, excluded.csv, assignments.jsonl,
    usage_log.jsonl, classified.csv. Resume-safe via the assignments sidecar
    (a resumed run reclassifies nothing). Classification runs to completion
    (the sanity-probe cap is disabled here).

    Returns a summary dict:
      {scheme_version, counts: {letter: n}, n_facets, excluded, usage}.
    """
    out_dir = Path(out_dir)
    facets = list(facets_from_annotations(annotations, moments, model=model, mode=mode))
    kept, excluded = build_pool(facets)
    write_pool_csv(kept, excluded, out_dir)
    # first_run_probe disabled: an integrated run classifies to completion in
    # one invocation rather than stopping for a manual re-run.
    assignments = classify_pool(
        kept,
        out_dir,
        client=client,
        first_run_probe=10**9,
    )
    classified = attach_categories(kept, assignments)
    write_classified_csv(classified, out_dir / "classified.csv")

    from collections import Counter

    counts = Counter(f.category for f in classified)

    # Roll categories up to their orientation (scaffolding / rigor / neutral)
    # for a compact, interpretable mix in the run summary.
    orientation = Counter()
    for letter, n in counts.items():
        orientation[ORIENTATION_BY_LETTER.get(letter, "neutral")] += n

    # Sum per-batch usage from the sidecar for the run's token bookkeeping.
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_path = out_dir / "usage_log.jsonl"
    if usage_path.exists():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in usage:
                usage[k] += int(rec.get(k, 0) or 0)

    return {
        "scheme_version": SCHEME_VERSION,
        "counts": dict(sorted(counts.items())),
        "orientation": {
            "scaffolding": orientation.get("scaffolding", 0),
            "rigor": orientation.get("rigor", 0),
            "neutral": orientation.get("neutral", 0),
        },
        "n_facets": len(classified),
        "excluded": len(excluded),
        "usage": usage,
    }


# ============================================================================
# 8. CLI dispatcher
# ============================================================================


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tutorsim taxonomy",
        description="Classify decomposed tutor actions into A-M and produce "
        "headline tables. Paper figures live in working-paper notebooks.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    input_kinds = ["key_moments", "tutorsim", "canonical"]

    p_cls = sub.add_parser("classify", help="filter + classify a single input")
    p_cls.add_argument("--kind", required=True, choices=input_kinds)
    p_cls.add_argument(
        "--input",
        required=True,
        help="key-moments jsonl, tutorsim results/<run_id>/, "
        "or a canonical facet jsonl",
    )
    p_cls.add_argument(
        "--scenarios", help="scenarios.jsonl (required for --kind tutorsim)"
    )
    p_cls.add_argument("--output", required=True, help="output directory")
    p_cls.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap LLM batches this run (e.g. for first-run probes)",
    )

    p_hl = sub.add_parser("headline", help="build the 5 headline CSVs")
    p_hl.add_argument(
        "--human",
        required=True,
        type=Path,
        help="classified.csv from `taxonomy classify` for the human pool",
    )
    p_hl.add_argument(
        "--lm",
        required=True,
        type=Path,
        help="classified.csv from `taxonomy classify` for the LM pool",
    )
    p_hl.add_argument("--output", required=True, type=Path)

    p_run = sub.add_parser("run", help="classify (twice) -> headline end-to-end")
    p_run.add_argument("--human-kind", default="key_moments", choices=input_kinds)
    p_run.add_argument("--human-input", required=True)
    p_run.add_argument("--human-scenarios")
    p_run.add_argument("--lm-kind", default="tutorsim", choices=input_kinds)
    p_run.add_argument("--lm-input", required=True)
    p_run.add_argument("--lm-scenarios")
    p_run.add_argument("--output", required=True, type=Path)
    p_run.add_argument("--max-batches", type=int, default=None)

    return parser


def cli_dispatch(argv: Optional[list[str]] = None) -> int:
    """Entry point for `tutorsim taxonomy ...`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_argparser().parse_args(argv)

    if args.cmd == "classify":
        spec: InputSpec = {"kind": args.kind, "path": args.input}
        if args.kind == "tutorsim":
            if not args.scenarios:
                raise SystemExit("--scenarios is required when --kind tutorsim")
            spec["scenarios"] = args.scenarios
        run_classify(spec, Path(args.output), max_batches=args.max_batches)
        return 0

    if args.cmd == "headline":
        run_headline(args.human, args.lm, args.output)
        return 0

    if args.cmd == "run":
        human_spec: InputSpec = {"kind": args.human_kind, "path": args.human_input}
        if args.human_kind == "tutorsim":
            if not args.human_scenarios:
                raise SystemExit(
                    "--human-scenarios is required when --human-kind tutorsim"
                )
            human_spec["scenarios"] = args.human_scenarios
        lm_spec: InputSpec = {"kind": args.lm_kind, "path": args.lm_input}
        if args.lm_kind == "tutorsim":
            if not args.lm_scenarios:
                raise SystemExit("--lm-scenarios is required when --lm-kind tutorsim")
            lm_spec["scenarios"] = args.lm_scenarios
        run_all(human_spec, lm_spec, args.output, max_batches=args.max_batches)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(cli_dispatch())
