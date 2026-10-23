"""Tutor action-facet taxonomy: classify decomposed actions into a 13-letter
scheme (A-M) and compute population-level macro/moment statistics.

This is the taxonomy *data-generation* layer, part of the tutormoments runtime.
Figure *rendering* lives in the analysis notebooks under
analysis/working-paper-*, which import these symbols
(`from tutormoments import taxonomy as tx`) and draw from the tables produced here.

Input contracts (use either; both are tutormoments-native, no new file format):

  1. Key-moments JSONL: one row per conversation, shape
       {conversation_id, num_turns, key_moments: [...]}.
     Each key_moment carries annotation_type, turn_start, turn_end,
     situation_label_agg, action_decomposed, .... This matches the
     ground-truth bundle distributed alongside the benchmark (the human
     reference distribution).

  2. Tutormoments run results: a directory of per-scenario score files
       results/<run_id>/scores/<scenario_id>.json, where each file is an
       Annotation produced by tutormoments.scoring.score. Pair with the
       scenarios.jsonl used for the run to recover the situation label,
       model, and prompt for each scenario. (During a run, the LM side is
       classified in-memory via `facets_from_annotations`.)

Pipeline (each stage idempotent and writable to/readable from disk):
  load_* / facets_from_annotations  -> Facet stream
  build_pool -> kept facets + excluded facets (with reason)
  classify_pool -> {statement -> category letter}, resume-safe via sidecar
  kl_situation -> scaffolding-vs-rigor KL, recorded in every run's summary
  human_reference -> the same KL for the human tutors, from the release

pandas is an optional extra (install with `pip install 'tutormoments[analysis]'`),
needed only by the DataFrame helpers the figure notebooks use
(facets_to_dataframe, macro_distribution, read_paper_distribution). Importing
this module never requires it, and the KL path is pure Python.
matplotlib/seaborn are not used here -- they belong to the analysis figures.
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

from tutormoments.resources import resource_text
from tutormoments.usage import EMPTY_USAGE, add_usage

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
    source: str = ""  # "hf" | "tutormoments" | "canonical"
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


def load_tutormoments_results(
    results_dir: Path, scenarios_path: Path
) -> Iterator[Facet]:
    """Stream Facets from a tutormoments run results directory.

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

    # config.json is written by `tutormoments.cli` with the run's tutor + mode.
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
                source="tutormoments",
            )


def facets_from_annotations(
    annotations: Iterable[Any],
    moments: Iterable[Any],
    *,
    model: str,
    mode: str,
) -> Iterator[Facet]:
    """Stream Facets from in-memory scored Annotations + their Moments.

    The in-run equivalent of `load_tutormoments_results` -- no disk round-trip and
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
                source="tutormoments",
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
    (see `tutormoments.moments`). Any moment without it is treated as
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

    Tutormoments scenario IDs typically end with the source transcript UUID; we
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
# Classifies unique statements against the frozen A-M scheme using the
# `taxonomy` config block's model with structured-output JSON (the category
# letter is a schema enum, so the model can't hallucinate one). Statements
# are deduplicated, batched, and assignments are appended to a sidecar JSONL
# so a crash or ctrl-C loses at most the in-flight batch.
#
# Designed for two-step operation: a small first-run probe (default 25
# batches, ~$1) for sanity-checking the distribution, then a resume call
# (re-run without --max-batches) to finish.

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
    thinking=None,
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
            `tutormoments.client.ModelClient` or a compatible test double). If
            None we build a ModelClient from the `taxonomy` config block
            (requires the provider API key).
        model / thinking / batch_size: override the `taxonomy` config block;
            each falls back to config when None (a missing or broken config
            raises -- there is no silent module default). `thinking` is a
            provider-native thinking mapping, same contract as
            ModelClient.generate.

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
    # tutormoments.taxonomy` stays lightweight (no client/provider-SDK import
    # unless we actually classify). A missing or broken config raises here:
    # the classifier must never silently substitute its own LM settings.
    if model is None or thinking is None or batch_size is None:
        from tutormoments.config import taxonomy_spec

        spec = taxonomy_spec()
        if model is None:
            model = spec.model
        if thinking is None:
            thinking = spec.thinking
        if batch_size is None:
            batch_size = spec.batch_size

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
    elif n_done == 0 and first_run_probe < total_batches:
        # Only a probe that actually stops short is announced; integrated
        # runs disable it (first_run_probe=10**9) and classify every batch.
        stop_after = first_run_probe
        logger.info(
            "First run with no prior progress -> stopping after %d of %d "
            "batches as a sanity probe. Re-run to continue.",
            stop_after,
            total_batches,
        )
    else:
        stop_after = total_batches

    if client is None:
        from tutormoments.client import ModelClient

        client = ModelClient(model)

    ask = _make_classifier(client, thinking=thinking)

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


def _make_classifier(client: Any, *, thinking=None):
    """Build a `(statements, label) -> ({stmt: letter}, usage_dict)` callable.

    `client` exposes `.generate(...)` (a `tutormoments.client.ModelClient` or a
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
        # Log the full usage dict (legacy counters + canonical cost vector +
        # provenance) so the sidecar carries everything costing needs; the
        # per-batch label rides along as call-level detail.
        usage = {"label": label, **resp.usage}
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
# 6. Macro distributions (moment-level tables for the figure notebooks)
# ============================================================================
#
# Every percentage here is a macro mean over moments: for each
# moment we compute the within-moment fraction of facets in each category,
# then average across moments. This controls for per-moment verbosity so a
# moment decomposed into more facets doesn't dominate.
#
# Pandas is required from here down. Importing this module without pandas
# is fine; only these functions raise if it's missing.


def _require_analysis_extras() -> None:
    """Raise ImportError with an install hint if pandas is missing.

    The DataFrame helpers use pandas only; matplotlib/seaborn are needed by the
    analysis figures (notebooks + benchmark_perf_cost.py), not by this module.
    """
    try:
        importlib.import_module("pandas")
    except ImportError:
        raise ImportError(
            "tutormoments taxonomy DataFrame helpers require pandas. "
            "Install with: pip install 'tutormoments[analysis]'"
        )


def facets_to_dataframe(facets: Iterable[Facet]):
    """Materialise a Facet stream as a pandas DataFrame.

    Adds an `orientation` column derived from `category` so callers don't
    have to re-derive it on every groupby.
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


# ============================================================================
# 7. Scaffolding-vs-rigor KL divergence
# ============================================================================
#
# How differently a tutor acts in scaffolding moments (S) vs rigor moments (R):
# KL(S||R) and KL(R||S) between the two macro distributions. The method is the
# paper's (analysis/working-paper-20260630, locked kl_divergence_table.tex):
# per situation, macro % -> pseudo-counts over that situation's moments
# (pct / 100 * n), add-one smoothing, KL in nats.
#
# Pure Python (no pandas), so every run records it without the analysis extra.
# The smoothing's pull toward uniform shrinks as n grows, so KL values are
# only comparable at similar n; n is recorded beside them for that reason.

KL_SITUATIONS: tuple[str, str] = ("scaffolding", "rigor")


def smoothed_probs(
    pcts: list[float], n_moments: int, alpha: float = 1.0
) -> list[float]:
    """Macro % -> add-`alpha` smoothed probabilities over `n_moments` moments."""
    counts = [p / 100.0 * n_moments + alpha for p in pcts]
    total = sum(counts)
    return [c / total for c in counts]


def kl_nats(p: list[float], q: list[float]) -> float:
    """KL(p||q) in nats. Both must be strictly positive (see smoothed_probs)."""
    import math

    return sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))


def macro_pcts(facets: Iterable[Facet]) -> tuple[list[float], int]:
    """Macro-mean % of each letter over moments, plus the moment count.

    The pure-Python `macro_distribution` mean: each moment's within-moment
    fraction per letter, averaged across moments. Facets without a category
    are ignored; a moment with none is not a moment.
    """
    from collections import Counter

    by_moment: dict[str, Counter] = {}
    for f in facets:
        if f.category is None:
            continue
        by_moment.setdefault(f.moment_id, Counter())[f.category] += 1
    n = len(by_moment)
    if n == 0:
        return [0.0] * len(CATEGORY_LETTERS), 0
    totals = [sum(c.values()) for c in by_moment.values()]
    pcts = [
        100.0 * sum(c[letter] / t for c, t in zip(by_moment.values(), totals)) / n
        for letter in CATEGORY_LETTERS
    ]
    return pcts, n


def kl_situation(facets: Iterable[Facet]) -> dict[str, Any]:
    """S-vs-R KL divergence of one cell's classified facets.

    Returns ``{s_r, r_s, n_scaffolding, n_rigor}``: KL(S||R) and KL(R||S) in
    nats, and the moment count each side was smoothed over. Moments are
    grouped by `moment_id` (a multi-trial run pools its trials under one id),
    and only `situation_label` scaffolding / rigor moments take part. Both KL
    values are None when either situation has no classified moment.
    """
    facets = list(facets)
    out: dict[str, Any] = {"s_r": None, "r_s": None}
    dists = {}
    for sit in KL_SITUATIONS:
        pcts, n = macro_pcts(f for f in facets if f.situation_label == sit)
        out[f"n_{sit}"] = n
        if n:
            dists[sit] = smoothed_probs(pcts, n)
    if len(dists) == len(KL_SITUATIONS):
        s, r = dists["scaffolding"], dists["rigor"]
        out["s_r"], out["r_s"] = kl_nats(s, r), kl_nats(r, s)
    return out


# -- Human reference ----------------------------------------------------------
#
# The human side of the comparison, on the same moments the runs use: the
# human-tutor actions at each `moments.jsonl` moment's span, read from the
# published `action_taxonomy` config (already classified -- no API calls).
# Every annotator at a span is pooled into that one moment, and the moment's
# own `dimension` is its situation label (moments.jsonl is authoritative,
# even where the ground-truth label at the span differs). Pinned to a full
# commit SHA so the reference cannot drift under a report.

HUMAN_REFERENCE: tuple[str, str] = (
    "allenai/tutormoments-preview",
    "3bb0c104d65e29facffca7ba864ef10fa22d0e78",
)
HUMAN_REFERENCE_LABEL = "balanced_520 human tutors"
HUMAN_REFERENCE_FILES = ("moments.jsonl", "action_taxonomy.jsonl")
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _hf_download(dataset: str, filename: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download  # ships with `datasets`

    return Path(
        hf_hub_download(dataset, filename, repo_type="dataset", revision=revision)
    )


def human_reference_facets(
    moments_path: Path, action_taxonomy_path: Path
) -> list[Facet]:
    """Classified human-tutor facets at each moment's span, one moment per span.

    A moment's span is (`provenance.conv_id`'s last ``_`` part,
    `provenance.turn_start`, `provenance.turn_end`); human `action_taxonomy`
    rows join on (`transcript_id`, `turn_start`, `turn_end`). Each facet takes
    the moment's id (pooling annotators) and the moment's `dimension`.
    """
    spans: dict[tuple[str, int, int], tuple[str, str]] = {}
    for m in _iter_jsonl(moments_path):
        prov = m.get("provenance") or {}
        conv_id = prov.get("conv_id")
        if not conv_id or prov.get("turn_start") is None:
            continue
        key = (conv_id.rsplit("_", 1)[-1], prov["turn_start"], prov["turn_end"])
        spans[key] = (m["id"], m.get("dimension") or "unknown")

    out: list[Facet] = []
    for r in _iter_jsonl(action_taxonomy_path):
        if r.get("source") != "human" or r.get("category") is None:
            continue
        hit = spans.get((r["transcript_id"], r["turn_start"], r["turn_end"]))
        if hit is None:
            continue
        moment_id, dimension = hit
        out.append(
            Facet(
                moment_id=moment_id,
                transcript_id=r["transcript_id"],
                turn_start=r["turn_start"],
                turn_end=r["turn_end"],
                statement_index=r["statement_index"],
                statement=r["statement"],
                annotation_type="scaffolding",
                situation_label=dimension,
                action_label=r.get("action_label"),
                result_label=r.get("result_label"),
                source="hf",
                stance_prefixed=bool(r.get("stance_prefixed")),
                category=r["category"],
            )
        )
    return out


def human_reference(
    dataset: str = HUMAN_REFERENCE[0],
    revision: str = HUMAN_REFERENCE[1],
    *,
    download=None,
) -> dict[str, Any]:
    """S-vs-R KL of the human tutors on the released moments, at `revision`.

    `revision` must be a full 40-character commit SHA. `download(dataset,
    filename, revision) -> Path` fetches one release file (default: the
    Hugging Face cache). Returns ``{label, dataset, revision, kl}`` with `kl`
    shaped like `kl_situation`'s. Raises on any download or parse failure;
    callers that treat the reference as optional catch it.
    """
    if not _FULL_SHA_RE.match(revision or ""):
        raise ValueError(
            f"human reference revision must be a full 40-character commit SHA, "
            f"got {revision!r}"
        )
    download = download or _hf_download
    moments_path, taxonomy_path = (
        download(dataset, name, revision) for name in HUMAN_REFERENCE_FILES
    )
    return {
        "label": HUMAN_REFERENCE_LABEL,
        "dataset": dataset,
        "revision": revision,
        "kl": kl_situation(human_reference_facets(moments_path, taxonomy_path)),
    }


# ============================================================================
# 8. Pipeline orchestration
# ============================================================================
#
# `run_classify` is idempotent and writes its outputs to disk so the
# next stage can read them back. They are usable from Python and from the CLI.

InputSpec = dict[
    str, Any
]  # {"kind": "key_moments"|"tutormoments"|"canonical", "path": "...", "scenarios": "..."}


def _adapter_for(spec: InputSpec) -> Iterator[Facet]:
    kind = spec.get("kind")
    if kind == "key_moments":
        return load_key_moments_jsonl(Path(spec["path"]))
    if kind == "tutormoments":
        return load_tutormoments_results(Path(spec["path"]), Path(spec["scenarios"]))
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

    This is the in-run entrypoint (called on every `tutormoments run`). `model`
    and `mode` are the tutor/prompt recorded on the facets; the classifier
    model comes from the `taxonomy` config block (see `classify_pool`).

    Writes into `out_dir`: pool.csv, excluded.csv, assignments.jsonl,
    usage_log.jsonl, classified.csv. Resume-safe via the assignments sidecar
    (a resumed run reclassifies nothing). Classification runs to completion
    (the sanity-probe cap is disabled here).

    Returns a summary dict:
      {scheme_version, counts: {letter: n}, orientation, kl, n_facets,
       excluded, usage}
    where `kl` is `kl_situation` over the run's classified facets.
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
    # add_usage sums every integer key (canonical vector included) and keeps
    # provenance; the per-batch "label" string is dropped by design.
    usage = dict(EMPTY_USAGE)
    usage_path = out_dir / "usage_log.jsonl"
    if usage_path.exists():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            add_usage(usage, rec)

    return {
        "scheme_version": SCHEME_VERSION,
        "counts": dict(sorted(counts.items())),
        "orientation": {
            "scaffolding": orientation.get("scaffolding", 0),
            "rigor": orientation.get("rigor", 0),
            "neutral": orientation.get("neutral", 0),
        },
        "kl": kl_situation(classified),
        "n_facets": len(classified),
        "excluded": len(excluded),
        "usage": usage,
    }


# ============================================================================
# 9. CLI dispatcher
# ============================================================================


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tutormoments taxonomy",
        description="Classify decomposed tutor actions into A-M. Runs record "
        "their own S-vs-R KL; `tutormoments report` tabulates it. Paper figures "
        "live in working-paper notebooks.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    input_kinds = ["key_moments", "tutormoments", "canonical"]

    p_cls = sub.add_parser("classify", help="filter + classify a single input")
    p_cls.add_argument("--kind", required=True, choices=input_kinds)
    p_cls.add_argument(
        "--input",
        required=True,
        help="key-moments jsonl, tutormoments results/<run_id>/, "
        "or a canonical facet jsonl",
    )
    p_cls.add_argument(
        "--scenarios", help="scenarios.jsonl (required for --kind tutormoments)"
    )
    p_cls.add_argument("--output", required=True, help="output directory")
    p_cls.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="cap LLM batches this run (e.g. for first-run probes)",
    )

    return parser


def cli_dispatch(argv: Optional[list[str]] = None) -> int:
    """Entry point for `tutormoments taxonomy ...`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_argparser().parse_args(argv)

    if args.cmd == "classify":
        spec: InputSpec = {"kind": args.kind, "path": args.input}
        if args.kind == "tutormoments":
            if not args.scenarios:
                raise SystemExit("--scenarios is required when --kind tutormoments")
            spec["scenarios"] = args.scenarios
        run_classify(spec, Path(args.output), max_batches=args.max_batches)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(cli_dispatch())
