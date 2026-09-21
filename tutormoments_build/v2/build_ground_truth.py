"""Build iteration/test ground-truth JSONL from doubly annotated v2 moments.

This script takes doubly-annotated moments, resolves the annotators into one
label set, and writes them to ``data/ground_truth/`` split by ``splits.json``

Each moment has five booleans:

  situation  scaffolding_appropriate   scaffolding was called for here
             rigor_appropriate         a push for rigor was called for here
  action     scaffolding_present       the tutor scaffolded
             rigor_present             the tutor pushed for rigor
             over_scaffolding_present  the tutor scaffolded too much

**Who votes.** 
Project staff (``EXCLUDED_ANNOTATORS``) are dropped. 
If the same person saved the same role twice, only their highest ``revision`` counts. 
An adjudicator (inside ``payload["final"]``) counts as one more vote.
Anyone who threw the moment out answered nothing and abstains rather than voting no.

**Disagreements are resolved differently for situation versus action:**

* *situation*: by **majority over every vote**.
* *action* -- what the tutor did -- by the **adjudicators alone** if they're there. 
  With no adjudicator, it's a majority over every vote. Either way a
  tie resolves by union over whichever group was deciding.

``over_scaffolding_declared`` is an action field and follows the action rule.
Each moment also carries ``situation_type``, the five-way split of the situation
votes (``scaffold_only``, ``scaffold_maj``, ``fifty_fifty``, ``rigor_maj``,
``rigor_only``) -- a different cut from the two situation booleans, not a
restatement of them.

If ``scaffolding_present and not scaffolding_appropriate``, it counts as
over-scaffolding, inferred from the *resolved* labels after the vote.

**What is dropped:** retracted moments, moments a second pass or an adjudicator
threw out, and moments whose cut point a later pass redrew (the two passes did
not judge the same excerpt).

**Boundaries.** Where adjudicators drew their own, the strictest of them is
adopted -- the draft that ends earliest. With no adjudication the second pass's
redraw stands, and the originals are preserved under ``original_boundaries``.

Usage::

    python -m tutormoments_build.v2.build_ground_truth --dry-run
    python -m tutormoments_build.v2.build_ground_truth
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.v2.splits import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_MANIFEST,
    load_manifest,
)

logger = logging.getLogger("tutormoments_build.v2.build_ground_truth")

DEFAULT_OUT_DIR = "data/ground_truth"

# Project staff, excluded from the ground truth. They annotate to pilot the
# rubric and the interface, not as expert raters, so their passes must not reach
# the labels -- a moment left with fewer than two annotators after they are
# removed falls out through the usual doubly-annotated requirement. Matched on
# the first name, since that is what the export carries; the build logs every
# name it drops, so a teacher who happens to share one is visible rather than
# silently discarded.
EXCLUDED_ANNOTATORS = frozenset({"lucy", "rebecca", "albert", "kajal"})
DEFAULT_ANNOTATOR_LABELS = "data/v2_annotations/annotator_labels.json"

# split name in splits.json -> output file stem
SPLIT_FILES = {"iterate": "iteration", "heldout": "test"}

# The moment status the tool sets on a selection its selector withdrew before it
# reached a second pass. A withdrawn judgment is not a judgment.
RETRACTED = "retracted"

# The five ground-truth booleans, as (output field, payload section, payload key).
# over_scaffolding_present is derived separately from scaffolding_amount.
BOOLEAN_FIELDS = (
    ("scaffolding_appropriate", "situation", "scaffolding_appropriate"),
    ("rigor_appropriate", "situation", "rigor_appropriate"),
    ("scaffolding_present", "action", "scaffolding_present"),
    ("rigor_present", "action", "rigor_present"),
)
LABEL_FIELDS = tuple(name for name, _, _ in BOOLEAN_FIELDS) + (
    "over_scaffolding_present",
)

# Which rule settles a disagreement. The situation fields are what the moment
# *called for* -- a judgment the annotators are answering the same question
# about, so the group's balance of opinion is the answer. The action fields are
# what the tutor *did*, where a single annotator spotting a move the others
# missed is evidence it happened, so any True carries.
SITUATION_FIELDS = ("scaffolding_appropriate", "rigor_appropriate")
ACTION_FIELDS = ("scaffolding_present", "rigor_present", "over_scaffolding_declared")

# What is resolved across annotators: the four they answer directly, plus their
# literal over-scaffolding choice. over_scaffolding_present is not here -- it is
# derived from these after the vote, not voted on.
RESOLVED_FIELDS = SITUATION_FIELDS + ACTION_FIELDS

OVER_SCAFFOLDING = "over_scaffolding"

BOUNDARY_FIELDS = (
    "start_turn",
    "end_turn",
    "cut_turn",
    "start_index",
    "end_index",
    "cut_index",
    "dialogue_turns",
)

# How the moment's situation votes split between the two sides, as a single
# category per moment. This is a *different* cut from the two situation
# booleans, not a restatement of them: the booleans answer "was scaffolding
# called for" and "was a push for rigor called for" separately, and an annotator
# may answer yes to both, while this counts how many annotators landed on each
# side and names which way the group leaned. A moment can be rigor_maj with both
# booleans resolved True.
SITUATION_TYPES = (
    "scaffold_only",
    "scaffold_maj",
    "fifty_fifty",
    "rigor_maj",
    "rigor_only",
)
SITUATION_TYPE_LABELS = {
    "scaffold_only": "all say scaffolding, none rigor",
    "scaffold_maj": "majority say scaffolding",
    "fifty_fifty": "even split",
    "rigor_maj": "majority say rigor",
    "rigor_only": "all say rigor, none scaffolding",
}

# How a tied situation vote was settled, recorded per moment so a reader can see
# which labels rest on the tie rule rather than on a majority.
# situation: a tie the adjudicators' own call settled, and the fallback where
# no adjudicator saw the moment or they split too.
TIE_ADJUDICATOR = "adjudicator"
TIE_DEFAULT_TRUE = "default_true"
# action: a tie resolved by union over the deciding group -- the adjudicators
# where there were any, otherwise every annotator.
TIE_ADJUDICATOR_UNION = "adjudicator_union"
TIE_UNION = "union"

# Drop counters that count *annotations* rather than moments. The moments they
# came from are usually still emitted, so the report keeps them in their own
# list instead of implying that many moments went missing.
ANNOTATION_DROPS = frozenset({"staff_annotations_removed", "resaves_collapsed"})


# ===========================================================================
# Annotator de-identification
# ===========================================================================


def normalise_name(name: str) -> str:
    """Annotator name in the form the label map and the exclusion list are keyed on."""
    return (name or "").strip().lower().replace(" ", "-")


def is_excluded(annotation: dict) -> bool:
    """Whether this annotation is project staff's (see ``EXCLUDED_ANNOTATORS``).

    Matches the whole normalised name and its first part, so "Lucy" and
    "Lucy Li" are both caught.
    """
    name = normalise_name(annotation.get("annotator_name", ""))
    return bool(name) and (
        name in EXCLUDED_ANNOTATORS or name.split("-")[0] in EXCLUDED_ANNOTATORS
    )


def load_annotator_labels(path: str) -> dict[str, str]:
    """Return {normalised annotator name: de-identified label}, e.g. {"paul": "A02"}."""
    if not os.path.exists(path):
        logger.warning(
            "no annotator label map at %s; falling back to annotator ids", path
        )
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def annotator_label(annotation: dict, labels: dict[str, str]) -> str:
    """De-identified stand-in for an annotator.

    Real names must not reach the ground-truth files. Anyone missing from the
    label map falls back to their annotator_id, which is already a de-identified
    UUID and is stable across runs.
    """
    return (
        labels.get(normalise_name(annotation.get("annotator_name", "")))
        or annotation["annotator_id"]
    )


# ===========================================================================
# Reading annotations
# ===========================================================================


def latest_annotations(annotations: list[dict]) -> list[dict]:
    """One annotation per (annotator, role): staff dropped, re-saves collapsed.

    A few moments carry the same person's annotation twice in the same role,
    saved seconds apart under successive ``revision`` numbers -- a re-save, not a
    second opinion. The highest revision is the one they meant, and the earlier
    draft must not be read as another annotator disagreeing with them.

    First-seen order is preserved, so the annotator lists on a record stay
    stable across runs.
    """
    latest: dict[tuple[str, str | None], dict] = {}
    for annotation in annotations:
        if is_excluded(annotation):
            continue
        key = (annotation["annotator_id"], annotation.get("role"))
        current = latest.get(key)
        if current is None or annotation.get("revision", 0) > current.get(
            "revision", 0
        ):
            latest[key] = annotation
    return list(latest.values())


def threw_out(annotation: dict) -> bool:
    """Whether this annotation's author threw the moment out.

    Reannotators and adjudicators both answer this in ``meta.throw_out``;
    selectors have no ``meta`` block and are never asked, since selecting the
    moment *is* their vote to keep it.
    """
    return bool((annotation["payload"].get("meta") or {}).get("throw_out"))


def redrew_cut_point(annotation: dict) -> bool:
    """Whether this annotation's author moved the cut point the selector drew."""
    return bool((annotation["payload"].get("meta") or {}).get("redrew_cut_point"))


def answered_payload(annotation: dict) -> dict | None:
    """The payload section this annotator actually filled in, or None.

    An adjudicator answers the label questions inside ``payload["final"]``;
    reading their top-level payload instead finds nothing and scores the person
    brought in to settle the disagreement as having voted no to everything.

    None means the annotator abstained: they threw the moment out, or left a
    section null. Either section null is an abstention -- testing only one would
    let a half-filled payload through, and its null half would then read as a
    row of negatives.
    """
    payload = annotation["payload"]
    if threw_out(annotation):
        return None
    if annotation.get("role") == "adjudicator":
        payload = payload.get("final")
        if not payload:
            return None
    if payload.get("situation") is None or payload.get("action") is None:
        return None
    return payload


def payload_labels(payload: dict) -> dict[str, bool]:
    """Pull one annotator's raw booleans out of the payload they answered.

    These are only what the annotator actually answered:
    ``over_scaffolding_declared`` is their literal amount choice. The inferred
    case is not applied here -- it is derived from the *resolved* labels, in
    ``resolve_labels``.
    """
    out = {
        name: bool((payload.get(section) or {}).get(key))
        for name, section, key in BOOLEAN_FIELDS
    }
    out["over_scaffolding_declared"] = (payload.get("action") or {}).get(
        "scaffolding_amount"
    ) == OVER_SCAFFOLDING
    return out


def votes(annotations: list[dict]) -> list[tuple[str | None, dict[str, bool]]]:
    """Every usable (role, labels) vote on one moment, abstentions dropped."""
    return [
        (annotation.get("role"), payload_labels(payload))
        for annotation, payload in (
            (a, answered_payload(a)) for a in annotations
        )
        if payload is not None
    ]


# ===========================================================================
# Label resolution
# ===========================================================================


def _majority(values: list[bool]) -> bool | None:
    """True or False by count; None where the vote ties or there is nothing to count."""
    if not values:
        return None
    yes = sum(1 for value in values if value)
    no = len(values) - yes
    return None if yes == no else yes > no


def resolve_situation(
    field: str, moment_votes: list[tuple[str | None, dict[str, bool]]]
) -> tuple[bool, str | None]:
    """One situation field by majority, with the adjudicator breaking a tie.

    Returns (value, how the tie was broken) -- the second element is None where
    a majority settled it outright.

    Most moments carry two votes and the adjudicated ones four, so a tie is the
    common case rather than an edge one. An adjudicator was brought in to settle
    exactly this, so their own call takes a tie where one exists. Where no
    adjudicator saw the moment -- or the two adjudicators split as well -- the
    tie resolves True: on a two-vote moment that is the union, which keeps a
    moment either annotator judged as calling for scaffolding (or for rigor) in
    that population rather than discarding a judgment nobody overruled.
    """
    call = _majority([labels[field] for _, labels in moment_votes])
    if call is not None:
        return call, None

    adjudicated = _majority(
        [labels[field] for role, labels in moment_votes if role == "adjudicator"]
    )
    if adjudicated is not None:
        return adjudicated, TIE_ADJUDICATOR
    return True, TIE_DEFAULT_TRUE


def situation_type(
    moment_votes: list[tuple[str | None, dict[str, bool]]],
) -> str | None:
    """Which way the moment's situation votes leaned, as one of ``SITUATION_TYPES``.

    Counted per side rather than per field: ``scaffold`` is how many annotators
    called for scaffolding and ``rigor`` how many called for a push for rigor,
    and one annotator can be in both counts. The unanimous rows are the strict
    ones -- ``scaffold_only`` means every annotator called for scaffolding and
    *nobody* called for rigor -- so a single annotator calling for both drops a
    moment from ``scaffold_only`` to ``scaffold_maj``.

    ``fifty_fifty`` is an even split, which by arithmetic also covers a moment
    nobody thought called for either. No such moment exists in the export; the
    report counts them so that one appearing later is visible rather than folded
    silently into the even splits.

    Returns None where nobody voted, so an empty moment is not reported as an
    even split between two sides that received no votes at all.
    """
    if not moment_votes:
        return None
    scaffold = sum(1 for _, v in moment_votes if v["scaffolding_appropriate"])
    rigor = sum(1 for _, v in moment_votes if v["rigor_appropriate"])
    if scaffold == len(moment_votes) and rigor == 0:
        return "scaffold_only"
    if rigor == len(moment_votes) and scaffold == 0:
        return "rigor_only"
    if scaffold > rigor:
        return "scaffold_maj"
    if rigor > scaffold:
        return "rigor_maj"
    return "fifty_fifty"


def resolve_action(
    field: str, moment_votes: list[tuple[str | None, dict[str, bool]]]
) -> tuple[bool, str | None]:
    """One action field, decided by the adjudicators where there are any.

    Returns (value, how a tie was broken) -- the second element is None where a
    majority settled it outright.

    Unlike situation, which counts every vote and only falls back to the
    adjudicator on a tie, action hands the decision to the adjudicators
    outright: they saw the same excerpt and the two passes' reads of it before
    ruling on what the tutor actually did, so where they ruled, theirs is the
    answer and the passes below do not dilute it. An adjudicator who says the
    tutor did not scaffold therefore overturns two annotators who say he did.

    A tie -- two adjudicators who split, or an unadjudicated moment whose
    annotators split -- resolves by union over whichever group was deciding.
    A tied group always holds at least one True, so this is True; it is written
    as a union because that is the rule, not the arithmetic of this export.
    """
    adjudicated = [labels[field] for role, labels in moment_votes if role == "adjudicator"]
    deciding = adjudicated or [labels[field] for _, labels in moment_votes]

    call = _majority(deciding)
    if call is not None:
        return call, None
    return any(deciding), TIE_ADJUDICATOR_UNION if adjudicated else TIE_UNION


def decided_by(moment_votes: list[tuple[str | None, dict[str, bool]]]) -> str | None:
    """Who settled this moment's action fields: the adjudicators, or everyone."""
    if not moment_votes:
        return None
    return (
        "adjudicator"
        if any(role == "adjudicator" for role, _ in moment_votes)
        else "annotators"
    )


def resolve_labels(
    annotations: list[dict],
) -> tuple[dict[str, bool], dict[str, bool], bool, dict]:
    """Resolve the annotators into one label set.

    Returns (labels, agreement, over_scaffolding_inferred, resolution) where
    ``agreement[field]`` is True when every voter gave the same value,
    ``over_scaffolding_inferred`` marks a True over-scaffolding label that no
    annotator declared outright, and ``resolution`` records how the vote was
    settled: ``situation_ties`` and ``action_ties`` name what broke a tie on
    each field (a field a majority settled outright is absent),
    ``action_decided_by`` names the group that decided the action fields, and
    ``situation_type`` is the five-way split of the situation votes.

    The two halves resolve differently; see ``resolve_situation`` and
    ``resolve_action``. ``over_scaffolding_declared`` is an action field and
    follows the action rule.

    **Agreement is measured over every voter**, whichever rule decided the
    field. It reports whether the annotators concurred, not whether the rule
    that resolved them had to choose -- so an action field the adjudicators
    settled against both passes below is recorded as a disagreement, which is
    what it was.

    **The inference rule is applied after the vote, not before.** It reads the
    resolved ``scaffolding_present``/``scaffolding_appropriate``, so it fires
    only where the resolved situation says no scaffolding was called for.
    Applied per annotator instead, it fired on one annotator's "not appropriate"
    even where the resolved label says the scaffolding was called for -- leaving
    moments labelled over-scaffolding by a rule whose premise the shipped labels
    contradict.

    ``agreement["over_scaffolding_present"]`` is agreement on the declared
    amount, which is the only over-scaffolding question annotators answer -- the
    inferred case is derived, so there is no per-annotator value to compare.
    """
    moment_votes = votes(annotations)
    labels: dict[str, bool] = {}
    agreement: dict[str, bool] = {}
    situation_ties: dict[str, str] = {}
    action_ties: dict[str, str] = {}

    for field in RESOLVED_FIELDS:
        values = [vote[field] for _, vote in moment_votes]
        agreement[field] = len(set(values)) == 1
        if field in SITUATION_FIELDS:
            labels[field], tie = resolve_situation(field, moment_votes)
            ties = situation_ties
        else:
            labels[field], tie = resolve_action(field, moment_votes)
            ties = action_ties
        if tie is not None:
            ties[field] = tie

    declared = labels.pop("over_scaffolding_declared")
    agreement["over_scaffolding_present"] = agreement.pop("over_scaffolding_declared")
    if "over_scaffolding_declared" in action_ties:
        action_ties["over_scaffolding_present"] = action_ties.pop(
            "over_scaffolding_declared"
        )
    inferred_case = (
        labels["scaffolding_present"] and not labels["scaffolding_appropriate"]
    )
    labels["over_scaffolding_present"] = declared or inferred_case

    resolution = {
        "situation_type": situation_type(moment_votes),
        "situation_ties": situation_ties,
        "action_ties": action_ties,
        "action_decided_by": decided_by(moment_votes),
    }
    return (
        labels,
        agreement,
        labels["over_scaffolding_present"] and not declared,
        resolution,
    )


# ===========================================================================
# Boundaries
# ===========================================================================


def _reannotator(annotations: list[dict]) -> dict | None:
    for annotation in annotations:
        if annotation.get("role") == "reannotator":
            return annotation
    return None


def reannotated_boundaries(moment: dict, reannotation: dict | None) -> dict:
    """The moment's boundaries as the second pass left them.

    The stored ``moment`` record keeps the boundaries the selector drew even
    after a reannotator moved them, so the redraw has to be applied here.
    Redraws are partial: ``new_*`` is null for whatever the reannotator left
    alone, and those fields keep the original value.
    """
    meta = (reannotation or {}).get("payload", {}).get("meta") or {}
    boundaries = {field: moment.get(field) for field in BOUNDARY_FIELDS}
    for field in BOUNDARY_FIELDS:
        new_value = meta.get(f"new_{field}")
        if new_value is not None:
            boundaries[field] = new_value
    return boundaries


def adjudicator_draft(annotation: dict) -> dict | None:
    """The boundaries one adjudicator drew, or None where they drew none.

    An adjudicator records their own boundaries in ``payload["final_*_turn"]``
    and the matching indices, not in the ``meta.new_*`` fields a reannotator
    uses. One who threw the moment out drew nothing.
    """
    payload = annotation["payload"]
    if not payload.get("final"):
        return None
    draft = {field: payload.get(f"final_{field}") for field in BOUNDARY_FIELDS}
    return draft if any(value is not None for value in draft.values()) else None


def _draft_span(draft: dict) -> tuple[float, float]:
    """(end, negated start) for the draft, as a sort key for strictness.

    ``end_index`` counts every transcript row and ``end_turn`` only the dialogue
    ones, so the index is the finer coordinate and is preferred where present.
    A draft that names neither sorts last rather than raising.
    """

    def coordinate(index_field: str, turn_field: str) -> float:
        for field in (index_field, turn_field):
            if draft.get(field) is not None:
                return draft[field]
        return math.inf

    return coordinate("end_index", "end_turn"), -coordinate("start_index", "start_turn")


def strictest_draft(drafts: list[dict]) -> dict:
    """The strictest of the adjudicators' boundaries: the one that ends earliest.

    Where two adjudicators drew the same end, the one that starts later wins --
    the narrower of two spans ending together. The whole draft is taken as a
    unit rather than stitching the tightest start onto the tightest end, so the
    excerpt that ships is always one an adjudicator actually drew and its
    start/end/cut stay consistent with each other.
    """
    return min(drafts, key=_draft_span)


def effective_boundaries(
    moment: dict, annotations: list[dict]
) -> tuple[dict, dict | None, str]:
    """The boundaries to ship, the originals if anything moved, and whose they are.

    An adjudicator sees the moment as the second pass left it, so their draft is
    laid over the reannotated boundaries rather than over the selector's. Where
    adjudicators drew, the strictest of them wins; otherwise the second pass's
    redraw stands.

    ``source`` names whose values ship, not who last looked at them. A pass that
    redraws a span exactly as it found it has moved nothing, and crediting it
    would report boundaries as the adjudicator's that are the selector's own --
    most adjudicators' drafts reproduce the span they were handed, so that is
    not an edge case but most of the adjudicator row in the report. The three
    candidates are therefore tested by value in authority order, which also
    makes ``source == "selector"`` and ``original_boundaries is None`` the same
    statement rather than two that can contradict each other.

    An adjudicator who draws the selector's original back over a second pass's
    redraw reads as "selector" for the same reason: what ships is the selector's
    span. That an adjudicator ruled on the moment at all stays legible in
    ``annotator_roles`` and ``action_decided_by``.

    Returns (boundaries, original_boundaries, source) with original_boundaries
    None when nothing moved, and source one of "selector", "reannotator" or
    "adjudicator".
    """
    original = {field: moment.get(field) for field in BOUNDARY_FIELDS}
    reannotated = reannotated_boundaries(moment, _reannotator(annotations))
    boundaries = dict(reannotated)

    drafts = [
        draft
        for draft in (
            adjudicator_draft(a) for a in annotations if a.get("role") == "adjudicator"
        )
        if draft is not None
    ]
    if drafts:
        draft = strictest_draft(drafts)
        for field in BOUNDARY_FIELDS:
            if draft[field] is not None:
                boundaries[field] = draft[field]

    if boundaries == original:
        source = "selector"
    elif boundaries == reannotated:
        source = "reannotator"
    else:
        source = "adjudicator"

    changed = boundaries != original
    return boundaries, (original if changed else None), source


# ===========================================================================
# Moment assembly
# ===========================================================================


def build_record(row: dict, split: str, labels_map: dict[str, str]) -> dict:
    """Assemble one ground-truth record from an annotations row.

    ``row["annotations"]`` is expected to have been through
    ``latest_annotations`` already, so staff are gone and re-saves collapsed.
    """
    moment = row["moment"]
    annotations = row["annotations"]

    labels, agreement, over_scaffolding_inferred, resolution = resolve_labels(
        annotations
    )
    boundaries, original, boundaries_source = effective_boundaries(moment, annotations)

    record = {
        "moment_id": moment["moment_id"],
        "transcript_id": row["transcript_id"],
        "split": SPLIT_FILES[split],
        **boundaries,
        "boundaries_redrawn": original is not None,
        "boundaries_source": boundaries_source,
        "cut_point_redrawn": any(redrew_cut_point(a) for a in annotations),
        "labels": labels,
        "agreement": agreement,
        **resolution,
        "over_scaffolding_inferred": over_scaffolding_inferred,
        "n_annotators": len(annotations),
        "n_votes": len(votes(annotations)),
        "annotators": [annotator_label(a, labels_map) for a in annotations],
        "annotator_roles": [a.get("role") for a in annotations],
        "moment_status": moment.get("status"),
        "moment_created_at": moment.get("created_at"),
        "thrown_out": any(threw_out(a) for a in annotations),
    }
    if original is not None:
        record["original_boundaries"] = original
    return record


# ===========================================================================
# Build
# ===========================================================================


def build(
    annotations_path: str,
    splits_path: str,
    labels_path: str,
    *,
    keep_thrown_out: bool = False,
    keep_cut_point_redrawn: bool = False,
) -> tuple[dict[str, list[dict]], Counter]:
    """Return ({output stem: [record, ...]}, per-reason drop counts)."""
    manifest = load_manifest(splits_path)
    assignments = manifest["assignments"]
    if not assignments:
        raise ValueError(
            f"{splits_path} has no split assignments; run "
            "`python -m tutormoments_build.v2.splits` first"
        )
    labels_map = load_annotator_labels(labels_path)

    out: dict[str, list[dict]] = {stem: [] for stem in SPLIT_FILES.values()}
    dropped: Counter = Counter()
    unmapped: set[str] = set()
    excluded_names: set[str] = set()

    with open(annotations_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "moment" not in row:
                dropped["no_key_moments_record"] += 1
                continue

            # A selection its selector withdrew before it reached a second pass.
            if row["moment"].get("status") == RETRACTED:
                dropped["retracted"] += 1
                continue

            raw = row["annotations"]
            annotations = latest_annotations(raw)
            staff = [a for a in raw if is_excluded(a)]
            if staff:
                excluded_names.update(a.get("annotator_name", "") for a in staff)
                dropped["staff_annotations_removed"] += len(staff)
            resaves = len(raw) - len(staff) - len(annotations)
            if resaves:
                dropped["resaves_collapsed"] += resaves
            row = {**row, "annotations": annotations}

            if _reannotator(annotations) is None or len(annotations) < 2:
                dropped["not_doubly_annotated"] += 1
                continue

            split = assignments.get(row["transcript_id"], {}).get("split")
            if split is None:
                dropped["transcript_not_in_splits"] += 1
                logger.warning(
                    "transcript %s is not in %s; skipping its moments",
                    row["transcript_id"],
                    splits_path,
                )
                continue

            record = build_record(row, split, labels_map)
            if record["thrown_out"] and not keep_thrown_out:
                dropped["thrown_out"] += 1
                continue
            # A moved cut point means the two passes did not read the same
            # excerpt, so their labels are not answers to the same question.
            if record["cut_point_redrawn"] and not keep_cut_point_redrawn:
                dropped["cut_point_redrawn"] += 1
                continue
            if not record["n_votes"]:
                dropped["no_usable_vote"] += 1
                continue

            for annotation in annotations:
                name = annotation.get("annotator_name", "")
                if normalise_name(name) not in labels_map:
                    unmapped.add(name)

            out[record["split"]].append(record)

    if excluded_names:
        logger.info(
            "removed %d staff annotation(s) from %s",
            dropped["staff_annotations_removed"],
            ", ".join(sorted(excluded_names)),
        )

    if unmapped:
        logger.warning(
            "%d annotator(s) missing from the label map, emitted as annotator ids: %s",
            len(unmapped),
            ", ".join(sorted(unmapped)),
        )

    for stem in out:
        out[stem].sort(key=lambda r: (r["transcript_id"], r["moment_id"]))
    return out, dropped


def write_split(out_dir: str, stem: str, records: list[dict]) -> str:
    """Write one split's JSONL atomically. Returns the path written."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.jsonl")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
    return path


# ===========================================================================
# Reporting
# ===========================================================================


def report(out: dict[str, list[dict]], dropped: Counter, dry_run: bool) -> str:
    lines = [
        "",
        "DRY RUN -- nothing written" if dry_run else "Ground truth written",
        "",
    ]

    header = f"  {'split':<12}{'moments':>9}{'transcripts':>13}"
    header += "".join(
        f"{f.replace('_present', '').replace('_appropriate', '_ok'):>22}"
        for f in LABEL_FIELDS
    )
    lines += [header, f"  {'-' * (len(header) - 2)}"]

    for stem, records in out.items():
        row = f"  {stem:<12}{len(records):>9}{len({r['transcript_id'] for r in records}):>13}"
        for field in LABEL_FIELDS:
            n = sum(1 for r in records if r["labels"][field])
            share = f"{n / len(records):.0%}" if records else "-"
            row += f"{f'{n} ({share})':>22}"
        lines.append(row)

    every = [r for records in out.values() for r in records]
    lines += ["", "  annotator agreement (every voter gave the same value):"]
    for field in LABEL_FIELDS:
        agreed = sum(1 for r in every if r["agreement"][field])
        pct = f"{agreed / len(every):.1%}" if every else "-"
        lines.append(f"    {field:<28}{agreed:>5} / {len(every):<5} {pct:>7}")

    lines += ["", "  situation type (votes per side):"]
    for name in SITUATION_TYPES:
        n = sum(1 for r in every if r["situation_type"] == name)
        share = f"{n / len(every):.1%}" if every else "-"
        lines.append(
            f"    {name:<14}{SITUATION_TYPE_LABELS[name]:<34}{n:>5} {share:>7}"
        )
    # An even split by arithmetic also covers a moment nobody thought called for
    # either side. There are none, and this is here so that stays checkable.
    neither = sum(
        1
        for r in every
        if r["situation_type"] == "fifty_fifty"
        and not r["labels"]["scaffolding_appropriate"]
        and not r["labels"]["rigor_appropriate"]
    )
    lines.append(f"    {'':<14}{'of which nobody called for either':<34}{neither:>5}")

    lines += ["", "  ties, and what settled them:"]
    for field, key in [(f, "situation_ties") for f in SITUATION_FIELDS] + [
        (f, "action_ties") for f in LABEL_FIELDS if f not in SITUATION_FIELDS
    ]:
        ties = Counter(r[key][field] for r in every if field in r[key])
        settled = ", ".join(f"{how} {n}" for how, n in sorted(ties.items())) or "none"
        lines.append(
            f"    {field:<28}{sum(ties.values()):>5} / {len(every):<5} {settled}"
        )

    decided = Counter(r["action_decided_by"] for r in every)
    lines.append(
        f"    {'action decided by':<28}{'':>5}   "
        + ", ".join(f"{who} {n}" for who, n in sorted(decided.items()))
    )

    sources = Counter(r["boundaries_source"] for r in every)
    lines += ["", "  boundaries taken from:"]
    for source in ("selector", "reannotator", "adjudicator"):
        lines.append(f"    {source:<28}{sources[source]:>5}")

    inferred = sum(1 for r in every if r["over_scaffolding_inferred"])
    lines += [
        "",
        f"  over-scaffolding resting only on the inference rule: {inferred} / "
        f"{sum(1 for r in every if r['labels']['over_scaffolding_present'])}",
    ]

    # Two different units, so they are not one list: the first counts moments
    # that never reached a split file, the second counts annotations removed
    # from moments that did.
    moments_dropped = {
        reason: n for reason, n in dropped.items() if reason not in ANNOTATION_DROPS
    }
    if moments_dropped:
        lines += ["", "  moments not emitted:"]
        for reason, count in sorted(moments_dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:<28}{count:>5}")

    annotations_dropped = {
        reason: n for reason, n in dropped.items() if reason in ANNOTATION_DROPS
    }
    if annotations_dropped:
        lines += ["", "  annotations dropped from the moments that were emitted:"]
        for reason, count in sorted(annotations_dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:<28}{count:>5}")
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.build_ground_truth",
        description="Build iteration/test ground truth from doubly annotated v2 moments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS, metavar="FILE")
    parser.add_argument(
        "--splits", default=DEFAULT_MANIFEST, metavar="FILE", help="Split manifest"
    )
    parser.add_argument(
        "--annotator-labels",
        default=DEFAULT_ANNOTATOR_LABELS,
        metavar="FILE",
        help="Name -> de-identified label map",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--keep-thrown-out",
        action="store_true",
        help="Keep moments a second pass or an adjudicator flagged meta.throw_out",
    )
    parser.add_argument(
        "--keep-cut-point-redrawn",
        action="store_true",
        help="Keep moments whose cut point a later pass moved",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without writing any file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    out, dropped = build(
        args.annotations,
        args.splits,
        args.annotator_labels,
        keep_thrown_out=args.keep_thrown_out,
        keep_cut_point_redrawn=args.keep_cut_point_redrawn,
    )

    if not args.dry_run:
        for stem, records in out.items():
            logger.info(
                "wrote %d moment(s) to %s",
                len(records),
                write_split(args.out_dir, stem, records),
            )

    print(report(out, dropped, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
