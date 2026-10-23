"""Build iteration/test ground-truth JSONL from multiply annotated v2 moments.

This script reads every provider's annotations, collapses each moment's
annotators into one label set, and writes them to ``data/ground_truth/`` split
by ``splits.json``. The rules follow the "Combined labels" section of
``annotation_stats.ipynb``.

Each moment gets one situation label and three action booleans:

  situation_type  one of scaffold_only, scaffold_maj, fifty_fifty, rigor_maj,
                  rigor_only (see ``situation_type``)
  labels          scaffolding_present       the tutor scaffolded (any level)
                  rigor_present             the tutor pushed for rigor
                  over_scaffolding_present  the tutor over-scaffolded

**Who votes.**
Project staff (``EXCLUDED_ANNOTATORS``) are dropped. If the same person saved
the same role twice, only their highest ``revision`` counts. Selector,
reannotator and adjudicators each count as one vote; an adjudicator's answers
are read from ``payload["final"]``. Anyone who threw the moment out answered
nothing and abstains rather than voting no.

**Situation.** Each situation checkbox is a vote for its side, so an annotator
who ticked both scaffolding and rigor votes for both. Comparing the two sides'
counts puts the moment in one of the five ``SITUATION_TYPES``.

**Action.** Each field is a majority over every vote, and a tie resolves True.
Over-scaffolding is the annotator's literal ``scaffolding_amount ==
"over_scaffolding"`` choice, resolved the same way; it is not inferred from the
other labels.

**What is dropped:** retracted moments, moments the second pass or any
adjudicator threw out, moments whose cut point anyone flagged as redrawn (the
annotators then judged different excerpts), and moments with fewer than two
usable votes.

**Boundaries.** See ``adjudicated_boundaries``.

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
    DEFAULT_ANNOTATIONS_FILES,
    DEFAULT_MANIFEST,
    load_manifest,
    source_of_file,
)

logger = logging.getLogger("tutormoments_build.v2.build_ground_truth")

DEFAULT_OUT_DIR = "data/ground_truth"

# Project staff, excluded from the ground truth. They annotate to pilot the
# rubric and the interface, not as expert raters. Matched on the first name,
# since that is what the export carries; the build logs every name it drops.
EXCLUDED_ANNOTATORS = frozenset({"lucy", "rebecca", "albert", "kajal"})
DEFAULT_ANNOTATOR_LABELS = "data/v2_annotations/annotator_labels.json"

# split name in splits.json -> output file stem
SPLIT_FILES = {"iterate": "iteration", "heldout": "test"}

# The moment status the tool sets on a selection its selector withdrew before it
# reached a second pass.
RETRACTED = "retracted"

# A single vote is one opinion, not an agreement to resolve.
MIN_VOTES = 2

OVER_SCAFFOLDING = "over_scaffolding"

# One annotator's booleans, as (vote field, payload section, payload key).
# over_scaffolding_declared is read separately from scaffolding_amount.
BOOLEAN_FIELDS = (
    ("scaffolding_appropriate", "situation", "scaffolding_appropriate"),
    ("rigor_appropriate", "situation", "rigor_appropriate"),
    ("scaffolding_present", "action", "scaffolding_present"),
    ("rigor_present", "action", "rigor_present"),
)

# Output action label -> the vote field it is resolved from.
ACTION_FIELDS = {
    "scaffolding_present": "scaffolding_present",
    "rigor_present": "rigor_present",
    "over_scaffolding_present": "over_scaffolding_declared",
}

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

BOUNDARY_FIELDS = (
    "start_turn",
    "end_turn",
    "cut_turn",
    "start_index",
    "end_index",
    "cut_index",
    "dialogue_turns",
)
TURN_FIELDS = ("start_turn", "end_turn", "cut_turn")

# Drop counters that count *annotations* rather than moments.
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
    UUID.
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

    A re-save carries a higher ``revision`` and is what the annotator meant;
    the earlier draft must not be read as a second opinion. First-seen order is
    preserved so the annotator lists on a record stay stable across runs.
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


def _meta(annotation: dict) -> dict:
    return annotation["payload"].get("meta") or {}


def threw_out(annotation: dict) -> bool:
    """Whether this annotation's author threw the moment out.

    Selectors have no ``meta`` block: selecting the moment is their vote to
    keep it. An adjudicator who reverted someone else's throw-out does not undo
    it here -- one throw-out from the second pass or any adjudicator drops the
    moment, as in the notebook.
    """
    return bool(_meta(annotation).get("throw_out"))


def redrew_cut_point(annotation: dict) -> bool:
    """Whether this annotation's author flagged moving the selector's cut point."""
    return bool(_meta(annotation).get("redrew_cut_point"))


def answered_payload(annotation: dict) -> dict | None:
    """The payload section this annotator filled in, or None if they abstained.

    Adjudicators answer inside ``payload["final"]``. Either section left null is
    an abstention, so a blank is never read as a row of "no"s.
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
    """One annotator's booleans, as answered."""
    labels = {
        name: bool((payload.get(section) or {}).get(key))
        for name, section, key in BOOLEAN_FIELDS
    }
    labels["over_scaffolding_declared"] = (payload.get("action") or {}).get(
        "scaffolding_amount"
    ) == OVER_SCAFFOLDING
    return labels


def votes(annotations: list[dict]) -> list[dict[str, bool]]:
    """Every usable vote on one moment, abstentions dropped."""
    payloads = (answered_payload(a) for a in annotations)
    return [payload_labels(p) for p in payloads if p is not None]


# ===========================================================================
# Label resolution
# ===========================================================================


def majority(values: list[bool], tie: bool) -> bool:
    """True or False by count; ``tie`` where the vote is even."""
    yes = sum(1 for value in values if value)
    no = len(values) - yes
    if yes == no:
        return tie
    return yes > no


def situation_type(moment_votes: list[dict[str, bool]]) -> str | None:
    """Which way the moment's situation votes leaned, as one of ``SITUATION_TYPES``.

    ``scaffold`` counts annotators who called for scaffolding and ``rigor`` those
    who called for rigor; one annotator can be in both. The unanimous rows are
    strict: ``scaffold_only`` means every annotator called for scaffolding and
    nobody called for rigor.

    Returns None where nobody voted.
    """
    if not moment_votes:
        return None
    scaffold = sum(1 for v in moment_votes if v["scaffolding_appropriate"])
    rigor = sum(1 for v in moment_votes if v["rigor_appropriate"])
    if scaffold == len(moment_votes) and rigor == 0:
        return "scaffold_only"
    if rigor == len(moment_votes) and scaffold == 0:
        return "rigor_only"
    if scaffold > rigor:
        return "scaffold_maj"
    if rigor > scaffold:
        return "rigor_maj"
    return "fifty_fifty"


def resolve_labels(moment_votes: list[dict[str, bool]]) -> tuple[dict, list[str]]:
    """The action labels by majority (tie→True), and which fields tied."""
    labels, ties = {}, []
    for name, field in ACTION_FIELDS.items():
        values = [v[field] for v in moment_votes]
        labels[name] = majority(values, tie=True)
        if majority(values, tie=False) != labels[name]:
            ties.append(name)
    return labels, ties


# ===========================================================================
# Boundaries
# ===========================================================================


def reannotation(annotations: list[dict]) -> dict | None:
    """The reannotator's annotation (the second pass), or None."""
    for annotation in annotations:
        if annotation.get("role") == "reannotator":
            return annotation
    return None


def adjudications(annotations: list[dict]) -> list[dict]:
    """The adjudicators' annotations, in the order they saved."""
    return sorted(
        (a for a in annotations if a.get("role") == "adjudicator"),
        key=lambda a: a.get("created_at") or "",
    )


def reannotated_boundaries(moment: dict, reannotation: dict | None) -> dict:
    """The moment's boundaries as the second pass left them.

    The ``moment`` record keeps the selector's boundaries, so the redraw is
    applied here. ``new_*`` is null for whatever the reannotator left alone.
    """
    meta = (reannotation or {}).get("payload", {}).get("meta") or {}
    boundaries = {field: moment.get(field) for field in BOUNDARY_FIELDS}
    for field in BOUNDARY_FIELDS:
        new_value = meta.get(f"new_{field}")
        if new_value is not None:
            boundaries[field] = new_value
    return boundaries


def adjudicator_draft(annotation: dict) -> dict | None:
    """The boundaries one adjudicator drew (``payload["final_*"]``), or None.

    One who threw the moment out has no ``final`` and drew nothing.
    """
    payload = annotation["payload"]
    if not payload.get("final"):
        return None
    draft = {field: payload.get(f"final_{field}") for field in BOUNDARY_FIELDS}
    return draft if any(value is not None for value in draft.values()) else None


def _draft_span(draft: dict) -> tuple[float, float]:
    """(end, negated start) as a sort key for strictness; index preferred over turn."""

    def coordinate(index_field: str, turn_field: str) -> float:
        for field in (index_field, turn_field):
            if draft.get(field) is not None:
                return draft[field]
        return math.inf

    return coordinate("end_index", "end_turn"), -coordinate("start_index", "start_turn")


def strictest_draft(drafts: list[dict]) -> dict:
    """The draft that ends earliest; on an equal end, the one that starts later.

    Taken whole, so the span that ships is one an adjudicator actually drew.
    """
    return min(drafts, key=_draft_span)


def adjudicated_boundaries(moment: dict, annotations: list[dict]) -> dict:
    """start_turn / end_turn / cut_turn as the adjudicators left them.

    * one adjudicator, or two who agree -> theirs
    * two who disagree -> the span that ends earliest (``strictest_draft``),
      with the cut most of the annotators chose (selector, second pass, and
      each adjudicator). On a tie, or a winning cut outside that span, the
      strictest adjudicator's own cut.
    * no adjudicator drew any -> the second pass's
    A value an adjudicator left blank keeps what they were shown (the second
    pass's version).
    """
    shown = reannotated_boundaries(moment, reannotation(annotations))
    drafts = []
    for annotation in adjudications(annotations):
        draft = adjudicator_draft(annotation)
        if draft is not None:
            drafts.append(
                {f: shown[f] if draft[f] is None else draft[f] for f in BOUNDARY_FIELDS}
            )

    if not drafts:
        return {f: shown[f] for f in TURN_FIELDS}
    strictest = strictest_draft(drafts)
    boundaries = {f: strictest[f] for f in TURN_FIELDS}
    if all(d[f] == drafts[0][f] for d in drafts for f in TURN_FIELDS):
        return boundaries

    cuts = Counter(
        [moment["cut_turn"], shown["cut_turn"], *(d["cut_turn"] for d in drafts)]
    )
    top = max(cuts.values())
    winners = [cut for cut, n in cuts.items() if n == top]
    if (
        len(winners) == 1
        and boundaries["start_turn"] <= winners[0] <= boundaries["end_turn"]
    ):
        boundaries["cut_turn"] = winners[0]
    return boundaries


def boundaries_source(moment: dict, annotations: list[dict], boundaries: dict) -> str:
    """Whose values ship: "selector", "reannotator" or "adjudicator".

    Tested by value, so a pass that redrew a span exactly as it found it is not
    credited with it.
    """
    shown = reannotated_boundaries(moment, reannotation(annotations))
    if all(boundaries[f] == moment.get(f) for f in TURN_FIELDS):
        return "selector"
    if all(boundaries[f] == shown[f] for f in TURN_FIELDS):
        return "reannotator"
    return "adjudicator"


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
    moment_votes = votes(annotations)
    labels, ties = resolve_labels(moment_votes)
    boundaries = adjudicated_boundaries(moment, annotations)
    source = boundaries_source(moment, annotations, boundaries)

    record = {
        "moment_id": moment["moment_id"],
        "source": row["source"],
        "transcript_id": row["transcript_id"],
        "split": SPLIT_FILES[split],
        **boundaries,
        "boundaries_source": source,
        "situation_type": situation_type(moment_votes),
        "situation_votes": {
            "scaffolding": sum(1 for v in moment_votes if v["scaffolding_appropriate"]),
            "rigor": sum(1 for v in moment_votes if v["rigor_appropriate"]),
        },
        "labels": labels,
        "action_ties": ties,
        "n_votes": len(moment_votes),
        "annotators": [annotator_label(a, labels_map) for a in annotations],
        "annotator_roles": [a.get("role") for a in annotations],
        "moment_status": moment.get("status"),
        "moment_created_at": moment.get("created_at"),
    }
    if source != "selector":
        record["original_boundaries"] = {f: moment.get(f) for f in TURN_FIELDS}
    return record


# ===========================================================================
# Build
# ===========================================================================


def read_rows(annotations_paths: list[str]):
    """Yield every annotations row, tagged with its provider in ``source``."""
    for path in annotations_paths:
        source = source_of_file(path)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.setdefault("source", source) != source:
                    raise ValueError(
                        f"{path}:{lineno}: row has source {row['source']!r}, "
                        f"but the file is {source!r}'s"
                    )
                yield row


def build(
    annotations_paths: list[str],
    splits_path: str,
    labels_path: str,
) -> tuple[dict[str, list[dict]], Counter]:
    """Return ({output stem: [record, ...]}, per-reason drop counts)."""
    assignments = load_manifest(splits_path)["assignments"]
    if not assignments:
        raise ValueError(
            f"{splits_path} has no split assignments; run "
            "`python -m tutormoments_build.v2.splits` first"
        )
    labels_map = load_annotator_labels(labels_path)

    out: dict[str, list[dict]] = {stem: [] for stem in SPLIT_FILES.values()}
    dropped: Counter = Counter()
    seen: set[str] = set()
    unmapped: set[str] = set()
    excluded_names: set[str] = set()

    for row in read_rows(annotations_paths):
        if "moment" not in row:
            dropped["no_key_moments_record"] += 1
            continue
        moment_id = row["moment"]["moment_id"]
        if moment_id in seen:
            raise ValueError(f"moment_id {moment_id} appears in more than one row")
        seen.add(moment_id)

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

        if any(threw_out(a) for a in annotations):
            dropped["thrown_out"] += 1
            continue
        if any(redrew_cut_point(a) for a in annotations):
            dropped["cut_point_redrawn"] += 1
            continue
        if len(votes(annotations)) < MIN_VOTES:
            dropped["fewer_than_two_votes"] += 1
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

        for annotation in annotations:
            name = annotation.get("annotator_name", "")
            if normalise_name(name) not in labels_map:
                unmapped.add(name)

        record = build_record(row, split, labels_map)
        out[record["split"]].append(record)

    if excluded_names:
        logger.info(
            "removed %d staff annotation(s) from %s",
            dropped["staff_annotations_removed"],
            ", ".join(sorted(excluded_names)),
        )
    if unmapped:
        logger.warning(
            "%d annotator(s) missing from the label map, emitted as annotator ids",
            len(unmapped),
        )

    for stem in out:
        out[stem].sort(key=lambda r: (r["source"], r["transcript_id"], r["moment_id"]))
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
    every = [r for records in out.values() for r in records]
    sources = sorted({r["source"] for r in every})

    header = f"  {'split':<12}{'source':<24}{'moments':>9}{'transcripts':>13}"
    header += "".join(f"{name:>28}" for name in ACTION_FIELDS)
    lines += [header, f"  {'-' * (len(header) - 2)}"]
    for stem, records in out.items():
        for source in sources:
            group = [r for r in records if r["source"] == source]
            line = (
                f"  {stem:<12}{source:<24}{len(group):>9}"
                f"{len({r['transcript_id'] for r in group}):>13}"
            )
            for name in ACTION_FIELDS:
                n = sum(1 for r in group if r["labels"][name])
                share = f"{n / len(group):.0%}" if group else "-"
                line += f"{f'{n} ({share})':>28}"
            lines.append(line)

    lines += ["", "  situation type (votes per side):"]
    for name in SITUATION_TYPES:
        n = sum(1 for r in every if r["situation_type"] == name)
        share = f"{n / len(every):.1%}" if every else "-"
        lines.append(
            f"    {name:<14}{SITUATION_TYPE_LABELS[name]:<34}{n:>5} {share:>7}"
        )

    lines += ["", "  action fields settled by the tie rule (tie→True):"]
    for name in ACTION_FIELDS:
        n = sum(1 for r in every if name in r["action_ties"])
        lines.append(f"    {name:<28}{n:>5} / {len(every)}")

    by_source = Counter(r["boundaries_source"] for r in every)
    lines += ["", "  boundaries taken from:"]
    for source in ("selector", "reannotator", "adjudicator"):
        lines.append(f"    {source:<28}{by_source[source]:>5}")

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
        description="Build iteration/test ground truth from multiply annotated v2 moments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=list(DEFAULT_ANNOTATIONS_FILES),
        metavar="FILE",
        help="Annotations JSONL, one per provider (<provider>_annotations.jsonl)",
    )
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
        "--dry-run", action="store_true", help="Report without writing any file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    out, dropped = build(args.annotations, args.splits, args.annotator_labels)

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
