"""Build iteration/test ground-truth JSONL from doubly annotated v2 moments.

This script takes doubly-annotated moments, resolves the two annotators into one
label set, and writes them to ``data/ground_truth/`` split by ``splits.json``

Each moment has five booleans:

  situation  scaffolding_appropriate   scaffolding was called for here
             rigor_appropriate         a push for rigor was called for here
  action     scaffolding_present       the tutor scaffolded
             rigor_present             the tutor pushed for rigor
             over_scaffolding_present  the tutor scaffolded too much

**Disagreements are resolved by union**: a field is True when *any* annotator
marked it True.

If ``scaffolding_present and not scaffolding_appropriate``, it counts as
over-scaffolding.

Project staff (``EXCLUDED_ANNOTATORS``) are filtered out before anything is
resolved. 

Moments the reannotator flagged ``meta.throw_out`` are dropped. 
Where the reannotator redrew boundaries, the redrawn values are emitted
and the originals preserved under ``original_boundaries``.

Usage::

    python -m tutormoments_build.v2.build_ground_truth --dry-run
    python -m tutormoments_build.v2.build_ground_truth
"""

import argparse
import json
import logging
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

# What is unioned across annotators: the four they answer directly, plus their
# literal over-scaffolding choice. over_scaffolding_present is not here -- it is
# derived from these after the union, not voted on.
RESOLVED_FIELDS = tuple(name for name, _, _ in BOOLEAN_FIELDS) + (
    "over_scaffolding_declared",
)

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
# Label resolution
# ===========================================================================


def _annotator_labels_for(annotation: dict) -> dict[str, bool]:
    """Pull one annotator's raw booleans out of their payload.

    These are only what the annotator actually answered:
    ``over_scaffolding_declared`` is their literal amount choice. The inferred
    case is not applied here -- it is derived from the *resolved* labels, in
    ``resolve_labels``.
    """
    payload = annotation["payload"]
    out = {}
    for name, section, key in BOOLEAN_FIELDS:
        out[name] = bool((payload.get(section) or {}).get(key))

    out["over_scaffolding_declared"] = (payload.get("action") or {}).get(
        "scaffolding_amount"
    ) == OVER_SCAFFOLDING
    return out


def resolve_labels(
    annotations: list[dict],
) -> tuple[dict[str, bool], dict[str, bool], bool]:
    """Union the annotators' labels.

    Returns (labels, agreement, over_scaffolding_inferred) where
    agreement[field] is True when every annotator gave the same value, and
    over_scaffolding_inferred marks a True over-scaffolding label that no
    annotator declared outright -- it rests entirely on the inference rule, and
    would be False without it.

    **The inference rule is applied after the union, not before.** It reads the
    resolved ``scaffolding_present``/``scaffolding_appropriate``, so it fires
    only where the annotators *agreed* no scaffolding was called for. Applied
    per annotator instead, it fired on one annotator's "not appropriate" even
    where the other said the scaffolding was called for -- which the union has
    already resolved to appropriate. That left moments labelled over-scaffolding
    by a rule whose premise the resolved labels contradict. Every label the rule
    still infers is one both annotators' situation judgments support; nothing it
    used to infer from an agreed "not appropriate" is lost.

    ``agreement["over_scaffolding_present"]`` is agreement on the declared
    amount, which is the only over-scaffolding question annotators answer -- the
    inferred case is derived, so there is no per-annotator value to compare.
    """
    per_annotator = [_annotator_labels_for(a) for a in annotations]
    labels, agreement = {}, {}
    for field in RESOLVED_FIELDS:
        values = [a[field] for a in per_annotator]
        labels[field] = any(values)
        agreement[field] = len(set(values)) == 1

    declared = labels.pop("over_scaffolding_declared")
    agreement["over_scaffolding_present"] = agreement.pop("over_scaffolding_declared")
    inferred_case = (
        labels["scaffolding_present"] and not labels["scaffolding_appropriate"]
    )
    labels["over_scaffolding_present"] = declared or inferred_case

    return labels, agreement, labels["over_scaffolding_present"] and not declared


# ===========================================================================
# Moment assembly
# ===========================================================================


def _reannotator(annotations: list[dict]) -> dict | None:
    for annotation in annotations:
        if annotation.get("role") == "reannotator":
            return annotation
    return None


def effective_boundaries(
    moment: dict, reannotation: dict | None
) -> tuple[dict, dict | None]:
    """Apply the reannotator's redrawn boundaries, if any.

    The stored ``moment`` record keeps the boundaries the selector drew even
    after a reannotator moved them, so the redraw has to be applied here.
    Redraws are partial: ``new_*`` is null for whatever the reannotator left
    alone, and those fields keep the original value.

    Returns (boundaries, original_boundaries) with original_boundaries None when
    nothing moved.
    """
    meta = (reannotation or {}).get("payload", {}).get("meta") or {}
    boundaries = {field: moment.get(field) for field in BOUNDARY_FIELDS}
    original = dict(boundaries)

    for field in BOUNDARY_FIELDS:
        new_value = meta.get(f"new_{field}")
        if new_value is not None:
            boundaries[field] = new_value

    changed = boundaries != original
    return boundaries, (original if changed else None)


def build_record(row: dict, split: str, labels_map: dict[str, str]) -> dict:
    """Assemble one ground-truth record from an annotations row."""
    moment = row["moment"]
    annotations = row["annotations"]
    reannotation = _reannotator(annotations)
    meta = (reannotation or {}).get("payload", {}).get("meta") or {}

    labels, agreement, over_scaffolding_inferred = resolve_labels(annotations)
    boundaries, original = effective_boundaries(moment, reannotation)

    record = {
        "moment_id": moment["moment_id"],
        "transcript_id": row["transcript_id"],
        "split": SPLIT_FILES[split],
        **boundaries,
        "boundaries_redrawn": original is not None,
        "cut_point_redrawn": bool(meta.get("redrew_cut_point")),
        "labels": labels,
        "agreement": agreement,
        "over_scaffolding_inferred": over_scaffolding_inferred,
        "n_annotators": len(annotations),
        "annotators": [annotator_label(a, labels_map) for a in annotations],
        "annotator_roles": [a.get("role") for a in annotations],
        "moment_status": moment.get("status"),
        "moment_created_at": moment.get("created_at"),
        "thrown_out": bool(meta.get("throw_out")),
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

            annotations = row["annotations"]
            staff = [a for a in annotations if is_excluded(a)]
            if staff:
                excluded_names.update(a.get("annotator_name", "") for a in staff)
                dropped["staff_annotations_removed"] += len(staff)
                annotations = [a for a in annotations if not is_excluded(a)]
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

    lines += ["", "  annotator agreement (both passes gave the same value):"]
    every = [r for records in out.values() for r in records]
    for field in LABEL_FIELDS:
        agreed = sum(1 for r in every if r["agreement"][field])
        pct = f"{agreed / len(every):.1%}" if every else "-"
        lines.append(f"    {field:<28}{agreed:>5} / {len(every):<5} {pct:>7}")

    if dropped:
        lines += ["", "  rows not emitted:"]
        for reason, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
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
        help="Keep moments the reannotator flagged meta.throw_out",
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
