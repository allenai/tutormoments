"""Build iteration/test ground-truth JSONL from doubly annotated v2 moments.

A v2 moment is annotated twice: a *selector* marks it and fills in the rubric,
then a *reannotator* independently fills in the same rubric and may redraw the
moment's boundaries or flag it for removal. This script keeps the moments that
carry both passes, resolves the two (occasionally more) annotators into one
label set, and writes them to ``data/ground_truth/`` split by the append-only
assignment in ``splits.json`` -- ``iterate`` -> ``iteration.jsonl``, ``heldout``
-> ``test.jsonl``.

Five booleans are emitted per moment:

  situation  scaffolding_appropriate   scaffolding was called for here
             rigor_appropriate         a push for rigor was called for here
  action     scaffolding_present       the tutor scaffolded
             rigor_present             the tutor pushed for rigor
             over_scaffolding_present  the tutor scaffolded too much

**Disagreements are resolved by union**: a field is True when *any* annotator
marked it True. Both annotators are experienced teachers reading the same
exchange, so a label only one of them saw is treated as a genuine reading of the
moment rather than as noise to be voted away -- with two annotators there is no
majority to take, and intersection would systematically under-count. Per-field
agreement is recorded alongside the label so the disagreement rate stays
measurable rather than being silently folded in.

``over_scaffolding_present`` reads ``action.scaffolding_amount ==
"over_scaffolding"``, which the interface only offers once an annotator has
marked scaffolding present; union over that field therefore always implies
``scaffolding_present``.

It also carries an inferred case: scaffolding delivered where that annotator
said none was called for (``scaffolding_present and not
scaffolding_appropriate``) counts as over-scaffolding whatever amount they
picked, because supporting a student who did not need supporting is
over-scaffolding by definition. The rule is applied to each annotator's own
payload before the union, so it reads one teacher's judgment of the moment
rather than mixing two. It overrides rather than fills a gap -- the amount is
never left unset once scaffolding is marked present -- and where it changes the
outcome the record is tagged ``over_scaffolding_inferred``, so a label that
rests only on the rule stays separable from one an annotator declared outright.

Moments the reannotator flagged ``meta.throw_out`` are dropped (they are
mis-selected moments, not hard cases) -- pass ``--keep-thrown-out`` to retain
them. Where the reannotator redrew boundaries, the redrawn values are emitted
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
    name = annotation.get("annotator_name", "")
    return (
        labels.get(name.strip().lower().replace(" ", "-")) or annotation["annotator_id"]
    )


# ===========================================================================
# Label resolution
# ===========================================================================


def _annotator_labels_for(annotation: dict) -> dict[str, bool]:
    """Pull one annotator's five booleans out of their payload.

    ``over_scaffolding_declared`` is the annotator's literal amount choice;
    ``over_scaffolding_present`` adds the inferred case (scaffolding where this
    annotator said none was called for). Both are returned so a label resting
    only on the inference stays distinguishable downstream.
    """
    payload = annotation["payload"]
    out = {}
    for name, section, key in BOOLEAN_FIELDS:
        out[name] = bool((payload.get(section) or {}).get(key))

    declared = (payload.get("action") or {}).get(
        "scaffolding_amount"
    ) == OVER_SCAFFOLDING
    inferred = out["scaffolding_present"] and not out["scaffolding_appropriate"]
    out["over_scaffolding_declared"] = declared
    out["over_scaffolding_present"] = declared or inferred
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
    """
    per_annotator = [_annotator_labels_for(a) for a in annotations]
    labels, agreement = {}, {}
    for field in LABEL_FIELDS:
        values = [a[field] for a in per_annotator]
        labels[field] = any(values)
        agreement[field] = len(set(values)) == 1

    inferred = labels["over_scaffolding_present"] and not any(
        a["over_scaffolding_declared"] for a in per_annotator
    )
    return labels, agreement, inferred


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
                if name.strip().lower().replace(" ", "-") not in labels_map:
                    unmapped.add(name)

            out[record["split"]].append(record)

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
