"""Assign v2-annotated transcripts to the `iterate` / `heldout` splits.

Splits are **append-only**: a re-run assigns only transcripts not yet in the
manifest and never moves an existing one.

One run reads every provider's annotations file (DEFAULT_ANNOTATIONS_FILES).
Each provider is split on its own, balanced against its own running totals, so
one provider never makes up for another's drift.

Current split (transcripts, as of batch 6):

    provider               iterate   heldout
    tutoring_provider_a        167       167
    tutoring_provider_c         53        52

How a batch is assigned, per provider (deterministic given seed + manifest):

1. Collect the provider's transcript ids not yet in the manifest, sorted.
2. Shuffle with ``random.Random(f"{seed}:{batch}:{provider}")``.
   (Batches 1-5 predate providers and used ``f"{seed}:{batch}"``; they are
   frozen, so the change does not affect them.)
3. Order heaviest-first (the shuffle breaks ties) and give each id to whichever
   split is furthest below its target share. Filling the gap, rather than
   flipping a coin per transcript, keeps the ratio on target across batches.

Usage:
    # preview without touching the manifest
    python -m tutormoments_build.v2.splits --dry-run

    # write/extend the manifest
    python -m tutormoments_build.v2.splits
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone

from tutormoments.logging_setup import logging_args_parent, setup_logging

logger = logging.getLogger("tutormoments_build.v2.splits")

SCHEMA_VERSION = "1.0"
SPLITS = ("iterate", "heldout")

DEFAULT_ANNOTATIONS = "data/v2_annotations/source/tutoring_provider_a_annotations.jsonl"
# Read together by this script. DEFAULT_ANNOTATIONS stays provider A only
# because build_ground_truth imports it.
DEFAULT_ANNOTATIONS_FILES = (
    DEFAULT_ANNOTATIONS,
    "data/v2_annotations/source/tutoring_provider_c_annotations.jsonl",
)
ANNOTATIONS_SUFFIX = "_annotations"
DEFAULT_MANIFEST = os.path.join(os.path.dirname(__file__), "splits.json")

# Frozen so a rebuilt manifest reproduces the committed one. Changing this
# re-randomises every *future* batch; assignments already in the manifest are
# unaffected, by design.
DEFAULT_SEED = 20260819
DEFAULT_HELDOUT_FRACTION = 0.5


# ===========================================================================
# Reading annotations
# ===========================================================================


def source_of_file(annotations_path: str) -> str:
    """``tutoring_provider_a_annotations.jsonl`` -> ``tutoring_provider_a``."""
    stem = os.path.splitext(os.path.basename(annotations_path))[0]
    return stem.removesuffix(ANNOTATIONS_SUFFIX)


def summarize_transcripts(annotations_path: str) -> dict[str, dict]:
    """Return {transcript_id: {"n_moments": int, "no_key_moments": bool, "source": str}}.

    The annotations JSONL carries one row per moment, plus rows shaped
    ``{"transcript_id", "no_key_moments_record"}`` for transcripts an annotator
    reviewed and found nothing in. Both kinds count as annotated transcripts and
    both are eligible for a split.

    A row's own ``source`` (some providers include one) must match the file's.
    """
    source = source_of_file(annotations_path)
    moments: Counter[str] = Counter()
    no_key_moments: set[str] = set()

    with open(annotations_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("transcript_id")
            if not tid:
                raise ValueError(
                    f"{annotations_path}:{lineno}: row has no transcript_id"
                )
            if row.get("source", source) != source:
                raise ValueError(
                    f"{annotations_path}:{lineno}: row {tid} has source "
                    f"{row['source']!r}, but the file is {source!r}'s"
                )
            if "moment" in row:
                moments[tid] += 1
            elif "no_key_moments_record" in row:
                no_key_moments.add(tid)
            else:
                raise ValueError(
                    f"{annotations_path}:{lineno}: row {tid} has neither 'moment' "
                    "nor 'no_key_moments_record'"
                )

    return {
        tid: {
            "n_moments": moments.get(tid, 0),
            "no_key_moments": tid in no_key_moments and moments.get(tid, 0) == 0,
            "source": source,
        }
        for tid in sorted(set(moments) | no_key_moments)
    }


def summarize_all(annotations_paths: list[str]) -> dict[str, dict]:
    """summarize_transcripts over every provider's file, merged.

    A transcript id in two providers' files is an error.
    """
    merged: dict[str, dict] = {}
    for path in annotations_paths:
        for tid, summary in summarize_transcripts(path).items():
            if tid in merged:
                raise ValueError(
                    f"transcript {tid} appears in both {merged[tid]['source']!r} "
                    f"and {summary['source']!r} annotations"
                )
            merged[tid] = summary
    return merged


# ===========================================================================
# Manifest I/O
# ===========================================================================


def load_manifest(path: str) -> dict:
    """Read the split manifest, or return an empty one if it does not exist."""
    if not os.path.exists(path):
        return {
            "schema_version": SCHEMA_VERSION,
            "seed": None,
            "heldout_fraction": None,
            "balance_by": None,
            "batches": [],
            "assignments": {},
        }
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {manifest.get('schema_version')!r} is not "
            f"{SCHEMA_VERSION!r}; refusing to edit a manifest this script does not understand"
        )
    return manifest


def write_manifest(path: str, manifest: dict) -> None:
    """Write the manifest atomically, with assignments sorted by transcript id."""
    manifest = dict(manifest)
    manifest["assignments"] = dict(sorted(manifest["assignments"].items()))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def read_splits(path: str = DEFAULT_MANIFEST) -> dict[str, str]:
    """Convenience accessor for downstream code: {transcript_id: split}."""
    manifest = load_manifest(path)
    return {tid: rec["split"] for tid, rec in manifest["assignments"].items()}


# ===========================================================================
# Assignment
# ===========================================================================


def _weight(record: dict, balance_by: str) -> float:
    return float(record["n_moments"]) if balance_by == "moments" else 1.0


def assigned_source(tid: str, record: dict, transcripts: dict[str, dict]) -> str | None:
    """The provider of an assigned transcript, or None if unknown.

    Batches 1-5 don't record one, so it falls back to the annotations.
    """
    return record.get("source") or transcripts.get(tid, {}).get("source")


def assign_new(
    transcripts: dict[str, dict],
    manifest: dict,
    *,
    seed: int,
    heldout_fraction: float,
    balance_by: str,
) -> tuple[dict[str, dict], list[str]]:
    """Assign every transcript not already in the manifest, provider by provider.

    Returns (new_assignments, ordered_new_ids). Existing assignments are read
    for the running totals but never modified.
    """
    assignments = manifest["assignments"]
    batch = len(manifest["batches"]) + 1

    new_assignments: dict[str, dict] = {}
    ordered_new_ids: list[str] = []
    for source in sorted({t["source"] for t in transcripts.values()}):
        provider = {tid: t for tid, t in transcripts.items() if t["source"] == source}
        provider_assignments = {
            tid: rec
            for tid, rec in assignments.items()
            if assigned_source(tid, rec, transcripts) == source
        }
        new, new_ids = _assign_provider(
            source,
            provider,
            provider_assignments,
            batch=batch,
            seed=seed,
            heldout_fraction=heldout_fraction,
            balance_by=balance_by,
        )
        new_assignments.update(new)
        ordered_new_ids.extend(new_ids)
    return new_assignments, ordered_new_ids


def _assign_provider(
    source: str,
    transcripts: dict[str, dict],
    assignments: dict[str, dict],
    *,
    batch: int,
    seed: int,
    heldout_fraction: float,
    balance_by: str,
) -> tuple[dict[str, dict], list[str]]:
    """Assign one provider's new transcripts against its own running totals."""
    new_ids = sorted(tid for tid in transcripts if tid not in assignments)
    random.Random(f"{seed}:{batch}:{source}").shuffle(new_ids)
    # Heaviest first, randomised within each weight. Greedy filling cannot undo
    # an earlier choice, so a 15-moment transcript arriving last would strand a
    # gap the remaining transcripts are too small to close; placing the heavy
    # ones while there is still slack to absorb them keeps the totals tight.
    # The shuffle above survives as the tie-break, so this is randomisation
    # blocked on moment count rather than a deterministic ordering. Under
    # --balance-by transcripts every weight is 1 and the sort is a no-op.
    new_ids.sort(key=lambda tid: -_weight(transcripts[tid], balance_by))

    targets = {"heldout": heldout_fraction, "iterate": 1.0 - heldout_fraction}
    weight = {s: 0.0 for s in SPLITS}
    count = {s: 0 for s in SPLITS}

    # Seed the running totals with what the manifest already holds, so this
    # batch corrects any drift left by earlier ones rather than repeating it.
    # A transcript whose annotations have since been withdrawn keeps its slot
    # but is weighted by what we can still see for it (0 moments).
    for tid, rec in assignments.items():
        split = rec["split"]
        weight[split] += _weight(transcripts.get(tid, {"n_moments": 0}), balance_by)
        count[split] += 1

    new_assignments: dict[str, dict] = {}
    for tid in new_ids:
        w = _weight(transcripts[tid], balance_by)
        total_w = sum(weight.values()) + w
        total_n = sum(count.values()) + 1
        # Pick the split furthest below its target share once this transcript
        # lands. A zero-weight transcript (one annotated as having no key
        # moments) cannot move the weight balance at all, so ranking it by the
        # weight deficit would send every single one to whichever split is
        # behind by even a moment. Those are ranked on the count deficit
        # instead; the final key keeps ties deterministic.
        chosen = max(
            SPLITS,
            key=lambda s: (
                (targets[s] * total_w - weight[s]) if w else 0.0,
                targets[s] * total_n - count[s],
                s,
            ),
        )
        weight[chosen] += w
        count[chosen] += 1
        new_assignments[tid] = {
            "split": chosen,
            "batch": batch,
            "n_moments": transcripts[tid]["n_moments"],
            "no_key_moments": transcripts[tid]["no_key_moments"],
            "source": source,
        }

    return new_assignments, new_ids


# ===========================================================================
# Reporting
# ===========================================================================


def _totals(assignments: dict[str, dict]) -> dict[str, dict]:
    out = {s: {"transcripts": 0, "moments": 0, "no_key_moments": 0} for s in SPLITS}
    for rec in assignments.values():
        bucket = out[rec["split"]]
        bucket["transcripts"] += 1
        bucket["moments"] += rec["n_moments"]
        bucket["no_key_moments"] += int(rec["no_key_moments"])
    return out


def report(
    manifest: dict, transcripts: dict[str, dict], n_new: int, dry_run: bool
) -> str:
    """Split totals per provider, then over all providers."""
    assignments = manifest["assignments"]
    by_source: dict[str, dict[str, dict]] = {}
    for tid, rec in assignments.items():
        source = assigned_source(tid, rec, transcripts) or "unknown provider"
        by_source.setdefault(source, {})[tid] = rec

    lines = [
        "",
        f"{'DRY RUN -- nothing written' if dry_run else 'Manifest updated'}"
        f"  ({n_new} newly assigned transcript(s), batch {len(manifest['batches'])})",
    ]
    for source in sorted(by_source):
        lines += _report_table(source, by_source[source])
    lines += _report_table("all providers", assignments)
    return "\n".join(lines)


def _report_table(title: str, assignments: dict[str, dict]) -> list[str]:
    totals = _totals(assignments)
    all_t = sum(v["transcripts"] for v in totals.values())
    all_m = sum(v["moments"] for v in totals.values())
    lines = [
        "",
        f"  {title}",
        f"  {'split':<10}{'transcripts':>13}{'':>8}{'moments':>10}{'':>8}{'no-key-moments':>16}",
        f"  {'-' * 65}",
    ]
    for split in SPLITS:
        t, m, nk = (
            totals[split]["transcripts"],
            totals[split]["moments"],
            totals[split]["no_key_moments"],
        )
        lines.append(
            f"  {split:<10}{t:>13}{f'({t / all_t:.1%})' if all_t else '':>8}"
            f"{m:>10}{f'({m / all_m:.1%})' if all_m else '':>8}{nk:>16}"
        )
    lines.append(f"  {'-' * 65}")
    lines.append(
        f"  {'total':<10}{all_t:>13}{'':>8}{all_m:>10}{'':>8}"
        f"{sum(v['no_key_moments'] for v in totals.values()):>16}"
    )
    return lines


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.splits",
        description="Append-only iterate/heldout split assignment for v2 annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=list(DEFAULT_ANNOTATIONS_FILES),
        metavar="FILE",
        help="Raw v2 annotations JSONL, one per provider (<provider>_annotations.jsonl)",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        metavar="FILE",
        help="Append-only split manifest (created if absent)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Shuffle seed for new transcripts",
    )
    parser.add_argument(
        "--heldout-fraction",
        type=float,
        default=DEFAULT_HELDOUT_FRACTION,
        metavar="F",
        help="Target share of the corpus held out (0 < F < 1)",
    )
    parser.add_argument(
        "--balance-by",
        choices=("transcripts", "moments"),
        default="transcripts",
        help="Unit whose totals the split targets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the assignment that would be made without writing the manifest",
    )
    parser.add_argument(
        "--allow-param-change",
        action="store_true",
        help="Permit seed/fraction/balance-by to differ from the manifest's recorded "
        "values (existing assignments still stay frozen)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    if not 0.0 < args.heldout_fraction < 1.0:
        logger.error("--heldout-fraction must be strictly between 0 and 1")
        return 2

    transcripts = summarize_all(args.annotations)
    manifest = load_manifest(args.manifest)
    logger.info(
        "%d annotated transcript(s) in %s; %d already assigned in %s",
        len(transcripts),
        ", ".join(args.annotations),
        len(manifest["assignments"]),
        args.manifest,
    )

    # Params are recorded on first write. Later runs must match, because the
    # frozen assignments were drawn under the recorded values.
    recorded = {
        "seed": manifest.get("seed"),
        "heldout_fraction": manifest.get("heldout_fraction"),
        "balance_by": manifest.get("balance_by"),
    }
    requested = {
        "seed": args.seed,
        "heldout_fraction": args.heldout_fraction,
        "balance_by": args.balance_by,
    }
    if (
        recorded["seed"] is not None
        and recorded != requested
        and not args.allow_param_change
    ):
        logger.error(
            "manifest was built with %s but %s was requested; re-run with "
            "--allow-param-change to proceed (existing assignments stay frozen)",
            recorded,
            requested,
        )
        return 2

    stale = sorted(set(manifest["assignments"]) - set(transcripts))
    if stale:
        logger.warning(
            "%d assigned transcript(s) no longer appear in any annotations file; "
            "keeping their assignments (append-only): %s",
            len(stale),
            ", ".join(stale[:5]) + (" ..." if len(stale) > 5 else ""),
        )

    new_assignments, new_ids = assign_new(
        transcripts,
        manifest,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
        balance_by=args.balance_by,
    )

    if not new_ids:
        logger.info("no new transcripts to assign; manifest is up to date")
        print(report(manifest, transcripts, 0, dry_run=True))
        return 0

    manifest["assignments"].update(new_assignments)
    manifest.update(requested)
    manifest["batches"].append(
        {
            "batch": len(manifest["batches"]) + 1,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sources": list(args.annotations),
            "seed": args.seed,
            "heldout_fraction": args.heldout_fraction,
            "balance_by": args.balance_by,
            "n_assigned": len(new_ids),
            "n_assigned_by_source": dict(
                sorted(
                    Counter(rec["source"] for rec in new_assignments.values()).items()
                )
            ),
        }
    )

    if args.dry_run:
        print(report(manifest, transcripts, len(new_ids), dry_run=True))
        return 0

    write_manifest(args.manifest, manifest)
    logger.info("wrote %s", args.manifest)
    print(report(manifest, transcripts, len(new_ids), dry_run=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
