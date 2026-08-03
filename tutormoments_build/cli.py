"""Maintainer CLI for tutormoments dataset construction (`tutormoments-build`).

Subcommands:
  dataset build-ground-truth  raw human annotations -> per-conversation GT JSON
  dataset build               GT + transcripts + ids -> release dir with moments.jsonl
  dataset validate            check a release dir's moments.jsonl against its manifest

These create or package the benchmark; the runtime `tutormoments` CLI only
consumes released datasets.
"""

import argparse
import logging
import os
import sys

from tutormoments.logging_setup import (
    logging_args_parent,
    per_run_log_file,
    setup_logging,
)

# Fixed name (not __name__): under `python -m tutormoments_build.cli` this module
# is "__main__", which would fall outside the "tutormoments_build" logger level.
logger = logging.getLogger("tutormoments_build.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tutormoments-build",
        description="Tutormoments dataset construction (maintainer-only)",
    )
    subs = parser.add_subparsers(dest="command")
    log_parent = logging_args_parent()

    dataset_p = subs.add_parser(
        "dataset",
        help="Build, validate, and package benchmark datasets",
    )
    dataset_subs = dataset_p.add_subparsers(dest="dataset_command")

    # -- dataset build ---------------------------------------------------------
    build_p = dataset_subs.add_parser(
        "build",
        help="Build a release dir (moments.jsonl + manifest) from ground truth",
        parents=[log_parent],
    )
    build_p.add_argument(
        "--set", required=True, metavar="NAME", help="Set name, e.g. balanced_520"
    )
    build_p.add_argument(
        "--ids", required=True, metavar="FILE", help="Path to JSON list of moment ids"
    )
    build_p.add_argument(
        "--ground-truth",
        required=True,
        dest="ground_truth",
        metavar="DIR",
        help="Ground truth directory",
    )
    build_p.add_argument(
        "--transcripts", required=True, metavar="DIR", help="Transcripts directory"
    )
    build_p.add_argument(
        "--tutoring-provider-a-jsonl",
        default=None,
        dest="tutoring_provider_a_jsonl",
        metavar="FILE",
        help="Normalized JSONL transcript file (optional)",
    )
    build_p.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Release directory to write moments.jsonl + moments.manifest.json",
    )
    build_p.add_argument(
        "--created",
        default="",
        metavar="DATE",
        help="ISO date string for manifest (default: empty)",
    )
    build_p.add_argument(
        "--version",
        default="0",
        metavar="VERSION",
        help="Dataset version string for manifest (default: 0)",
    )

    # -- dataset build-from-run --------------------------------------------------
    bfr_p = dataset_subs.add_parser(
        "build-from-run",
        help="Rebuild the frozen moments set from a published benchmark run "
        "(the canonical record of the benchmark-time detections)",
        parents=[log_parent],
    )
    bfr_p.add_argument(
        "--set", required=True, metavar="NAME", help="Set name, e.g. balanced_520"
    )
    bfr_p.add_argument(
        "--reference-run",
        required=True,
        dest="reference_run",
        metavar="FILE",
        help="Published run JSONL (one row per replay)",
    )
    bfr_p.add_argument(
        "--transcripts",
        default=None,
        metavar="DIR",
        help="Transcripts directory (optional if JSONL given)",
    )
    bfr_p.add_argument(
        "--tutoring-provider-a-jsonl",
        default=None,
        dest="tutoring_provider_a_jsonl",
        metavar="FILE",
        help="Normalized JSONL transcript file (optional)",
    )
    bfr_p.add_argument(
        "--ids",
        default=None,
        metavar="FILE",
        help="Optional canonical id list to cross-check coverage against",
    )
    bfr_p.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Release directory to write moments.jsonl + moments.manifest.json",
    )
    bfr_p.add_argument(
        "--created",
        default="",
        metavar="DATE",
        help="ISO date string for manifest (default: empty)",
    )
    bfr_p.add_argument(
        "--version",
        default="0",
        metavar="VERSION",
        help="Dataset version string for manifest (default: 0)",
    )

    # -- dataset validate ------------------------------------------------------
    val_p = dataset_subs.add_parser(
        "validate",
        help="Validate a release dir's moments.jsonl against its manifest",
        parents=[log_parent],
    )
    val_p.add_argument(
        "--data_path",
        required=True,
        dest="data_path",
        metavar="DIR",
        help="Release directory containing moments.jsonl + moments.manifest.json",
    )

    # -- dataset build-ground-truth ---------------------------------------------
    gt_p = dataset_subs.add_parser(
        "build-ground-truth",
        help="Build ground truth from raw human annotations (LLM batch pipeline)",
        parents=[log_parent],
    )
    gt_p.add_argument(
        "--input",
        default=None,
        metavar="FILE",
        help="Annotations JSONL (default: packaged release path)",
    )
    gt_p.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        metavar="DIR",
        help="Output directory (default: data/ground_truth_<labeller>)",
    )
    gt_p.add_argument(
        "--labeller",
        default="hybrid",
        metavar="NAME",
        help="Labeller template name (default: hybrid)",
    )
    gt_p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Plan only; no LLM calls or writes",
    )
    gt_p.add_argument(
        "--scaffolding-only",
        action="store_true",
        dest="scaffolding_only",
        help="Restrict to scaffolding records (merge-preserve rapport)",
    )
    gt_p.add_argument(
        "--refresh-agg",
        nargs="?",
        const="both",
        default=None,
        choices=["action", "result", "both"],
        dest="refresh_agg",
        help="Reclassify action/result aggregations",
    )
    gt_p.add_argument(
        "--refresh-decomp",
        nargs="?",
        const="both",
        default=None,
        choices=["action", "result", "both"],
        dest="refresh_decomp",
        help="Re-decompose action/result facets (cache keys on text, not prompt)",
    )
    gt_p.add_argument(
        "--refresh-overscaffold",
        action="store_true",
        dest="refresh_overscaffold",
        help="Re-decompose over-scaffolding for all scaffolding moments",
    )
    gt_p.add_argument(
        "--consolidate",
        action="store_true",
        help="Also write a consolidated <out-dir>.jsonl after building",
    )

    return parser


def _cmd_build_ground_truth(args, command_line: str) -> None:
    """Dispatch to groundtruth.build_ground_truth."""
    from contextlib import nullcontext

    from tutormoments_build import groundtruth

    input_path = args.input or groundtruth.ANNOTATIONS_JSONL
    out_dir = args.out_dir or groundtruth.default_out_dir(args.labeller)
    # --dry-run promises no writes, so it gets no build.log either.
    log_ctx = (
        nullcontext()
        if args.dry_run
        else per_run_log_file(
            os.path.join(str(out_dir), "build.log"),
            current_thread_only=False,
            header=command_line,
        )
    )
    with log_ctx:
        groundtruth.build_ground_truth(
            input_path=input_path,
            out_dir=out_dir,
            labeller=args.labeller,
            dry_run=args.dry_run,
            scaffolding_only=args.scaffolding_only,
            refresh_agg=args.refresh_agg,
            refresh_decomp=args.refresh_decomp,
            refresh_overscaffold=args.refresh_overscaffold,
            consolidate=args.consolidate,
        )


def main(argv=None) -> None:
    """Parse args and dispatch subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "dataset" or not getattr(args, "dataset_command", None):
        parser.print_help()
        sys.exit(0 if args.command is None else 1)

    setup_logging(level=args.log_level, log_file=args.log_file)
    command_line = "Command: tutormoments-build " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    logger.info("%s", command_line)

    from tutormoments_build.moments_build import (
        _cli_build,
        _cli_build_from_run,
        _cli_validate,
    )

    if args.dataset_command == "build":
        with per_run_log_file(
            os.path.join(args.out, "build.log"),
            current_thread_only=False,
            header=command_line,
        ):
            _cli_build(args)
    elif args.dataset_command == "build-from-run":
        with per_run_log_file(
            os.path.join(args.out, "build.log"),
            current_thread_only=False,
            header=command_line,
        ):
            _cli_build_from_run(args)
    elif args.dataset_command == "validate":
        _cli_validate(args)
    elif args.dataset_command == "build-ground-truth":
        _cmd_build_ground_truth(args, command_line)
    else:
        raise SystemExit(
            "Choose a dataset subcommand: build, build-from-run, "
            "build-ground-truth, or validate"
        )


if __name__ == "__main__":
    main()
