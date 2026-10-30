"""Decompose adjudicators' situation descriptions into atomic facets about the student.

Each adjudicator writes a ``situation.why`` account of the moment before the
cut. Per moment those accounts are merged into one situation description:

  one adjudicator             their account as written
  two, very similar           the longer account (length ties broken at random)
  two, different              both accounts, concatenated in the order saved

"Very similar" is a ``difflib`` similarity ratio of at least
``SIMILARITY_THRESHOLD``. The ratio is recorded on every record, so a different
cutoff can be judged without re-running anything.

The merged description fills ``prompts/v2_helper/student_state.md``, which asks
for a JSON array of standalone facets describing the student's state, behaviour
and actions. The array is parsed with the ground-truth pipeline's lenient facet
parser (``tutormoments.scoring._parse_decomposed``).

Only moments in the ground truth are decomposed. Output is one file,
``data/ground_truth/student_state.jsonl``, one record per ground-truth moment in
ground-truth file order. Each record carries ``moment_id`` and ``split``
(``iteration`` or ``test``), so it joins on ``moment_id`` against
``data/ground_truth/<split>.jsonl``. A moment no adjudicator described is still
written, with ``facets`` None and ``skipped`` saying why, so the join never
comes up short.

``--model`` defaults to ``claude-opus-5-5`` and must be configured as for
``classify_excerpts`` (``v2`` block, ``v2.models``, or a ``benchmark_models``
arm); its provider-native reasoning parameters come from that entry.

Usage::

    python -m tutormoments_build.v2.decompose_situation --dry-run
    python -m tutormoments_build.v2.decompose_situation --print
    python -m tutormoments_build.v2.decompose_situation --limit 20
    python -m tutormoments_build.v2.decompose_situation
    python -m tutormoments_build.v2.decompose_situation --batch-id msgbatch_01ABC...
"""

import argparse
import difflib
import itertools
import json
import logging
import os
import random
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.resources import resource_text
from tutormoments_build.v2.build_ground_truth import (
    DEFAULT_ANNOTATOR_LABELS,
    DEFAULT_EXCLUDED_ANNOTATORS,
    adjudications,
    annotator_label,
    answered_payload,
    latest_annotations,
    load_annotator_labels,
    load_excluded_annotators,
    read_rows,
)
from tutormoments_build.v2.build_ground_truth import DEFAULT_OUT_DIR as DEFAULT_GT_DIR
from tutormoments_build.v2.classify_excerpts import (
    DEFAULT_STUDENT_STATE,
    PLACEHOLDER,
    RANDOM,
    SPLIT_STEMS,
    ZERO_USAGE,
    V2Spec,
    _sum_usage,
    phase_config,
    run_entries,
)
from tutormoments_build.v2.splits import DEFAULT_ANNOTATIONS_FILES

logger = logging.getLogger("tutormoments_build.v2.decompose_situation")


# Resolves to its `benchmark_models` arm, so it decomposes at the parameters
# the benchmark runs it with.
DEFAULT_MODEL = "claude-opus-5-5"

PROMPT_PATH = "prompts/v2_helper/student_state.md"

# Accounts at least this similar count as the same account, and the longer is
# kept; below it, both are kept. A judgement call: adjudicators often paste the
# same text and edit it lightly, and a light edit should not double the
# description. The ratio is recorded per moment (see merge_descriptions).
SIMILARITY_THRESHOLD = 0.8

# Seeds the length-tie pick. Mixed with the moment id, so a moment's pick is the
# same on every run and does not depend on which other moments were loaded.
TIE_SEED = "v2-student-state"

# How two or more accounts are joined when they differ.
CONCAT_SEPARATOR = "\n\n"

# Batch-entry key prefix. A moment_id is a UUID, so a key splits on "__".
ENTRY_PREFIX = "state"

NO_DESCRIPTION = "no_description"


# ===========================================================================
# Reading
# ===========================================================================


def load_adjudicator_descriptions(
    annotations_paths=DEFAULT_ANNOTATIONS_FILES,
    excluded_path: str = DEFAULT_EXCLUDED_ANNOTATORS,
    labels_path: str = DEFAULT_ANNOTATOR_LABELS,
) -> dict[str, list[dict]]:
    """Return {moment_id: [{"adjudicator": label, "text": why}, ...]}.

    Each adjudicator's ``situation.why`` from their final answers, in the order
    they saved. Staff and superseded re-saves are dropped exactly as the ground
    truth build drops them; an adjudicator who threw the moment out or left the
    box blank contributes nothing. Adjudicators are named by their de-identified
    label, never their name.
    """
    excluded = load_excluded_annotators(excluded_path)
    labels = load_annotator_labels(labels_path)
    out: dict[str, list[dict]] = {}
    for row in read_rows(list(annotations_paths)):
        if "moment" not in row:
            continue
        found = []
        annotations = latest_annotations(row.get("annotations") or [], excluded)
        for annotation in adjudications(annotations):
            situation = (answered_payload(annotation) or {}).get("situation") or {}
            text = str(situation.get("why") or "").strip()
            if text:
                found.append(
                    {"adjudicator": annotator_label(annotation, labels), "text": text}
                )
        if found:
            out[row["moment"]["moment_id"]] = found
    return out


def load_ground_truth(gt_dir: str, stems=SPLIT_STEMS) -> list[dict]:
    """Ground-truth records across ``stems``, in file order.

    A missing split file is not an error -- a build may have produced only one.
    """
    out: list[dict] = []
    for stem in stems:
        path = os.path.join(gt_dir, f"{stem}.jsonl")
        if not os.path.exists(path):
            logger.warning("no ground-truth file at %s; skipping split", path)
            continue
        with open(path, encoding="utf-8") as fh:
            out.extend(json.loads(line) for line in fh if line.strip())
    if not out:
        raise FileNotFoundError(
            f"no ground-truth splits found in {gt_dir}; run "
            "`python -m tutormoments_build.v2.build_ground_truth` first"
        )
    return out


# ===========================================================================
# Merging
# ===========================================================================


def similarity(a: str, b: str) -> float:
    """difflib similarity ratio between two accounts, 0 to 1."""
    # autojunk=False: otherwise difflib ignores common letters in long text.
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def merge_descriptions(moment_id: str, candidates: list[dict]) -> dict:
    """One situation description from a moment's adjudicator accounts.

    Returns ``situation`` (the text sent to the model), ``method`` (``single``,
    ``longer`` or ``concatenated``), ``similarity`` (the lowest pairwise ratio;
    None for a single account) and ``adjudicators`` (whose accounts the text
    holds, in order).

    The accounts are kept whole only if every pair is similar, so this holds
    for more than two adjudicators as well, though the data has at most two.
    """
    if len(candidates) == 1:
        only = candidates[0]
        return {
            "situation": only["text"],
            "method": "single",
            "similarity": None,
            "adjudicators": [only["adjudicator"]],
        }

    lowest = min(
        similarity(a["text"], b["text"])
        for a, b in itertools.combinations(candidates, 2)
    )
    if lowest >= SIMILARITY_THRESHOLD:
        longest = max(len(c["text"]) for c in candidates)
        tied = [c for c in candidates if len(c["text"]) == longest]
        pick = random.Random(f"{TIE_SEED}:{moment_id}").choice(tied)
        return {
            "situation": pick["text"],
            "method": "longer",
            "similarity": lowest,
            "adjudicators": [pick["adjudicator"]],
        }

    return {
        "situation": CONCAT_SEPARATOR.join(c["text"] for c in candidates),
        "method": "concatenated",
        "similarity": lowest,
        "adjudicators": [c["adjudicator"] for c in candidates],
    }


# ===========================================================================
# Prompting and parsing
# ===========================================================================


def fill(template: str, situation: str) -> str:
    """Substitute ``{situation}`` into the template.

    One pass with the shared placeholder pattern, so an adjudicator's text is
    never itself searched for slot names, and any other placeholder raises
    rather than reaching the model as a literal.
    """
    unfilled = [name for name in PLACEHOLDER.findall(template) if name != "situation"]
    if unfilled:
        raise ValueError(f"template slot(s) left unfilled: {', '.join(unfilled)}")
    return PLACEHOLDER.sub(lambda _: situation, template)


def build_entries(merged: dict[str, dict], template: str) -> list[dict]:
    """One batch entry per described moment."""
    from tutormoments.client import build_batch_entry

    return [
        build_batch_entry(
            f"{ENTRY_PREFIX}__{moment_id}",
            fill(template, m["situation"]),
            json_mode=True,
        )
        for moment_id, m in merged.items()
    ]


def parse_facets(text: str) -> tuple[list[str] | None, bool]:
    """Parse a response into facet strings. Returns (facets, had_error).

    ``facets`` is None when the response could not be parsed, which is
    distinct from ``[]``: the prompt asks for an empty array when the
    description says nothing about the student.
    """
    from tutormoments.scoring import _parse_decomposed

    facets, had_error = _parse_decomposed(text or "")
    if had_error:
        return None, True
    return [f.strip() for f in facets if f.strip()], False


# ===========================================================================
# Assembly
# ===========================================================================


def build_record(gt: dict, merged: dict | None, raw: dict | None, cfg: V2Spec) -> dict:
    """One output record for a ground-truth moment."""
    record = {
        "moment_id": gt["moment_id"],
        "transcript_id": gt["transcript_id"],
        "source": gt.get("source"),
        "split": gt["split"],
        "model": cfg.model,
        "provider": cfg.provider,
        "thinking": dict(cfg.thinking),
        "prompt": PROMPT_PATH,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
    if merged is None:
        return {
            **record,
            "situation": None,
            "situation_method": None,
            "situation_similarity": None,
            "adjudicators": [],
            "skipped": NO_DESCRIPTION,
            "facets": None,
            "parse_error": False,
            "error": None,
            "raw": None,
            "usage": dict(ZERO_USAGE),
        }

    raw = raw or {}
    text = raw.get("text", "")
    facets, had_error = parse_facets(text)
    return {
        **record,
        "situation": merged["situation"],
        "situation_method": merged["method"],
        "situation_similarity": merged["similarity"],
        "adjudicators": merged["adjudicators"],
        "skipped": None,
        "facets": facets,
        "parse_error": had_error,
        "error": raw.get("error"),
        "raw": text,
        "usage": raw.get("usage", dict(ZERO_USAGE)),
    }


def merge_all(ground_truth: list[dict], descriptions: dict[str, list[dict]]) -> dict:
    """{moment_id: merged description} for every described ground-truth moment."""
    return {
        gt["moment_id"]: merge_descriptions(
            gt["moment_id"], descriptions[gt["moment_id"]]
        )
        for gt in ground_truth
        if descriptions.get(gt["moment_id"])
    }


def write_records(path: str, records: list[dict]) -> str:
    """Write the records JSONL atomically. Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
    return path


# ===========================================================================
# Reporting
# ===========================================================================


def report(
    ground_truth: list[dict],
    merged: dict[str, dict],
    cfg: V2Spec,
    records: list[dict] | None = None,
    out_path: str | None = None,
) -> str:
    dry_run = records is None
    knobs = ", ".join(f"{k}: {v}" for k, v in sorted(cfg.thinking.items()))
    lines = [
        "",
        "DRY RUN -- no API calls made, nothing written"
        if dry_run
        else "Facets written",
        "",
        f"  model:  {cfg.model} ({knobs or 'no reasoning params sent'})",
        f"  prompt: {PROMPT_PATH}",
    ]
    if out_path:
        lines.append(f"  output: {out_path}")
    lines.append("")

    splits = Counter(gt["split"] for gt in ground_truth)
    lines.append("  moments: " + ", ".join(f"{n} {s}" for s, n in splits.items()))
    lines.append(
        f"  not described by any adjudicator: {len(ground_truth) - len(merged)}"
    )
    methods = Counter(m["method"] for m in merged.values())
    lines.append(
        f"  situation (similarity cutoff {SIMILARITY_THRESHOLD}): "
        + ", ".join(f"{methods[k]} {k}" for k in ("single", "longer", "concatenated"))
    )

    if records is not None:
        sent = [r for r in records if not r["skipped"]]
        unparsed = sum(1 for r in sent if r["parse_error"])
        empty = sum(1 for r in sent if r["facets"] == [])
        n_facets = sum(len(r["facets"] or []) for r in sent)
        usage = _sum_usage(*(r["usage"] for r in records))
        lines += [
            "",
            f"  facets: {n_facets} across {len(sent)} moment(s); "
            f"{empty} with none; {unparsed} unparsed",
            f"  tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out "
            f"/ {usage['total_tokens']:,} total",
        ]
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.decompose_situation",
        description="Decompose adjudicators' situation descriptions into "
        "facets about the student",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--gt-dir", default=DEFAULT_GT_DIR, metavar="DIR")
    parser.add_argument(
        "--out",
        default=DEFAULT_STUDENT_STATE,
        metavar="PATH",
        help="Output file; classify_excerpts reads its facets from here",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help="Model to decompose with. Must be the config's `v2` model, under "
        "`v2.models`, or a `benchmark_models` arm; its provider-native "
        "reasoning parameters come from that entry.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_STEMS,
        default=list(SPLIT_STEMS),
        help="Which ground-truth splits to decompose.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Decompose only the first N moments of each split, for a smoke test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file. Without it a run that would "
        "overwrite one refuses before making any API call.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        metavar="ID",
        help="Resume polling an in-flight batch instead of submitting a new one. "
        "Must be a batch submitted with the same moments and flags.",
    )
    parser.add_argument(
        "--print",
        dest="print_moment",
        nargs="?",
        const=RANDOM,
        metavar="MOMENT_ID",
        help="Print one moment's filled prompt to stdout and exit (id prefix "
        "accepted). Pass it bare to print a random described moment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Merge the descriptions, report, and print one random moment's "
        "filled prompt; make no API calls and write nothing",
    )
    return parser


def _print_prompt(merged: dict[str, dict], prefix, template: str) -> int:
    if prefix is RANDOM:
        moment_id = random.choice(list(merged)) if merged else None
    else:
        moment_id = next((m for m in merged if m.startswith(prefix)), None)
    if moment_id is None:
        print(f"no described moment matching {prefix!r}", file=sys.stderr)
        return 1
    m = merged[moment_id]
    print(
        f"===== moment {moment_id} ({m['method']}, "
        f"adjudicators: {', '.join(m['adjudicators'])}) =====\n"
    )
    print(fill(template, m["situation"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    template = resource_text(PROMPT_PATH)
    out_path = args.out

    ground_truth = []
    for stem in args.splits:
        split = load_ground_truth(args.gt_dir, stems=(stem,))
        ground_truth += split[: args.limit] if args.limit is not None else split
    merged = merge_all(ground_truth, load_adjudicator_descriptions())

    if args.print_moment:
        return _print_prompt(merged, args.print_moment, template)

    # Resolved before the dry run too, so an unknown --model fails there.
    cfg = phase_config(model=args.model)

    if args.dry_run:
        print(report(ground_truth, merged, cfg, out_path=out_path))
        # One random moment's filled prompt, as `--print` would show it.
        if merged:
            _print_prompt(merged, RANDOM, template)
        return 0

    # Before the batch, not after it: see classify_excerpts.check_writable.
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass --overwrite to replace it, or "
            "--out to write elsewhere."
        )

    entries = build_entries(merged, template)
    logger.info(
        "decomposing %d moment(s) with model=%s thinking=%s",
        len(entries),
        cfg.model,
        cfg.thinking,
    )
    raw_entries = (
        run_entries(
            entries,
            cfg,
            batch_id=args.batch_id,
            display_name="v2_decompose_situation",
        )
        if entries
        else {}
    )

    records = [
        build_record(
            gt,
            merged.get(gt["moment_id"]),
            raw_entries.get(f"{ENTRY_PREFIX}__{gt['moment_id']}"),
            cfg,
        )
        for gt in ground_truth
    ]
    logger.info(
        "wrote %d record(s) to %s", len(records), write_records(out_path, records)
    )
    print(report(ground_truth, merged, cfg, records=records, out_path=out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
