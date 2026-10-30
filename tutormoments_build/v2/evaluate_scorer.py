"""Score v2 predictions against the gold labels: precision, recall, F1.

``classify_excerpts.py`` writes predictions to
``<pred-dir>/<model>/<prompt version>/<excerpt width>/<split>.jsonl``, each
record carrying the gold ``labels`` it was made against, so those files are all
that scoring needs.

Every task is reported in up to three blocks:

  all moments        every moment with a usable prediction
  agreed             only the moments the ground truth flags as agreed for that
                     construct (the per-record ``agreement`` flags)
  short moments      only the moments with ``--short-turns`` (5 by default) or
                     fewer dialogue turns after the cut point

The agreed block needs ``agreement`` flags on the ground-truth records. The
current build (``build_ground_truth.py``) resolves each label by majority vote
over every annotator and does not write them, so against today's ground truth
the block is skipped and the run says so, rather than scoring an empty subset
as zeros. Where the flags exist the subset is per construct: a moment can be
agreed for rigor and contested for scaffolding, and it is scored in the agreed
block of the one and not the other.

The short-moment block is a different cut of the same predictions: how the
model does when there is little tutoring after the cut to read. It is a slice on
the excerpt, not on the labels, so it drops positives and negatives alike and
its precision and recall both move freely. The turn count
is ``post_cut_dialogue_rows``, written by the excerpt builder -- dialogue turns
only, so the screen-activity rows in the span are not counted as turns.

Per-annotator values are not in the prediction records, so the agreement flags
are read back from the ground truth (``--ground-truth-dir``) and joined on
``moment_id``. Where a moment's gold labels there no longer match the ones the
prediction was made against, the predictions are stale relative to the ground
truth and the run says so.

Three binary tasks are scored, positive class = True:

  scaffolding        action_direction.scaffolding      vs labels.scaffolding_present
  rigor              action_direction.rigor            vs labels.rigor_present
  over-scaffolding   over_scaffolding.over_scaffolding vs labels.over_scaffolding_present

Over-scaffolding is scored only on the moments it was asked about (the gold gate
in ``classify_excerpts``): elsewhere there is no prediction, not a "no".

A moment with no usable prediction -- never classified, not asked, a failed
request, or a response that would not parse -- is left out of the counts and
reported separately, broken down by reason under each block. Scoring an
unparsed response as "no" would credit a parse failure as a correct negative on
the mostly-negative tasks, which reads as a better model rather than a broken
one.

Usage::

    python -m tutormoments_build.v2.evaluate_scorer
    python -m tutormoments_build.v2.evaluate_scorer --model claude-opus-5
    python -m tutormoments_build.v2.evaluate_scorer --prompt-version 1
    python -m tutormoments_build.v2.evaluate_scorer --context-turns 20
    python -m tutormoments_build.v2.evaluate_scorer --splits iteration test
    python -m tutormoments_build.v2.evaluate_scorer --short-turns 3
    python -m tutormoments_build.v2.evaluate_scorer --ground-truth-dir data/ground_truth

With no ``--prompt-version`` every version found under a model is scored, so a
bare run reports each revision of the prompts against the same gold. Likewise
with no ``--context-turns``: every excerpt width found under a version.
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.v2 import excerpts
from tutormoments_build.v2.build_ground_truth import (
    DEFAULT_OUT_DIR as DEFAULT_GROUND_TRUTH_DIR,
)
from tutormoments_build.v2.classify_excerpts import (
    DEFAULT_OUT_DIR,
    DEFAULT_SPLITS,
    PASSES,
)

logger = logging.getLogger("tutormoments_build.v2.evaluate_scorer")

# Moments with this many dialogue turns after the cut or fewer are reported
# again as their own block: a short post-cut span is less evidence to classify
# from, and the question is whether the model's errors concentrate there.
SHORT_POST_CUT_TURNS = 5

# (task name, prediction pass, field in that pass, gold field in labels)
TASKS = (
    ("scaffolding", "action_direction", "scaffolding", "scaffolding_present"),
    ("rigor", "action_direction", "rigor", "rigor_present"),
    (
        "over-scaffolding",
        "over_scaffolding",
        "over_scaffolding",
        "over_scaffolding_present",
    ),
)


# Why a moment has no (gold, predicted) pair for a task, in the order they are
# checked, with the report's column header for each. A missing gold label has no
# column: it should not happen, so it gets a line of its own when it does.
NOT_SCORED_COLUMNS = {
    "not_run": "not run",
    "no_post_cut_content": "empty post-cut",
    "not_asked": "not asked",
    "response_failed": "response failed",
    "unparsed": "response parse failed",
}


# ===========================================================================
# Metrics
# ===========================================================================


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def score(pairs: list[tuple[bool, bool]]) -> dict:
    """Confusion counts and precision/recall/F1 over (gold, predicted) pairs."""
    tp = sum(1 for gold, pred in pairs if gold and pred)
    fp = sum(1 for gold, pred in pairs if not gold and pred)
    fn = sum(1 for gold, pred in pairs if gold and not pred)
    tn = sum(1 for gold, pred in pairs if not gold and not pred)

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "n": len(pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": tp + fn,  # gold positives
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "accuracy": _ratio(tp + tn, len(pairs)),
    }


def not_scored_reason(record: dict, pass_name: str, field: str, gold_field: str):
    """Why ``record`` has nothing to score for one task, or None if it does.

    A ``NOT_SCORED_COLUMNS`` key, or ``"no_gold"``. "not_asked" is the
    over-scaffolding gold gate; "response_failed" covers both a request that
    errored and one whose result never made it into the file. "not_run" is a
    pass the round left out with --passes; records from before that flag ran
    every pass.
    """
    if pass_name not in record.get("passes", PASSES):
        return "not_run"
    result = record.get(pass_name)
    if result is None:
        if record.get("skipped"):
            return "no_post_cut_content"
        if pass_name == "over_scaffolding" and not record.get("over_scaffolding_asked"):
            return "not_asked"
        return "response_failed"
    if result.get(field) is None:
        return "response_failed" if result.get("error") else "unparsed"
    if (record.get("labels") or {}).get(gold_field) is None:
        return "no_gold"
    return None


def collect(records: list[dict], pass_name: str, field: str, gold_field: str) -> tuple:
    """Return ([(gold, predicted), ...], Counter of not-scored reasons) for one task.

    A record is not scored when there is no prediction or no gold label to pair
    it with; the reason is counted rather than guessed at as a "no".
    """
    pairs, not_scored = [], Counter()
    for record in records:
        reason = not_scored_reason(record, pass_name, field, gold_field)
        if reason:
            not_scored[reason] += 1
            continue
        pairs.append(
            (bool(record["labels"][gold_field]), bool(record[pass_name][field]))
        )
    return pairs, not_scored


def _not_scored(reasons: Counter) -> dict:
    """The per-block not-scored fields: the total and its breakdown by reason."""
    return {"missing": sum(reasons.values()), "not_scored": dict(reasons)}


def agreed_records(
    records: list[dict], agreement: dict[str, dict], gold_field: str
) -> tuple[list[dict], int]:
    """Records the ground truth flags as agreed for one construct.

    Returns (kept, n_unknown); unknown is a moment the ground truth has no
    agreement flags for, which is a moment that has left the ground truth since
    it was classified rather than a contested one -- it belongs in neither
    subset, so it is counted and dropped.

    For over-scaffolding the flag is agreement on the *declared* amount, the
    only over-scaffolding question annotators answer, and the only thing the
    gold ``over_scaffolding_present`` is built from.
    """
    kept, unknown = [], 0
    for record in records:
        flags = agreement.get(record.get("moment_id"))
        if flags is None:
            unknown += 1
            continue
        if flags.get(gold_field):
            kept.append(record)
    return kept, unknown


def short_records(records: list[dict], max_turns: int) -> tuple[list[dict], int]:
    """Records with ``max_turns`` or fewer dialogue turns after the cut.

    Returns (kept, n_unknown); unknown is a record written before
    ``post_cut_dialogue_rows`` existed, which cannot be placed on either side of
    the threshold, so it is counted and dropped rather than guessed at.
    """
    kept, unknown = [], 0
    for record in records:
        turns = record.get("post_cut_dialogue_rows")
        if turns is None:
            unknown += 1
            continue
        if turns <= max_turns:
            kept.append(record)
    return kept, unknown


# ===========================================================================
# Reading
# ===========================================================================


def read_predictions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_ground_truth(ground_truth_dir: str, split: str) -> list[dict] | None:
    """One split's ground-truth records, or None when the file is not there.

    Scoring the whole set needs nothing but the predictions; only the agreed
    subset needs the ground truth back, so a missing file is a skipped subset
    rather than an error.
    """
    path = os.path.join(ground_truth_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        logger.warning(
            "no ground truth at %s; reporting all moments only, without the "
            "annotator-agreement subset",
            path,
        )
        return None
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def agreement_by_moment(ground_truth: list[dict]) -> dict[str, dict] | None:
    """{moment_id: agreement flags} -- per construct, did the annotators match.

    None when no record carries ``agreement``: the majority-vote build does not
    write it, and reading every moment as "no flags" would empty the agreed
    block and report it as zeros.
    """
    if not any("agreement" in r for r in ground_truth):
        return None
    return {r["moment_id"]: (r.get("agreement") or {}) for r in ground_truth}


def stale_moments(predictions: list[dict], ground_truth: list[dict]) -> list[str]:
    """Moments whose gold labels have moved since the predictions were made.

    The agreement flags are joined on moment_id, so they only describe these
    predictions while the labels beside them still match. Where they do not, the
    predictions want re-running against the current ground truth; scoring goes
    ahead either way, but not silently.
    """
    gold = {r["moment_id"]: r.get("labels") or {} for r in ground_truth}
    return sorted(
        record["moment_id"]
        for record in predictions
        if record["moment_id"] in gold
        and (record.get("labels") or {}) != gold[record["moment_id"]]
    )


def _subdirectories(path: str) -> list[str]:
    return sorted(
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    )


def find_models(pred_dir: str, model: str | None) -> list[str]:
    """Model subdirectory names to evaluate: the one asked for, or all present."""
    if model:
        return [model.replace("/", "_")]
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(
            f"no predictions at {pred_dir}; run "
            "`python -m tutormoments_build.v2.classify_excerpts` first"
        )
    return _subdirectories(pred_dir)


def find_prompt_versions(model_dir: str, prompt_version: str | None) -> list[str]:
    """Prompt-version subdirectories to score under one model.

    Default is every version present, so a bare run reports each revision of the
    prompts side by side -- which is the comparison a prompt edit is made to
    settle. Versions sort numerically where they are numbers, so v10 lands after
    v9 rather than after v1.
    """
    if prompt_version:
        return [prompt_version]
    if not os.path.isdir(model_dir):
        return []
    return sorted(
        _subdirectories(model_dir),
        key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else n),
    )


def find_widths(version_dir: str, width: str | None) -> list[str]:
    """Excerpt-width subdirectories to score under one prompt version.

    Default is every width present, widest first, so a bare run puts a narrow
    window beside the full transcript it is measured against. A directory that
    is not a width key is not a result set and is left out.
    """
    if width:
        return [width]
    if not os.path.isdir(version_dir):
        return []
    return excerpts.sort_widths(
        name
        for name in _subdirectories(version_dir)
        if name == excerpts.FULL_KEY or name.isdigit()
    )


def evaluate(
    records: list[dict],
    agreement: dict[str, dict] | None = None,
    short_turns: int = SHORT_POST_CUT_TURNS,
) -> list[dict]:
    """Score every task over one split's prediction records.

    Each row carries three subsets: ``all``, every moment with a usable
    prediction; ``agreed``, only the moments the ground truth flags as agreed on
    that construct, and None when there are no agreement flags to join against;
    and ``short``, only the moments with ``short_turns`` or fewer dialogue
    turns after the cut point.
    """
    short, short_unknown = short_records(records, short_turns)

    rows = []
    for name, pass_name, field, gold_field in TASKS:
        pairs, reasons = collect(records, pass_name, field, gold_field)
        row = {"task": name, "all": {**_not_scored(reasons), **score(pairs)}}

        if agreement is None:
            row["agreed"] = None
        else:
            kept, unknown = agreed_records(records, agreement, gold_field)
            pairs, reasons = collect(kept, pass_name, field, gold_field)
            row["agreed"] = {
                **_not_scored(reasons),
                "unknown": unknown,
                "contested": len(records) - len(kept) - unknown,
                **score(pairs),
            }

        pairs, reasons = collect(short, pass_name, field, gold_field)
        row["short"] = {
            **_not_scored(reasons),
            "unknown": short_unknown,
            "turns": short_turns,
            "moments": len(short),
            "of_moments": len(records) - short_unknown,
            **score(pairs),
        }
        rows.append(row)
    return rows


# ===========================================================================
# Reporting
# ===========================================================================


def _score_line(task: str, scores: dict) -> str:
    return (
        f"  {task:<18}{scores['n']:>6}{scores['support']:>10}"
        f"{scores['tp']:>5}{scores['fp']:>5}{scores['fn']:>5}{scores['tn']:>5}"
        f"{scores['precision']:>8.2f}{scores['recall']:>7.2f}{scores['f1']:>7.2f}"
    ) + "".join(
        f"{scores['not_scored'].get(key, 0):>{len(header) + 2}}"
        for key, header in NOT_SCORED_COLUMNS.items()
    )


def _no_gold_lines(rows: list[dict], block: str) -> list[str]:
    """A warning per task whose ``block`` has moments with no gold label."""
    return [
        f"  {row['task']}: {row[block]['not_scored']['no_gold']} moment(s) have "
        "no gold label and were not scored"
        for row in rows
        if row[block]["not_scored"].get("no_gold")
    ]


def report(
    model: str, prompt_version: str, width: str, split: str, rows: list[dict]
) -> str:
    """One block per subset: all moments, the ones flagged as agreed, and
    the ones with a short post-cut span."""
    header = (
        f"  {'task':<18}{'n':>6}{'gold yes':>10}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
        f"{'prec':>8}{'rec':>7}{'F1':>7}"
    ) + "".join(
        f"{header:>{len(header) + 2}}" for header in NOT_SCORED_COLUMNS.values()
    )
    lines = [
        "",
        f"  {model}  --  prompts v{prompt_version}  --  @{width} excerpt  --  {split}",
        "",
        header,
        f"  {'-' * (len(header) - 2)}",
        "  all moments",
    ]
    lines += [_score_line(row["task"], row["all"]) for row in rows]
    lines += _no_gold_lines(rows, "all")

    scored = [row for row in rows if row["agreed"] is not None]
    if scored:
        lines += ["", "  annotators agreed on the construct (ground-truth flags)"]
        lines += [_score_line(row["task"], row["agreed"]) for row in scored]
        lines += _no_gold_lines(scored, "agreed")
        dropped = ",  ".join(
            f"{row['task']} {row['agreed']['contested']}" for row in scored
        )
        lines.append(f"  contested moments left out:  {dropped}")
        unknown = max(row["agreed"]["unknown"] for row in scored)
        if unknown:
            lines.append(
                f"  {unknown} moment(s) are not in the ground truth and are in "
                "neither block"
            )
    else:
        lines += [
            "",
            "  no agreement flags in the ground truth; the agreement subset was skipped",
        ]

    short = rows[0]["short"]
    lines += ["", f"  {short['turns']} or fewer turns after the cut"]
    lines += [_score_line(row["task"], row["short"]) for row in rows]
    lines += _no_gold_lines(rows, "short")
    lines.append(
        f"  {short['moments']} of {short['of_moments']} moment(s); turns are "
        "dialogue turns after the cut, screen activity not counted"
    )
    if short["unknown"]:
        lines.append(
            f"  {short['unknown']} moment(s) predate the post-cut turn count "
            "and are left out of this block"
        )

    lines += [
        "",
        "  positive class = True; moments in the last four columns have no "
        "prediction and are left out of n, not counted as 'no'",
    ]
    lines.append(
        "  the short block slices the excerpt, not the labels, so it drops "
        "positives and negatives alike"
    )
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.evaluate_scorer",
        description="Score v2 predictions against the gold labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--pred-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--ground-truth-dir",
        default=DEFAULT_GROUND_TRUTH_DIR,
        metavar="DIR",
        help="Ground truth to check the predictions' gold labels against, and "
        "to read per-construct agreement flags from where its records carry them.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model whose predictions to score. Default: every model found "
        "under --pred-dir.",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        metavar="VERSION",
        help="Prompt version whose predictions to score. Default: every "
        "version found under each model.",
    )
    parser.add_argument(
        "--context-turns",
        default=None,
        metavar="WIDTH",
        help="Excerpt width whose predictions to score (`full` or a turn "
        "count). Default: every width found under each prompt version.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Which prediction splits to score",
    )
    parser.add_argument(
        "--short-turns",
        type=int,
        default=SHORT_POST_CUT_TURNS,
        metavar="N",
        help="Report a separate block over the moments with N or fewer "
        "dialogue turns after the cut point",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        metavar="FILE",
        help="Also write the scores as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    # One read per split, shared across every model and prompt version: they are
    # all scored against the same gold.
    agreement, ground_truths = {}, {}
    for split in args.splits:
        ground_truth = read_ground_truth(args.ground_truth_dir, split)
        agreement[split] = (
            agreement_by_moment(ground_truth) if ground_truth is not None else None
        )
        if ground_truth is not None:
            ground_truths[split] = ground_truth
            if agreement[split] is None:
                logger.info(
                    "%s ground truth carries no agreement flags; skipping the "
                    "agreement subset",
                    split,
                )

    results = []
    for model in find_models(args.pred_dir, args.model):
        model_dir = os.path.join(args.pred_dir, model)
        versions = find_prompt_versions(model_dir, args.prompt_version)
        if not versions:
            logger.warning("no prompt-version directories under %s", model_dir)
            continue
        for version in versions:
            version_dir = os.path.join(model_dir, version)
            widths = find_widths(version_dir, args.context_turns)
            if not widths:
                logger.warning("no excerpt-width directories under %s", version_dir)
                continue
            for width in widths:
                for split in args.splits:
                    path = os.path.join(version_dir, width, f"{split}.jsonl")
                    if not os.path.exists(path):
                        logger.warning("no predictions at %s; skipping", path)
                        continue
                    predictions = read_predictions(path)
                    if split in ground_truths:
                        stale = stale_moments(predictions, ground_truths[split])
                        if stale:
                            logger.warning(
                                "%d moment(s) in %s carry gold labels that differ "
                                "from the current ground truth (e.g. %s); re-run "
                                "classify_excerpts to score against today's gold",
                                len(stale),
                                path,
                                ", ".join(stale[:3]),
                            )
                    rows = evaluate(predictions, agreement[split], args.short_turns)
                    results.append(
                        {
                            "model": model,
                            "prompt_version": version,
                            "context_turns": width,
                            "split": split,
                            "tasks": rows,
                        }
                    )
                    print(report(model, version, width, split, rows))

    if not results:
        logger.error("no prediction files found under %s", args.pred_dir)
        return 1

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
            fh.write("\n")
        logger.info("wrote %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
