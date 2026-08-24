"""Score v2 predictions against the gold labels: precision, recall, F1.

``classify_excerpts.py`` writes predictions to
``<pred-dir>/<model>/<split>.jsonl``, each record carrying the gold ``labels``
it was made against, so this script needs nothing but those files.

Three binary tasks are scored, positive class = True:

  scaffolding        action_direction.scaffolding      vs labels.scaffolding_present
  rigor              action_direction.rigor            vs labels.rigor_present
  over-scaffolding   over_scaffolding.over_scaffolding vs labels.over_scaffolding_present

Over-scaffolding is scored only on the moments it was asked about (the gold gate
in ``classify_excerpts``): elsewhere there is no prediction, not a "no".

A moment with no usable prediction -- never classified, or a response that would
not parse -- is left out of the counts and reported separately. Scoring an
unparsed response as "no" would credit a parse failure as a correct negative on
the mostly-negative tasks, which reads as a better model rather than a broken
one.

Usage::

    python -m tutormoments_build.v2.evaluate
    python -m tutormoments_build.v2.evaluate --model claude-opus-5
    python -m tutormoments_build.v2.evaluate --splits iteration test
"""

import argparse
import json
import logging
import os
import sys

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.v2.classify_excerpts import DEFAULT_OUT_DIR, DEFAULT_SPLITS

logger = logging.getLogger("tutormoments_build.v2.evaluate")

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


def collect(records: list[dict], pass_name: str, field: str, gold_field: str) -> tuple:
    """Return ([(gold, predicted), ...], n_missing) for one task.

    A record is missing when the pass did not run for it or its response did not
    parse -- either way there is no label to score.
    """
    pairs, missing = [], 0
    for record in records:
        gold = (record.get("labels") or {}).get(gold_field)
        predicted = (record.get(pass_name) or {}).get(field)
        if gold is None or predicted is None:
            missing += 1
            continue
        pairs.append((bool(gold), bool(predicted)))
    return pairs, missing


# ===========================================================================
# Reading
# ===========================================================================


def read_predictions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def find_models(pred_dir: str, model: str | None) -> list[str]:
    """Model subdirectory names to evaluate: the one asked for, or all present."""
    if model:
        return [model.replace("/", "_")]
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(
            f"no predictions at {pred_dir}; run "
            "`python -m tutormoments_build.v2.classify_excerpts` first"
        )
    return sorted(
        name
        for name in os.listdir(pred_dir)
        if os.path.isdir(os.path.join(pred_dir, name))
    )


def evaluate(records: list[dict]) -> list[dict]:
    """Score every task over one split's prediction records."""
    rows = []
    for name, pass_name, field, gold_field in TASKS:
        pairs, missing = collect(records, pass_name, field, gold_field)
        rows.append({"task": name, "missing": missing, **score(pairs)})
    return rows


# ===========================================================================
# Reporting
# ===========================================================================


def report(model: str, split: str, rows: list[dict]) -> str:
    header = (
        f"  {'task':<18}{'n':>6}{'gold yes':>10}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
        f"{'prec':>8}{'rec':>7}{'F1':>7}{'not scored':>13}"
    )
    lines = ["", f"  {model}  --  {split}", "", header, f"  {'-' * (len(header) - 2)}"]
    for row in rows:
        lines.append(
            f"  {row['task']:<18}{row['n']:>6}{row['support']:>10}"
            f"{row['tp']:>5}{row['fp']:>5}{row['fn']:>5}{row['tn']:>5}"
            f"{row['precision']:>8.2f}{row['recall']:>7.2f}{row['f1']:>7.2f}"
            f"{row['missing']:>13}"
        )
    lines += [
        "",
        "  positive class = True; 'not scored' = no prediction (not asked, "
        "not classified, or unparsed)",
        "",
    ]
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.evaluate",
        description="Score v2 predictions against the gold labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--pred-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model whose predictions to score. Default: every model found "
        "under --pred-dir.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Which prediction splits to score",
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

    results = []
    for model in find_models(args.pred_dir, args.model):
        for split in args.splits:
            path = os.path.join(args.pred_dir, model, f"{split}.jsonl")
            if not os.path.exists(path):
                logger.warning("no predictions at %s; skipping", path)
                continue
            rows = evaluate(read_predictions(path))
            results.append({"model": model, "split": split, "tasks": rows})
            print(report(model, split, rows))

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
