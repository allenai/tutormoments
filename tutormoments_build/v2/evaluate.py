"""Score v2 predictions against the gold labels: precision, recall, F1.

``classify_excerpts.py`` writes predictions to
``<pred-dir>/<model>/<prompt version>/<split>.jsonl``, each record carrying the
gold ``labels`` it was made against, so those files are all that scoring needs.

Every task is reported three times:

  all moments        every moment with a usable prediction
  both passes agreed only the moments where the first-pass annotator and the
                     reannotator gave the *same* value for that construct
  short moments      only the moments with fewer than ``--short-turns`` (10 by
                     default) dialogue turns after the cut point

The gold labels are a union across the two passes, so a moment either pass
called True is labelled True. The agreed subset drops those single-vote moments
and leaves the ones both teachers saw the same way -- a cleaner target, and the
gap between the two numbers is how much of the model's error sits on moments the
humans themselves split on. The subset is per construct: a moment can be agreed
for rigor and contested for scaffolding, and it is scored in the agreed block of
the one and not the other.

**Read the two blocks as answering different questions, not as one number
improving on another.** Under a union, a gold *negative* is a moment neither
pass called True, so it is agreed by construction: the subset can only ever drop
contested *positives*. So the agreed block keeps every false positive and true
negative the full block had -- those columns are identical -- while shedding the
TPs and FNs from single-vote moments. Recall therefore rises and precision falls
mechanically, and the interesting quantity is how much. Recall on the agreed
moments is the model's hit rate on the constructs both teachers actually saw;
precision there says how much of what the model flags lands outside anything the
teachers jointly endorsed.

The short-moment block is a different cut of the same predictions: how the
model does when there is little tutoring after the cut to read. It is a slice on
the excerpt, not on the labels, so unlike the agreed subset it drops positives
and negatives alike and its precision and recall both move freely. The turn count
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

A moment with no usable prediction -- never classified, or a response that would
not parse -- is left out of the counts and reported separately. Scoring an
unparsed response as "no" would credit a parse failure as a correct negative on
the mostly-negative tasks, which reads as a better model rather than a broken
one.

Usage::

    python -m tutormoments_build.v2.evaluate
    python -m tutormoments_build.v2.evaluate --model claude-opus-5
    python -m tutormoments_build.v2.evaluate --prompt-version 1
    python -m tutormoments_build.v2.evaluate --splits iteration test
    python -m tutormoments_build.v2.evaluate --short-turns 5
    python -m tutormoments_build.v2.evaluate --ground-truth-dir data/ground_truth

With no ``--prompt-version`` every version found under a model is scored, so a
bare run reports each revision of the prompts against the same gold.
"""

import argparse
import json
import logging
import os
import sys

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.v2.build_ground_truth import (
    DEFAULT_OUT_DIR as DEFAULT_GROUND_TRUTH_DIR,
)
from tutormoments_build.v2.classify_excerpts import DEFAULT_OUT_DIR, DEFAULT_SPLITS

logger = logging.getLogger("tutormoments_build.v2.evaluate")

# Moments with fewer than this many dialogue turns after the cut are reported
# again as their own block: a short post-cut span is less evidence to classify
# from, and the question is whether the model's errors concentrate there.
SHORT_POST_CUT_TURNS = 10

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


def agreed_records(
    records: list[dict], agreement: dict[str, dict], gold_field: str
) -> tuple[list[dict], int]:
    """Records where both annotation passes gave the same value for one construct.

    Returns (kept, n_unknown); unknown is a moment the ground truth has no
    agreement flags for, which is a moment that has left the ground truth since
    it was classified rather than a contested one -- it belongs in neither
    subset, so it is counted and dropped.

    For over-scaffolding the flag is agreement on the *declared* amount, the
    only over-scaffolding question annotators answer. That is the right flag
    here: the moments whose gold over-scaffolding is inferred from the rule
    rather than declared are exactly the ones the classifier is never asked
    about (both gate on scaffolding_appropriate), so none of them reach these
    pairs to be mislabelled agreed.
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
    """Records with fewer than ``max_turns`` dialogue turns after the cut.

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
        if turns < max_turns:
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


def agreement_by_moment(ground_truth: list[dict]) -> dict[str, dict]:
    """{moment_id: agreement flags} -- per construct, did both passes match."""
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


def evaluate(
    records: list[dict],
    agreement: dict[str, dict] | None = None,
    short_turns: int = SHORT_POST_CUT_TURNS,
) -> list[dict]:
    """Score every task over one split's prediction records.

    Each row carries three subsets: ``all``, every moment with a usable
    prediction; ``agreed``, only the moments where both annotation passes gave
    that construct the same value, and None when no ground truth was found to
    join against; and ``short``, only the moments with fewer than
    ``short_turns`` dialogue turns after the cut point.
    """
    short, short_unknown = short_records(records, short_turns)

    rows = []
    for name, pass_name, field, gold_field in TASKS:
        pairs, missing = collect(records, pass_name, field, gold_field)
        row = {"task": name, "all": {"missing": missing, **score(pairs)}}

        if agreement is None:
            row["agreed"] = None
        else:
            kept, unknown = agreed_records(records, agreement, gold_field)
            pairs, missing = collect(kept, pass_name, field, gold_field)
            row["agreed"] = {
                "missing": missing,
                "unknown": unknown,
                "contested": len(records) - len(kept) - unknown,
                **score(pairs),
            }

        pairs, missing = collect(short, pass_name, field, gold_field)
        row["short"] = {
            "missing": missing,
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
        f"{scores['missing']:>13}"
    )


def report(model: str, prompt_version: str, split: str, rows: list[dict]) -> str:
    """One block per subset: all moments, the ones both passes agreed on, and
    the ones with a short post-cut span."""
    header = (
        f"  {'task':<18}{'n':>6}{'gold yes':>10}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
        f"{'prec':>8}{'rec':>7}{'F1':>7}{'not scored':>13}"
    )
    lines = [
        "",
        f"  {model}  --  prompts v{prompt_version}  --  {split}",
        "",
        header,
        f"  {'-' * (len(header) - 2)}",
        "  all moments",
    ]
    lines += [_score_line(row["task"], row["all"]) for row in rows]

    scored = [row for row in rows if row["agreed"] is not None]
    if scored:
        lines += ["", "  both annotation passes agreed on the construct"]
        lines += [_score_line(row["task"], row["agreed"]) for row in scored]
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
        lines += ["", "  no ground truth found; the agreement subset was skipped"]

    short = rows[0]["short"]
    lines += ["", f"  fewer than {short['turns']} turns after the cut"]
    lines += [_score_line(row["task"], row["short"]) for row in rows]
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
        "  positive class = True; 'not scored' = no prediction (not asked, "
        "not classified, or unparsed)",
    ]
    if scored:
        lines.append(
            "  gold is a union of the two passes, so only contested positives "
            "leave the agreed block:"
        )
        lines.append(
            "  FP/TN carry over unchanged, and recall rises where precision "
            "falls by construction"
        )
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
        prog="python -m tutormoments_build.v2.evaluate",
        description="Score v2 predictions against the gold labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--pred-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--ground-truth-dir",
        default=DEFAULT_GROUND_TRUTH_DIR,
        metavar="DIR",
        help="Where to read the annotator-agreement flags from, joined on "
        "moment_id. Without it only the all-moments numbers are reported.",
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
        help="Report a separate block over the moments with fewer than N "
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

    results = []
    for model in find_models(args.pred_dir, args.model):
        model_dir = os.path.join(args.pred_dir, model)
        versions = find_prompt_versions(model_dir, args.prompt_version)
        if not versions:
            logger.warning("no prompt-version directories under %s", model_dir)
            continue
        for version in versions:
            for split in args.splits:
                path = os.path.join(model_dir, version, f"{split}.jsonl")
                if not os.path.exists(path):
                    logger.warning("no predictions at %s; skipping", path)
                    continue
                predictions = read_predictions(path)
                if split in ground_truths:
                    stale = stale_moments(predictions, ground_truths[split])
                    if stale:
                        logger.warning(
                            "%d moment(s) in %s carry gold labels that differ from "
                            "the current ground truth (e.g. %s); re-run "
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
                        "split": split,
                        "tasks": rows,
                    }
                )
                print(report(model, version, split, rows))

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
