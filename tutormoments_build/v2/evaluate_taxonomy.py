"""Compare ``taxonomy.py``'s predicted action labels with the adjudicators' gold.

Reports, per model / excerpt width / split, and alongside it how well the first
and second pass annotators agree with each other on the same moments:

  jaccard    mean per-moment |A & B| / |A | B| (1.0 where both sets are empty)
  micro      precision, recall and F1 over the pooled label decisions of
             every moment
  labels     mean labels per moment on each side
  confusion  a multi-label confusion matrix (``confusion``)

The metrics are reported over all compared moments, with both families' labels
together, and then per family (``family_sets``): ``scaffolding_present`` over
the moments where scaffolding is compared, with both sides cut to scaffolding
strategies, and ``rigor_present`` likewise. A moment comparing both families is
in both rows. There is one confusion matrix per family, from those same sets.
For the model, the compared families are the ones the gold
``scaffolding_present`` / ``rigor_present`` labels mark, the same ones the
prompt was told.

**Short moments.** The model's predictions are also scored over only the moments
with ``--short-turns`` (5 by default) or fewer dialogue turns after the cut, as
``evaluate_scorer`` does: a short post-cut span gives less to label from, so this
checks whether errors cluster there. The count is the record's
``post_cut_dialogue_rows`` (dialogue turns only, screen activity not counted); a
record without it cannot be placed and is counted and left out of the block.

Gold is read from the annotation exports (``DEFAULT_ANNOTATIONS_FILES``), using
the adjudicators only. Each adjudicator's ``final.action`` carries
``scaffolding_strategies`` and ``rigor_strategies`` lists; a moment's gold set is
the union over its adjudicators. Staff and superseded re-saves are dropped as in
the ground-truth build, and an adjudicator who threw the moment out or left the
action section blank contributes nothing. A moment no adjudicator answered has
no gold.

Gold names are mapped onto the prompt's (``GOLD_ALIASES``): ``re_explain`` and
``re_explain_procedure`` collapse into ``re_explain``,
``different_representation`` is the prompt's ``diff_rep``, and gold
``simpler_language`` is scored as ``re_explain``. Only gold is remapped, so a
predicted ``simpler_language`` never matches. Free-text "other" strategies have
no category and are carried alongside, not compared. The history of these
mappings is in ``prompts/v2_helper/classify_action/README.md``.

A family's strategies count as gold only where the moment's gold label marks
that family present (``FAMILY_LABELS``): scaffolding strategies when
``scaffolding_present``, rigor strategies when ``rigor_present``. Those labels
are a majority over every annotator, while the strategies come from the
adjudicators alone, so the two can disagree; the labels are also what the
prompt was told, so they decide which families are compared.

Predicted sets keep only labels the prompt lists; any others are returned
separately so they can be counted without counting against a category.

**First vs second pass.** The selector (first pass) and the reannotator (second
pass) fill in the same strategy lists. A family is compared on a moment only
where both of them marked it present; families they disagree on, or both left
absent, are left out, and a moment with no family left is not compared. The
moments are the split's ground-truth moments (``--ground-truth-dir``), so the
passes are read on the same population the model is. Free-text "other" is left
out here too.

Usage::

    python -m tutormoments_build.v2.evaluate_taxonomy
    python -m tutormoments_build.v2.evaluate_taxonomy --model claude-opus-5
    python -m tutormoments_build.v2.evaluate_taxonomy --splits iteration test
    python -m tutormoments_build.v2.evaluate_taxonomy --json scores.json
    python -m tutormoments_build.v2.evaluate_taxonomy --prompt-version 1
    python -m tutormoments_build.v2.evaluate_taxonomy --short-turns 3

Predictions are read from ``<pred-dir>/<model>/<prompt-version>/<width>/``, and
each version directory is scored with that version's own label vocabulary. By
default every version present is scored, side by side; ``--prompt-version``
picks one. The first/second pass comparison uses the newest prompt's vocabulary.
"""

import argparse
import json
import logging
import os
import statistics
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
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
from tutormoments_build.v2.classify_excerpts import DEFAULT_SPLITS
from tutormoments_build.v2.evaluate_scorer import (
    SHORT_POST_CUT_TURNS,
    find_prompt_versions,
    find_widths,
    read_ground_truth,
    short_records,
)
from tutormoments_build.v2.splits import DEFAULT_ANNOTATIONS_FILES
from tutormoments_build.v2.taxonomy import (
    ACTION_WORDING,
    DEFAULT_OUT_DIR,
    PROMPT_ROOT,
    load_template,
    parse_categories,
)

logger = logging.getLogger("tutormoments_build.v2.evaluate_taxonomy")

DEFAULT_GROUND_TRUTH_DIR = "data/ground_truth"

# Gold strategy name -> the prompt's category name. Gold only: predictions are
# not remapped. Keep prompts/v2_helper/classify_action/README.md in sync.
GOLD_ALIASES = {
    "re_explain_procedure": "re_explain",
    "different_representation": "diff_rep",
    # Annotators file "simpler alternative" moves under re_explain.
    "simpler_language": "re_explain",
}

# (gold list key, free-text key, family) in an adjudicator's final.action.
GOLD_STRATEGY_FIELDS = (
    ("scaffolding_strategies", "scaffolding_strategy_other", "scaffolding"),
    ("rigor_strategies", "rigor_strategy_other", "rigor"),
)

# Strategy family -> the gold label that marks it present.
FAMILY_LABELS = {
    "scaffolding": "scaffolding_present",
    "rigor": "rigor_present",
}

# Why a moment drops out of the gold/prediction pairing.
NO_GOLD = "no_answered_adjudicator"
NOT_ASKED = "not_asked"
UNPARSED = "prediction_unparsed"

# The two passes compared with each other.
FIRST_PASS = "selector"
SECOND_PASS = "reannotator"

# Why a moment drops out of the first/second pass pairing.
MISSING_PASS = "missing_pass"
PASS_ABSTAINED = "pass_abstained"
NO_AGREED_FAMILY = "no_family_both_marked_present"

# The confusion matrix's row / column for "no label to pair with".
NONE = "(none)"

ALL = "all"


# ===========================================================================
# Gold
# ===========================================================================


def gold_label(name: str) -> str:
    """A gold strategy name in the prompt's vocabulary."""
    name = name.strip().lower()
    return GOLD_ALIASES.get(name, name)


def annotation_actions(annotation: dict) -> dict | None:
    """One annotator's action labels, or None if they answered nothing.

    Any role: ``answered_payload`` reads an adjudicator's ``final``. Returns
    ``{"actions": set, "by_family": {family: set}, "present": {family: bool},
    "other": {family: text}}``, with names mapped through ``GOLD_ALIASES``.
    """
    payload = answered_payload(annotation)
    if payload is None:
        return None
    action = payload["action"]
    by_family: dict[str, set[str]] = {}
    other: dict[str, str] = {}
    for list_key, other_key, family in GOLD_STRATEGY_FIELDS:
        by_family[family] = {gold_label(name) for name in action.get(list_key) or []}
        text = str(action.get(other_key) or "").strip()
        if text:
            other[family] = text
    actions = set().union(*by_family.values())
    present = {
        family: bool(action.get(label)) for family, label in FAMILY_LABELS.items()
    }
    return {
        "actions": actions,
        "by_family": by_family,
        "present": present,
        "other": other,
    }


def moment_gold(annotations: list[dict], labels: dict[str, str]) -> dict | None:
    """The union of a moment's answered adjudicators' labels, or None if none.

    ``annotations`` must already be staff-filtered and re-save-collapsed
    (``latest_annotations``).
    """
    answered = []
    for annotation in adjudications(annotations):
        found = annotation_actions(annotation)
        if found is not None:
            answered.append((annotator_label(annotation, labels), found))
    if not answered:
        return None
    return {
        # Every family, before the gold labels select which ones count.
        "actions": frozenset().union(*(found["actions"] for _, found in answered)),
        "by_family": {
            family: frozenset().union(
                *(found["by_family"][family] for _, found in answered)
            )
            for _, _, family in GOLD_STRATEGY_FIELDS
        },
        "adjudicators": [label for label, _ in answered],
        # Per adjudicator, so a disagreement between the two stays readable.
        "per_adjudicator": {
            label: sorted(found["actions"]) for label, found in answered
        },
        "other": {label: found["other"] for label, found in answered if found["other"]},
    }


def load_annotations(
    annotations_paths=DEFAULT_ANNOTATIONS_FILES,
    excluded_path: str = DEFAULT_EXCLUDED_ANNOTATORS,
) -> dict[str, list[dict]]:
    """Return {moment_id: annotations}, staff dropped and re-saves collapsed."""
    excluded = load_excluded_annotators(excluded_path)
    return {
        row["moment"]["moment_id"]: latest_annotations(
            row.get("annotations") or [], excluded
        )
        for row in read_rows(list(annotations_paths))
        if "moment" in row
    }


def gold_by_moment(
    annotations: dict[str, list[dict]], labels: dict[str, str]
) -> dict[str, dict]:
    """Return {moment_id: gold record} for every moment an adjudicator answered."""
    out: dict[str, dict] = {}
    for moment_id, moment_annotations in annotations.items():
        gold = moment_gold(moment_annotations, labels)
        if gold is not None:
            out[moment_id] = gold
    return out


def load_gold(
    annotations_paths=DEFAULT_ANNOTATIONS_FILES,
    excluded_path: str = DEFAULT_EXCLUDED_ANNOTATORS,
    labels_path: str = DEFAULT_ANNOTATOR_LABELS,
) -> dict[str, dict]:
    """Return {moment_id: gold record} for every moment an adjudicator answered."""
    return gold_by_moment(
        load_annotations(annotations_paths, excluded_path),
        load_annotator_labels(labels_path),
    )


# ===========================================================================
# Predictions
# ===========================================================================


def prediction_actions(record: dict) -> dict | None:
    """One taxonomy.py record's predicted labels, or None if there are none to use.

    None covers a moment that was not asked and a response that did not parse;
    ``record["skipped"]`` / ``record["taxonomy"]["parse_error"]`` say which.
    Labels the prompt does not list (family None) go to ``unknown``.
    """
    taxonomy = record.get("taxonomy")
    if not taxonomy or taxonomy.get("actions") is None:
        return None
    known = {a["category"] for a in taxonomy["actions"] if a.get("family")}
    unknown = {a["category"] for a in taxonomy["actions"] if not a.get("family")}
    return {"actions": frozenset(known), "unknown": frozenset(unknown)}


def load_predictions(path: str) -> dict[str, dict]:
    """Return {moment_id: taxonomy.py record} from one predictions JSONL."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no predictions at {path}; run `python -m tutormoments_build.v2.taxonomy`"
        )
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                out[record["moment_id"]] = record
    return out


# ===========================================================================
# Pairing
# ===========================================================================


def present_gold(moment: dict, labels: dict) -> frozenset:
    """The union of the families' strategies whose gold label is present."""
    return frozenset().union(
        *(
            moment["by_family"][family]
            for family, label in FAMILY_LABELS.items()
            if labels.get(label)
        )
    )


def pair(
    gold: dict[str, dict], predictions: dict[str, dict]
) -> tuple[list[dict], Counter]:
    """Join gold and predicted label sets per predicted moment.

    Returns (pairs, dropped). Each pair has ``moment_id``, ``split``, the gold
    ``action`` wording the prompt was told, and ``gold`` / ``pred`` frozensets
    (plus ``pred_unknown``). ``gold`` holds only the families the record's gold
    labels mark present (``present_gold``). Moments that cannot be compared are counted in
    ``dropped`` by reason rather than paired with an empty set, which would
    read as "no actions".
    """
    pairs: list[dict] = []
    dropped: Counter = Counter()
    for moment_id, record in predictions.items():
        if not record.get("asked"):
            dropped[NOT_ASKED] += 1
            continue
        pred = prediction_actions(record)
        if pred is None:
            dropped[UNPARSED] += 1
            continue
        moment = gold.get(moment_id)
        if moment is None:
            dropped[NO_GOLD] += 1
            continue
        pairs.append(
            {
                "moment_id": moment_id,
                "split": record.get("split"),
                "action": record.get("action"),
                "gold": present_gold(moment, record.get("labels") or {}),
                "pred": pred["actions"],
                "pred_unknown": pred["unknown"],
                "n_adjudicators": len(moment["adjudicators"]),
            }
        )
    return pairs, dropped


# ===========================================================================
# First vs second pass
# ===========================================================================


def _role(annotations: list[dict], role: str) -> dict | None:
    for annotation in annotations:
        if annotation.get("role") == role:
            return annotation
    return None


def pass_pairs(
    annotations: dict[str, list[dict]], moment_ids
) -> tuple[list[dict], Counter]:
    """Join the first and second pass's strategies on each of ``moment_ids``.

    Returns (pairs, dropped). Each pair has ``moment_id``, ``action`` (the
    ``ACTION_WORDING`` of the families compared), and ``first`` / ``second``
    frozensets holding only the families both passes marked present.
    ``dropped`` counts moments by reason, plus ``disagreed_<family>`` for each
    family left out because the passes disagreed on whether it was present.
    """
    pairs: list[dict] = []
    dropped: Counter = Counter()
    for moment_id in moment_ids:
        moment_annotations = annotations.get(moment_id) or []
        first = _role(moment_annotations, FIRST_PASS)
        second = _role(moment_annotations, SECOND_PASS)
        if first is None or second is None:
            dropped[MISSING_PASS] += 1
            continue
        first, second = annotation_actions(first), annotation_actions(second)
        if first is None or second is None:
            dropped[PASS_ABSTAINED] += 1
            continue
        families = []
        for family in FAMILY_LABELS:
            if first["present"][family] and second["present"][family]:
                families.append(family)
            elif first["present"][family] != second["present"][family]:
                dropped[f"disagreed_{family}"] += 1
        if not families:
            dropped[NO_AGREED_FAMILY] += 1
            continue
        pairs.append(
            {
                "moment_id": moment_id,
                "action": ACTION_WORDING[
                    ("scaffolding" in families, "rigor" in families)
                ],
                "first": frozenset().union(*(first["by_family"][f] for f in families)),
                "second": frozenset().union(
                    *(second["by_family"][f] for f in families)
                ),
            }
        )
    return pairs, dropped


# ===========================================================================
# Metrics
# ===========================================================================


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    """F1, or None where neither side used the label."""
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else None


def jaccard(a: frozenset, b: frozenset) -> float:
    """|a & b| / |a | b|; two empty sets agree perfectly."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def confusion(sets: list[tuple[frozenset, frozenset]]) -> dict[str, Counter]:
    """{reference label: Counter(compared label)} over (reference, compared) sets.

    A label on both sides is a hit on the diagonal. Of the rest, every reference
    label the compared side missed is paired with every label the compared side
    added in the same moment; a miss with nothing added goes to the ``NONE``
    column, an addition with nothing missed to the ``NONE`` row. So a row's
    off-diagonal cells say what the compared side said instead.
    """
    matrix: dict[str, Counter] = {}
    for reference, compared in sets:
        for label in reference & compared:
            matrix.setdefault(label, Counter())[label] += 1
        missed, added = reference - compared, compared - reference
        for row in missed or {NONE}:
            for column in added or {NONE}:
                if row != NONE or column != NONE:
                    matrix.setdefault(row, Counter())[column] += 1
    return matrix


def set_metrics(sets: list[tuple[frozenset, frozenset]]) -> dict:
    """Jaccard, micro precision/recall/F1 and label counts over (reference, compared)
    label sets."""
    per_label: dict[str, Counter] = {}
    for reference, compared in sets:
        for label in reference | compared:
            counts = per_label.setdefault(label, Counter())
            counts[
                "tp"
                if label in reference and label in compared
                else "fn"
                if label in reference
                else "fp"
            ] += 1
    totals = Counter()
    for counts in per_label.values():
        totals.update(counts)
    label_scores = {
        label: {
            "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
            "recall": _ratio(c["tp"], c["tp"] + c["fn"]),
            "f1": _f1(c["tp"], c["fp"], c["fn"]),
        }
        for label, c in sorted(per_label.items())
    }

    return {
        "n": len(sets),
        "jaccard": statistics.fmean(jaccard(r, c) for r, c in sets) if sets else None,
        "micro_precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
        "micro_recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
        "micro_f1": _f1(totals["tp"], totals["fp"], totals["fn"]),
        "mean_reference_labels": (
            statistics.fmean(len(r) for r, _ in sets) if sets else None
        ),
        "mean_compared_labels": (
            statistics.fmean(len(c) for _, c in sets) if sets else None
        ),
        "per_label": {
            label: {**dict(per_label[label]), **scores}
            for label, scores in label_scores.items()
        },
    }


def families_compared(action: str) -> list[str]:
    """The families an ``ACTION_WORDING`` value names, e.g. "scaffold" -> scaffolding."""
    flags = next(key for key, wording in ACTION_WORDING.items() if wording == action)
    return [family for family, flag in zip(FAMILY_LABELS, flags) if flag]


def family_sets(
    pairs: list[dict],
    family: str,
    reference: str,
    compared: str,
    categories: dict[str, str],
) -> list[tuple[frozenset, frozenset]]:
    """(reference, compared) sets cut to ``family``'s labels, over the pairs
    that compare that family.

    Cutting both sides means a miss in one family is never paired with an
    addition in the other. A compared label from a family the moment does not
    compare is in no family's sets (``off_family_labels``).
    """
    return [
        tuple(
            frozenset(label for label in p[side] if categories.get(label) == family)
            for side in (reference, compared)
        )
        for p in pairs
        if family in families_compared(p["action"])
    ]


def off_family_labels(
    pairs: list[dict], compared: str, categories: dict[str, str]
) -> Counter:
    """Compared labels from a family the moment does not compare, by label.

    For the model: e.g. a rigor strategy on a moment the prompt was told was
    scaffolding only. They count against the all-moments row, but are in
    neither family's row or confusion matrix.
    """
    out: Counter = Counter()
    for p in pairs:
        families = families_compared(p["action"])
        out.update(l for l in p[compared] if categories.get(l) not in families)
    return out


def grouped_metrics(
    pairs: list[dict], reference: str, compared: str, categories: dict[str, str]
) -> dict:
    """``set_metrics`` over every pair, then per family (keyed by its gold
    label, e.g. ``scaffolding_present``) with that family's confusion matrix."""
    groups = {ALL: set_metrics([(p[reference], p[compared]) for p in pairs])}
    for family, label in FAMILY_LABELS.items():
        sets = family_sets(pairs, family, reference, compared, categories)
        groups[label] = {
            **set_metrics(sets),
            "family": family,
            "confusion": {
                row: dict(columns) for row, columns in confusion(sets).items()
            },
        }
    return groups


# ===========================================================================
# Reporting
# ===========================================================================


def _fmt(value: float | None, width: int) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.2f}"


_PRF = ("precision", "recall", "f1")


def _metrics_table(groups: dict, reference: str, compared: str) -> list[str]:
    """Precision and recall read ``compared`` against ``reference``."""
    header = (
        f"  {'':<22}{'n':>6}{'jaccard':>9}"
        + "".join(f"{h:>11}" for h in ("micro-P", "micro-R", "micro-F1"))
        + f"{reference + '/moment':>16}{compared + '/moment':>16}"
    )
    lines = [header, f"  {'-' * (len(header) - 2)}"]
    for name, m in groups.items():
        lines.append(
            f"  {name:<22}{m['n']:>6}{_fmt(m['jaccard'], 9)}"
            + "".join(_fmt(m[f"micro_{k}"], 11) for k in _PRF)
            + f"{_fmt(m['mean_reference_labels'], 16)}"
            f"{_fmt(m['mean_compared_labels'], 16)}"
        )
    lines.append(f"  P and R read {compared} against {reference}")
    return lines


def _confusion_table(
    matrix: dict, vocabulary: list[str], reference: str, compared: str
) -> list[str]:
    """Rows ``reference``, columns ``compared``, numbered by the legend."""
    used = set(matrix) | {c for row in matrix.values() for c in row}
    labels = [label for label in vocabulary if label in used]
    labels += sorted(used - set(labels) - {NONE})
    labels.append(NONE)
    width = max(4, len(str(len(labels))) + 2)
    lines = [
        f"  rows: {reference}; columns: {compared}; '.' = 0",
        f"  {'':<30}" + "".join(f"{i:>{width}}" for i in range(1, len(labels) + 1)),
    ]
    for i, row in enumerate(labels, 1):
        cells = matrix.get(row) or {}
        lines.append(
            f"  {f'{i:>2} {row}':<30}"
            + "".join(f"{cells.get(c, 0) or '.':>{width}}" for c in labels)
        )
    return lines


def _dropped_line(dropped: Counter) -> str:
    return ",  ".join(f"{reason} {n}" for reason, n in sorted(dropped.items()))


def _block(
    title: str,
    groups: dict,
    dropped: Counter,
    vocabulary: list[str],
    reference: str,
    compared: str,
) -> list[str]:
    lines = ["", title, ""]
    lines += _metrics_table(groups, reference, compared)
    if dropped:
        lines.append(f"  not compared:  {_dropped_line(dropped)}")
    for name, m in groups.items():
        if "confusion" not in m or not m["confusion"]:
            continue
        lines += ["", f"  {m['family']} confusion matrix -- {name} moments"]
        lines += _confusion_table(m["confusion"], vocabulary, reference, compared)
    return lines


def report_model(result: dict, vocabulary: list[str]) -> str:
    title = (
        f"  {result['model']}  --  prompt v{result['prompt_version']}  --  "
        f"@{result['context_turns']} excerpt  --  "
        f"{result['split']}  --  model vs adjudicator gold"
    )
    lines = _block(
        title,
        result["groups"],
        Counter(result["dropped"]),
        vocabulary,
        "gold",
        "model",
    )
    if result["off_family_labels"]:
        lines.append(
            "  labels from a family the gold marks absent (in the all row "
            "only):  "
            + ",  ".join(f"{k} {n}" for k, n in result["off_family_labels"].items())
        )
    if result["unknown_labels"]:
        lines.append(
            "  labels not in the prompt, left out:  "
            + ",  ".join(f"{k} {n}" for k, n in result["unknown_labels"].items())
        )

    short = result["short"]
    lines += ["", f"  {short['turns']} or fewer turns after the cut", ""]
    lines += _metrics_table(short["groups"], "gold", "model")
    if short["dropped"]:
        lines.append(f"  not compared:  {_dropped_line(Counter(short['dropped']))}")
    lines.append(
        f"  {short['moments']} of {short['of_moments']} moment(s); turns are "
        "dialogue turns after the cut, screen activity not counted"
    )
    if short["unknown"]:
        lines.append(
            f"  {short['unknown']} moment(s) have no post-cut turn count "
            "and are left out of this block"
        )
    lines.append("")
    return "\n".join(lines)


def report_passes(result: dict, vocabulary: list[str]) -> str:
    title = (
        f"  {result['split']}  --  first pass ({FIRST_PASS}) vs second pass "
        f"({SECOND_PASS}), families both marked present"
    )
    lines = _block(
        title,
        result["groups"],
        Counter(result["dropped"]),
        vocabulary,
        "first",
        "second",
    )
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# Evaluation
# ===========================================================================


def evaluate_predictions(
    gold: dict[str, dict],
    predictions: dict[str, dict],
    categories: dict[str, str],
    short_turns: int = SHORT_POST_CUT_TURNS,
) -> dict:
    pairs, dropped = pair(gold, predictions)
    unknown = Counter(label for p in pairs for label in p["pred_unknown"])
    short, short_unknown = short_records(list(predictions.values()), short_turns)
    short_pairs, short_dropped = pair(gold, {r["moment_id"]: r for r in short})
    return {
        "short": {
            "turns": short_turns,
            "moments": len(short),
            "of_moments": len(predictions) - short_unknown,
            "unknown": short_unknown,
            "groups": grouped_metrics(short_pairs, "gold", "pred", categories),
            "dropped": dict(short_dropped),
        },
        "groups": grouped_metrics(pairs, "gold", "pred", categories),
        "dropped": dict(dropped),
        "off_family_labels": dict(
            off_family_labels(pairs, "pred", categories).most_common()
        ),
        "unknown_labels": dict(unknown.most_common()),
    }


def evaluate_passes(
    annotations: dict[str, list[dict]], moment_ids, categories: dict[str, str]
) -> dict:
    pairs, dropped = pass_pairs(annotations, moment_ids)
    return {
        "groups": grouped_metrics(pairs, "first", "second", categories),
        "dropped": dict(dropped),
    }


# ===========================================================================
# CLI
# ===========================================================================


def find_models(pred_dir: str, model: str | None) -> list[str]:
    """Model subdirectory names to evaluate: the one asked for, or all present."""
    if model:
        return [model.replace("/", "_")]
    if not os.path.isdir(pred_dir):
        return []
    return sorted(
        name
        for name in os.listdir(pred_dir)
        if os.path.isdir(os.path.join(pred_dir, name))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.evaluate_taxonomy",
        description="Score taxonomy.py's action labels against the adjudicators, "
        "and the first pass against the second",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--pred-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--ground-truth-dir",
        default=DEFAULT_GROUND_TRUTH_DIR,
        metavar="DIR",
        help="Ground truth whose moments the first/second pass comparison reads.",
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=list(DEFAULT_ANNOTATIONS_FILES),
        metavar="FILE",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model whose predictions to score. Default: every model under --pred-dir.",
    )
    parser.add_argument(
        "--context-turns",
        default=None,
        metavar="WIDTH",
        help="Excerpt width to score (`full` or a turn count). Default: every "
        "width found.",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        metavar="VERSION",
        help="Prompt version whose predictions to score, each against its own "
        f"prompt under tutormoments_build/{PROMPT_ROOT}/. Default: every version "
        "present.",
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--short-turns",
        type=int,
        default=SHORT_POST_CUT_TURNS,
        metavar="N",
        help="Also score the moments with N or fewer dialogue turns after the cut",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        metavar="FILE",
        help="Also write every number (per-label counts and F1 included) as JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    # The passes have no prompt version; score them in the newest one's terms.
    pass_categories = parse_categories(load_template())
    annotations = load_annotations(args.annotations)
    gold = gold_by_moment(annotations, load_annotator_labels(DEFAULT_ANNOTATOR_LABELS))

    results = []
    for split in args.splits:
        ground_truth = read_ground_truth(args.ground_truth_dir, split)
        if ground_truth is not None:
            passes = {
                "kind": "passes",
                "split": split,
                **evaluate_passes(
                    annotations,
                    [r["moment_id"] for r in ground_truth],
                    pass_categories,
                ),
            }
            results.append(passes)
            print(report_passes(passes, list(pass_categories)))

        for model in find_models(args.pred_dir, args.model):
            model_dir = os.path.join(args.pred_dir, model)
            for version in find_prompt_versions(model_dir, args.prompt_version):
                version_dir = os.path.join(model_dir, version)
                categories = parse_categories(load_template(version))
                for width in find_widths(version_dir, args.context_turns):
                    path = os.path.join(version_dir, width, f"{split}.jsonl")
                    if not os.path.exists(path):
                        logger.warning("no predictions at %s; skipping", path)
                        continue
                    result = {
                        "kind": "model",
                        "model": model,
                        "prompt_version": version,
                        "context_turns": width,
                        "split": split,
                        **evaluate_predictions(
                            gold, load_predictions(path), categories, args.short_turns
                        ),
                    }
                    results.append(result)
                    print(report_model(result, list(categories)))

    if not any(r["kind"] == "model" for r in results):
        logger.error(
            "no prediction files found under %s; run "
            "`python -m tutormoments_build.v2.taxonomy` first",
            args.pred_dir,
        )
        return 1

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
            fh.write("\n")
        logger.info("wrote %s", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
