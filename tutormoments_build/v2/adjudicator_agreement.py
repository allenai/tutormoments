"""How well adjudicators agree with each other and with the annotators.

Reports four things:

- Cohen's kappa between adjudicators, on the moments two adjudicators each
  reviewed. Scored on the five ground-truth labels plus ``throw_out``. A moment
  someone threw out has no labels, so the count differs per label. One table
  pools everyone; then one table per adjudicator compares them against whoever
  they were paired with. Each moment shows up in two of those per-person tables,
  so don't average them.
- How often the adjudicator keeps a label both annotators agreed on, and, when
  the annotators disagreed, how often the adjudicator says yes.
- How often the adjudicator agrees with the reannotator's throw-out decision.
- How much the adjudicator changes the annotators' written explanations.

Usage:
    python -m tutormoments_build.v2.adjudicator_agreement
    python -m tutormoments_build.v2.adjudicator_agreement --verbose
    python -m tutormoments_build.v2.adjudicator_agreement --json-out agreement.json

Each annotation file (by default, providers C and A) is reported separately,
never pooled. Without ``--verbose`` only the pooled-adjudicator kappa table is
printed per file; ``--json-out`` always carries every number, keyed by file.
"""

import argparse
import difflib
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.v2.build_ground_truth import (
    DEFAULT_ANNOTATOR_LABELS,
    DEFAULT_EXCLUDED_ANNOTATORS,
    RETRACTED,
    annotator_label,
    latest_annotations,
    load_annotator_labels,
    load_excluded_annotators,
    redrew_cut_point,
)

logger = logging.getLogger("tutormoments_build.v2.adjudicator_agreement")

DEFAULT_ANNOTATIONS = (
    "data/v2_annotations/source/tutoring_provider_c_annotations.jsonl",
    "data/v2_annotations/source/tutoring_provider_a_annotations.jsonl",
)

# (label, section, key) for the four yes/no questions.
BOOLEAN_FIELDS = (
    ("scaffolding_appropriate", "situation", "scaffolding_appropriate"),
    ("rigor_appropriate", "situation", "rigor_appropriate"),
    ("scaffolding_present", "action", "scaffolding_present"),
    ("rigor_present", "action", "rigor_present"),
)
# The five labels an adjudicator gives when they keep the moment.
FINAL_FIELDS = tuple(name for name, _, _ in BOOLEAN_FIELDS) + (
    "over_scaffolding_declared",
)
# throw_out first: if a moment is thrown out, the other labels are blank.
LABEL_FIELDS = ("throw_out",) + FINAL_FIELDS

OVER_SCAFFOLDING = "over_scaffolding"

# Written-explanation boxes both annotators and adjudicators fill in, as
# (section, key). The adjudicator-only ``rationale`` box is reported separately.
TEXT_BOXES = (
    ("situation", "why"),
    ("action", "explanation"),
    ("result", "explanation"),
    (None, "other_observations"),
)

# Texts less similar than this count as rewritten rather than edited. The cutoff
# is a judgement call, so character counts are reported too.
REWRITE_SIMILARITY = 0.5

EDIT_KINDS = ("verbatim", "edited", "rewritten", "added", "cleared", "blank")


def payload_labels(payload: dict) -> dict[str, bool]:
    """The five labels from one set of answers (annotator or adjudicator)."""
    labels = {
        name: bool((payload.get(section) or {}).get(key))
        for name, section, key in BOOLEAN_FIELDS
    }
    labels["over_scaffolding_declared"] = (payload.get("action") or {}).get(
        "scaffolding_amount"
    ) == OVER_SCAFFOLDING
    return labels


def threw_out(payload: dict) -> bool:
    """Whether this person threw the moment out. Selectors are never asked."""
    return bool((payload.get("meta") or {}).get("throw_out"))


def adjudicator_labels(annotation: dict) -> dict[str, bool | None]:
    """One adjudicator's ``throw_out`` answer plus the five labels.

    If they threw the moment out, the labels are None rather than False, so a
    throw-out isn't counted as "no" to every label.
    """
    payload = annotation["payload"]
    labels = {"throw_out": threw_out(payload)}

    final = payload.get("final")
    if not final:
        return labels | {field: None for field in FINAL_FIELDS}

    return labels | payload_labels(final)


def moment_records(path: str, excluded: frozenset[str]):
    """Yield moment records, skipping the same moments the ground-truth build does.

    Skips "no key moments" transcripts, retracted moments, and moments where
    someone moved the cut point (the annotators then judged different excerpts).
    Staff are dropped, and if someone saved twice only their latest save is kept.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if "moment" not in record:  # a "no key moments" transcript record
                continue
            if record["moment"].get("status") == RETRACTED:
                continue
            annotations = latest_annotations(record["annotations"], excluded)
            if any(redrew_cut_point(a) for a in annotations):
                continue
            yield record | {"annotations": annotations}


def latest_by_role(annotations: list[dict], role: str) -> dict | None:
    """The annotation for ``role``, or None. If two people have the same role,
    warns and uses the first."""
    candidates = [a for a in annotations if a.get("role") == role]
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            "%d annotators share the %s role on one moment; using the first",
            len(candidates),
            role,
        )
    return candidates[0]


def doubly_adjudicated(
    path: str, labels: dict[str, str], excluded: frozenset[str]
) -> list[dict]:
    """One row per moment with exactly two adjudicators.

    Each row is {"moment_id", "raters": (label, label), "labels": [dict, dict]}.
    The two adjudicators are sorted by label so the same person is always
    rater 1 for a given pair, which kappa needs.
    """
    rows = []
    for record in moment_records(path, excluded):
        adjudications = [
            a for a in record["annotations"] if a.get("role") == "adjudicator"
        ]
        if len(adjudications) < 2:
            continue
        if len(adjudications) > 2:
            logger.warning(
                "moment %s has %d adjudications; skipping (Cohen's kappa is "
                "a two-rater statistic)",
                record["moment"]["moment_id"],
                len(adjudications),
            )
            continue
        adjudications.sort(key=lambda a: annotator_label(a, labels))
        rows.append(
            {
                "moment_id": record["moment"]["moment_id"],
                "raters": tuple(annotator_label(a, labels) for a in adjudications),
                "labels": [adjudicator_labels(a) for a in adjudications],
            }
        )
    return rows


def by_adjudicator(rows: list[dict]) -> dict[str, list[dict]]:
    """For each adjudicator, their shared moments with them as rater 1.

    Rater 2 is whoever else reviewed each moment, so it can be different people.
    Every moment appears under both of its adjudicators, so don't average the
    results.
    """
    oriented = defaultdict(list)
    for row in rows:
        for index, label in enumerate(row["raters"]):
            other = row["raters"][1 - index]
            oriented[label].append(
                row
                | {
                    "raters": (label, other),
                    "labels": [row["labels"][index], row["labels"][1 - index]],
                }
            )
    return dict(sorted(oriented.items()))


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa over (rater A, rater B) pairs. None if there are no pairs
    or everyone gave the same answer every time."""
    n = len(pairs)
    if n == 0:
        return None

    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    categories = first.keys() | second.keys()
    expected = sum((first[c] / n) * (second[c] / n) for c in categories)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def score(rows: list[dict]) -> dict[str, dict]:
    """Per label: kappa, percent agreement and yes counts, using only moments
    where both adjudicators answered."""
    results = {}
    for field in LABEL_FIELDS:
        pairs = [
            (row["labels"][0][field], row["labels"][1][field])
            for row in rows
            if row["labels"][0][field] is not None
            and row["labels"][1][field] is not None
        ]
        results[field] = {
            "n": len(pairs),
            "kappa": cohens_kappa(pairs),
            "observed_agreement": (
                sum(a == b for a, b in pairs) / len(pairs) if pairs else None
            ),
            "positives": [sum(a for a, _ in pairs), sum(b for _, b in pairs)],
        }
    return results


def report(title: str, results: dict[str, dict]) -> str:
    """A row per label. ``pos 1``/``pos 2`` count each rater's yes answers."""
    lines = [
        title,
        f"{'label':<28}{'n':>4}{'kappa':>9}{'agree':>8}{'pos 1':>7}{'pos 2':>7}",
    ]
    for field, row in results.items():
        kappa = "n/a" if row["kappa"] is None else f"{row['kappa']:.3f}"
        agreement = (
            "n/a"
            if row["observed_agreement"] is None
            else f"{row['observed_agreement']:.1%}"
        )
        lines.append(
            f"{field:<28}{row['n']:>4}{kappa:>9}{agreement:>8}"
            f"{row['positives'][0]:>7}{row['positives'][1]:>7}"
        )
    return "\n".join(lines)


# ===========================================================================
# How the adjudicator settled the two annotators
# ===========================================================================


def adjudications(
    path: str, labels: dict[str, str], excluded: frozenset[str]
) -> list[dict]:
    """One row per adjudication, with both annotators' answers.

    ``final`` is None if the adjudicator threw the moment out. Moments missing
    an annotator are skipped with a warning.
    """
    rows = []
    for record in moment_records(path, excluded):
        annotations = record["annotations"]
        adjudicators = [a for a in annotations if a.get("role") == "adjudicator"]
        if not adjudicators:
            continue

        selector = latest_by_role(annotations, "selector")
        reannotator = latest_by_role(annotations, "reannotator")
        if selector is None or reannotator is None:
            logger.warning(
                "moment %s was adjudicated but has no %s; skipping",
                record["moment"]["moment_id"],
                "selector" if selector is None else "reannotator",
            )
            continue

        for adjudicator in adjudicators:
            payload = adjudicator["payload"]
            rows.append(
                {
                    "moment_id": record["moment"]["moment_id"],
                    "adjudicator": annotator_label(adjudicator, labels),
                    "selector": selector["payload"],
                    "reannotator": reannotator["payload"],
                    "final": payload.get("final") or None,
                    "rationale": (payload.get("rationale") or "").strip(),
                }
            )
    return rows


def resolution_counts(rows: list[dict]) -> dict[str, dict]:
    """Per label: how often the adjudicator went along with the annotators.

    When both annotators gave the same answer, ``upheld`` counts the adjudicator
    giving it too, broken out by yes and no. When they disagreed, ``picked``
    counts the adjudicator saying yes. Skips moments the adjudicator or the
    reannotator threw out.
    """
    results = {
        field: dict.fromkeys(
            (
                "n",
                "agreed",
                "upheld",
                "agreed_yes",
                "upheld_yes",
                "agreed_no",
                "upheld_no",
                "disagreed",
                "picked",
                "with_selector",
                "with_reannotator",
            ),
            0,
        )
        for field in FINAL_FIELDS
    }

    for row in rows:
        if not row["final"]:
            continue
        # The reannotator threw it out, so they gave no labels to compare.
        if threw_out(row["reannotator"]):
            continue
        selector = payload_labels(row["selector"])
        reannotator = payload_labels(row["reannotator"])
        final = payload_labels(row["final"])

        for field in FINAL_FIELDS:
            counts = results[field]
            counts["n"] += 1
            if selector[field] == reannotator[field]:
                side = "yes" if selector[field] else "no"
                upheld = final[field] == selector[field]
                counts["agreed"] += 1
                counts["upheld"] += upheld
                counts[f"agreed_{side}"] += 1
                counts[f"upheld_{side}"] += upheld
            else:
                counts["disagreed"] += 1
                counts["picked"] += final[field]
                counts["with_selector"] += final[field] == selector[field]
                counts["with_reannotator"] += final[field] == reannotator[field]
    return results


def _count_pct(count: int, total: int) -> str:
    return "-" if not total else f"{count} ({count / total:.0%})"


def _share(count: int, total: int) -> str:
    """``count/total (pct)``."""
    return "-" if not total else f"{count}/{total} ({count / total:.0%})"


def report_resolution(title: str, results: dict[str, dict]) -> str:
    """A row per label. Percentages are out of the agreed or disagreed count, not ``n``."""
    lines = [
        title,
        f"{'':<28}{'':>4}{'---- the two annotators agreed ----':^42}"
        f"{'--- they split ---':^24}",
        f"{'label':<28}{'n':>4}{'upheld':>12}{'both said yes':>15}"
        f"{'both said no':>15}{'n':>10}{'picked':>14}",
    ]
    for field, row in results.items():
        lines.append(
            f"{field:<28}{row['n']:>4}"
            f"{_share(row['upheld'], row['agreed']):>12}"
            f"{_share(row['upheld_yes'], row['agreed_yes']):>15}"
            f"{_share(row['upheld_no'], row['agreed_no']):>15}"
            f"{row['disagreed']:>10}"
            f"{_count_pct(row['picked'], row['disagreed']):>14}"
        )
    return "\n".join(lines)


def throw_out_resolution(rows: list[dict]) -> dict:
    """How the adjudicator's throw-out decision compares to the reannotator's.

    Only the reannotator is compared, since selectors are never asked.
    """
    pairs = [(threw_out(row["reannotator"]), row["final"] is None) for row in rows]
    thrown = [adjudicator for reannotator, adjudicator in pairs if reannotator]
    kept = [adjudicator for reannotator, adjudicator in pairs if not reannotator]

    return {
        "n": len(pairs),
        "kappa": cohens_kappa(pairs),
        "observed_agreement": (
            sum(a == b for a, b in pairs) / len(pairs) if pairs else None
        ),
        "reannotator_threw": len(thrown),
        "upheld": sum(thrown),
        "reinstated": len(thrown) - sum(thrown),
        "reannotator_kept": len(kept),
        "kept_standing": len(kept) - sum(kept),
        "thrown_anyway": sum(kept),
    }


def report_throw_out(title: str, results: dict) -> str:
    """The throw-out summary as text."""
    kappa = "n/a" if results["kappa"] is None else f"{results['kappa']:.3f}"
    agreement = (
        "n/a"
        if results["observed_agreement"] is None
        else f"{results['observed_agreement']:.1%}"
    )
    return "\n".join(
        [
            title,
            f"  the second pass threw it out    {results['reannotator_threw']:>4}  "
            f"adjudicator agreed {_share(results['upheld'], results['reannotator_threw'])}"
            f", reinstated {results['reinstated']}",
            f"  the second pass kept it         {results['reannotator_kept']:>4}  "
            f"adjudicator agreed "
            f"{_share(results['kept_standing'], results['reannotator_kept'])}"
            f", threw it out {results['thrown_anyway']}",
            f"  kappa {kappa}, raw agreement {agreement} over "
            f"{results['n']} adjudications",
        ]
    )


# ===========================================================================
# How much the written explanations were edited
# ===========================================================================


def box_text(payload: dict, section: str | None, key: str) -> str:
    """The text in one box."""
    source = (payload.get(section) or {}) if section else payload
    return source.get(key) or ""


def diff_stats(base: str, final: str) -> dict:
    """Similarity, change in length, and number of characters changed."""
    # autojunk=False: otherwise difflib ignores common letters in long text.
    matcher = difflib.SequenceMatcher(None, base, final, autojunk=False)
    changed = sum(
        (i2 - i1) + (j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    return {
        "similarity": matcher.ratio(),
        "net": len(final) - len(base),
        "changed": changed,
    }


def classify_edit(final: str, sources: dict[str, str]) -> dict:
    """Label what the adjudicator did to one box (one of ``EDIT_KINDS``).

    We don't know which annotator's text the adjudicator started from, so we
    compare against whichever is closest.
    """
    # Ignore leading/trailing whitespace.
    final = final.strip()
    written = {name: text.strip() for name, text in sources.items() if text.strip()}
    if not final:
        return {"kind": "cleared" if written else "blank", "base": None}
    if not written:
        return {
            "kind": "added",
            "base": None,
            "similarity": 0.0,
            "net": len(final),
            "changed": len(final),
        }

    base = max(written, key=lambda name: diff_stats(written[name], final)["similarity"])
    stats = diff_stats(written[base], final)
    if written[base] == final:
        kind = "verbatim"
    elif stats["similarity"] >= REWRITE_SIMILARITY:
        kind = "edited"
    else:
        kind = "rewritten"
    return {"kind": kind, "base": base, **stats}


def _median(values) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def text_edits(rows: list[dict]) -> dict[str, dict]:
    """Per box: how often each kind of edit happened, and median size of the
    change where the adjudicator started from an annotator's text."""
    results = {}
    for section, key in TEXT_BOXES:
        name = f"{section}.{key}" if section else key
        kinds: Counter = Counter()
        bases: Counter = Counter()
        stats: list[dict] = []

        for row in rows:
            if not row["final"]:
                continue
            edit = classify_edit(
                box_text(row["final"], section, key),
                {
                    "selector": box_text(row["selector"], section, key),
                    "reannotator": box_text(row["reannotator"], section, key),
                },
            )
            kinds[edit["kind"]] += 1
            if edit["base"] is not None:
                bases[edit["base"]] += 1
                stats.append(edit)

        results[name] = {
            "n": sum(kinds.values()),
            "kinds": {kind: kinds[kind] for kind in EDIT_KINDS},
            "base": {
                "selector": bases["selector"],
                "reannotator": bases["reannotator"],
            },
            "median_net_chars": _median(s["net"] for s in stats),
            "median_changed_chars": _median(s["changed"] for s in stats),
            "median_similarity": _median(s["similarity"] for s in stats),
        }
    return results


def _number(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def report_text_edits(title: str, results: dict[str, dict]) -> tuple[str, str]:
    """Two tables: kinds of edit per box, then how much the text changed."""
    kinds = [
        title,
        f"{'text box':<24}{'n':>4}" + "".join(f"{kind:>12}" for kind in EDIT_KINDS),
    ]
    distances = [
        "Distance from the text the adjudicator started with (verbatim, edited "
        "and rewritten boxes only)",
        f"{'text box':<24}{'n':>4}{'from sel':>12}{'from rea':>12}"
        f"{'net chars':>11}{'chars changed':>15}{'similarity':>12}",
    ]

    for name, row in results.items():
        kinds.append(
            f"{name:<24}{row['n']:>4}"
            + "".join(
                f"{_count_pct(row['kinds'][kind], row['n']):>12}" for kind in EDIT_KINDS
            )
        )
        based = row["base"]["selector"] + row["base"]["reannotator"]
        distances.append(
            f"{name:<24}{based:>4}"
            f"{_count_pct(row['base']['selector'], based):>12}"
            f"{_count_pct(row['base']['reannotator'], based):>12}"
            f"{_number(row['median_net_chars'], '+.0f'):>11}"
            f"{_number(row['median_changed_chars'], '.0f'):>15}"
            f"{_number(row['median_similarity'], '.2f'):>12}"
        )
    return "\n".join(kinds), "\n".join(distances)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        default=list(DEFAULT_ANNOTATIONS),
        help="one or more annotation files; each is reported separately",
    )
    parser.add_argument(
        "--annotator-labels",
        default=DEFAULT_ANNOTATOR_LABELS,
        help="map of annotator_id -> de-identified label, so the output carries "
        "labels rather than ids",
    )
    parser.add_argument(
        "--excluded-annotators",
        default=DEFAULT_EXCLUDED_ANNOTATORS,
        help="JSON list of project staff annotator_ids to drop",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every table, not just the pooled adjudicator kappa table",
    )
    parser.add_argument(
        "--json-out",
        help="also write the numbers to this path, keyed by annotation file",
    )
    return parser


def _moments(n: int) -> str:
    return f"{n} moment" if n == 1 else f"{n} moments"


def report_file(
    path: str, labels: dict[str, str], excluded: frozenset[str], verbose: bool
) -> dict | None:
    """Print the tables for one annotation file and return its numbers, or None
    if it has no adjudicated moments. Without ``verbose`` only the pooled kappa
    table is printed; the returned numbers are the same either way."""
    rows = doubly_adjudicated(path, labels, excluded)
    all_adjudications = adjudications(path, labels, excluded)
    if not rows and not all_adjudications:
        logger.error("no adjudicated moments in %s", path)
        return None

    overall, per_adjudicator = {}, {}
    if rows:
        thrown_out = sum(any(a["throw_out"] for a in row["labels"]) for row in rows)
        logger.info(
            "%d doubly adjudicated moment(s) in %s; %d had at least one adjudicator "
            "throw the moment out, so only the throw_out row is scored on all %d",
            len(rows),
            path,
            thrown_out,
            len(rows),
        )

        overall = score(rows)
        print(report(f"All doubly adjudicated moments ({_moments(len(rows))})", overall))

        for name, own_rows in by_adjudicator(rows).items():
            per_adjudicator[name] = score(own_rows)
            if not verbose:
                continue
            partners = Counter(row["raters"][1] for row in own_rows)
            print()
            print(
                report(
                    f"{name} vs every other adjudicator "
                    f"({_moments(len(own_rows))})\n  paired with: "
                    + ", ".join(
                        f"{partner} {count}"
                        for partner, count in sorted(partners.items())
                    ),
                    per_adjudicator[name],
                )
            )
    else:
        logger.warning(
            "no doubly adjudicated moments in %s; skipping the kappa tables", path
        )

    finalised = [row for row in all_adjudications if row["final"]]
    # Skip ones the reannotator threw out, since they gave no labels.
    comparable = [row for row in finalised if not threw_out(row["reannotator"])]
    resolution = resolution_counts(all_adjudications)
    throw_out_rates = throw_out_resolution(all_adjudications)
    edits = text_edits(all_adjudications)
    adjudicated_moments = _moments(len({row["moment_id"] for row in all_adjudications}))

    logger.info(
        "%d adjudication(s) over %s; %d threw the moment out and answered nothing. "
        "The label table is scored on the %d final calls whose reannotator left "
        "labels to compare (%d more reinstated a moment the reannotator threw out, "
        "so there are none)",
        len(all_adjudications),
        adjudicated_moments,
        len(all_adjudications) - len(finalised),
        len(comparable),
        len(finalised) - len(comparable),
    )

    if verbose:
        print()
        print(
            report_resolution(
                f"How often the adjudicator backed the first and second pass "
                f"({len(comparable)} final calls over {adjudicated_moments})",
                resolution,
            )
        )

        print()
        print(
            report_throw_out(
                "How the adjudicator answered the second pass's throw-out call "
                "(the selector is never asked)",
                throw_out_rates,
            )
        )

        kinds_table, distance_table = report_text_edits(
            f"What the adjudicator did with the annotators' free text "
            f"({len(finalised)} final calls)",
            edits,
        )
        print()
        print(kinds_table)
        print()
        print(distance_table)

        written = [row["rationale"] for row in finalised if row["rationale"]]
        print()
        print(
            "Adjudicator's own rationale note (no annotator draft to compare "
            f"against): {_count_pct(len(written), len(finalised))} of final calls "
            "carry one, median "
            f"{_number(_median(len(text) for text in written), '.0f')} characters."
        )

    return {
        "n_moments": len(rows),
        "overall": overall,
        "by_adjudicator": per_adjudicator,
        "n_adjudications": len(all_adjudications),
        "n_final_calls": len(finalised),
        "n_comparable_final_calls": len(comparable),
        "resolution": resolution,
        "throw_out_resolution": throw_out_rates,
        "text_edits": edits,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    labels = load_annotator_labels(args.annotator_labels)
    excluded = load_excluded_annotators(args.excluded_annotators)

    # Each file is reported on its own; numbers are never pooled across files.
    results, failed = {}, False
    for i, path in enumerate(args.annotations):
        if i:
            print("\n")
        print(f"=== {path} ===\n")
        numbers = report_file(path, labels, excluded, args.verbose)
        if numbers is None:
            failed = True
        else:
            results[path] = numbers

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
            fh.write("\n")
        logger.info("wrote %s", args.json_out)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
