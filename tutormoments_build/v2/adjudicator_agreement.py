"""Cohen's kappa between adjudicators on the moments two of them adjudicated.

Most moments are seen by a selector and a reannotator. A small number go to
adjudication, and a smaller number still are adjudicated *twice* -- two
adjudicators independently issuing a final call on the same moment. Those doubly
adjudicated moments are the only place in the export where two annotators
answered the same questions in the same role, so they are what agreement is
measured on here.

The five labels are the ones ``build_ground_truth`` resolves, read from the
adjudicator's ``payload["final"]``:

    situation.scaffolding_appropriate    scaffolding was called for here
    situation.rigor_appropriate          a push for rigor was called for here
    action.scaffolding_present           the tutor scaffolded
    action.rigor_present                 the tutor pushed for rigor
    action.scaffolding_amount == over_scaffolding
                                         the tutor scaffolded too much

The last one is the adjudicator's literal amount choice. The *inferred*
over-scaffolding case is derived from resolved labels rather than answered by
anyone, so there is nothing per-annotator to compare and it is not scored.

A sixth row, ``throw_out``, is the adjudicator's call that the moment does not
belong in the benchmark at all. It is scored separately because it is the
decision that gates the rest: an adjudicator who throws a moment out leaves
``final: null`` and answers no label questions, so that moment drops out of the
other five rows rather than being read as five negatives. ``n`` is therefore per
row -- every moment reaches ``throw_out``, fewer reach the labels.

Cohen's kappa is a two-rater statistic, but the pairs vary across moments (three
distinct adjudicators appear). The headline number pools every doubly adjudicated
moment and treats "the two adjudicators on this moment" as the two raters, which
is the usual pooled reading; the per-pair block below it shows the same
calculation restricted to one pair at a time. Read the per-pair kappas with their
n in view -- a pair with a handful of moments is not an estimate of anything.

Two further sections look at the adjudicator against the pair they were called
in to settle, and so are scored on *every* adjudication, not just the doubly
adjudicated ones:

**How often the adjudicator backs the pair below them.** For each label: where
the first and second pass agreed, how often the adjudicator let that stand --
split by whether they agreed the label was there or agreed it was not, since a
pooled rate on a lopsided label mostly reports how often everyone said no. And
where the two passes split, how often the adjudicator picked the label rather
than leaving it off. One side of a split said True by definition, so that
"picked" rate is also the rate at which the adjudicator lands where the union
rule ``build_ground_truth`` resolves the shipped labels does.

**How much the free-text boxes are edited.** The adjudicator's final call carries
the same prose boxes the annotators filled in, and adjudicating one is a matter
of keeping a text, editing it, or writing a new one. Each box is compared against
whichever of the two source texts it is closest to -- the export does not record
which one the interface put in front of them -- and reported as verbatim, edited,
rewritten or added, with the character distance behind that call.

Usage:
    python -m tutormoments_build.v2.adjudicator_agreement
    python -m tutormoments_build.v2.adjudicator_agreement --json-out agreement.json
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
    annotator_label,
    is_excluded,
    load_annotator_labels,
)

logger = logging.getLogger("tutormoments_build.v2.adjudicator_agreement")

DEFAULT_ANNOTATIONS = "data/v2_annotations/source/tutoring_provider_a_annotations.jsonl"

# (label, payload section, payload key) -- the four booleans adjudicators answer.
BOOLEAN_FIELDS = (
    ("scaffolding_appropriate", "situation", "scaffolding_appropriate"),
    ("rigor_appropriate", "situation", "rigor_appropriate"),
    ("scaffolding_present", "action", "scaffolding_present"),
    ("rigor_present", "action", "rigor_present"),
)
# The five labels that only exist inside a final call.
FINAL_FIELDS = tuple(name for name, _, _ in BOOLEAN_FIELDS) + (
    "over_scaffolding_declared",
)
# throw_out comes first: it is the decision that determines whether the rest of
# the labels exist at all.
LABEL_FIELDS = ("throw_out",) + FINAL_FIELDS

OVER_SCAFFOLDING = "over_scaffolding"

# The free-text boxes an adjudicator inherits, as (payload section, key). These
# are the annotators' own prose: the adjudicator's final call carries the same
# four, filled with whatever they decided to keep, edit or replace. The
# adjudicator-only ``rationale`` note is not here -- nobody wrote a first draft
# of it, so there is nothing to measure it against; it is summarised separately.
TEXT_BOXES = (
    ("situation", "why"),
    ("action", "explanation"),
    ("result", "explanation"),
    (None, "other_observations"),
)

# Similarity below which a text is called rewritten rather than edited. A
# threshold is a reading of a continuum, not a fact about the data, so the raw
# character distances are reported alongside the buckets: half the characters
# surviving is a generous floor for "this is still the same paragraph".
REWRITE_SIMILARITY = 0.5

EDIT_KINDS = ("verbatim", "edited", "rewritten", "added", "cleared", "blank")


def payload_labels(payload: dict) -> dict[str, bool]:
    """The five labels as answered inside one ``situation``/``action`` payload.

    Selectors and reannotators answer these at the top level of their payload;
    an adjudicator answers the identical questions inside ``payload["final"]``,
    so both sides read with this one function and are directly comparable.
    """
    labels = {
        name: bool((payload.get(section) or {}).get(key))
        for name, section, key in BOOLEAN_FIELDS
    }
    labels["over_scaffolding_declared"] = (payload.get("action") or {}).get(
        "scaffolding_amount"
    ) == OVER_SCAFFOLDING
    return labels


def adjudicator_labels(annotation: dict) -> dict[str, bool | None]:
    """Pull one adjudicator's decisions out of their adjudication.

    ``throw_out`` -- the adjudicator's call that the moment does not belong in
    the benchmark at all -- is always present. The other five are None when they
    threw the moment out, because an adjudication with ``final: null`` answered
    no label questions.

    **The None matters.** Reading a null final as all-False would score "this
    moment should not be in the benchmark" as "the tutor did not scaffold, and
    no scaffolding was called for", turning a colleague who declined to label
    into one who labelled everything negative -- and manufacturing disagreement
    with whoever did label the moment.
    """
    payload = annotation["payload"]
    labels = {"throw_out": bool((payload.get("meta") or {}).get("throw_out"))}

    final = payload.get("final")
    if not final:
        return labels | {field: None for field in FINAL_FIELDS}

    return labels | payload_labels(final)


def moment_records(path: str):
    """Yield the export's moment records, skipping "no key moments" transcripts.

    Staff annotations are dropped here so every reader downstream sees the same
    annotator set the ground-truth build does.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if "moment" not in record:  # a "no key moments" transcript record
                continue
            yield record | {
                "annotations": [a for a in record["annotations"] if not is_excluded(a)]
            }


def latest_by_role(annotations: list[dict], role: str) -> dict | None:
    """The one annotation for ``role``, or None.

    A few moments carry the same person's annotation twice, saved seconds apart
    under successive ``revision`` numbers -- a re-save, not a second opinion. The
    highest revision is the one they meant, and the earlier draft must not be
    read as a second annotator disagreeing with them.
    """
    candidates = [a for a in annotations if a.get("role") == role]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.get("revision", 0))


def doubly_adjudicated(path: str, labels: dict[str, str]) -> list[dict]:
    """Return one row per moment that exactly two adjudicators finalised.

    Each row is {"moment_id", "raters": (label, label), "labels": [dict, dict]},
    with the two adjudicators sorted by label so that "rater 1" and "rater 2" mean
    the same person on every moment a given pair shares -- otherwise the marginals
    kappa expects chance agreement from would mix the two raters together.

    Moments with one adjudicator have nothing to compare; the export contains no
    moment with three, and any that appeared would be skipped and logged.
    """
    rows = []
    for record in moment_records(path):
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


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for a list of (rater A value, rater B value).

    Returns None where kappa is undefined: no pairs, or chance agreement of 1
    because every rating on both sides fell in a single category (with nothing
    to disagree about, the statistic has no denominator).
    """
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
    """Kappa, raw agreement and positive counts for each label over ``rows``.

    A moment counts towards a label only where *both* adjudicators answered it,
    so ``n`` is per label: every moment reaches ``throw_out``, while the other
    five are scored on the moments neither adjudicator threw out.
    """
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
    """One table: a row per label, kappa alongside the raw numbers behind it.

    ``pos 1``/``pos 2`` are how many moments each rater called True -- the
    marginals kappa corrects for, and the reason a label both raters almost never
    call True can hold high agreement and a low kappa.
    """
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
# How the adjudicator resolved the pair below them
# ===========================================================================


def adjudications(path: str, labels: dict[str, str]) -> list[dict]:
    """One row per adjudication, alongside the two annotations it was settling.

    Each row is {"moment_id", "adjudicator", "selector", "reannotator", "final",
    "rationale"} -- the last four raw payloads (``final`` is None where the
    adjudicator threw the moment out). Every adjudication counts, including both
    halves of a doubly adjudicated moment: each is one person's independent
    resolution of the same split, which is exactly the unit being counted.

    A moment missing a selector or a reannotator has no pair to resolve and is
    skipped with a warning.
    """
    rows = []
    for record in moment_records(path):
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
    """Per label: how often the adjudicator backed the two annotators below them.

    Two questions, one for each thing the pair below can do:

    **They agreed.** ``agreed`` counts those adjudications and ``upheld`` how
    many the adjudicator left standing. Split by what was agreed, because a
    pooled uphold rate on a lopsided label is mostly a report of how often
    everyone said no: ``agreed_yes``/``upheld_yes`` are the adjudications where
    both annotators marked the label, ``agreed_no``/``upheld_no`` where neither
    did. An overturn -- the difference between ``agreed`` and ``upheld`` -- is a
    call no rule over the two annotators could have produced, since both of them
    said the same thing.

    **They split.** ``disagreed`` counts those, and ``picked`` how often the
    adjudicator marked the label rather than leaving it off. One side of a split
    said True by definition, so ``picked`` is also the rate at which the
    adjudicator lands where the union rule ``build_ground_truth`` resolves the
    shipped labels does.

    ``with_selector``/``with_reannotator`` record which pass the adjudicator's
    call matched on a split. They are carried in the JSON rather than the printed
    tables: on a two-value label they say the same thing as ``picked`` about
    whose read prevailed, only keyed by role instead of by answer.

    Thrown-out adjudications answer no label questions and are not counted.
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
    """``count/total (pct)`` -- the rate with the numbers it rests on in view."""
    return "-" if not total else f"{count}/{total} ({count / total:.0%})"


def report_resolution(title: str, results: dict[str, dict]) -> str:
    """One row per label: how often the adjudicator backed the pair below them.

    Every percentage is of the group in its own half of the table, never of
    ``n``: the uphold rates are of the adjudications where the two annotators
    agreed (and of the yes/no halves of that), ``picked`` is of the ones where
    they split.
    """
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


# ===========================================================================
# How much the free-text boxes were edited
# ===========================================================================


def box_text(payload: dict, section: str | None, key: str) -> str:
    """One free-text box out of a payload."""
    source = (payload.get(section) or {}) if section else payload
    return source.get(key) or ""


def diff_stats(base: str, final: str) -> dict:
    """How far ``final`` sits from ``base``, in characters.

    ``net`` is the change in length -- positive where the adjudicator added more
    than they cut -- and ``changed`` is the characters actually touched (deleted
    plus inserted), which distinguishes a rewrite of the same length from a text
    left alone. ``similarity`` is difflib's ratio over characters.

    ``autojunk`` is off: it treats characters appearing in more than 1% of a
    sequence over 200 elements as junk, which for prose is most of the alphabet,
    and would collapse the similarity of texts that are nearly identical.
    """
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
    """What the adjudicator did to one box, against the nearest source text.

    The export does not record which annotator's text the interface put in the
    box, so the base is inferred: whichever source the final text is closest to.
    Where the two annotators wrote the same thing the attribution is arbitrary
    and the first source wins -- read ``base`` as "whose text this most resembles",
    not as a record of what was on screen.

    The kinds:

        verbatim   kept one annotator's text character for character
        edited     started from one and changed part of it
        rewritten  shares less than REWRITE_SIMILARITY with either -- a new text
        added      wrote something where neither annotator wrote anything
        cleared    left the box empty though an annotator had filled it
        blank      nobody filled it, adjudicator included
    """
    # Stripped: surrounding whitespace is what a text box collects on its own,
    # not something an adjudicator decided, and comparing it would report edits
    # nobody made.
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
    """Per free-text box: what the adjudicator did to it, and by how many characters.

    Counted over the adjudications that reached a final call -- a thrown-out
    moment has no boxes to compare.

    The character medians cover only the boxes that started from a source text
    (verbatim, edited, rewritten), so they answer "how far did the adjudicator
    move the text they were given". A box written from nothing has no distance
    to report and would otherwise enter the median as a whole text's worth of
    change.
    """
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
    """Two tables: what happened to each box, then how far the kept texts moved.

    They are split because the second only describes the boxes with a source
    text behind them, and putting both in one row would invite reading the
    character medians as covering every adjudication of that box.
    """
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
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--annotator-labels",
        default=DEFAULT_ANNOTATOR_LABELS,
        help="map of annotator name -> de-identified label, so real names stay "
        "out of the output",
    )
    parser.add_argument("--json-out", help="also write the numbers to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    labels = load_annotator_labels(args.annotator_labels)
    rows = doubly_adjudicated(args.annotations, labels)
    all_adjudications = adjudications(args.annotations, labels)
    if not rows and not all_adjudications:
        logger.error("no adjudicated moments in %s", args.annotations)
        return 1

    def moments(n: int) -> str:
        return f"{n} moment" if n == 1 else f"{n} moments"

    overall, pairs = {}, {}
    if rows:
        thrown_out = sum(any(a["throw_out"] for a in row["labels"]) for row in rows)
        logger.info(
            "%d doubly adjudicated moment(s) in %s; %d had at least one adjudicator "
            "throw the moment out, so only the throw_out row is scored on all %d",
            len(rows),
            args.annotations,
            thrown_out,
            len(rows),
        )

        overall = score(rows)
        print(report(f"All doubly adjudicated moments ({moments(len(rows))})", overall))

        by_pair = defaultdict(list)
        for row in rows:
            by_pair[row["raters"]].append(row)

        for pair, pair_rows in sorted(by_pair.items()):
            name = "+".join(pair)
            pairs[name] = score(pair_rows)
            print()
            print(
                report(
                    f"{pair[0]} vs {pair[1]} ({len(pair_rows)} moments)", pairs[name]
                )
            )
    else:
        logger.warning(
            "no doubly adjudicated moments in %s; skipping the kappa tables",
            args.annotations,
        )

    finalised = [row for row in all_adjudications if row["final"]]
    resolution = resolution_counts(all_adjudications)
    edits = text_edits(all_adjudications)
    adjudicated_moments = moments(len({row["moment_id"] for row in all_adjudications}))

    logger.info(
        "%d adjudication(s) over %s; %d threw the moment out and answered nothing, "
        "so the sections below are scored on the other %d",
        len(all_adjudications),
        adjudicated_moments,
        len(all_adjudications) - len(finalised),
        len(finalised),
    )

    print()
    print(
        report_resolution(
            f"How often the adjudicator backed the first and second pass "
            f"({len(finalised)} final calls over {adjudicated_moments})",
            resolution,
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
        "Adjudicator's own rationale note (no annotator draft to compare against): "
        f"{_count_pct(len(written), len(finalised))} of final calls carry one, "
        f"median {_number(_median(len(text) for text in written), '.0f')} characters."
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_moments": len(rows),
                    "overall": overall,
                    "by_pair": pairs,
                    "n_adjudications": len(all_adjudications),
                    "n_final_calls": len(finalised),
                    "resolution": resolution,
                    "text_edits": edits,
                },
                fh,
                indent=1,
            )
            fh.write("\n")
        logger.info("wrote %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
