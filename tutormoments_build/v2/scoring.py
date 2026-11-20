"""Scores replay classifications: did the LM tutor make the move the moment called for?

Reads one or more `classify_replays.py` output files and joins each record to
its gold `situation_type` in `data/ground_truth/{iteration,test}.jsonl`.

Metrics (each a % of moments in its situation set):

- `scaffold_only`: scaffold_only moments where the tutor scaffolded and did
  not push for rigor.
- `rigor_only`: rigor_only moments where the tutor pushed for rigor and did
  not scaffold.
- `rigor_present`: rigor_maj + rigor_only moments where the tutor pushed for rigor.
- `scaffolding_present`: scaffold_maj + scaffold_only moments where the tutor
  scaffolded.
- `no_over_scaffolding`: all moments where the tutor did not over-scaffold. A
  replay predicted to have no scaffolding was never asked the over-scaffolding
  question and counts as not over-scaffolding.

A moment whose needed prediction is missing (nothing after the cut, a failed
request, an unparseable reply) is left out of that metric's denominator and
counted under `excluded`, never scored as a yes or a no.

Action divergence (`--taxonomy`, `taxonomy_replays.py` output): the
Jensen-Shannon divergence (base 2, in [0, 1]) between the fine-grained action
type distributions of scaffolding moments (scaffold_only + scaffold_maj) and
rigor moments (rigor_maj + rigor_only). Higher means the tutor acts more
differently in the two kinds of moment. Each moment weighs the same: its
distinct action types share it equally (a type labelled at two turns counts
once), and a group's distribution is the mean over its moments. Labels the
prompt doesn't list are dropped. A moment with no listed type still counts, as
a whole share of a `none` bucket: action_direction predicted neither
scaffolding nor rigor (so the taxonomy was never asked), or the taxonomy found
no listed type.

Action diversity (same `--taxonomy` files): how many different strategies of
the called-for family the tutor uses across a group's moments -- scaffolding
types in scaffolding moments, rigor types in rigor moments. The Shannon entropy
(bits) of that family's type distribution, built the same way (each moment
weighs the same, its distinct types of the family share it equally). Moments
with no type of the family (including `none`) are left out of it.

For both, moments whose answer is unknown (a failed request or unparseable
reply at either stage, or nothing after the cut) are left out and counted by
reason.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass

from tutormoments.taxonomy import js_divergence
from tutormoments_build.v2 import classify_replays, excerpts, jsonl_runs
from tutormoments_build.v2 import taxonomy as T
from tutormoments_build.v2.taxonomy_replays import PREDICTED_NEITHER


@dataclass(frozen=True)
class Metric:
    name: str
    # Gold situation types the metric covers; empty means every moment.
    situations: tuple[str, ...]
    # Which predictions the metric reads; a moment missing any is excluded.
    needs: tuple[str, ...]
    # Given the predictions (all non-None), did the tutor do the right thing?
    hit: object


METRICS = (
    Metric(
        "scaffold_only",
        ("scaffold_only",),
        ("scaffolding", "rigor"),
        lambda p: p["scaffolding"] and not p["rigor"],
    ),
    Metric(
        "rigor_only",
        ("rigor_only",),
        ("scaffolding", "rigor"),
        lambda p: p["rigor"] and not p["scaffolding"],
    ),
    Metric(
        "rigor_present",
        ("rigor_maj", "rigor_only"),
        ("rigor",),
        lambda p: p["rigor"],
    ),
    Metric(
        "scaffolding_present",
        ("scaffold_maj", "scaffold_only"),
        ("scaffolding",),
        lambda p: p["scaffolding"],
    ),
    Metric(
        "no_over_scaffolding",
        (),  # every situation
        ("over_scaffolding",),
        lambda p: not p["over_scaffolding"],
    ),
)


def load_situations(ground_truth_dir: str = excerpts.DEFAULT_GROUND_TRUTH_DIR):
    """`{moment_id: situation_type}` over every ground-truth split."""
    out = {}
    for records in excerpts.load_ground_truth(ground_truth_dir).values():
        for record in records:
            out[record["moment_id"]] = record["situation_type"]
    return out


def predictions(record: dict) -> dict:
    """The replay's predicted scaffolding / rigor / over-scaffolding (None = unknown)."""
    action = record.get("action_direction") or {}
    scaffolding = action.get("scaffolding")
    rigor = action.get("rigor")
    routing = record.get("over_scaffolding_routing") or {}
    if routing.get("reason") == classify_replays.NOT_ROUTED_NO_SCAFFOLDING:
        over = False
    elif routing.get("routed"):
        over = (record.get("over_scaffolding") or {}).get("over_scaffolding")
    else:
        over = None
    return {"scaffolding": scaffolding, "rigor": rigor, "over_scaffolding": over}


def gold_situation(record: dict, situations: dict[str, str]) -> str:
    """The record's gold situation_type, read from the ground truth.

    Raises if the moment is missing from the ground truth, or the record's stored
    `situation_type` disagrees with it (a replay built from stale ground truth).
    """
    moment_id = record["moment_id"]
    if moment_id not in situations:
        raise KeyError(f"{moment_id} is not in the ground truth")
    gold = situations[moment_id]
    stored = record.get("situation_type")
    if stored is not None and stored != gold:
        raise ValueError(
            f"{moment_id}: record says {stored!r}, ground truth says {gold!r}"
        )
    return gold


def score(records: list[dict], situations: dict[str, str]) -> dict:
    """Every metric for one classification file.

    Returns `{metric: {hits, n, excluded, rate}}`; `rate` is None when `n` is 0.
    """
    joined = [(gold_situation(r, situations), predictions(r)) for r in records]

    out = {}
    for metric in METRICS:
        hits = n = excluded = 0
        for gold, pred in joined:
            if metric.situations and gold not in metric.situations:
                continue
            if any(pred[k] is None for k in metric.needs):
                excluded += 1
                continue
            n += 1
            hits += bool(metric.hit(pred))
        out[metric.name] = {
            "situations": list(metric.situations) or "all",
            "hits": hits,
            "n": n,
            "excluded": excluded,
            "rate": hits / n if n else None,
        }
    return out


# ===========================================================================
# Action divergence and diversity
# ===========================================================================

# Situation groups; each name is also the action family it calls for.
SITUATION_GROUPS = {
    "scaffolding": ("scaffold_only", "scaffold_maj"),
    "rigor": ("rigor_maj", "rigor_only"),
}


NONE = "none"


def moment_types(record: dict, categories: dict[str, str]) -> set[str] | str:
    """The moment's distinct listed types, or the reason it is left out.

    Labels the prompt doesn't list are dropped. Predicted neither (taxonomy not
    asked) is an empty set, like an empty list.
    """
    taxonomy = record.get("taxonomy")
    if not record.get("asked") or taxonomy is None:
        if record.get("skipped") == PREDICTED_NEITHER:
            return set()
        return f"not asked: {record.get('skipped')}"
    if taxonomy.get("error") and not taxonomy.get("raw"):
        return "request failed"
    if taxonomy.get("actions") is None:
        return "unparseable"
    return {a["category"] for a in taxonomy["actions"]} & categories.keys()


def shares(types: set[str]) -> dict[str, float]:
    """Equal shares of one moment; no types is all `none`."""
    return {t: 1 / len(types) for t in types} if types else {NONE: 1.0}


def mean_distribution(moments: list[dict], order: list[str]) -> dict | None:
    """The per-moment distributions averaged over moments; None if there are none."""
    if not moments:
        return None
    return {c: sum(m.get(c, 0.0) for m in moments) / len(moments) for c in order}


def grouped_moments(records: list[dict], situations: dict[str, str]):
    """`(categories, {group: [moment types]}, {reason: excluded moments})`."""
    versions = {r["prompt_version"] for r in records}
    if len(versions) != 1:
        raise ValueError(f"taxonomy records mix prompt versions {sorted(versions)}")
    categories = T.parse_categories(T.load_template(versions.pop()))
    group_of = {s: g for g, sits in SITUATION_GROUPS.items() for s in sits}

    moments: dict[str, list[set]] = {g: [] for g in SITUATION_GROUPS}
    excluded: dict[str, int] = {}
    for record in records:
        group = group_of.get(gold_situation(record, situations))
        if group is None:
            continue
        types = moment_types(record, categories)
        if isinstance(types, str):
            excluded[types] = excluded.get(types, 0) + 1
        else:
            moments[group].append(types)
    return categories, moments, dict(sorted(excluded.items()))


def action_divergence(records: list[dict], situations: dict[str, str]) -> dict:
    """JSD between scaffolding and rigor moments' action type distributions.

    Returns `{jsd, n: {group: moments}, distributions: {group: {type: p}},
    excluded: {reason: moments}}`. `jsd` is None if either group is empty.
    """
    categories, moments, excluded = grouped_moments(records, situations)
    order = [*categories, NONE]
    distributions = {
        g: mean_distribution([shares(t) for t in ms], order)
        for g, ms in moments.items()
    }
    s, r = distributions["scaffolding"], distributions["rigor"]
    jsd = None
    if s and r:
        jsd = js_divergence([s[c] for c in order], [r[c] for c in order])
    return {
        "jsd": jsd,
        "groups": {g: list(sits) for g, sits in SITUATION_GROUPS.items()},
        "n": {g: len(ms) for g, ms in moments.items()},
        "distributions": distributions,
        "excluded": excluded,
    }


def entropy_bits(distribution: dict[str, float]) -> float:
    # `+ 0.0` turns the -0.0 of a single type into 0.0.
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0) + 0.0


def action_diversity(records: list[dict], situations: dict[str, str]) -> dict:
    """Per group, the diversity of the called-for family's types across moments.

    Returns `{group: {family, situations, entropy_bits, distribution}}`, plus
    `excluded`. `entropy_bits` is None if no moment used a type of the family.
    """
    categories, moments, excluded = grouped_moments(records, situations)
    out: dict = {}
    for group, ms in moments.items():
        family = [c for c, f in categories.items() if f == group]
        with_family = [t & set(family) for t in ms if t & set(family)]
        distribution = mean_distribution([shares(t) for t in with_family], family)
        h = None if distribution is None else entropy_bits(distribution)
        out[group] = {
            "family": group,
            "situations": list(SITUATION_GROUPS[group]),
            "entropy_bits": h,
            "distribution": distribution,
        }
    out["excluded"] = excluded
    return out


def divergence_report(divergence_by_file: dict[str, dict]) -> str:
    lines = []
    for path, d in divergence_by_file.items():
        jsd = "n/a" if d["jsd"] is None else f"{d['jsd']:.4f}"
        n = d["n"]
        lines.append(path)
        lines.append(
            f"  action JSD (scaffolding vs rigor)  {jsd}  "
            f"(n = {n['scaffolding']} / {n['rigor']} moments)"
        )
        for reason, count in d["excluded"].items():
            lines.append(f"    excluded, {reason}: {count}")
    return "\n".join(lines)


def diversity_report(diversity_by_file: dict[str, dict]) -> str:
    lines = []
    for path, d in diversity_by_file.items():
        lines.append(path)
        for group in SITUATION_GROUPS:
            g = d[group]
            h = g["entropy_bits"]
            value = "n/a" if h is None else f"{h:.3f} bits"
            lines.append(f"  {group} diversity in {group} moments  {value}")
        for reason, count in d["excluded"].items():
            lines.append(f"    excluded, {reason}: {count}")
    return "\n".join(lines)


def report(scores_by_file: dict[str, dict]) -> str:
    lines = []
    for path, scores in scores_by_file.items():
        lines.append(path)
        for name, s in scores.items():
            rate = "  n/a" if s["rate"] is None else f"{100 * s['rate']:5.1f}%"
            excluded = f"  ({s['excluded']} excluded)" if s["excluded"] else ""
            lines.append(f"  {name:<22}{rate}  {s['hits']}/{s['n']}{excluded}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.scoring",
        description="Score classified replays against the gold situation labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "classifications",
        nargs="*",
        metavar="CLASSIFIED.jsonl",
        help="classify_replays.py output file(s)",
    )
    parser.add_argument(
        "--taxonomy",
        nargs="+",
        default=[],
        metavar="TAXONOMY.jsonl",
        help="taxonomy_replays.py output file(s), for the action divergence",
    )
    parser.add_argument(
        "--ground-truth", default=excerpts.DEFAULT_GROUND_TRUTH_DIR, metavar="DIR"
    )
    parser.add_argument(
        "--json", dest="json_out", default=None, metavar="FILE", help="Also write JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.classifications and not args.taxonomy:
        parser.error("give classification file(s), --taxonomy file(s), or both")
    situations = load_situations(args.ground_truth)
    out: dict[str, dict] = {
        "decisions": {},
        "action_divergence": {},
        "action_diversity": {},
    }
    for key, paths, run in (
        ("decisions", args.classifications, score),
        ("action_divergence", args.taxonomy, action_divergence),
        ("action_diversity", args.taxonomy, action_diversity),
    ):
        for path in paths:
            records = list(jsonl_runs.load_by_id(path).values())
            if not records:
                print(f"no records in {path}", file=sys.stderr)
                return 1
            out[key][path] = run(records, situations)
    if out["decisions"]:
        print(report(out["decisions"]))
    if out["action_divergence"]:
        print(divergence_report(out["action_divergence"]))
        print(diversity_report(out["action_diversity"]))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
