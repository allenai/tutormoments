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
"""

import argparse
import json
import sys
from dataclasses import dataclass

from tutormoments_build.v2 import classify_replays, excerpts, jsonl_runs


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


def score(records: list[dict], situations: dict[str, str]) -> dict:
    """Every metric for one classification file.

    Returns `{metric: {hits, n, excluded, rate}}`; `rate` is None when `n` is 0.
    Raises if a record's moment is missing from the ground truth, or its stored
    `situation_type` disagrees with it (a replay built from stale ground truth).
    """
    joined = []
    for record in records:
        moment_id = record["moment_id"]
        if moment_id not in situations:
            raise KeyError(f"{moment_id} is not in the ground truth")
        gold = situations[moment_id]
        stored = record.get("situation_type")
        if stored is not None and stored != gold:
            raise ValueError(
                f"{moment_id}: record says {stored!r}, ground truth says {gold!r}"
            )
        joined.append((gold, predictions(record)))

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
        nargs="+",
        metavar="CLASSIFIED.jsonl",
        help="classify_replays.py output file(s)",
    )
    parser.add_argument(
        "--ground-truth", default=excerpts.DEFAULT_GROUND_TRUTH_DIR, metavar="DIR"
    )
    parser.add_argument(
        "--json", dest="json_out", default=None, metavar="FILE", help="Also write JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    situations = load_situations(args.ground_truth)
    scores_by_file = {}
    for path in args.classifications:
        records = list(jsonl_runs.load_by_id(path).values())
        if not records:
            print(f"no records in {path}", file=sys.stderr)
            return 1
        scores_by_file[path] = score(records, situations)
    print(report(scores_by_file))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(scores_by_file, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
