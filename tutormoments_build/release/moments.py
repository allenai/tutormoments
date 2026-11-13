"""Write ``benchmark/moments.jsonl``: everything needed to replay each benchmark moment.

One row per row of ``benchmark/dev.jsonl`` and ``test.jsonl`` (dev first, each in file
order), built only from files this build has already written, so the release can
always rebuild it:

* ``transcripts/<source>.jsonl``: the whole session, every turn and screen event
  copied unchanged, events placed before the turn their ``before_turn`` names;
  provider C's ``problems``.
* ``benchmark/dev.jsonl``, ``test.jsonl``: boundaries and the situation.
* ``annotations/<source>.jsonl``: provider A's ``annotator_view``, the
  ``transcript_selection`` of the annotator whose boundaries the moment uses.

The view keeps the cut where that annotator drew it, which can be after a screen-event
row; ``cut_turn`` is then the tool's conversion to a turn number and can be one turn
off either way. Only where the moment's cut is not the annotator's own (two
adjudicators drew different spans and the cut most annotators chose won) is the cut
placed after the last row holding ``cut_turn``.

The schema carries copies of the provider schemas' turn, screen-event and problem
definitions; every build checks they still match.
"""

import json
import logging

from tutormoments_build.release.annotations import PROVIDERS
from tutormoments_build.release.benchmark import ANNOTATIONS_DIR, BENCHMARK_DIR
from tutormoments_build.release.build import BuildContext, Output, packaged_schemas
from tutormoments_build.v2 import build_ground_truth as gt

logger = logging.getLogger("tutormoments_build.release.moments")

SCHEMA = "moments.schema.json"
PATH = f"{BENCHMARK_DIR}moments.jsonl"
SPLITS = ("dev", "test")
TRANSCRIPTS_DIR = "transcripts/"

# moments.schema.json definition -> (provider schema, definition) it copies.
COPIED_DEFINITIONS = {
    "TurnProviderA": ("tutoring_provider_a.schema.json", "Turn"),
    "EnrichmentProviderA": ("tutoring_provider_a.schema.json", "Enrichment"),
    "TurnProviderC": ("tutoring_provider_c.schema.json", "Turn"),
    "ProblemProviderC": ("tutoring_provider_c.schema.json", "Problem"),
    "LearningObjective": ("tutoring_provider_c.schema.json", "LearningObjective"),
}
BOUNDARIES = ("start_turn", "cut_turn", "end_turn")


def check_copied_definitions(schemas: dict[str, dict]) -> None:
    """Fail unless each copied definition equals its provider's (bar its ``$comment``)."""
    ours = schemas[SCHEMA]["$defs"]
    for name, (provider_schema, definition) in COPIED_DEFINITIONS.items():
        copy = {k: v for k, v in ours[name].items() if k != "$comment"}
        if copy != schemas[provider_schema]["$defs"][definition]:
            raise SystemExit(
                f"{SCHEMA} $defs.{name} no longer matches {provider_schema} "
                f"$defs.{definition}; copy the provider's definition across"
            )


def session(record: dict) -> list[dict]:
    """Turns and screen events in one list: events before the turn ``before_turn``
    names, in ``enrichment_number`` order; those naming no turn after the last."""
    waiting: dict = {}
    for event in sorted(
        record.get("enrichments") or [], key=lambda e: e["enrichment_number"]
    ):
        waiting.setdefault(event["before_turn"], []).append(event)
    items = []
    for turn in record["turns"]:
        items += [{"enrichment": e} for e in waiting.pop(turn["turn_number"], [])]
        items.append({"turn": turn})
    for key in sorted(waiting, key=lambda k: (k is None, k or 0)):
        items += [{"enrichment": e} for e in waiting[key]]
    return items


def drawn_boundaries(entry: dict, row: dict) -> tuple:
    """(start, cut, end) as this annotator left the moment, by the ground-truth rules."""
    moment = row["moment"]
    shown = gt.reannotated_boundaries(moment, gt.reannotation(row["annotations"]))
    if entry["role"] == "selector":
        drawn = {f: moment[f] for f in BOUNDARIES}
    elif entry["role"] == "reannotator":
        drawn = shown
    else:
        draft = gt.adjudicator_draft(entry) or {}
        drawn = {f: shown[f] if draft.get(f) is None else draft[f] for f in BOUNDARIES}
    return tuple(drawn[f] for f in BOUNDARIES)


def annotator_view(label: dict, row: dict, items: list[dict]) -> dict:
    """The view of the annotator whose boundaries ``label`` uses."""
    where = f"moment {label['moment_id']}"
    want = tuple(label[f] for f in BOUNDARIES)
    candidates = sorted(
        (
            e
            for e in row["annotations"]
            if e["role"] == label["boundaries_source"] and "transcript_selection" in e
        ),
        key=lambda e: e["created_at"],
    )
    own = [e for e in candidates if drawn_boundaries(e, row) == want]
    same_span = [
        e for e in candidates if drawn_boundaries(e, row)[::2] == (want[0], want[2])
    ]
    if not (own or same_span):
        raise SystemExit(
            f"{where}: no {label['boundaries_source']} drew boundaries {want}"
        )
    entry = (own or same_span)[0]
    rows = [dict(r) for r in entry["transcript_selection"]["rows"]]
    if not own:
        holding = [i for i, r in enumerate(rows) if r["turn_number"] == want[1]]
        if not holding:
            raise SystemExit(f"{where}: no row of the view holds cut turn {want[1]}")
        for i, r in enumerate(rows):
            r["in_situation"] = i <= holding[-1]

    turns = {i["turn"]["turn_number"] for i in items if "turn" in i}
    events = {i["enrichment"]["enrichment_number"] for i in items if "enrichment" in i}
    for r in rows:
        if r["turn_number"] is not None and r["turn_number"] not in turns:
            raise SystemExit(
                f"{where}: view names turn {r['turn_number']}, not in the transcript"
            )
        for n in r["enrichment_numbers"]:
            if n not in events:
                raise SystemExit(
                    f"{where}: view names screen event {n}, not in the transcript"
                )
    return {
        "annotator": entry["annotator"],
        "role": entry["role"],
        "interface_version": entry["interface_version"],
        "rows": rows,
    }


def read_jsonl(ctx: BuildContext, path: str) -> list[dict]:
    key = ctx.build_prefix + path
    if not ctx.store.exists(key):
        raise SystemExit(
            f"{path} is not in the build; moments are built from it, so its writer "
            "must run first"
        )
    return [json.loads(line) for line in ctx.store.iter_lines(key)]


def write_moments(ctx: BuildContext) -> list[Output]:
    check_copied_definitions(
        {name: json.loads(data) for name, data in packaged_schemas().items()}
    )
    for provider in PROVIDERS:
        if not ctx.store.exists(f"{ctx.build_prefix}{TRANSCRIPTS_DIR}{provider}.jsonl"):
            raise SystemExit(
                f"{TRANSCRIPTS_DIR}{provider}.jsonl is not in the build; moments are "
                "built from it, so write_transcripts must run first"
            )
    labels = [
        label
        for split in SPLITS
        for label in read_jsonl(ctx, f"{BENCHMARK_DIR}{split}.jsonl")
    ]
    wanted = {label["moment_id"] for label in labels}
    annotations = {
        row["moment"]["moment_id"]: row
        for provider in PROVIDERS
        for row in read_jsonl(ctx, f"{ANNOTATIONS_DIR}{provider}.jsonl")
        if row["moment"] and row["moment"]["moment_id"] in wanted
    }

    sessions: dict[str, dict] = {}
    for provider in PROVIDERS:
        needed = {m["transcript_id"] for m in labels if m["source"] == provider}
        key = f"{ctx.build_prefix}{TRANSCRIPTS_DIR}{provider}.jsonl"
        for line in ctx.store.iter_lines(key):
            record = json.loads(line)
            if record["transcript_id"] in needed:
                sessions[record["transcript_id"]] = {
                    "transcript": session(record),
                    "problems": record.get("problems"),
                }

    local = ctx.workdir / "moments.jsonl"
    with open(local, "w", encoding="utf-8") as out:
        for label in labels:
            where = f"moment {label['moment_id']}"
            if label["transcript_id"] not in sessions:
                raise SystemExit(
                    f"{where}: transcript {label['transcript_id']} not found"
                )
            if label["moment_id"] not in annotations:
                raise SystemExit(f"{where}: no annotations row")
            items = sessions[label["transcript_id"]]["transcript"]
            row = annotations[label["moment_id"]]
            has_view = any("transcript_selection" in e for e in row["annotations"])
            moment = {
                **{
                    k: label[k]
                    for k in (
                        "moment_id",
                        "source",
                        "transcript_id",
                        "split",
                        *BOUNDARIES,
                        "boundaries_source",
                        "situation_type",
                        "situation_votes",
                    )
                },
                "transcript": items,
                "problems": sessions[label["transcript_id"]]["problems"],
                "annotator_view": annotator_view(label, row, items)
                if has_view
                else None,
            }
            out.write(json.dumps(moment, ensure_ascii=False) + "\n")
    ctx.write(PATH, local)
    local.unlink()
    logger.info("wrote %s: %d moments", PATH, len(labels))
    return [Output(PATH, (SCHEMA,))]
