"""Write ``annotations/<provider>.jsonl`` from the annotation tool's frozen export.

One row per moment, with every annotator's entry on it, or one per "no key moments"
record. Annotators become their release labels (a missing label fails the build);
names, the tool's internal fields and retracted moments are dropped; where one
annotator saved a role more than once, only the highest revision is kept, as
``tutormoments_build.v2.build_ground_truth`` does. Answers ship as exported.

Every field the export carries is listed below as kept or dropped. A field the
export starts carrying fails the build until someone decides which it is, so
nothing new ships, or vanishes, unreviewed.
"""

import json
import logging

from tutormoments_build.release.build import BuildContext, Output

logger = logging.getLogger("tutormoments_build.release.annotations")

SCHEMA = "annotations.schema.json"
LABELS = "v2_annotations/annotator_labels.json"
PROVIDERS = ("tutoring_provider_a", "tutoring_provider_c")
PURPOSE = "rigor_scaffolding"
BATCH = "v2"

RETRACTED = "retracted"

# Export fields, by level. Kept fields are renamed or copied below; dropped ones are
# the transcript fingerprints (they match no released transcript file), names, the
# tool's revision counter, and provider A's flat ``enrichment_numbers`` (always the
# events listed in ``transcript_selection``, which ships).
ROW_FIELDS = {
    "transcript_id",
    "source",
    "corpus_revision",
    "moment",
    "annotations",
    "no_key_moments_record",
    "v1_rigor_hinted",
    "v1_rigor_regions",
}
MOMENT_FIELDS = {
    "moment_id",
    "start_turn",
    "cut_turn",
    "end_turn",
    "dialogue_turns",
    "status",
    "created_by",
    "created_by_name",
    "created_at",
    "in_v1_rigor_region",
}
ENTRY_FIELDS = {
    "annotator_id",
    "annotator_name",
    "role",
    "revision",
    "created_at",
    "interface_version",
    "schema_version",
    "payload",
    "selection_source_sha256",
    "transcript_selection",
    "enrichment_numbers",
}

# Provider A's record of how the tool showed the moment: its rows, top to bottom.
SELECTION_FIELDS = {"rows"}
SELECTION_ROW_FIELDS = {"turn_number", "enrichment_numbers", "is_cut"}

# Tool-computed turn counts (end - start + 1), wrong on some early provider A moments.
PAYLOAD_DROPS = ("final_dialogue_turns",)
META_DROPS = ("new_dialogue_turns",)


def check_fields(obj: dict, known: set[str], where: str) -> None:
    unknown = set(obj) - known
    if unknown:
        raise SystemExit(
            f"{where}: unknown export field(s) {sorted(unknown)}; decide whether the "
            "release keeps or drops them in annotations.py"
        )


class Labels:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def __call__(self, annotator_id: str, where: str) -> str:
        if annotator_id not in self.mapping:
            raise SystemExit(f"{where}: annotator {annotator_id} has no release label")
        return self.mapping[annotator_id]


def last_saves(entries: list[dict]) -> list[dict]:
    """Each annotator's highest revision per role, in export order."""
    latest: dict[tuple[str, str], dict] = {}
    for entry in entries:
        key = (entry["annotator_id"], entry["role"])
        if key not in latest or entry["revision"] > latest[key]["revision"]:
            latest[key] = entry
    return [e for e in entries if latest[(e["annotator_id"], e["role"])] is e]


def release_selection(raw: dict, where: str) -> dict:
    """The tool's selection with its one cut flag turned into a per-row side.

    ``is_cut`` marks the row the cut falls after. ``in_situation`` says on every row
    which side of the cut it is: true up to and including the cut row (what the
    annotator judged the situation from), false after it (the tutor's action).
    """
    check_fields(raw, SELECTION_FIELDS, f"{where} transcript_selection")
    rows = raw["rows"]
    for row in rows:
        check_fields(row, SELECTION_ROW_FIELDS, f"{where} transcript_selection row")
    cuts = [i for i, row in enumerate(rows) if row["is_cut"]]
    if len(cuts) != 1:
        raise SystemExit(
            f"{where}: transcript_selection has {len(cuts)} cut rows; expected exactly one"
        )
    return {
        "rows": [
            {
                "turn_number": row["turn_number"],
                "enrichment_numbers": row["enrichment_numbers"],
                "in_situation": i <= cuts[0],
            }
            for i, row in enumerate(rows)
        ]
    }


def release_entry(raw: dict, provider: str, labels: Labels, where: str) -> dict:
    check_fields(raw, ENTRY_FIELDS, where)
    entry = {
        "annotator": labels(raw["annotator_id"], where),
        "role": raw["role"],
        "created_at": raw["created_at"],
        "interface_version": raw["interface_version"],
        "schema_version": raw["schema_version"],
    }
    if provider == "tutoring_provider_a" and "transcript_selection" in raw:
        entry["transcript_selection"] = release_selection(
            raw["transcript_selection"], where
        )
    payload = {k: v for k, v in raw["payload"].items() if k not in PAYLOAD_DROPS}
    if isinstance(payload.get("meta"), dict):
        payload["meta"] = {
            k: v for k, v in payload["meta"].items() if k not in META_DROPS
        }
    entry["payload"] = payload
    return entry


def release_row(raw: dict, provider: str, labels: Labels) -> dict | None:
    """The release row for one export row, or None for a retracted moment."""
    where = f"{provider} {raw.get('transcript_id')}"
    check_fields(raw, ROW_FIELDS, where)
    moment = raw.get("moment")
    if moment is not None:
        where = f"{where} moment {moment.get('moment_id')}"
        check_fields(moment, MOMENT_FIELDS, where)
        if moment["status"] == RETRACTED:
            return None

    row = {
        "transcript_id": raw["transcript_id"],
        "source": provider,
        "purpose": PURPOSE,
        "batch": BATCH,
        "moment": None,
        "v1_rigor_hinted": raw["v1_rigor_hinted"],
        "v1_rigor_regions": raw["v1_rigor_regions"],
    }
    if moment is None:
        entries = [raw["no_key_moments_record"]]
    else:
        row["moment"] = {
            "moment_id": moment["moment_id"],
            "start_turn": moment["start_turn"],
            "cut_turn": moment["cut_turn"],
            "end_turn": moment["end_turn"],
            "status": moment["status"],
            "selected_by": labels(moment["created_by"], where),
            "created_at": moment["created_at"],
            "in_v1_rigor_region": moment["in_v1_rigor_region"],
        }
        entries = last_saves(raw["annotations"])
    row["annotations"] = [release_entry(e, provider, labels, where) for e in entries]
    return row


def write_annotations(ctx: BuildContext) -> list[Output]:
    labels = Labels(json.loads(ctx.store.read_bytes(ctx.snapshot_key(LABELS))))
    outputs = []
    for provider in PROVIDERS:
        source = f"v2_annotations/source/{provider}_annotations.jsonl"
        path = f"annotations/{provider}.jsonl"
        local = ctx.workdir / f"{provider}_annotations.jsonl"
        kept = dropped = 0
        with open(local, "w", encoding="utf-8") as out:
            for line in ctx.store.iter_lines(ctx.snapshot_key(source)):
                row = release_row(json.loads(line), provider, labels)
                if row is None:
                    dropped += 1
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
        ctx.write(path, local)
        local.unlink()
        logger.info("wrote %s: %d rows (%d retracted dropped)", path, kept, dropped)
        outputs.append(Output(path, (SCHEMA,)))
    return outputs
