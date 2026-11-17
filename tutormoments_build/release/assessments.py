"""Write ``assessments/<provider>.jsonl`` and ``reference/eedi_*.jsonl``.

The assessment files ship as the bucket has them, apart from the changes declared
here; the Eedi reference tables ship unchanged. As for transcripts, every build
applies the declared changes to the frozen bucket schema and requires the rules to
match the packaged release schema exactly (``check_declared_changes``).

Declared changes:

* Provider A (Eedi quiz sessions): ``batches`` is dropped (it lists every delivery a
  quiz session appeared in; ``batch``, the first of them, stays). An empty
  ``date_completed`` becomes null. ``question_url`` becomes null unless the question
  is in the reference questions, so the release links only questions Eedi has
  published, and on unanswered entries, where it is the bare host.
* Provider C: only rows joining a released transcript ship. A program's assessments
  need a transcript with its ``program_id``; an exit card (``post_session``) needs a
  transcript with its ``session_id``. The rest belong to sessions the transcript
  filters excluded. ``batch`` is pinned, as in C's transcripts.

Provider C's filter reads the transcripts this build wrote, so this writer runs after
``write_transcripts``.
"""

import json
import logging
from dataclasses import dataclass
from typing import Callable

from tutormoments_build.release.build import BuildContext, Output, packaged_schemas
from tutormoments_build.release.transcripts import check_declared_changes

logger = logging.getLogger("tutormoments_build.release.assessments")

ASSESSMENTS_DIR = "assessments/"
REFERENCE_DIR = "reference/"
TRANSCRIPTS_DIR = "transcripts/"

# Public Eedi tables, shipped unchanged under the same names.
REFERENCE = (
    "eedi_constructs",
    "eedi_diagnostic_questions",
    "eedi_misconceptions",
    "eedi_subjects",
)
QUESTIONS = "eedi_diagnostic_questions"

RowFilter = Callable[[dict], dict | None]


def _a_schema_changes(rules: dict) -> None:
    del rules["properties"]["batches"]
    rules["required"].remove("batches")
    rules["properties"]["date_completed"]["type"] = ["string", "null"]
    rules["$defs"]["Question"]["properties"]["question_url"]["type"] = [
        "string",
        "null",
    ]


def _c_schema_changes(rules: dict) -> None:
    # The partner delivered once and will not deliver again.
    rules["properties"]["batch"]["const"] = "2026-07-06"


def a_rows(ctx: BuildContext) -> RowFilter:
    """Provider A's declared changes, given the build's reference questions."""
    published = {
        json.loads(line)["parent_question_id"]
        for line in ctx.store.iter_lines(ctx.snapshot_key(f"source/{QUESTIONS}.jsonl"))
    }

    def change(row: dict) -> dict:
        deliveries = row.pop("batches")
        if row["batch"] != min(deliveries):
            raise SystemExit(
                f"{row['quiz_session_id']}: batch {row['batch']!r} is not the first "
                f"delivery in {deliveries}; the release describes it as the first"
            )
        if row["date_completed"] == "":
            row["date_completed"] = None
        for question in row["questions"]:
            if (
                question["answer"] == ""
                or question["parent_question_id"] not in published
            ):
                question["question_url"] = None
        return row

    return change


def c_rows(ctx: BuildContext) -> RowFilter:
    """Provider C: keep only rows that join a transcript this build released."""
    programs, sessions = set(), set()
    key = ctx.build_prefix + TRANSCRIPTS_DIR + "tutoring_provider_c.jsonl"
    for line in ctx.store.iter_lines(key):
        session = json.loads(line)["session"]
        programs.add(session["program_id"])
        sessions.add(session["session_id"])

    def keep(row: dict) -> dict | None:
        if row["program_id"] not in programs:
            return None
        if (
            row["assessment_type"] == "post_session"
            and row["session_id"] not in sessions
        ):
            return None
        return row

    return keep


@dataclass(frozen=True)
class Assessment:
    bucket: str  # bucket file name: source/<bucket>.jsonl, schemas/<bucket>.schema.json
    rows: Callable[[BuildContext], RowFilter]
    schema_changes: Callable[[dict], None]


# Provider B adds an entry here, with its own release schema.
ASSESSMENTS = {
    "tutoring_provider_a": Assessment(
        "tutoring_provider_a_eedi_quiz_sessions", a_rows, _a_schema_changes
    ),
    "tutoring_provider_c": Assessment(
        "tutoring_provider_c_assessments", c_rows, _c_schema_changes
    ),
}
_BY_BUCKET = {spec.bucket: provider for provider, spec in ASSESSMENTS.items()}


def release_schema(bucket: str) -> str:
    """The packaged release schema for a bucket file."""
    if bucket in _BY_BUCKET:
        return f"assessments_{_BY_BUCKET[bucket]}.schema.json"
    return f"{bucket}.schema.json"


def check_assessment_drift(
    bucket_schema: dict, release_schema: dict, bucket: str
) -> None:
    """Fail unless the bucket file's rules + declared changes == release rules."""
    provider = _BY_BUCKET.get(bucket)
    changes = ASSESSMENTS[provider].schema_changes if provider else (lambda rules: None)
    check_declared_changes(
        bucket_schema, release_schema, changes, bucket, "assessments.py"
    )


def write_assessments(ctx: BuildContext) -> list[Output]:
    schemas = packaged_schemas()

    def checked_schema(bucket: str) -> str:
        name = release_schema(bucket)
        bucket_schema = json.loads(
            ctx.store.read_bytes(ctx.snapshot_key(f"schemas/{bucket}.schema.json"))
        )
        check_assessment_drift(bucket_schema, json.loads(schemas[name]), bucket)
        return name

    outputs = []
    for provider, spec in ASSESSMENTS.items():
        schema = checked_schema(spec.bucket)
        row_filter = spec.rows(ctx)
        path = f"{ASSESSMENTS_DIR}{provider}.jsonl"
        local = ctx.workdir / f"assessments_{provider}.jsonl"
        read = kept = 0
        with open(local, "w", encoding="utf-8") as out:
            for line in ctx.store.iter_lines(
                ctx.snapshot_key(f"source/{spec.bucket}.jsonl")
            ):
                read += 1
                row = row_filter(json.loads(line))
                if row is not None:
                    kept += 1
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
        ctx.write(path, local)
        local.unlink()
        logger.info("wrote %s: %d of %d rows", path, kept, read)
        outputs.append(Output(path, (schema,)))

    for bucket in REFERENCE:
        schema = checked_schema(bucket)
        path = f"{REFERENCE_DIR}{bucket}.jsonl"
        ctx.copy_from_snapshot(f"source/{bucket}.jsonl", path)
        logger.info("wrote %s", path)
        outputs.append(Output(path, (schema,)))
    return outputs
