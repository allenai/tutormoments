"""Assessments writer: provider A's declared changes, provider C's transcript join,
reference tables copied unchanged, and schema-drift detection.

All rows are synthetic. The "bucket" schemas are rebuilt from the packaged release
schemas by undoing the declared changes, so the tests track the real schemas.
"""

import copy
import json

import pytest

pytest.importorskip("jsonschema")

from tests.tutormoments_build.test_release_transcripts import (
    CLEAN,
    NOW,
    U,
    jsonl,
    row_a,
    row_c,
)
from tests.tutormoments_build.test_release_transcripts import (
    INPUTS as TRANSCRIPT_INPUTS,
)
from tests.tutormoments_build.test_release_transcripts import (
    bucket_schemas as transcript_bucket_schemas,
)
from tutormoments_build.release import assessments
from tutormoments_build.release import build as rb
from tutormoments_build.release.assessments import (
    ASSESSMENTS,
    REFERENCE,
    check_assessment_drift,
    write_assessments,
)
from tutormoments_build.release.snapshot import create_snapshot
from tutormoments_build.release.transcripts import write_transcripts

PUBLISHED = "100561"  # in the reference questions
UNPUBLISHED = "156172"  # not in them
IMG = "https://images.diagnosticquestions.com/uploads/questions/{}.jpg".format
BARE_HOST = "https://images.diagnosticquestions.com"

# row_c()'s transcript: program U(15), session U(12).
PROGRAM, SESSION = U(15), U(12)


def answer(seq, question_id, letter="B", url=None):
    return {
        "question_sequence": seq,
        "answer_sequence": seq,
        "parent_question_id": question_id,
        "answer_type": "Check-in" if letter else "",
        "predicted_correctness": "-",
        "actual_correctness": "Correct" if letter else "Incorrect",
        "answer": letter,
        "explanation": "Synthetic feedback: add the two parts." if letter else "",
        "question_url": url or (IMG(seq) if letter else BARE_HOST),
    }


def quiz(**extra):
    row = {
        "source": "tutoring_provider_a",
        "batch": "2026-03-24",
        "batches": ["2026-03-24", "2026-05-29"],
        "student_id": U(4),
        "quiz_session_id": U(5),
        "quiz_name": "Synthetic quiz",
        "type": "Independent",
        "date_started": "2025-11-22",
        "date_completed": "2025-11-26",
        "lesson_length": "00:24:38",
        "lesson_type": "TopicPathwayQuiz",
        "predicted_score": "-",
        "actual_score": "80%",
        "questions": [
            answer(1, PUBLISHED),
            answer(2, UNPUBLISHED),
            answer(3, PUBLISHED, letter=""),
        ],
    }
    row.update(extra)
    return row


def assessment(kind, program=PROGRAM, session=None, questions=()):
    row = {
        "source": "tutoring_provider_c",
        "batch": "2026-07-06",
        "assessment_type": kind,
        "student_id": U(14),
        "program_id": program,
        "questions": list(questions),
    }
    if session:
        row["session_id"] = session
    return row


C_QUESTION = {
    "question_id": 15024,
    "question_text": '<p>Half of 8 is <img alt="\\frac{1}{2}" src="https://latex.codecogs.com/gif.latex?x" /></p>',
    "answers": [
        {"id": 1, "answer": "<p>4</p>", "given_answer": True, "correct_answer": 1},
        {"id": 2, "answer": "<p>2</p>", "given_answer": False, "correct_answer": 0},
    ],
    "answer_id": 1,
    "is_correct": True,
    "difficulty_level": 1,
    "lesson_id": 900,
    "assignment_date": "2024-12-16",
}

C_ROWS = {
    "kept_pre": assessment("pre_program", questions=[C_QUESTION]),
    "kept_exit_card": assessment("post_session", session=SESSION),
    "kept_post": assessment("post_program"),
    "excluded_session": assessment("post_session", session=U(99)),
    "program_without_transcript": assessment("pre_program", program=U(98)),
}

REFERENCE_ROWS = {
    "eedi_constructs": {
        "construct_id": "402",
        "construct_name": "Synthetic construct",
        "subject_id": "200",
        "prerequisite_construct_ids": ["479"],
    },
    "eedi_diagnostic_questions": {
        "parent_question_id": PUBLISHED,
        "intervention_id": "6358",
        "question_text": "Synthetic question text",
        "answers": [{"label": "A", "text": "\\( \\frac{1}{10} \\)"}],
        "subjects": [
            {"SubjectName": "Number", "SubjectLevel": 1, "SubjectType": "Subject"}
        ],
        "construct_id": "402",
        "construct_name": None,
        "year_group": "8",
        "curriculum_level": "35",
    },
    "eedi_misconceptions": {
        "misconception_id": "0",
        "misconception_name": "Synthetic misconception",
    },
    "eedi_subjects": {
        "subject_id": "3",
        "name": "Maths",
        "parent_id": None,
        "level": 0,
    },
}


def bucket_schemas():
    """The packaged release schemas with the declared changes undone, keyed by bucket name."""
    packaged = rb.packaged_schemas()
    a = json.loads(packaged["assessments_tutoring_provider_a.schema.json"])
    a["properties"]["batches"] = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "uniqueItems": True,
    }
    a["required"].insert(2, "batches")
    a["properties"]["date_completed"]["type"] = "string"
    a["$defs"]["Question"]["properties"]["question_url"]["type"] = "string"
    c = json.loads(packaged["assessments_tutoring_provider_c.schema.json"])
    del c["properties"]["batch"]["const"]
    schemas = {
        "tutoring_provider_a_eedi_quiz_sessions": a,
        "tutoring_provider_c_assessments": c,
    }
    for name in REFERENCE:
        schemas[name] = json.loads(packaged[f"{name}.schema.json"])
    return schemas


INPUTS = TRANSCRIPT_INPUTS + tuple(
    key
    for name in bucket_schemas()
    for key in (f"source/{name}.jsonl", f"schemas/{name}.schema.json")
)


@pytest.fixture
def store(tmp_path):
    from tutormoments_build.release.store import LocalStore

    s = LocalStore(tmp_path)
    s.write_bytes("source/tutoring_provider_a.jsonl", jsonl(row_a()))
    s.write_bytes("source/tutoring_provider_c.jsonl", jsonl(row_c()))
    for name, schema in transcript_bucket_schemas().items():
        s.write_bytes(f"schemas/{name}.schema.json", json.dumps(schema).encode())
    for name, schema in bucket_schemas().items():
        s.write_bytes(f"schemas/{name}.schema.json", json.dumps(schema).encode())
    s.write_bytes(
        "source/tutoring_provider_a_eedi_quiz_sessions.jsonl",
        jsonl(quiz(), quiz(date_completed="", quiz_session_id=U(6))),
    )
    s.write_bytes(
        "source/tutoring_provider_c_assessments.jsonl", jsonl(*C_ROWS.values())
    )
    for name, row in REFERENCE_ROWS.items():
        s.write_bytes(f"source/{name}.jsonl", jsonl(row))
    return s


def build(store, snapshot_id="s1"):
    create_snapshot(store, snapshot_id=snapshot_id, inputs=INPUTS)
    return rb.run_build(
        store,
        snapshot_id,
        writers=(write_transcripts, write_assessments),
        code=CLEAN,
        now=NOW,
    )


def released(store, manifest, path):
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    return [json.loads(line) for line in store.read_bytes(prefix + path).splitlines()]


def test_a_applies_only_the_declared_changes(store):
    manifest = build(store)
    first, no_date = released(store, manifest, "assessments/tutoring_provider_a.jsonl")

    expected = quiz()
    del expected["batches"]
    expected["questions"][1]["question_url"] = None  # not a published question
    expected["questions"][2]["question_url"] = None  # unanswered: the bare host
    assert first == expected
    assert no_date["date_completed"] is None
    assert no_date["questions"][0]["question_url"] == IMG(1)
    # The feedback stays on every answer, published question or not.
    assert first["questions"][1]["explanation"] == quiz()["questions"][1]["explanation"]

    entry = manifest["files"]["assessments/tutoring_provider_a.jsonl"]
    assert entry["rows"] == 2
    assert entry["schemas"] == ["assessments_tutoring_provider_a.schema.json"]


def test_a_batch_that_is_not_the_first_delivery_fails(store):
    late = quiz(batch="2026-05-29")
    store.write_bytes(
        "source/tutoring_provider_a_eedi_quiz_sessions.jsonl", jsonl(late)
    )
    with pytest.raises(SystemExit, match="first delivery"):
        build(store)


def test_c_keeps_only_rows_that_join_a_released_transcript(store):
    manifest = build(store)
    rows = released(store, manifest, "assessments/tutoring_provider_c.jsonl")
    assert rows == [C_ROWS["kept_pre"], C_ROWS["kept_exit_card"], C_ROWS["kept_post"]]
    entry = manifest["files"]["assessments/tutoring_provider_c.jsonl"]
    assert entry["schemas"] == ["assessments_tutoring_provider_c.schema.json"]


def test_reference_tables_are_byte_identical_copies(store):
    manifest = build(store)
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    for name, row in REFERENCE_ROWS.items():
        assert store.read_bytes(f"{prefix}reference/{name}.jsonl") == jsonl(row)
        assert manifest["files"][f"reference/{name}.jsonl"]["schemas"] == [
            f"{name}.schema.json"
        ]


def test_c_row_breaking_the_release_schema_fails_the_build(store):
    late = assessment("pre_program")
    late["batch"] = "2027-01-01"
    store.write_bytes("source/tutoring_provider_c_assessments.jsonl", jsonl(late))
    with pytest.raises(SystemExit, match="failed its checks"):
        build(store)


def test_text_only_upstream_changes_are_not_drift():
    for name, bucket in bucket_schemas().items():
        bucket["description"] = "reworded upstream"
        release = json.loads(rb.packaged_schemas()[assessments.release_schema(name)])
        check_assessment_drift(bucket, release, name)


@pytest.mark.parametrize(
    "name,field",
    [
        ("tutoring_provider_a_eedi_quiz_sessions", "quiz_name"),
        ("tutoring_provider_c_assessments", "assessment_type"),
        ("eedi_subjects", "level"),
    ],
)
def test_upstream_rule_change_is_drift(name, field):
    bucket = copy.deepcopy(bucket_schemas()[name])
    bucket["properties"][field]["minLength"] = 99
    release = json.loads(rb.packaged_schemas()[assessments.release_schema(name)])
    with pytest.raises(SystemExit, match="no longer match"):
        check_assessment_drift(bucket, release, name)


def test_every_bucket_input_is_snapshotted_and_has_a_release_schema():
    from tutormoments_build.release.snapshot import SNAPSHOT_INPUTS

    packaged = rb.packaged_schemas()
    for name in [spec.bucket for spec in ASSESSMENTS.values()] + list(REFERENCE):
        assert f"source/{name}.jsonl" in SNAPSHOT_INPUTS
        assert f"schemas/{name}.schema.json" in SNAPSHOT_INPUTS
        assert assessments.release_schema(name) in packaged


def test_assessments_run_after_the_transcripts_they_join():
    names = [w.__name__ for w in rb.default_writers()]
    assert names.index("write_assessments") == names.index("write_transcripts") + 1
