"""Transcript writer: declared field drops, straight copies, and schema-drift detection.

All rows are synthetic. The "bucket" schemas are rebuilt from the packaged release
schemas by undoing the declared changes, so the tests track the real schemas.
"""

import copy
import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("jsonschema")

from tutormoments_build.release import build as rb
from tutormoments_build.release.snapshot import create_snapshot
from tutormoments_build.release.store import LocalStore
from tutormoments_build.release.transcripts import (
    check_schema_drift,
    write_transcripts,
)

CLEAN = {"commit": "abc1234def5678", "dirty": False}
NOW = datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc)
U = "00000000-0000-5000-8000-0000000000{:02d}".format


def row_a(**session_extra):
    return {
        "transcript_id": U(1),
        "source": "tutoring_provider_a",
        "batch": "2025-10-16",
        "has_video": True,
        "turns": [
            {
                "turn_number": 1,
                "start_seconds": 0,
                "end_seconds": 2,
                "role": "Tutor",
                "text": "Hi",
            },
            {
                "turn_number": 2,
                "start_seconds": 2,
                "end_seconds": 3,
                "role": "Student",
                "text": "Hey",
            },
        ],
        "session": {
            "session_id": U(2),
            "tutor_id": U(3),
            "student_id": U(4),
            "transcription_by": "LS",
            "primary_language": "en",
            **session_extra,
        },
        "eedi_content_link": [
            {"type": "quiz_session", "quiz_session_id": U(5), "start_seconds": -30}
        ],
    }


def row_c():
    return {
        "transcript_id": U(11),
        "source": "tutoring_provider_c",
        "batch": "2026-07-06",
        "has_video": False,
        "turns": [
            {"turn_number": 1, "start_seconds": 0, "role": "Tutor", "text": "Hello"},
            {
                "turn_number": 2,
                "start_seconds": 1,
                "role": "Unattributed",
                "text": "[unclear]",
            },
        ],
        "session": {
            "session_id": U(12),
            "tutor_id": U(13),
            "student_id": U(14),
            "program_id": U(15),
        },
        "problems": [
            {
                "slide_id": 7,
                "slide_position": 1,
                "learning_objective": {"id": 900, "title": "Adding money.", "year": 2},
                "problem_text": "Add the coins.",
            }
        ],
    }


def bucket_schemas():
    """The packaged release schemas with the declared changes undone."""
    packaged = rb.packaged_schemas()
    a = json.loads(packaged["tutoring_provider_a.schema.json"])
    a["$defs"]["Session"]["properties"]["prior_session_count"] = {
        "type": "integer",
        "minimum": 0,
    }
    link = a["$defs"]["EediContentLink"]
    link["properties"]["type"] = {"type": "string"}
    link["required"].insert(0, "type")
    c = json.loads(packaged["tutoring_provider_c.schema.json"])
    del c["properties"]["batch"]["const"]
    return {"tutoring_provider_a": a, "tutoring_provider_c": c}


def jsonl(*rows):
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


@pytest.fixture
def store(tmp_path):
    s = LocalStore(tmp_path)
    s.write_bytes(
        "source/tutoring_provider_a.jsonl", jsonl(row_a(prior_session_count=3), row_a())
    )
    s.write_bytes("source/tutoring_provider_c.jsonl", jsonl(row_c()))
    for name, schema in bucket_schemas().items():
        s.write_bytes(f"schemas/{name}.schema.json", json.dumps(schema).encode())
    return s


INPUTS = (
    "source/tutoring_provider_a.jsonl",
    "source/tutoring_provider_c.jsonl",
    "schemas/tutoring_provider_a.schema.json",
    "schemas/tutoring_provider_c.schema.json",
)


def build(store, snapshot_id="s1"):
    create_snapshot(store, snapshot_id=snapshot_id, inputs=INPUTS)
    return rb.run_build(
        store, snapshot_id, writers=(write_transcripts,), code=CLEAN, now=NOW
    )


def released(store, manifest, provider):
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    lines = store.read_bytes(f"{prefix}transcripts/{provider}.jsonl").splitlines()
    return [json.loads(line) for line in lines]


def test_a_rows_lose_only_the_declared_fields(store):
    manifest = build(store)
    rows = released(store, manifest, "tutoring_provider_a")

    expected = row_a()
    del expected["eedi_content_link"][0]["type"]
    assert rows == [expected, expected]
    entry = manifest["files"]["transcripts/tutoring_provider_a.jsonl"]
    assert entry["rows"] == 2
    assert entry["schemas"] == [
        "tutoring_provider_a.schema.json",
        "transcript_core.schema.json",
    ]


def test_c_is_a_byte_identical_copy(store):
    manifest = build(store)
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    assert store.read_bytes(f"{prefix}transcripts/tutoring_provider_c.jsonl") == jsonl(
        row_c()
    )
    assert manifest["files"]["transcripts/tutoring_provider_c.jsonl"]["schemas"] == [
        "tutoring_provider_c.schema.json",
        "transcript_core.schema.json",
    ]


def test_unexpected_link_type_fails_instead_of_being_dropped(store):
    bad = row_a()
    bad["eedi_content_link"][0]["type"] = "assessment"
    store.write_bytes("source/tutoring_provider_a.jsonl", jsonl(bad))
    with pytest.raises(SystemExit, match="'assessment'"):
        build(store)


def test_row_breaking_the_provider_schema_fails_the_build(store):
    late = row_c()
    late["batch"] = "2027-01-01"
    store.write_bytes("source/tutoring_provider_c.jsonl", jsonl(late))
    with pytest.raises(SystemExit, match="failed its checks"):
        build(store)


def test_text_only_upstream_changes_are_not_drift():
    bucket = bucket_schemas()["tutoring_provider_c"]
    bucket["description"] = "reworded upstream"
    release = json.loads(rb.packaged_schemas()["tutoring_provider_c.schema.json"])
    check_schema_drift(bucket, release, "tutoring_provider_c")


@pytest.mark.parametrize("provider", ["tutoring_provider_a", "tutoring_provider_c"])
def test_upstream_rule_change_is_drift(provider):
    bucket = copy.deepcopy(bucket_schemas()[provider])
    bucket["properties"]["transcript_id"]["pattern"] = "^.*$"
    release = json.loads(rb.packaged_schemas()[f"{provider}.schema.json"])
    with pytest.raises(SystemExit, match="no longer match"):
        check_schema_drift(bucket, release, provider)
