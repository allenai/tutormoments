"""Release plumbing: snapshot, build, checks, verify, and the transcript core contract.

Runs against a local-directory store (and moto for the S3 backend); no AWS calls.
All rows here are synthetic.
"""

import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("jsonschema")

from tutormoments_build.release import build as rb
from tutormoments_build.release import checks
from tutormoments_build.release.snapshot import (
    SNAPSHOT_FILE,
    create_snapshot,
    load_snapshot,
)
from tutormoments_build.release.store import LocalStore, open_store

CLEAN = {"commit": "abc1234def5678", "dirty": False}
NOW = datetime(2026, 11, 3, 15, 2, tzinfo=timezone.utc)

TID = "00000000-0000-5000-8000-000000000001"
SID = "00000000-0000-5000-8000-000000000002"


def turn(n, role="Tutor", **extra):
    return {
        "turn_number": n,
        "role": role,
        "text": f"turn {n}",
        "start_seconds": n,
        **extra,
    }


def transcript(**overrides):
    row = {
        "transcript_id": TID,
        "source": "tutoring_provider_x",
        "batch": "2026-01-01",
        "has_video": False,
        "turns": [turn(1), turn(2, "Student")],
        "session": {"session_id": SID, "tutor_id": SID, "student_id": SID},
    }
    row.update(overrides)
    return row


def jsonl(*rows):
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


@pytest.fixture
def store(tmp_path):
    s = LocalStore(tmp_path)
    s.write_bytes("source/t.jsonl", jsonl(transcript()))
    s.write_bytes("gt/a.jsonl", b"{}\n")
    s.write_bytes("gt/b.jsonl", b"{}\n{}\n")
    return s


INPUTS = ("source/t.jsonl", "gt/")


@pytest.fixture
def snapshot(store):
    create_snapshot(store, snapshot_id="s1", inputs=INPUTS)
    return "s1"


# -- name check ----------------------------------------------------------------


def test_name_keys_flags_nested_person_name_fields():
    row = {
        "annotations": [{"annotator_name": "x", "payload": {"created_by_name": "y"}}]
    }
    assert checks.name_keys(row, checks.allowed_name_keys("annotations/a.jsonl")) == [
        "annotations[].annotator_name",
        "annotations[].payload.created_by_name",
    ]


def test_name_keys_allows_known_non_person_fields_only_where_listed():
    assert (
        checks.name_keys({"construct_name": "Fractions"}, checks.allowed_name_keys("x"))
        == []
    )
    subjects = {"name": "Number"}
    assert (
        checks.name_keys(
            subjects, checks.allowed_name_keys("reference/eedi_subjects.jsonl")
        )
        == []
    )
    assert checks.name_keys(
        subjects, checks.allowed_name_keys("annotations/a.jsonl")
    ) == ["name"]


def test_name_keys_ignores_values_and_lookalike_keys():
    row = {"rename_count": 1, "names_seen": 2, "text": "annotator_name"}
    assert checks.name_keys(row, frozenset()) == []


# -- snapshot ------------------------------------------------------------------


def test_snapshot_freezes_inputs_with_hashes(store, snapshot):
    record = load_snapshot(store, "release-v2", snapshot)
    assert sorted(record["inputs"]) == ["gt/a.jsonl", "gt/b.jsonl", "source/t.jsonl"]
    assert record["inputs"]["gt/b.jsonl"]["lines"] == 2
    frozen = "release-v2/snapshots/s1/gt/b.jsonl"
    assert record["inputs"]["gt/b.jsonl"]["sha256"] == store.digest(frozen)["sha256"]


def test_snapshot_never_overwrites(store, snapshot):
    with pytest.raises(SystemExit, match="already exists"):
        create_snapshot(store, snapshot_id=snapshot, inputs=INPUTS)


def test_snapshot_missing_input_is_an_error(store):
    with pytest.raises(SystemExit, match="missing"):
        create_snapshot(store, snapshot_id="s2", inputs=("source/nope.jsonl",))
    assert store.list("release-v2/snapshots/s2/") == []


def test_snapshot_dry_run_copies_nothing(store):
    record = create_snapshot(store, snapshot_id="s3", inputs=INPUTS, dry_run=True)
    assert len(record["inputs"]) == 3
    assert store.list("release-v2/snapshots/s3/") == []


def test_build_refuses_incomplete_snapshot(store, snapshot):
    (store.root / "release-v2/snapshots/s1" / SNAPSHOT_FILE).unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        rb.run_build(store, snapshot, code=CLEAN)


# -- build ---------------------------------------------------------------------


def copy_transcripts(ctx):
    ctx.copy_from_snapshot("source/t.jsonl", "transcripts/t.jsonl")
    return [rb.Output("transcripts/t.jsonl", ("transcript_core.schema.json",))]


def test_build_writes_schemas_and_manifest(store, snapshot):
    manifest = rb.run_build(
        store, snapshot, writers=(copy_transcripts,), code=CLEAN, now=NOW
    )

    assert manifest["build_id"] == "20261103T1502Z-abc1234"
    assert manifest["code"] == CLEAN
    snap_json = store.digest("release-v2/snapshots/s1/" + SNAPSHOT_FILE)["sha256"]
    assert manifest["snapshot"] == {
        "snapshot_id": "s1",
        "snapshot_json_sha256": snap_json,
    }

    files = manifest["files"]
    assert files["transcripts/t.jsonl"]["rows"] == 1
    assert files["transcripts/t.jsonl"]["schemas"] == ["transcript_core.schema.json"]
    assert "schemas/transcript_core.schema.json" in files
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    assert json.loads(store.read_bytes(prefix + rb.MANIFEST_FILE)) == manifest
    assert rb.verify_build(store, manifest["build_id"]) == []


def test_build_refuses_dirty_tree_unless_allowed(store, snapshot):
    dirty = {**CLEAN, "dirty": True}
    with pytest.raises(SystemExit, match="uncommitted"):
        rb.run_build(store, snapshot, code=dirty)
    manifest = rb.run_build(
        store, snapshot, writers=(), code=dirty, allow_dirty=True, now=NOW
    )
    assert manifest["code"]["dirty"] is True


def test_failed_checks_leave_no_manifest(store, snapshot):
    def bad_rows(ctx):
        src = ctx.workdir / "t.jsonl"
        src.write_bytes(jsonl(transcript(turns=[]), transcript(annotator_name="x")))
        ctx.write("transcripts/t.jsonl", src)
        return [rb.Output("transcripts/t.jsonl", ("transcript_core.schema.json",))]

    with pytest.raises(SystemExit, match="failed its checks"):
        rb.run_build(store, snapshot, writers=(bad_rows,), code=CLEAN, now=NOW)

    prefix = rb.build_prefix("release-v2", "20261103T1502Z-abc1234")
    assert not store.exists(prefix + rb.MANIFEST_FILE)
    (report,) = json.loads(store.read_bytes(prefix + rb.CHECKS_FILE))
    assert report["errors"] == 2
    assert any("annotator_name" in e for e in report["examples"])
    assert rb.verify_build(store, "20261103T1502Z-abc1234") != []


def test_output_without_schema_fails(store, snapshot):
    def no_schema(ctx):
        ctx.copy_from_snapshot("gt/a.jsonl", "benchmark/a.jsonl")
        return [rb.Output("benchmark/a.jsonl", ())]

    with pytest.raises(SystemExit, match="failed its checks"):
        rb.run_build(store, snapshot, writers=(no_schema,), code=CLEAN, now=NOW)


def test_verify_catches_tampering_and_strays(store, snapshot):
    manifest = rb.run_build(
        store, snapshot, writers=(copy_transcripts,), code=CLEAN, now=NOW
    )
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    store.write_bytes(prefix + "transcripts/t.jsonl", jsonl(transcript(batch="other")))
    store.write_bytes(prefix + "stray.txt", b"x")

    problems = rb.verify_build(store, manifest["build_id"])
    assert "hash mismatch: transcripts/t.jsonl" in problems
    assert "not in manifest: stray.txt" in problems


# -- transcript core contract --------------------------------------------------


@pytest.fixture(scope="module")
def core():
    schema = rb.packaged_schemas()["transcript_core.schema.json"]
    return checks.load_validator(json.loads(schema))


def test_core_accepts_provider_extras(core):
    row = transcript(
        turns=[turn(1, end_seconds=2), turn(2, "Unattributed")],
        session={
            "session_id": SID,
            "tutor_id": SID,
            "student_id": SID,
            "program_id": SID,
        },
        problems=[{"slide_id": 1}],
    )
    assert list(core.iter_errors(row)) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"turns": []},
        {"turns": [turn(1, "Narrator")]},
        {"turns": [turn(0)]},
        {"session": {"session_id": SID, "tutor_id": SID}},
        {"transcript_id": "not-a-uuid"},
        {"source": "provider_a"},
    ],
)
def test_core_rejects_contract_violations(core, overrides):
    assert list(core.iter_errors(transcript(**overrides))) != []


# -- S3 backend ----------------------------------------------------------------


def test_s3_store_round_trip():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="release-test")
        s3 = open_store("s3://release-test/pre")
        s3.write_bytes("x/one.jsonl", b"{}\n{}\n")
        s3.copy("x/one.jsonl", "y/two.jsonl")

        assert s3.list("") == ["x/one.jsonl", "y/two.jsonl"]
        assert s3.exists("y/two.jsonl") and not s3.exists("y/none")
        assert s3.digest("y/two.jsonl")["lines"] == 2
        assert list(s3.iter_lines("y/two.jsonl")) == [b"{}\n", b"{}\n"]
