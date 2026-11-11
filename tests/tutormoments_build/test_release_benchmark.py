"""Benchmark labels writer: annotation exports -> ``benchmark/dev.jsonl``, ``test.jsonl``.

Rows are synthetic, built with the annotations writer's export fixtures. Builds run
against a local store, so every output also passes the packaged schemas and the
name check, and the annotations writer runs first as it does in a real build.
"""

import json

import pytest

pytest.importorskip("jsonschema")

from tests.tutormoments_build.test_release_annotations import (
    CLEAN,
    INPUTS,
    NOW,
    U,
    no_key_a,
    raw_a,
    raw_c,
    store_with,
)
from tutormoments_build.release import benchmark
from tutormoments_build.release import build as rb
from tutormoments_build.release.annotations import write_annotations
from tutormoments_build.release.snapshot import create_snapshot
from tutormoments_build.v2 import build_ground_truth as gt

# raw_a's transcript is U(1); raw_c's is U(3); no_key_a's is U(5).
ASSIGNMENTS = {
    U(1): {"split": "iterate", "source": "tutoring_provider_a"},
    U(3): {"split": "heldout", "source": "tutoring_provider_c"},
    U(5): {"split": "iterate", "source": "tutoring_provider_a"},
}


@pytest.fixture
def splits(tmp_path, monkeypatch):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"schema_version": "1.0", "assignments": ASSIGNMENTS}))
    monkeypatch.setattr(benchmark, "SPLITS", str(path))
    return path


def build(store, writers=(write_annotations, benchmark.write_benchmark_labels)):
    create_snapshot(store, snapshot_id="s1", inputs=INPUTS)
    return rb.run_build(store, "s1", writers=writers, code=CLEAN, now=NOW)


def released(store, manifest, path):
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    return store.read_bytes(prefix + path)


def test_build_writes_dev_and_test_from_the_ground_truth_rules(tmp_path, splits):
    store = store_with(tmp_path / "bucket", [raw_a(), no_key_a()], [raw_c()])
    manifest = build(store)

    dev = [
        json.loads(x)
        for x in released(store, manifest, "benchmark/dev.jsonl").splitlines()
    ]
    test = [
        json.loads(x)
        for x in released(store, manifest, "benchmark/test.jsonl").splitlines()
    ]
    assert [(r["moment_id"], r["split"]) for r in dev] == [(U(2), "dev")]
    assert [(r["moment_id"], r["split"]) for r in test] == [(U(4), "test")]
    assert dev[0]["annotators"] == [
        {"annotator": "A01", "role": "selector"},
        {"annotator": "A02", "role": "reannotator"},
        {"annotator": "A03", "role": "adjudicator"},
    ]
    for path in ("benchmark/dev.jsonl", "benchmark/test.jsonl"):
        assert manifest["files"][path]["schemas"] == [benchmark.SCHEMA]


def test_release_files_match_the_ground_truth_script_run_on_the_export(
    tmp_path, splits
):
    store = store_with(tmp_path / "bucket", [raw_a()], [raw_c()])
    manifest = build(store)

    export = tmp_path / "export"
    export.mkdir()
    paths = []
    for provider, key in zip(benchmark.PROVIDERS, INPUTS[:2], strict=True):
        path = export / f"{provider}_annotations.jsonl"
        path.write_bytes(store.read_bytes(key))
        paths.append(str(path))
    labels = export / "labels.json"
    labels.write_bytes(store.read_bytes(INPUTS[2]))
    no_staff = export / "no_staff.json"
    no_staff.write_text("[]")
    out, _ = gt.build(paths, str(splits), str(labels), str(no_staff))

    for stem, name in (("iteration", "dev"), ("test", "test")):
        records = [benchmark.release_record(r) for r in out[stem]]
        expected = "".join(json.dumps(r) + "\n" for r in records).encode()
        assert released(store, manifest, f"benchmark/{name}.jsonl") == expected


def ground_truth_record(**overrides):
    record = {
        "moment_id": U(2),
        "source": "tutoring_provider_a",
        "transcript_id": U(1),
        "split": "iteration",
        "start_turn": 2,
        "end_turn": 6,
        "cut_turn": 4,
        "boundaries_source": "selector",
        "situation_type": "scaffold_only",
        "situation_votes": {"scaffolding": 2, "rigor": 0},
        "labels": {
            "scaffolding_present": True,
            "rigor_present": False,
            "over_scaffolding_present": False,
        },
        "action_ties": [],
        "n_votes": 2,
        "annotators": ["A01", "A02"],
        "annotator_roles": ["selector", "reannotator"],
        "moment_status": "adjudicated",
        "moment_created_at": "2026-10-01T11:00:00Z",
    }
    record.update(overrides)
    return record


def test_release_record_renames_the_split_and_pairs_annotators_with_roles():
    row = benchmark.release_record(ground_truth_record())
    assert row["split"] == "dev"
    assert row["annotators"] == [
        {"annotator": "A01", "role": "selector"},
        {"annotator": "A02", "role": "reannotator"},
    ]
    assert row["original_boundaries"] is None
    for dropped in ("annotator_roles", "moment_status", "moment_created_at"):
        assert dropped not in row


def test_release_record_keeps_moved_boundaries():
    original = {"start_turn": 2, "end_turn": 8, "cut_turn": 4}
    row = benchmark.release_record(
        ground_truth_record(
            split="test",
            boundaries_source="reannotator",
            original_boundaries=original,
        )
    )
    assert row["split"] == "test"
    assert row["original_boundaries"] == original


def test_release_record_refuses_unpaired_roles():
    with pytest.raises(ValueError):
        benchmark.release_record(ground_truth_record(annotator_roles=["selector"]))


def test_writer_needs_the_annotations_release(tmp_path, splits):
    store = store_with(tmp_path / "bucket", [raw_a()], [raw_c()])
    with pytest.raises(SystemExit, match="annotations/tutoring_provider_a.jsonl"):
        build(store, writers=(benchmark.write_benchmark_labels,))


def test_build_fails_when_the_released_annotations_do_not_reproduce_the_labels(
    tmp_path, splits
):
    def write_annotations_losing_an_answer(ctx):
        """The annotations writer, but the adjudicator's entry goes missing."""
        outputs = write_annotations(ctx)
        key = ctx.build_prefix + "annotations/tutoring_provider_a.jsonl"
        row = json.loads(ctx.store.read_bytes(key))
        assert row["annotations"][-1]["role"] == "adjudicator"
        row["annotations"].pop()
        ctx.store.write_bytes(key, (json.dumps(row) + "\n").encode())
        return outputs

    store = store_with(tmp_path / "bucket", [raw_a()], [raw_c()])
    with pytest.raises(SystemExit, match="do not reproduce benchmark/dev.jsonl"):
        build(
            store,
            writers=(
                write_annotations_losing_an_answer,
                benchmark.write_benchmark_labels,
            ),
        )


def test_default_build_writes_benchmark_labels_after_annotations():
    names = [w.__name__ for w in rb.default_writers()]
    assert names.index("write_benchmark_labels") > names.index("write_annotations")
