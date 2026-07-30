"""Tests for --scaffolding-only support in build_ground_truth.py.

Covers the two custom pieces: load_from_jsonl's annotation-type filter (so a
scaffolding-only run skips rapport records entirely) and _merge_scaffolding_only
(so writing a scaffolding-only rebuild preserves rapport moments already on disk).
"""

import json

from tutorsim_build.groundtruth import _merge_scaffolding_only, load_from_jsonl


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _record(transcript_id, annotation_type, annotator_id, turn_start, turn_end):
    return {
        "transcript_id": transcript_id,
        "annotation_type": annotation_type,
        "annotator_id": annotator_id,
        "turn_annotations": [
            {
                "turn_number_start": turn_start,
                "turn_number_end": turn_end,
                "situation": "s",
                "action": "a",
                "result": "r",
                "annotation_timestamp": "2026-01-01",
            }
        ],
    }


# --- load_from_jsonl annotation-type filter --------------------------------


def test_load_from_jsonl_default_keeps_scaffolding_and_rapport(tmp_path):
    path = tmp_path / "ann.jsonl"
    _write_jsonl(
        path,
        [
            _record("c1", "scaffolding", "t1", 1, 5),
            _record("c1", "rapport", "t2", 2, 6),
            _record("c1", "caption", "t3", 0, 0),
        ],
    )
    convs = dict(load_from_jsonl(str(path)))
    types = {a["annotation_type"] for a in convs["c1"]["annotations"]}
    assert types == {"scaffolding", "rapport"}


def test_load_from_jsonl_scaffolding_only_drops_rapport(tmp_path):
    path = tmp_path / "ann.jsonl"
    _write_jsonl(
        path,
        [
            _record("c1", "scaffolding", "t1", 1, 5),
            _record("c1", "rapport", "t2", 2, 6),
        ],
    )
    convs = dict(load_from_jsonl(str(path), annotation_types=("scaffolding",)))
    types = {a["annotation_type"] for a in convs["c1"]["annotations"]}
    assert types == {"scaffolding"}


def test_load_from_jsonl_scaffolding_only_omits_rapport_only_conversation(tmp_path):
    path = tmp_path / "ann.jsonl"
    _write_jsonl(
        path,
        [
            _record("c_rapport_only", "rapport", "t1", 1, 5),
            _record("c_scaf", "scaffolding", "t2", 1, 5),
        ],
    )
    convs = dict(load_from_jsonl(str(path), annotation_types=("scaffolding",)))
    assert "c_rapport_only" not in convs
    assert "c_scaf" in convs


# --- _merge_scaffolding_only -----------------------------------------------


def test_merge_preserves_existing_rapport_moments():
    new_scaf = [{"annotation_type": "scaffolding", "turn_start": 1, "turn_end": 5}]
    existing = [
        {
            "annotation_type": "scaffolding",
            "turn_start": 1,
            "turn_end": 5,
            "stale": True,
        },
        {"annotation_type": "rapport", "turn_start": 2, "turn_end": 8},
    ]
    merged = _merge_scaffolding_only(new_scaf, existing)
    # new scaffolding moments come first, stale scaffolding dropped, rapport preserved
    assert merged[0] == new_scaf[0]
    rapport = [m for m in merged if m["annotation_type"] == "rapport"]
    assert len(rapport) == 1 and rapport[0]["turn_end"] == 8
    assert all(not m.get("stale") for m in merged)


def test_merge_with_no_existing_moments_returns_new():
    new_scaf = [{"annotation_type": "scaffolding", "turn_start": 1, "turn_end": 5}]
    assert _merge_scaffolding_only(new_scaf, []) == new_scaf
