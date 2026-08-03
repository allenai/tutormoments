"""Headline reproducibility acceptance test for the ground-truth build.

A warm rebuild must be deterministic and free of LLM calls: once the per-conversation
cache files exist, re-running build_ground_truth on the same annotations + out_dir must
make ZERO batch-API calls and re-emit byte-identical JSON. This pins the cache machinery
(strategy/situation labels, action/result/overscaffold decomposition, and the
action/result aggregation cache reconstructed by load_existing_action_result_agg) that
makes rebuilding the published ground_truth_hybrid.jsonl reproducible.

The first (cold) build stubs run_batch with canned outputs so it is offline and
deterministic; the second (warm) build patches run_batch to RAISE, so any cache miss
fails the test loudly instead of silently calling the API.
"""

import json

import pytest

from tutorsim_build import groundtruth


def _canned_text(key: str) -> str:
    """Deterministic canned model output keyed by the request-key suffix.

    The exact labels don't matter for the round-trip property -- whatever the cold
    build caches, the warm build must reproduce -- but using parseable values keeps
    the fixture realistic.
    """
    if key.endswith("__sit"):
        return '{"scaffolding": "yes", "rigor": "no"}'
    if key.endswith("__action_agg"):
        return '{"scaffolding": "yes", "rigor": "no"}'
    if key.endswith("__result_agg"):
        return "pos"
    if key.endswith("__action"):
        return '["the tutor explained the full answer"]'
    if key.endswith("__result"):
        return '["the student copied the answer down"]'
    if key.endswith("__overscaffold"):
        return '["explained the full answer"]'
    return "effective"  # strategy/effectiveness classification


def _fake_run_batch(client, entries, **kwargs):
    return {e["key"]: {"text": _canned_text(e["key"]), "usage": {}} for e in entries}


class _FakeClient:
    def __init__(self, model):
        self.model = model


def _record(
    transcript_id,
    annotation_type,
    annotator_id,
    ts,
    te,
    situation,
    action,
    result,
    timestamp,
):
    return {
        "transcript_id": transcript_id,
        "annotation_type": annotation_type,
        "annotator_id": annotator_id,
        "turn_annotations": [
            {
                "turn_number_start": ts,
                "turn_number_end": te,
                "situation": situation,
                "action": action,
                "result": result,
                "annotation_timestamp": timestamp,
            }
        ],
    }


@pytest.fixture
def annotations_path(tmp_path):
    # Two scaffolding annotators on the SAME turn range (-> one IoU==1.0 cluster,
    # exercising the agg path) plus a rapport moment.
    records = [
        _record(
            "conv-A",
            "scaffolding",
            "ann1",
            5,
            9,
            "student is stuck on factoring",
            "tutor gave a leading hint",
            "student factored the next term",
            "2026-01-01T00:00:00",
        ),
        _record(
            "conv-A",
            "scaffolding",
            "ann2",
            5,
            9,
            "student stuck on the same step",
            "tutor asked a probing question",
            "student made progress",
            "2026-01-01T00:00:01",
        ),
        _record(
            "conv-A",
            "rapport",
            "ann3",
            2,
            4,
            "student seems frustrated",
            "tutor acknowledged the difficulty",
            "student relaxed",
            "2026-01-01T00:00:02",
        ),
    ]
    path = tmp_path / "ann.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _read_all(out_dir):
    return {
        p.name: p.read_text(encoding="utf-8") for p in sorted(out_dir.glob("*.json"))
    }


def test_warm_rebuild_is_byte_identical_and_makes_zero_llm_calls(
    annotations_path, tmp_path, monkeypatch
):
    out_dir = tmp_path / "gt_out"

    # --- cold build: stub the API with canned outputs ---
    monkeypatch.setattr("tutorsim.client.run_batch", _fake_run_batch)
    monkeypatch.setattr("tutorsim.client.ModelClient", _FakeClient)
    summary1 = groundtruth.build_ground_truth(
        input_path=annotations_path, out_dir=out_dir
    )
    assert summary1["gt_written"] == 1
    cold = _read_all(out_dir)
    assert set(cold) == {"conv-A.json"}

    # Sanity: every documented enrichment field is present on the scaffolding moments.
    conv = json.loads(cold["conv-A.json"])
    scaf = [m for m in conv["key_moments"] if m["annotation_type"] == "scaffolding"]
    assert scaf, "expected scaffolding moments"
    for m in scaf:
        for k in (
            "strategy_label",
            "situation_label",
            "situation_label_agg",
            "action_decomposed",
            "result_decomposed",
            "overscaffold_decomposed",
            "action_direction_agg",
            "student_outcome_agg",
        ):
            assert k in m, f"missing {k} on scaffolding moment"

    # --- warm rebuild: any cache miss would call run_batch, which now RAISES ---
    def _boom(*a, **k):
        raise AssertionError(
            "warm rebuild made an LLM call -- cache was not fully warm"
        )

    monkeypatch.setattr("tutorsim.client.run_batch", _boom)
    monkeypatch.setattr(
        "tutorsim.client.ModelClient",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("warm rebuild constructed a ModelClient")
        ),
    )
    summary2 = groundtruth.build_ground_truth(
        input_path=annotations_path, out_dir=out_dir
    )
    warm = _read_all(out_dir)

    assert warm == cold, "warm rebuild was not byte-identical to the cold build"
    # Everything reused; nothing queued for classification.
    assert summary2["classify_strategy"] == 0
    assert summary2["classify_situation"] == 0
    assert summary2["new_decompositions"] == 0
    assert summary2["classify_action_agg"] == 0
    assert summary2["classify_result_agg"] == 0


def test_dry_run_makes_zero_llm_calls_and_writes_nothing(
    annotations_path, tmp_path, monkeypatch
):
    out_dir = tmp_path / "gt_out"

    def _boom(*a, **k):
        raise AssertionError("dry run made an LLM call")

    monkeypatch.setattr("tutorsim.client.run_batch", _boom)
    monkeypatch.setattr(
        "tutorsim.client.ModelClient",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("dry run built a client")),
    )
    summary = groundtruth.build_ground_truth(
        input_path=annotations_path, out_dir=out_dir, dry_run=True
    )

    assert summary["dry_run"] is True
    assert summary["total_moments"] == 3  # 2 scaffolding + 1 rapport
    assert not out_dir.exists(), "dry run must not write any files"


def test_consolidate_writes_jsonl_matching_per_conversation_files(
    annotations_path, tmp_path, monkeypatch
):
    out_dir = tmp_path / "gt_out"
    monkeypatch.setattr("tutorsim.client.run_batch", _fake_run_batch)
    monkeypatch.setattr("tutorsim.client.ModelClient", _FakeClient)

    summary = groundtruth.build_ground_truth(
        input_path=annotations_path, out_dir=out_dir, consolidate=True
    )

    jsonl_path = out_dir.parent / f"{out_dir.name}.jsonl"
    assert jsonl_path.exists()
    lines = [
        l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert len(lines) == summary["consolidated_count"] == 1
    # The consolidated record equals the per-conversation file's content.
    per_conv = json.loads((out_dir / "conv-A.json").read_text(encoding="utf-8"))
    assert json.loads(lines[0]) == per_conv
