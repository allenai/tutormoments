"""Tests for tutormoments.cli -- run_cell end-to-end (TDD: Red phase).

All tests mock run_conversation and score so there are no network calls.
Uses tmp_path as results_root so nothing writes to the real results/ directory.
"""

import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tutormoments.config import ArmSpec, ScorerSpec, StudentSpec
from tutormoments.moments import Moment
from tutormoments.scoring import Annotation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scenario(sid: str, dimension: str = "scaffolding") -> Moment:
    """Build a minimal Moment for testing."""
    return Moment(
        id=sid,
        context=[{"turn_number": 1, "role": "tutor", "text": "Hello"}],
        dimension=dimension,
        student={
            "context": "Student context",
            "reference": "",
            "trait": {
                "persona": "fixture persona",
                "trait_mode": "joined-3",
                "generator_model": "claude-opus-4-6",
                "generated_at": "2026-06-18T00:00:00",
            },
        },
        rubric={"gold": dimension, "hint": "test hint"},
        provenance={"conv_id": f"conv_{sid}", "cut_turn": 1},
    )


def _make_annotation(sid: str) -> Annotation:
    """Build a minimal Annotation for testing."""
    return Annotation(
        scenario_id=sid,
        annotation_type="scaffolding",
        turn_start=2,
        turn_end=4,
        situation="situation text",
        action="action text",
        result="result text",
        action_decomposed=["facet a"],
        overscaffold_decomposed=[],
        action_label="scaffolding",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def _make_transcript(
    sid: str,
    tutor_latencies=None,
    student_latencies=None,
    tutor_usage=None,
    student_usage=None,
) -> MagicMock:
    """Build a mock Transcript-like object with optional latency/usage data."""
    t = MagicMock()
    t.scenario_id = sid
    t.tutor_model = "claude-opus-4-8"
    t.generated_turns = [{"turn_number": 2, "role": "TUTOR", "text": "Hi"}]
    t.to_dict.return_value = {
        "scenario_id": sid,
        "completed": True,
        "generated_turns": [],
    }
    t.tutor_latencies = tutor_latencies if tutor_latencies is not None else []
    t.student_latencies = student_latencies if student_latencies is not None else []
    t.tutor_usage = (
        tutor_usage
        if tutor_usage is not None
        else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    t.student_usage = (
        student_usage
        if student_usage is not None
        else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    return t


FIXTURE_SCENARIOS = [
    _make_scenario("scenario_001", "scaffolding"),
    _make_scenario("scenario_002", "rigor"),
]


# ---------------------------------------------------------------------------
# Helper: patch targets
# ---------------------------------------------------------------------------

_CONV_PATCH = "tutormoments.cli.conversation.run_conversation"
_SCORE_PATCH = "tutormoments.cli.scoring.score_batch"


def _score_batch_from(annotations):
    """Adapt per-moment Annotation fixtures to the pooled score_batch API.

    Each call returns annotations for exactly the pairs requested, consuming
    one fixture per scenario id per call (supports trials>1 fixtures lists).
    """
    by_sid = {}
    for a in annotations:
        by_sid.setdefault(a.scenario_id, []).append(a)

    def _fake(pairs):
        return {s.id: by_sid[s.id].pop(0) for s, _t in pairs}

    return _fake


_LOAD_PATCH = "tutormoments.cli.load_moments"


def _load_result(moments):
    """load_moments returns (moments, source_meta)."""
    return (
        list(moments),
        {
            "dataset_id": None,
            "revision": None,
            "data_path": None,
            "config": "moments",
            "record_count": len(moments),
            "content_hash": "0" * 64,
        },
    )


_CFG_PATCH = "tutormoments.cli.build_run_config"
# The always-on taxonomy classification hook makes LLM calls; patch it out in
# run_cell tests so the suite never hits the network. Zero usage keeps the
# token-total assertions unaffected.
_TAX_PATCH = "tutormoments.cli._classify_run_taxonomy"
_TAX_RESULT = {
    "scheme_version": "lm_extended_v1",
    "counts": {},
    "n_facets": 0,
    "excluded": 0,
    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
}


def _make_run_config(
    sample=2,
    dataset="test_ds",
    max_turns=4,
    replay_concurrency=1,
    arm="claude-opus-4-8",
    model=None,
    thinking="dynamic",
):
    """Build a fake RunConfig-like object.

    `arm` is the roster key results are keyed by; `model` is the resolved
    provider model id (defaults to the arm name, as in the config schema).
    """
    model = model or arm
    cfg = MagicMock()
    cfg.dataset = dataset
    cfg.data_path = None
    cfg.dataset_revision = None
    cfg.dataset_config = "moments"
    cfg.sample = sample
    cfg.max_turns = max_turns
    cfg.tutors = [arm]
    cfg.modes = ["plain"]
    cfg.trials = 1
    cfg.replay_concurrency = replay_concurrency
    cfg.student = StudentSpec(model="claude-haiku", mode="oracle", thinking="dynamic")
    cfg.scorer = ScorerSpec(model="claude-opus-4-6", thinking="dynamic")
    cfg.resolved_tutors = {
        arm: ArmSpec(name=arm, model=model, provider="anthropic", thinking=thinking)
    }
    cfg.config_source = "test"
    return cfg


# ---------------------------------------------------------------------------
# Test 1: run_cell writes all expected files and correct summary
# ---------------------------------------------------------------------------


def test_run_cell_writes_all_files(tmp_path):
    """run_cell over a 2-scenario fixture writes 2 transcripts + 2 scores +
    config.json + summary.json; summary metrics == aggregate of the 2 scores."""
    import tutormoments.report as report_mod
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]
    expected_summary = report_mod.aggregate(scenarios, annotations)

    cfg_mock = _make_run_config(sample=2)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(annotations)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    # run_id must be non-empty
    assert run_id

    run_dir = tmp_path / run_id

    # config.json
    assert (run_dir / "config.json").exists()
    cfg_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert isinstance(cfg_data, dict)
    # --seed was removed (never seeded anything); must not be recorded
    assert "seed" not in cfg_data
    # Arm-based identity: `tutor`/`arm` are the roster key, `model` the
    # resolved provider model id.
    assert cfg_data["tutor"] == "claude-opus-4-8"
    assert cfg_data["arm"] == "claude-opus-4-8"
    assert cfg_data["model"] == "claude-opus-4-8"
    # Frozen specs are serialized as plain dicts with thinking as a string.
    assert cfg_data["student"] == {
        "model": "claude-haiku",
        "mode": "oracle",
        "thinking": "dynamic",
    }
    assert cfg_data["scorer"] == {"model": "claude-opus-4-6", "thinking": "dynamic"}
    assert cfg_data["resolved_tutors"] == {
        "claude-opus-4-8": {
            "name": "claude-opus-4-8",
            "model": "claude-opus-4-8",
            "provider": "anthropic",
            "thinking": "dynamic",
        }
    }

    # 2 transcripts
    for s in scenarios:
        assert (run_dir / "transcripts" / f"{s.id}.json").exists(), (
            f"Missing transcript for {s.id}"
        )

    # 2 scores
    for s in scenarios:
        assert (run_dir / "scores" / f"{s.id}.json").exists(), (
            f"Missing score for {s.id}"
        )

    # summary.json
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # Identity block: tutor_model is the ARM name; model the resolved id.
    assert summary["tutor_model"] == "claude-opus-4-8"
    assert summary["model"] == "claude-opus-4-8"
    assert summary["mode"] == "plain"

    # summary metrics == aggregate of the 2 annotations
    assert summary["n_scenarios"] == expected_summary["n_scenarios"]
    assert "outcome_pos_rate" not in summary  # dropped from the paper
    assert "scaffolding_did" not in summary  # did-rates dropped from the paper
    assert "rigor_did" not in summary
    scaf = summary["scaffold_calibrated"]
    exp_scaf = expected_summary["scaffold_calibrated"]
    assert scaf["n_clean_yes"] == exp_scaf["n_clean_yes"]
    assert scaf["n_total"] == exp_scaf["n_total"]

    # per-run log written automatically next to config.json / summary.json,
    # narrating the two phases with the paper's terminology
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert f"Run id: {run_id}" in run_log
    assert "Starting Replay (trial 1/1): 2 moments" in run_log
    assert "Starting Classification (trial 1/1): 2 replays" in run_log
    assert "Run complete: 2/2 moments succeeded" in run_log
    # every record inside the cell carries the [tutor/mode] tag
    assert "[claude-opus-4-8/plain] " in run_log


# ---------------------------------------------------------------------------
# Test 1b: run_cell writes latency + tokens blocks into summary.json (spec S7)
# ---------------------------------------------------------------------------


def test_run_cell_writes_latency_and_tokens(tmp_path):
    """run_cell aggregates tutor_latencies/usage from transcripts into
    summary.json latency.tutor and tokens.total blocks (spec S7).

    Mocked transcripts carry non-zero latencies and usage so the aggregation
    path is exercised end-to-end.
    """
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)

    # Two transcripts with known latencies and token counts
    transcripts = [
        _make_transcript(
            "scenario_001",
            tutor_latencies=[1.0, 3.0],
            student_latencies=[0.5],
            tutor_usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            student_usage={
                "input_tokens": 80,
                "output_tokens": 20,
                "total_tokens": 100,
            },
        ),
        _make_transcript(
            "scenario_002",
            tutor_latencies=[2.0],
            student_latencies=[0.8],
            tutor_usage={
                "input_tokens": 200,
                "output_tokens": 100,
                "total_tokens": 300,
            },
            student_usage={
                "input_tokens": 150,
                "output_tokens": 50,
                "total_tokens": 200,
            },
        ),
    ]
    annotations = [_make_annotation(s.id) for s in scenarios]

    cfg_mock = _make_run_config(sample=2)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(annotations)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # latency.tutor block must be present and non-None
    assert "latency" in summary, "summary must have 'latency' key"
    assert "tutor" in summary["latency"], "summary['latency'] must have 'tutor' key"
    lat = summary["latency"]["tutor"]
    assert lat is not None, "latency.tutor must not be None (transcripts had latencies)"
    assert "p50_seconds" in lat, "latency.tutor must have p50_seconds"
    assert "p95_seconds" in lat, "latency.tutor must have p95_seconds"
    # tutor_latencies = [1.0, 3.0, 2.0] -> sorted = [1.0, 2.0, 3.0]
    # n=3, p50 = s[1] = 2.0, p95_idx = max(0, min(2, round(2.85)-1)) = max(0,min(2,2)) = 2 -> 3.0
    assert lat["p50_seconds"] == pytest.approx(2.0), (
        f"p50 expected 2.0, got {lat['p50_seconds']}"
    )
    assert lat["p95_seconds"] == pytest.approx(3.0), (
        f"p95 expected 3.0, got {lat['p95_seconds']}"
    )

    # tokens.total block must be present
    assert "tokens" in summary, "summary must have 'tokens' key"
    assert "total" in summary["tokens"], "summary['tokens'] must have 'total' key"
    tok = summary["tokens"]["total"]
    assert "total_tokens" in tok, "tokens.total must have total_tokens"
    # tutor: 150+300=450, student: 100+200=300, total: 750
    assert tok["total_tokens"] == 750, (
        f"total_tokens expected 750, got {tok['total_tokens']}"
    )


# ---------------------------------------------------------------------------
# Test 2: run_cell resumes (second call skips already-done scenarios)
# ---------------------------------------------------------------------------


def test_run_cell_resumes(tmp_path):
    """Second run_cell call finds both scenarios already done; 0 new conversation
    calls are made (is_done returns True for both)."""
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]

    cfg_mock = _make_run_config(sample=2)

    # First run: write all files
    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=list(transcripts)),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(annotations)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    # Second run: both scenarios are done, no new conv/score calls
    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH) as mock_conv2,
        patch(_SCORE_PATCH) as mock_score2,
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id2 = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    assert run_id2 == run_id
    mock_conv2.assert_not_called()
    mock_score2.assert_not_called()

    # summary.json still written on resume
    run_dir = tmp_path / run_id
    assert (run_dir / "summary.json").exists()


# ---------------------------------------------------------------------------
# Test 3: score error -> logged + skipped; run completes with partial summary
# ---------------------------------------------------------------------------


def test_run_cell_skips_on_score_error(tmp_path):
    """A scenario whose score() raises is logged + skipped; run still completes
    and writes summary.json over the successful scenarios only."""
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    good_scenario = scenarios[0]
    bad_scenario = scenarios[1]

    good_transcript = _make_transcript(good_scenario.id)
    bad_transcript = _make_transcript(bad_scenario.id)
    good_annotation = _make_annotation(good_scenario.id)

    cfg_mock = _make_run_config(sample=2)

    def _score_batch_side_effect(pairs):
        # Pooled semantics: the batch succeeds but yields no annotation for
        # the bad moment (per-entry failure) -- run_cell must fail just it.
        return {s.id: good_annotation for s, _t in pairs if s.id == good_scenario.id}

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=[good_transcript, bad_transcript]),
        patch(_SCORE_PATCH, side_effect=_score_batch_side_effect),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    run_dir = tmp_path / run_id

    # Good scenario: transcript + score written
    assert (run_dir / "transcripts" / f"{good_scenario.id}.json").exists()
    assert (run_dir / "scores" / f"{good_scenario.id}.json").exists()

    # Bad scenario: transcript may or may not exist, score must NOT exist
    assert not (run_dir / "scores" / f"{bad_scenario.id}.json").exists()

    # summary.json written (over the 1 successful scenario)
    assert (run_dir / "summary.json").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    # Only 1 scenario made it to summary
    assert summary["n_scenarios"] == 1
    assert summary["run_counts"] == {
        "attempted": 2,
        "succeeded": 1,
        "failed": 1,
        "resumed": 0,
    }
    assert summary["failed_scenarios"][0]["id"] == bad_scenario.id


def test_run_cell_raises_when_all_scenarios_fail(tmp_path):
    """A run with zero completed scenarios must not write an empty valid summary."""
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    cfg_mock = _make_run_config(sample=2)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=RuntimeError("Scorer exploded!")),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        with pytest.raises(RuntimeError, match="No scenarios completed"):
            run_cell(
                tutor="claude-opus-4-8",
                mode="plain",
                run_cfg=None,
                date="20260626",
                results_root=str(tmp_path),
            )


# ---------------------------------------------------------------------------
# Test 4: cell expansion -- tutors x modes = cells with correct lane assignment
# ---------------------------------------------------------------------------

# An arm-based roster covering all three providers, plus an arm whose key is
# NOT a model id (lane must come from its resolved model's provider).
_ARM_CONFIG_YAML = """
providers:
  anthropic: { env: ANTHROPIC_API_KEY }
  openai:    { env: OPENAI_API_KEY }
  gemini:    { env: GEMINI_API_KEY }

models:
  claude-opus-4-8:   { thinking: dynamic }
  claude-sonnet-4-6: { thinking: dynamic }
  gemini-2.5-pro:    { thinking: dynamic }
  gpt-5.5:           { thinking: high }
  sonnet-high:       { model: claude-sonnet-4-6, thinking: high }

student: { model: claude-haiku-4-5, mode: oracle, thinking: none }
scorer:  { model: claude-opus-4-6, thinking: dynamic }

defaults: { trials: 1, max_turns: 4 }
retry: { max_retries: 1, base_delay: 1 }
batch: { timeout: 60 }
""".strip()


@pytest.fixture
def arm_config(tmp_path, monkeypatch):
    """Point TUTORMOMENTS_CONFIG at a roster of arms for lane-resolution tests."""
    from tutormoments.config import _reset_config_cache

    config_path = tmp_path / "arm_config.yaml"
    config_path.write_text(_ARM_CONFIG_YAML, encoding="utf-8")
    monkeypatch.setenv("TUTORMOMENTS_CONFIG", str(config_path))
    _reset_config_cache()
    yield
    _reset_config_cache()


def test_cell_expansion_and_lane_assignment(arm_config):
    """3 tutors x 2 modes = 6 cells; each cell gets the correct provider lane."""
    from tutormoments.cli import expand_cells

    tutors = ["claude-opus-4-8", "gemini-2.5-pro", "gpt-5.5"]
    modes = ["plain", "scaffolding_rigor"]

    cells = expand_cells(tutors, modes)

    assert len(cells) == 6

    # Check all (tutor, mode) pairs present
    pairs = {(c["tutor"], c["mode"]) for c in cells}
    for t in tutors:
        for m in modes:
            assert (t, m) in pairs, f"Missing cell ({t}, {m})"

    # Check lane assignment by the arm's resolved model's provider
    lane_map = {c["tutor"]: c["lane"] for c in cells}
    assert lane_map["claude-opus-4-8"] == "anthropic"
    assert lane_map["gemini-2.5-pro"] == "gemini"
    assert lane_map["gpt-5.5"] == "openai"


def test_cell_expansion_arm_key_that_is_not_a_model_id(arm_config):
    """An arm named after its condition (not a model id) still lands on the
    lane of its resolved model's provider."""
    from tutormoments.cli import expand_cells

    cells = expand_cells(["sonnet-high"], ["plain"])

    assert cells == [{"tutor": "sonnet-high", "mode": "plain", "lane": "anthropic"}]


# ---------------------------------------------------------------------------
# Test 5: scheduler -- within-lane sequential, lanes can run independently
# ---------------------------------------------------------------------------


def test_scheduler_within_lane_sequential(tmp_path, arm_config):
    """Cells within a lane are called in order; all 6 run_cell calls complete."""
    from tutormoments.cli import expand_cells, run_sweep

    # 2 tutors in the same lane (anthropic), 1 mode each -> 2 cells in 1 lane
    tutors = ["claude-opus-4-8", "claude-sonnet-4-6"]
    modes = ["plain"]

    cells = expand_cells(tutors, modes)
    # Both claude models -> anthropic lane
    assert all(c["lane"] == "anthropic" for c in cells)

    call_order = []

    def fake_run_cell(tutor, mode, run_cfg, *, date, results_root):
        call_order.append((tutor, mode))
        return f"{tutor}_{mode}_run_id"

    run_ids = run_sweep(
        cells=cells,
        run_cfg=MagicMock(),
        date="20260626",
        results_root=str(tmp_path),
        _run_cell_fn=fake_run_cell,
    )

    # All 2 cells produced a run_id
    assert len(run_ids) == 2

    # Within-lane sequential: claude-opus-4-8 before claude-sonnet-4-6 (sweep order)
    tutors_in_order = [t for t, m in call_order]
    assert tutors_in_order == ["claude-opus-4-8", "claude-sonnet-4-6"]


def test_scheduler_multiple_lanes_all_cells_run(tmp_path):
    """3 tutors x 2 modes = 6 cells; all 6 get a run_id regardless of parallelism."""
    from tutormoments.cli import expand_cells, run_sweep

    tutors = ["claude-opus-4-8", "gemini-3.1-pro-preview", "gpt-5.4"]
    modes = ["plain", "scaffolding_rigor"]

    cells = expand_cells(tutors, modes)

    def fake_run_cell(tutor, mode, run_cfg, *, date, results_root):
        return f"{tutor}_{mode}_run_id"

    run_ids = run_sweep(
        cells=cells,
        run_cfg=MagicMock(),
        date="20260626",
        results_root=str(tmp_path),
        _run_cell_fn=fake_run_cell,
    )

    assert len(run_ids) == 6


# ---------------------------------------------------------------------------
# Test 6: --trials N -- conversation+score called N times; summary has mean+spread
# ---------------------------------------------------------------------------


def _make_run_config_trials(
    n_trials: int, sample=2, dataset="test_ds", max_turns=4, replay_concurrency=1
):
    cfg = MagicMock()
    cfg.dataset = dataset
    cfg.data_path = None
    cfg.dataset_revision = None
    cfg.dataset_config = "moments"
    cfg.sample = sample
    cfg.max_turns = max_turns
    cfg.tutors = ["claude-opus-4-8"]
    cfg.modes = ["plain"]
    cfg.trials = n_trials
    cfg.replay_concurrency = replay_concurrency
    cfg.student = StudentSpec(model="claude-haiku", mode="oracle", thinking="dynamic")
    cfg.scorer = ScorerSpec(model="claude-opus-4-6", thinking="dynamic")
    cfg.resolved_tutors = {
        "claude-opus-4-8": ArmSpec(
            name="claude-opus-4-8",
            model="claude-opus-4-8",
            provider="anthropic",
            thinking="dynamic",
        )
    }
    cfg.config_source = "test"
    return cfg


def test_trials_3_calls_conversation_n_times(tmp_path):
    """trials=3: run_conversation is called 3x per scenario (not 1x)."""
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    n_scenarios = len(scenarios)
    n_trials = 3

    # We need n_trials * n_scenarios transcripts and annotations
    transcripts = [_make_transcript(s.id) for s in scenarios for _ in range(n_trials)]
    annotations = [_make_annotation(s.id) for s in scenarios for _ in range(n_trials)]

    cfg_mock = _make_run_config_trials(n_trials=n_trials, sample=n_scenarios)

    conv_mock = MagicMock(side_effect=transcripts)
    score_mock = MagicMock(side_effect=_score_batch_from(annotations))

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, new=conv_mock),
        patch(_SCORE_PATCH, new=score_mock),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    # conversation called n_trials times per scenario;
    # scoring pooled into ONE score_batch call per trial
    assert conv_mock.call_count == n_trials * n_scenarios
    assert score_mock.call_count == n_trials


def test_trials_summary_has_mean_and_spread(tmp_path):
    """trials=3: summary.json has mean and spread keys for numeric metrics."""
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    n_scenarios = len(scenarios)
    n_trials = 3

    transcripts = [_make_transcript(s.id) for s in scenarios for _ in range(n_trials)]
    annotations = [_make_annotation(s.id) for s in scenarios for _ in range(n_trials)]

    cfg_mock = _make_run_config_trials(n_trials=n_trials, sample=n_scenarios)

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(annotations)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # Multi-trial summaries must carry mean and spread at the top level
    assert "trials" in summary, "summary must record number of trials"
    assert summary["trials"] == n_trials
    assert "mean" in summary, "summary must have 'mean' sub-dict for trials>1"
    assert "spread" in summary, "summary must have 'spread' sub-dict for trials>1"
    # The mean block should have the same shape as a single-run summary
    assert "n_scenarios" in summary["mean"]
    assert "outcome_pos_rate" not in summary["mean"]  # dropped from the paper


def test_trials_1_summary_matches_single_run(tmp_path):
    """trials=1: summary.json must be byte-compatible with the Task-4 single-run format
    (no 'mean'/'spread' keys -- it is the plain aggregate dict)."""
    import tutormoments.report as report_mod
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]
    expected_summary = report_mod.aggregate(scenarios, annotations)

    cfg_mock = _make_run_config_trials(n_trials=1, sample=len(scenarios))

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(annotations)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # trials=1: no mean/spread wrapper -- plain aggregate
    assert "mean" not in summary, "trials=1 must NOT have 'mean' key"
    assert "spread" not in summary, "trials=1 must NOT have 'spread' key"
    # Metrics match expected
    assert summary["n_scenarios"] == expected_summary["n_scenarios"]
    assert "outcome_pos_rate" not in summary  # dropped from the paper


# ---------------------------------------------------------------------------
# Helper: build a fake run directory with a summary.json
# ---------------------------------------------------------------------------


def _make_fake_run(
    root: Path,
    run_id: str,
    tutor_model: str,
    mode: str,
    n: int = 10,
    scaffold_cal: float = 0.6,
) -> Path:
    """Write a minimal summary.json inside root/run_id/."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "tutor_model": tutor_model,
        "mode": mode,
        "n_scenarios": n,
        "scaffold_calibrated": {
            "score": scaffold_cal,
            "n_clean_yes": 6,
            "n_total": n,
            "n_overscaffold": 1,
        },
        "rigor_calibrated": {"score": 0.5, "n_clean_yes": 5, "n_total": n},
        "overscaffold": {"rate": 0.1, "n_yes": 1, "n_total": n, "available": True},
        "latency": {
            "tutor": {
                "p50_seconds": 1.0,
                "p95_seconds": 2.5,
                "mean_seconds": 1.2,
                "n": n,
            }
        },
        "tokens": {
            "total": {
                "total_tokens": 100000,
                "input_tokens": 80000,
                "output_tokens": 20000,
            }
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# Test 7: report subcommand writes leaderboard.md + .csv with both rows
# ---------------------------------------------------------------------------


def test_report_writes_leaderboard_md_and_csv(tmp_path):
    """main(['report', '--results-root', ..., '--out', ...]) writes leaderboard.md + .csv
    containing both run rows."""
    from tutormoments.cli import main

    results_root = tmp_path / "results"
    _make_fake_run(
        results_root,
        "model_alpha_plain_ds_20260626",
        "model-alpha",
        "plain",
        scaffold_cal=0.75,
    )
    _make_fake_run(
        results_root,
        "model_beta_plain_ds_20260626",
        "model-beta",
        "plain",
        scaffold_cal=0.50,
    )

    out_stem = str(tmp_path / "leaderboard")

    main(["report", "--results-root", str(results_root), "--out", out_stem])

    md_path = tmp_path / "leaderboard.md"
    csv_path = tmp_path / "leaderboard.csv"

    assert md_path.exists(), "leaderboard.md must be written"
    assert csv_path.exists(), "leaderboard.csv must be written"

    md_text = md_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")

    # Both model names must appear in both files
    assert "model-alpha" in md_text
    assert "model-beta" in md_text
    assert "model-alpha" in csv_text
    assert "model-beta" in csv_text

    # CSV must have header + 2 data rows (+ possible trailing newline)
    csv_lines = [l for l in csv_text.strip().splitlines() if l.strip()]
    assert len(csv_lines) == 3, f"Expected header + 2 rows, got: {csv_lines}"


def _make_probe_run(root: Path, run_id: str, tutor_model: str, mode: str) -> None:
    """Write a minimal `tutormoments latency` result inside root/run_id/."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latency.json").write_text(
        json.dumps(
            {
                "source": "probe",
                "tutor_model": tutor_model,
                "mode": mode,
                "tutor": {
                    "ttft": {"all": {"n": 336, "p50_seconds": 9.043}},
                    "ttlt": {"all": {"n": 336, "p50_seconds": 10.52}},
                },
                "samples": [
                    {"ttft_seconds": 13.217, "turn_index": 0},
                    {"ttft_seconds": 7.9, "turn_index": 1},
                    {"ttft_seconds": 7.993, "turn_index": 1},
                    {"ttft_seconds": 8.1, "turn_index": 2},
                ],
                "subsample": {
                    "subsample_source": "frozen_packaged",
                    "subsample_id": "589e8acf8ac761f2",
                    "subsample_complete": True,
                },
                "measurement_environment": {"measured_at": "2026-08-18T14:21:39"},
            }
        ),
        encoding="utf-8",
    )


def test_report_joins_probe_ttft_onto_the_matching_run(tmp_path):
    """End to end: a probe run sitting beside a benchmark run in the same
    results root fills that cell's TTFT columns, and only that cell's."""
    from tutormoments.cli import main

    results_root = tmp_path / "results"
    _make_fake_run(
        results_root,
        "model-alpha_scaffolding_rigor_ds_20260818",
        "model-alpha",
        "scaffolding_rigor",
    )
    _make_fake_run(
        results_root, "model-alpha_plain_ds_20260818", "model-alpha", "plain"
    )
    _make_probe_run(
        results_root,
        "model-alpha_scaffolding_rigor_latency_20260818",
        "model-alpha",
        "scaffolding_rigor",
    )

    out_stem = str(tmp_path / "leaderboard")
    main(["report", "--results-root", str(results_root), "--out", out_stem])

    rows = {
        line.split("|")[2].strip(): line
        for line in (tmp_path / "leaderboard.md").read_text("utf-8").splitlines()
        if line.startswith("| model-alpha")
    }
    assert "9.043" in rows["scaffolding_rigor"]
    assert "13.217" in rows["scaffolding_rigor"]
    assert "9.043" not in rows["plain"]


def test_report_recovers_the_cell_from_config_not_the_run_id(tmp_path):
    """Runs predating tutor_model/mode in summary.json must still join. The run
    id cannot supply the mode: make_run_id joins fields with underscores that
    occur inside a mode too, so splitting it yields "scaffolding" for
    "scaffolding_rigor" -- which would silently miss the probe."""
    from tutormoments.cli import main

    results_root = tmp_path / "results"
    run_id = "claude-opus-4-8_scaffolding_rigor_tutormoments-preview_20260807"
    _make_fake_run(results_root, run_id, "claude-opus-4-8", "scaffolding_rigor")
    summary_path = results_root / run_id / "summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    del summary["tutor_model"]
    del summary["mode"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (results_root / run_id / "config.json").write_text(
        json.dumps({"tutor": "claude-opus-4-8", "mode": "scaffolding_rigor"}),
        encoding="utf-8",
    )
    _make_probe_run(
        results_root,
        "claude-opus-4-8_scaffolding_rigor_latency_20260818",
        "claude-opus-4-8",
        "scaffolding_rigor",
    )

    out_stem = str(tmp_path / "leaderboard")
    main(["report", "--results-root", str(results_root), "--out", out_stem])

    row = next(
        line
        for line in (tmp_path / "leaderboard.md").read_text("utf-8").splitlines()
        if line.startswith("| claude-opus-4-8")
    )
    assert "| scaffolding_rigor |" in row
    assert "9.043" in row


def test_report_warns_when_probes_mix_subsamples(tmp_path, caplog):
    """Two frozen-but-different samples each pass per-probe eligibility, so the
    table would compare figures measured over different prompts."""
    from tutormoments.cli import main

    results_root = tmp_path / "results"
    _make_fake_run(
        results_root,
        "model-alpha_scaffolding_rigor_ds_20260818",
        "model-alpha",
        "scaffolding_rigor",
    )
    _make_probe_run(
        results_root,
        "model-alpha_scaffolding_rigor_latency_20260818",
        "model-alpha",
        "scaffolding_rigor",
    )
    _make_probe_run(
        results_root,
        "model-beta_scaffolding_rigor_latency_20260817",
        "model-beta",
        "scaffolding_rigor",
    )
    stale = results_root / "model-beta_scaffolding_rigor_latency_20260817"
    block = json.loads((stale / "latency.json").read_text("utf-8"))
    block["subsample"]["subsample_id"] = "84b4ad5615876a3e"
    (stale / "latency.json").write_text(json.dumps(block), encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        main(
            [
                "report",
                "--results-root",
                str(results_root),
                "--out",
                str(tmp_path / "leaderboard"),
            ]
        )

    assert "mix 2 latency subsamples" in caplog.text


# ---------------------------------------------------------------------------
# Test 8: view subcommand writes non-empty HTML
# ---------------------------------------------------------------------------


def test_view_writes_html(tmp_path):
    """main(['view', '--results-root', ..., '--out', ...]) writes a non-empty HTML file."""
    from tutormoments.cli import main

    results_root = tmp_path / "results"
    _make_fake_run(
        results_root, "model_alpha_plain_ds_20260626", "model-alpha", "plain"
    )

    out_html = str(tmp_path / "viewer.html")

    main(["view", "--results-root", str(results_root), "--out", out_html])

    html_path = tmp_path / "viewer.html"
    assert html_path.exists(), "viewer.html must be written"

    html_text = html_path.read_text(encoding="utf-8")
    assert len(html_text) > 100, "HTML must be non-trivially non-empty"
    assert "<!DOCTYPE html>" in html_text or "<html" in html_text


# ---------------------------------------------------------------------------
# Test 10: --help and run --help exit 0 and list subcommands
# ---------------------------------------------------------------------------


def test_main_help_exits_0():
    """main(['--help']) raises SystemExit(0)."""
    from tutormoments.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_run_help_exits_0():
    """main(['run', '--help']) raises SystemExit(0)."""
    from tutormoments.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Test 11: end-to-end smoke -- main(['run', ...]) with mocks
# ---------------------------------------------------------------------------


def test_main_run_smoke(tmp_path):
    """main(['run', '--tutors', 'claude-opus-4-8', '--sample', '1', ...]) with mocked
    run_conversation and score produces a run dir with summary.json."""
    from tutormoments.cli import main

    scenario = _make_scenario("smoke_001", "scaffolding")
    transcript = _make_transcript("smoke_001")
    annotation = _make_annotation("smoke_001")

    cfg_mock = _make_run_config(sample=1, dataset="test_ds")

    with (
        patch(_CFG_PATCH, return_value=cfg_mock) as _mock_cfg,
        patch(_LOAD_PATCH, return_value=_load_result([scenario])),
        patch(_CONV_PATCH, return_value=transcript),
        patch(_SCORE_PATCH, return_value={annotation.scenario_id: annotation}),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
        patch("tutormoments.cli.run_sweep") as mock_sweep,
    ):
        # run_sweep returns a list of run_ids; stub it so we test main() wiring
        mock_sweep.return_value = ["claude-opus-4-8_plain_test_ds_20260626"]
        main(
            [
                "run",
                "--tutors",
                "claude-opus-4-8",
                "--sample",
                "1",
                "--dataset",
                "test_ds",
            ]
        )

    # Verify run_sweep was called (proving main() wired through correctly)
    mock_sweep.assert_called_once()


def test_main_run_missing_dataset_exits_cleanly(capsys):
    """DatasetNotFoundError should be reported without a Python traceback."""
    from tutormoments.cli import main
    from tutormoments.moments import DatasetNotFoundError

    cfg_mock = _make_run_config(sample=1, dataset="missing_ds")

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(
            "tutormoments.cli.run_sweep",
            side_effect=DatasetNotFoundError("missing dataset"),
        ),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "run",
                    "--tutors",
                    "claude-opus-4-8",
                    "--sample",
                    "1",
                    "--dataset",
                    "missing_ds",
                ]
            )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Error: missing dataset" in captured.err
    assert "Traceback" not in captured.err


def test_run_config_argument_becomes_active_config_for_late_resolution(
    tmp_path, monkeypatch
):
    """`tutormoments run --config FILE` also affects later no-arg config lookups."""
    from tutormoments.cli import main
    from tutormoments.config import _reset_config_cache

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  openai: { env: OPENAI_API_KEY }

models:
  gpt-5.5: { thinking: high }

student: { model: claude-haiku-4-5, mode: oracle, thinking: none }
scorer: { model: gpt-5.5, thinking: high }

defaults: { trials: 1, max_turns: 1 }
retry: { max_retries: 1, base_delay: 1 }
batch: { timeout: 60 }
""".strip(),
        encoding="utf-8",
    )

    observed = {}

    def fake_run_sweep(cells, run_cfg, *, date, results_root):
        from tutormoments.config import resolve_arm, scorer_spec, student_spec

        observed["provider"] = resolve_arm("gpt-5.5").provider
        observed["student"] = student_spec().model
        observed["scorer"] = scorer_spec().model
        observed["run_cfg_source"] = run_cfg.config_source
        return ["fake_run"]

    monkeypatch.setattr("tutormoments.cli.run_sweep", fake_run_sweep)
    original_config = os.environ.pop("TUTORMOMENTS_CONFIG", None)
    _reset_config_cache()

    try:
        main(
            [
                "run",
                "--config",
                str(config_path),
                "--tutors",
                "gpt-5.5",
                "--dataset",
                "readme_mock",
            ]
        )

        assert observed == {
            "provider": "openai",
            "student": "claude-haiku-4-5",
            "scorer": "gpt-5.5",
            "run_cfg_source": str(config_path),
        }
        assert os.environ["TUTORMOMENTS_CONFIG"] == str(config_path)
    finally:
        if original_config is None:
            os.environ.pop("TUTORMOMENTS_CONFIG", None)
        else:
            os.environ["TUTORMOMENTS_CONFIG"] = original_config
        _reset_config_cache()


# ---------------------------------------------------------------------------
# Pooled scoring: run_cell scores each trial's moments in ONE score_batch call
# ---------------------------------------------------------------------------

_SCORE_BATCH_PATCH = "tutormoments.cli.scoring.score_batch"


def test_run_cell_pools_scoring_into_one_call(tmp_path):
    """run_cell calls scoring.score_batch once with all completed pairs."""
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = {s.id: _make_transcript(s.id) for s in scenarios}
    annotations = {s.id: _make_annotation(s.id) for s in scenarios}

    cfg_mock = _make_run_config(sample=2)
    batch_calls = []

    def fake_score_batch(pairs):
        batch_calls.append([s.id for s, _ in pairs])
        return {s.id: annotations[s.id] for s, _ in pairs}

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=lambda s, **kw: transcripts[s.id]),
        patch(_SCORE_BATCH_PATCH, side_effect=fake_score_batch),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260701",
            results_root=str(tmp_path),
        )

    assert batch_calls == [[s.id for s in scenarios]], (
        f"Expected ONE pooled score_batch call with both moments, got {batch_calls}"
    )
    # Per-moment score files still written (resume granularity unchanged)
    for s in scenarios:
        assert (tmp_path / run_id / "scores" / f"{s.id}.json").exists()
    summary = json.loads((tmp_path / run_id / "summary.json").read_text())
    assert summary["n_scenarios"] == 2
    assert summary["run_counts"]["succeeded"] == 2


def test_run_cell_transcript_without_score_resumes_into_pool(tmp_path):
    """A moment with a transcript on disk but no score skips its conversation
    and is scored via the pool (the interrupted-run resume case)."""
    from tutormoments import results as res
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    done_conv, needs_score = scenarios[0], scenarios[1]
    annotations = {s.id: _make_annotation(s.id) for s in scenarios}

    cfg_mock = _make_run_config(sample=2)
    run_id = res.make_run_id("claude-opus-4-8", "plain", "test_ds", "20260701")

    # Pre-write ONLY the transcript for needs_score (simulates an interrupt
    # between transcript write and score write).
    pre_tx = {
        "scenario_id": needs_score.id,
        "tutor_model": "claude-opus-4-8",
        "generated_turns": [{"turn_number": 2, "role": "TUTOR", "text": "Hi"}],
        "completed": True,
        "ended_via": "MAX_TURNS",
    }
    res.write_transcript(run_id, needs_score.id, pre_tx, results_root=str(tmp_path))

    conv_calls = []
    batch_calls = []

    def fake_conv(s, **kw):
        conv_calls.append(s.id)
        return _make_transcript(s.id)

    def fake_score_batch(pairs):
        batch_calls.append([(s.id, t.generated_turns) for s, t in pairs])
        return {s.id: annotations[s.id] for s, _ in pairs}

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=fake_conv),
        patch(_SCORE_BATCH_PATCH, side_effect=fake_score_batch),
    ):
        out_run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260701",
            results_root=str(tmp_path),
        )

    assert out_run_id == run_id
    # Conversation ran only for the fresh moment
    assert conv_calls == [done_conv.id]
    # Both moments were scored in the single pooled call
    assert len(batch_calls) == 1
    scored_ids = [sid for sid, _ in batch_calls[0]]
    assert set(scored_ids) == {done_conv.id, needs_score.id}
    # The pre-written transcript's turns reached the scorer (from disk)
    turns_by_id = dict(batch_calls[0])
    assert turns_by_id[needs_score.id] == pre_tx["generated_turns"]
    # And its score file now exists
    assert (tmp_path / run_id / "scores" / f"{needs_score.id}.json").exists()
    summary = json.loads((tmp_path / run_id / "summary.json").read_text())
    assert summary["run_counts"]["succeeded"] == 2


# ---------------------------------------------------------------------------
# Taxonomy integration: the always-on hook folds into summary.json
# ---------------------------------------------------------------------------


def test_run_cell_folds_taxonomy_block_into_summary(tmp_path):
    """The always-on taxonomy hook's result lands in summary.json, and its
    token usage is added into tokens.total."""
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]
    cfg_mock = _make_run_config(sample=2)

    tax_result = {
        "scheme_version": "lm_extended_v1",
        "counts": {"A": 3, "E": 1},
        "n_facets": 4,
        "excluded": 1,
        "usage": {"input_tokens": 700, "output_tokens": 300, "total_tokens": 1000},
    }
    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(annotations)),
        patch(_TAX_PATCH, return_value=tax_result) as mock_tax,
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    assert mock_tax.called
    summary = json.loads(
        (tmp_path / run_id / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["taxonomy"] == tax_result
    # taxonomy usage folded into the token bookkeeping
    assert summary["tokens"]["taxonomy"] == tax_result["usage"]
    assert summary["tokens"]["total"]["total_tokens"] >= 1000


# ---------------------------------------------------------------------------
# Arm identity: an arm keyed by a condition name, not a model id
# ---------------------------------------------------------------------------


def test_run_cell_keys_results_by_arm_name_not_model_id(tmp_path):
    """For an arm whose key is not a model id (sonnet-high -> claude-sonnet-4-6):
    the run id is keyed by the ARM name, config.json records arm == tutor ==
    the arm name with model == the resolved id, and the summary identity block
    carries both."""
    import tutormoments.results as results_mod
    from tutormoments.cli import run_cell

    scenarios = list(FIXTURE_SCENARIOS)
    transcripts = [_make_transcript(s.id) for s in scenarios]
    annotations = [_make_annotation(s.id) for s in scenarios]

    cfg_mock = _make_run_config(
        sample=2, arm="sonnet-high", model="claude-sonnet-4-6", thinking="high"
    )

    with (
        patch(_CFG_PATCH, return_value=cfg_mock),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=transcripts),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(annotations)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
        patch(
            "tutormoments.cli.results.make_run_id", wraps=results_mod.make_run_id
        ) as mock_run_id,
    ):
        run_id = run_cell(
            tutor="sonnet-high",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    # make_run_id received the ARM name, not the resolved model id
    mock_run_id.assert_called_once_with("sonnet-high", "plain", "test_ds", "20260626")
    assert "sonnet-high" in run_id
    assert "claude-sonnet-4-6" not in run_id

    run_dir = tmp_path / run_id
    cfg_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg_data["arm"] == "sonnet-high"
    assert cfg_data["tutor"] == "sonnet-high"
    assert cfg_data["model"] == "claude-sonnet-4-6"
    assert cfg_data["resolved_tutors"]["sonnet-high"] == {
        "name": "sonnet-high",
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "thinking": "high",
    }

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["tutor_model"] == "sonnet-high"
    assert summary["model"] == "claude-sonnet-4-6"
    assert summary["mode"] == "plain"


# ---------------------------------------------------------------------------
# Replay concurrency (result-preserving parallel replay of moments)
# ---------------------------------------------------------------------------


def _conv_by_sid(scenarios, delays=None, fail_ids=()):
    """run_conversation fake that returns the right transcript for its scenario.

    Keyed by scenario, so it is robust to the nondeterministic call order that
    threads introduce (a list side_effect would mis-assign under concurrency).
    Optional per-sid delays shuffle completion order; fail_ids raise.
    """
    delays = delays or {}
    tmap = {s.id: _make_transcript(s.id) for s in scenarios}

    def _fake(scenario, **kwargs):
        d = delays.get(scenario.id, 0)
        if d:
            time.sleep(d)
        if scenario.id in fail_ids:
            raise RuntimeError(f"boom {scenario.id}")
        return tmap[scenario.id]

    return _fake


def test_replay_concurrency_matches_serial(tmp_path):
    """Concurrency=4 yields byte-identical summary + file set to concurrency=1,
    even when completion order is shuffled (reverse per-sid delays)."""
    from tutormoments.cli import run_cell

    scenarios = [
        _make_scenario(f"scenario_{i:03d}", "scaffolding") for i in range(1, 6)
    ]
    # Reverse delays: last-submitted finishes first under concurrency, so
    # completion order != submission order -- exercises index-order reassembly.
    delays = {s.id: 0.02 * (len(scenarios) - i) for i, s in enumerate(scenarios)}

    def _run(root, concurrency):
        cfg = _make_run_config(sample=len(scenarios), replay_concurrency=concurrency)
        anns = [_make_annotation(s.id) for s in scenarios]
        with (
            patch(_CFG_PATCH, return_value=cfg),
            patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
            patch(_CONV_PATCH, side_effect=_conv_by_sid(scenarios, delays)),
            patch(_SCORE_PATCH, side_effect=_score_batch_from(anns)),
            patch(_TAX_PATCH, return_value=_TAX_RESULT),
        ):
            return run_cell(
                tutor="claude-opus-4-8",
                mode="plain",
                run_cfg=None,
                date="20260626",
                results_root=str(root),
            )

    root_serial = tmp_path / "serial"
    root_conc = tmp_path / "concurrent"
    rid_serial = _run(root_serial, 1)
    rid_conc = _run(root_conc, 4)

    assert rid_serial == rid_conc  # run_id is concurrency-independent

    def _summary(root, rid):
        return json.loads((root / rid / "summary.json").read_text(encoding="utf-8"))

    def _files(root, rid):
        base = root / rid
        return sorted(
            p.relative_to(base).as_posix()
            for sub in ("transcripts", "scores")
            for p in (base / sub).glob("*.json")
        )

    # Scores must be byte-identical across concurrency settings. The latency
    # block is deliberately excluded: concurrency genuinely does change
    # latency (shared decode batches, edge queueing near rate limits), which
    # is why the block records the concurrency it was gathered at and why the
    # reportable figure comes from `tutormoments latency` instead.
    serial_summary = _summary(root_serial, rid_serial)
    conc_summary = _summary(root_conc, rid_conc)
    assert serial_summary.pop("latency")["concurrency"] == 1
    assert conc_summary.pop("latency")["concurrency"] == 4
    assert serial_summary == conc_summary
    assert _files(root_serial, rid_serial) == _files(root_conc, rid_conc)


def test_replay_concurrency_error_isolation(tmp_path):
    """Under concurrency, a single failing replay is isolated: it is counted as
    failed with phase 'run', no score file is written for it, and every other
    moment still succeeds."""
    from tutormoments.cli import run_cell

    scenarios = [
        _make_scenario(f"scenario_{i:03d}", "scaffolding") for i in range(1, 5)
    ]
    bad = scenarios[1]
    good = [s for s in scenarios if s.id != bad.id]

    cfg = _make_run_config(sample=len(scenarios), replay_concurrency=4)
    anns = [_make_annotation(s.id) for s in good]

    with (
        patch(_CFG_PATCH, return_value=cfg),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=_conv_by_sid(scenarios, fail_ids={bad.id})),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(anns)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["run_counts"] == {
        "attempted": len(scenarios),
        "succeeded": len(good),
        "failed": 1,
        "resumed": 0,
    }
    failed = summary["failed_scenarios"]
    assert len(failed) == 1
    assert failed[0]["id"] == bad.id
    assert failed[0]["phase"] == "run"

    # Bad moment: no score written. Every good moment: transcript + score written.
    assert not (run_dir / "scores" / f"{bad.id}.json").exists()
    for s in good:
        assert (run_dir / "transcripts" / f"{s.id}.json").exists()
        assert (run_dir / "scores" / f"{s.id}.json").exists()


def test_replay_concurrency_resume(tmp_path):
    """With concurrency>1, resume still only replays never-run moments and counts
    the already-done ones as resumed."""
    from tutormoments.cli import run_cell

    scenarios = [
        _make_scenario(f"scenario_{i:03d}", "scaffolding") for i in range(1, 4)
    ]

    # First run: complete only the first moment (sample=1).
    cfg1 = _make_run_config(sample=1, replay_concurrency=4)
    with (
        patch(_CFG_PATCH, return_value=cfg1),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios[:1])),
        patch(_CONV_PATCH, side_effect=_conv_by_sid(scenarios[:1])),
        patch(
            _SCORE_PATCH,
            side_effect=_score_batch_from([_make_annotation(scenarios[0].id)]),
        ),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    # Second run: all 3 moments; only the 2 new ones should be replayed.
    called_ids = []

    def _tracking_conv(scenario, **kwargs):
        called_ids.append(scenario.id)
        return _make_transcript(scenario.id)

    cfg2 = _make_run_config(sample=3, replay_concurrency=4)
    with (
        patch(_CFG_PATCH, return_value=cfg2),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=_tracking_conv),
        patch(
            _SCORE_PATCH,
            side_effect=_score_batch_from(
                [_make_annotation(s.id) for s in scenarios[1:]]
            ),
        ),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id2 = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    assert run_id2 == run_id
    assert sorted(called_ids) == [scenarios[1].id, scenarios[2].id]

    summary = json.loads(
        (tmp_path / run_id2 / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["run_counts"]["resumed"] == 1
    assert summary["run_counts"]["succeeded"] == 3
    assert summary["run_counts"]["failed"] == 0


def test_replay_concurrency_transcripts_survive_interrupt(tmp_path):
    """Transcripts are written as each replay completes (main thread, in
    completion order): if the run is killed mid-pool, every already-finished
    conversation is on disk for resume instead of being lost."""
    from tutormoments.cli import run_cell

    scenarios = [
        _make_scenario(f"scenario_{i:03d}", "scaffolding") for i in range(1, 3)
    ]
    fast, doomed = scenarios

    def _conv(scenario, **kwargs):
        if scenario.id == doomed.id:
            time.sleep(0.5)  # let the fast moment complete + persist first
            raise KeyboardInterrupt()
        return _make_transcript(scenario.id)

    cfg = _make_run_config(sample=len(scenarios), replay_concurrency=2)
    with (
        patch(_CFG_PATCH, return_value=cfg),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=_conv),
        patch(_SCORE_PATCH, side_effect=AssertionError("scoring must not run")),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
        pytest.raises(KeyboardInterrupt),
    ):
        run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    # The completed moment was persisted before the interrupt; the doomed one
    # never produced a transcript.
    written = sorted(p.stem for p in tmp_path.glob("*/transcripts/*.json"))
    assert written == [fast.id]


def test_replay_concurrency_worker_logs_reach_run_log(tmp_path):
    """Records logged inside run_conversation on pool worker threads (e.g.
    client retry warnings) are captured in results/<run_id>/run.log."""
    from tutormoments.cli import run_cell

    scenarios = [
        _make_scenario(f"scenario_{i:03d}", "scaffolding") for i in range(1, 4)
    ]

    def _conv(scenario, **kwargs):
        logging.getLogger("tutormoments.client").warning(
            "simulated retry for %s", scenario.id
        )
        return _make_transcript(scenario.id)

    cfg = _make_run_config(sample=len(scenarios), replay_concurrency=3)
    anns = [_make_annotation(s.id) for s in scenarios]
    with (
        patch(_CFG_PATCH, return_value=cfg),
        patch(_LOAD_PATCH, return_value=_load_result(scenarios)),
        patch(_CONV_PATCH, side_effect=_conv),
        patch(_SCORE_PATCH, side_effect=_score_batch_from(anns)),
        patch(_TAX_PATCH, return_value=_TAX_RESULT),
    ):
        run_id = run_cell(
            tutor="claude-opus-4-8",
            mode="plain",
            run_cfg=None,
            date="20260626",
            results_root=str(tmp_path),
        )

    content = (tmp_path / run_id / "run.log").read_text(encoding="utf-8")
    for s in scenarios:
        assert f"simulated retry for {s.id}" in content
