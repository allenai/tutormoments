"""Tests for tutormoments.results — the run-results store (TDD: write tests first)."""

from tutormoments.results import (
    is_done,
    list_runs,
    make_run_id,
    read_config,
    read_score,
    read_summary,
    read_transcript,
    write_config,
    write_score,
    write_summary,
    write_transcript,
)

# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------


def test_make_run_id_basic():
    result = make_run_id("claude-opus-4-8", "plain", "balanced_520", "20260626")
    assert result == "claude-opus-4-8_plain_balanced_520_20260626"


def test_make_run_id_slash_replaced():
    result = make_run_id(
        "deepseek-ai/DeepSeek-V4-Pro", "plain", "balanced_520", "20260626"
    )
    assert "/" not in result
    assert result == "deepseek-ai_DeepSeek-V4-Pro_plain_balanced_520_20260626"


def test_make_run_id_multiple_slashes():
    result = make_run_id("org/sub/model", "cot", "ds100", "20260101")
    assert result == "org_sub_model_cot_ds100_20260101"


# ---------------------------------------------------------------------------
# config round-trip
# ---------------------------------------------------------------------------


def test_write_read_config(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    cfg = {"tutor": "claude", "mode": "plain", "dataset": "ds"}
    write_config(run_id, cfg, results_root=str(tmp_path))
    loaded = read_config(run_id, results_root=str(tmp_path))
    assert loaded == cfg


def test_read_config_missing_returns_none(tmp_path):
    assert read_config("nonexistent_run", results_root=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# transcript round-trip
# ---------------------------------------------------------------------------


def test_write_read_transcript(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    scenario_id = "conv-abc_m1"
    transcript = {
        "scenario_id": scenario_id,
        "turns": [{"role": "tutor", "text": "Hello"}],
    }
    write_transcript(run_id, scenario_id, transcript, results_root=str(tmp_path))
    loaded = read_transcript(run_id, scenario_id, results_root=str(tmp_path))
    assert loaded == transcript


def test_read_transcript_missing_returns_none(tmp_path):
    assert read_transcript("run_x", "s1", results_root=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# score round-trip
# ---------------------------------------------------------------------------


def test_write_read_score(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    scenario_id = "conv-abc_m1"
    annotation = {"scaffolding": "effective", "rapport": "partial"}
    write_score(run_id, scenario_id, annotation, results_root=str(tmp_path))
    loaded = read_score(run_id, scenario_id, results_root=str(tmp_path))
    assert loaded == annotation


def test_read_score_missing_returns_none(tmp_path):
    assert read_score("run_x", "s1", results_root=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# summary round-trip
# ---------------------------------------------------------------------------


def test_write_read_summary(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    summary = {"n_scenarios": 10, "scaffolding_rate": 0.8}
    write_summary(run_id, summary, results_root=str(tmp_path))
    loaded = read_summary(run_id, results_root=str(tmp_path))
    assert loaded == summary


def test_read_summary_missing_returns_none(tmp_path):
    assert read_summary("run_x", results_root=str(tmp_path)) is None


# ---------------------------------------------------------------------------
# is_done (resume guard)
# ---------------------------------------------------------------------------


def test_is_done_false_when_neither_exists(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    assert is_done(run_id, "conv-abc_m1", results_root=str(tmp_path)) is False


def test_is_done_false_when_only_transcript(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    scenario_id = "conv-abc_m1"
    transcript = {"scenario_id": scenario_id, "completed": True}
    write_transcript(run_id, scenario_id, transcript, results_root=str(tmp_path))
    assert is_done(run_id, scenario_id, results_root=str(tmp_path)) is False


def test_is_done_false_when_only_score(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    scenario_id = "conv-abc_m1"
    write_score(run_id, scenario_id, {"score": 1.0}, results_root=str(tmp_path))
    assert is_done(run_id, scenario_id, results_root=str(tmp_path)) is False


def test_is_done_true_when_both_exist(tmp_path):
    run_id = "testrun_plain_ds_20260626"
    scenario_id = "conv-abc_m1"
    write_transcript(
        run_id,
        scenario_id,
        {"scenario_id": scenario_id, "completed": True},
        results_root=str(tmp_path),
    )
    write_score(run_id, scenario_id, {"score": 1.0}, results_root=str(tmp_path))
    assert is_done(run_id, scenario_id, results_root=str(tmp_path)) is True


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_empty(tmp_path):
    assert list_runs(results_root=str(tmp_path)) == []


def test_list_runs_returns_run_ids(tmp_path):
    run_ids = ["run_a_plain_ds_20260601", "run_b_cot_ds_20260602"]
    for rid in run_ids:
        write_config(rid, {"key": "val"}, results_root=str(tmp_path))
    found = list_runs(results_root=str(tmp_path))
    assert set(found) == set(run_ids)


def test_list_runs_ignores_non_directories(tmp_path):
    # A stray file in results_root should not appear as a run_id
    (tmp_path / "stray_file.txt").write_text("x", encoding="utf-8")
    write_config("real_run_plain_ds_20260601", {"k": "v"}, results_root=str(tmp_path))
    found = list_runs(results_root=str(tmp_path))
    assert found == ["real_run_plain_ds_20260601"]
