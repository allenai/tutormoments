"""Unit tests for the TTFT half of website/scripts/refresh-data.py.

The script previously carried its own copy of the probe's publishability rules
and got them wrong -- it gated on cache hit *rate*, which the runtime refuses to
do, and would have published a "warm" figure for a provider whose hits read back
one shared block of system prompt. These tests pin the corrected behaviour:
figures come from `tutormoments.latency`, the site's model ids are matched to the
probe's, and a stale figure is never left standing next to a fresh one.

Score and end-to-end-latency assembly is thin I/O over an analysis export and is
covered by tests/analysis/test_benchmark_perf_cost.py.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "website" / "scripts" / "refresh-data.py"


@pytest.fixture(scope="module")
def refresh():
    """Load the script by path: its filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location("refresh_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe_dir(
    root: Path,
    run_id: str,
    *,
    tutor: str,
    mode: str = "scaffolding_rigor",
    p50_all: float = 9.0,
    first: tuple = (12.0, 12.5, 13.0),
    later: tuple = (7.5, 8.0, 8.5),
    measured_at: str = "2026-08-18T10:00:00",
    sub_id: str = "589e8acf8ac761f2",
    usage: dict | None = None,
) -> None:
    samples = [{"ttft_seconds": v, "turn_index": 0} for v in first]
    samples += [
        {"ttft_seconds": v, "turn_index": 1 + i % 2} for i, v in enumerate(later)
    ]
    if usage is not None:
        # A probe written after per-call usage capture.
        samples = [{**s, "usage": usage} for s in samples]
    run = root / run_id
    run.mkdir(parents=True)
    (run / "latency.json").write_text(
        json.dumps(
            {
                "source": "probe",
                "tutor_model": tutor,
                "mode": mode,
                "tutor": {
                    "n_samples": 336,
                    "ttft": {"all": {"n": 336, "p50_seconds": p50_all}},
                    "ttlt": {"all": {"n": 336, "p50_seconds": p50_all + 1}},
                },
                "samples": samples,
                "subsample": {
                    "subsample_source": "frozen_packaged",
                    "subsample_id": sub_id,
                    "subsample_complete": True,
                },
                "measurement_environment": {"measured_at": measured_at},
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# probe_ttft
# ---------------------------------------------------------------------------


def test_probe_ttft_reads_pooled_p50_and_the_split(refresh, tmp_path):
    _probe_dir(
        tmp_path, "a_scaffolding_rigor_latency_20260818", tutor="claude-opus-4-8"
    )
    figures, provenance = refresh.probe_ttft(REPO, tmp_path, "scaffolding_rigor")
    assert figures["claude-opus-4-8"] == {
        "ttft_s": 9.0,
        "ttlt_s": 10.0,
        "ttft_first_s": 12.5,
        "ttft_later_s": 8.0,
    }
    assert provenance["subsample_id"] == "589e8acf8ac761f2"
    assert provenance["measured_at"]["claude-opus-4-8"] == "2026-08-18T10:00:00"


def test_probe_ttft_flattens_the_provider_slash_to_the_site_id(refresh, tmp_path):
    """Site ids mirror result-directory names, not the model id the probe
    records: deepseek-ai/DeepSeek-V4-Pro -> deepseek-ai_DeepSeek-V4-Pro."""
    _probe_dir(
        tmp_path,
        "ds_scaffolding_rigor_latency_20260818",
        tutor="deepseek-ai/DeepSeek-V4-Pro",
    )
    figures, _ = refresh.probe_ttft(REPO, tmp_path, "scaffolding_rigor")
    assert "deepseek-ai_DeepSeek-V4-Pro" in figures


def test_probe_ttft_splits_by_turn_regardless_of_cache_reporting(refresh, tmp_path):
    """The split keys on turn position, which the probe recorded itself -- so
    Gemini (no cache tokens) and DeepSeek (automatic prefix cache whose labels
    the runtime rejects) get first/later figures like everyone else."""
    _probe_dir(
        tmp_path,
        "gem_scaffolding_rigor_latency_20260818",
        tutor="gemini-2.5-pro",
        p50_all=14.08,
        first=(14.6, 14.7, 14.8),
        later=(13.6, 13.7, 13.8),
    )
    figures, _ = refresh.probe_ttft(REPO, tmp_path, "scaffolding_rigor")
    assert figures["gemini-2.5-pro"] == {
        "ttft_s": 14.08,
        "ttlt_s": 15.08,
        "ttft_first_s": 14.7,
        "ttft_later_s": 13.7,
    }


def test_probe_ttft_omits_the_split_when_a_probe_has_no_samples(refresh, tmp_path):
    """Pooled comes from the stored aggregate; the split needs samples. An
    empty samples list yields a pooled-only row rather than a crash."""
    _probe_dir(
        tmp_path,
        "gem_scaffolding_rigor_latency_20260818",
        tutor="gemini-2.5-pro",
        p50_all=14.94,
        first=(),
        later=(),
    )
    figures, _ = refresh.probe_ttft(REPO, tmp_path, "scaffolding_rigor")
    assert figures["gemini-2.5-pro"] == {"ttft_s": 14.94, "ttlt_s": 15.94}


def test_probe_ttft_ignores_other_prompt_modes(refresh, tmp_path):
    _probe_dir(
        tmp_path, "a_plain_latency_20260818", tutor="claude-opus-4-8", mode="plain"
    )
    figures, _ = refresh.probe_ttft(REPO, tmp_path, "scaffolding_rigor")
    assert figures == {}


def test_probe_ttft_flags_a_mixed_subsample(refresh, tmp_path, capsys):
    """Two frozen-but-different samples each pass per-probe eligibility; the
    chart must not silently plot them on one axis."""
    _probe_dir(
        tmp_path,
        "a_scaffolding_rigor_latency_20260818",
        tutor="claude-opus-4-8",
        sub_id="589e8acf8ac761f2",
    )
    _probe_dir(
        tmp_path,
        "b_scaffolding_rigor_latency_20260817",
        tutor="claude-sonnet-4-6",
        sub_id="84b4ad5615876a3e",
    )
    _, provenance = refresh.probe_ttft(REPO, tmp_path, "scaffolding_rigor")
    assert provenance["subsample_id"] is None
    assert "mix 2 latency subsamples" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# apply_ttft / refresh_ttft_only
# ---------------------------------------------------------------------------


def test_apply_ttft_drops_a_stale_figure(refresh):
    """A model that lost its probe run must not keep the number from a sample
    that is no longer being measured -- that is the drift subsample_id exists
    to prevent."""
    rows = [
        {"id": "kept", "latency_s": 10.0, "ttft_s": 99.0, "ttft_warm_s": 98.0},
        {"id": "gone", "latency_s": 7.0, "ttft_s": 99.0, "ttft_first_s": 99.5},
    ]
    assert refresh.apply_ttft(rows, {"kept": {"ttft_s": 9.4}}) == 1
    assert rows[0] == {"id": "kept", "latency_s": 10.0, "ttft_s": 9.4}
    assert rows[1] == {"id": "gone", "latency_s": 7.0}


def test_refresh_ttft_only_keeps_the_scores_it_did_not_measure(
    refresh, tmp_path, monkeypatch
):
    """A checkout can have probe runs without a full scored sweep. Rebuilding
    latency.json wholesale there would discard the paper's scores."""
    out = tmp_path / "data"
    out.mkdir()
    (out / "latency.json").write_text(
        json.dumps(
            {
                "source": "Figure 7, paper",
                "models": [
                    {
                        "id": "claude-opus-4-8",
                        "name": "Claude Opus 4.8",
                        "latency_s": 12.5,
                        "latency_estimated": True,
                        "score": 0.8445,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(refresh, "OUT_DIR", out)
    probes = tmp_path / "results"
    _probe_dir(probes, "a_scaffolding_rigor_latency_20260818", tutor="claude-opus-4-8")

    refresh.refresh_ttft_only(REPO, probes)

    payload = json.loads((out / "latency.json").read_text("utf-8"))
    row = payload["models"][0]
    assert row["score"] == 0.8445
    assert row["latency_s"] == 12.5
    assert row["ttft_s"] == 9.0
    assert payload["source"] == "Figure 7, paper"
    assert payload["ttft"]["subsample_id"] == "589e8acf8ac761f2"


def test_refresh_ttft_only_without_probe_runs_leaves_the_file_alone(
    refresh, tmp_path, monkeypatch
):
    out = tmp_path / "data"
    out.mkdir()
    original = {"source": "Figure 7, paper", "models": [{"id": "x", "ttft_s": 1.0}]}
    (out / "latency.json").write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(refresh, "OUT_DIR", out)

    refresh.refresh_ttft_only(REPO, tmp_path / "empty")

    assert json.loads((out / "latency.json").read_text("utf-8")) == original


# ---------------------------------------------------------------------------
# measured_cost / cost_rows / write_cost
# ---------------------------------------------------------------------------


def _usage(model="gpt-5.4-mini-2026-03-17", uncached=500, cache_read=6000, output=400):
    total = uncached + cache_read + output
    return {
        "input_tokens": uncached + cache_read,
        "output_tokens": output,
        "total_tokens": total,
        "input_uncached": uncached,
        "cache_read": cache_read,
        "cache_write": 0,
        "output": output,
        "reasoning": 0,
        "total": total,
        "provider": "openai",
        "model": model,
        "endpoint": "stream",
    }


def _bench_run(
    root: Path,
    run_id: str,
    *,
    tutor: str,
    mode: str = "scaffolding_rigor",
    n_calls: int = 1560,
    output: int = 400,
) -> None:
    """A benchmark run directory: summary.json + config.json, as cli.py writes."""
    usage = _usage(
        model=tutor,
        uncached=500 * n_calls,
        cache_read=6000 * n_calls,
        output=output * n_calls,
    )
    run = root / run_id
    run.mkdir(parents=True)
    summary = {
        "tutor_model": tutor,
        "mode": mode,
        "run_counts": {"attempted": 520, "succeeded": 520, "failed": 0},
        "latency": {"source": "run", "tutor": {"n": n_calls}},
        "tokens": {"tutor": usage},
        "cost": {"n_conversations": 520},
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"sample": None}), encoding="utf-8")


def test_measured_cost_prices_a_probe_at_current_rates(refresh, tmp_path):
    from tutormoments.models import get_pricing, pricing_version

    _probe_dir(
        tmp_path,
        "mini_scaffolding_rigor_latency_20261001",
        tutor="gpt-5.4-mini-2026-03-17",
        usage=_usage(),
    )
    figures, provenance = refresh.measured_cost(REPO, tmp_path, "scaffolding_rigor")
    rates = get_pricing("gpt-5.4-mini-2026-03-17")
    row = figures["gpt-5.4-mini-2026-03-17"]
    # Cached prompt tokens are priced at the full input rate: uncached cost.
    expected = (6500 * rates["input"] + 400 * rates["output"]) / 1_000_000
    assert row["uncached_cost_per_response_usd"] == pytest.approx(expected)
    assert row["prompt_tokens_per_response"] == 6500
    assert row["output_tokens_per_response"] == 400
    assert row["n_calls"] == 6
    assert row["rates"] == {
        "input_per_mtok": rates["input"],
        "output_per_mtok": rates["output"],
        "as_of": rates["as_of"],
    }
    assert row["source"]["kind"] == "probe"
    assert row["source"]["subsample_id"] == "589e8acf8ac761f2"
    assert provenance["pricing_version"] == pricing_version()


def test_measured_cost_reads_a_full_benchmark_run(refresh, tmp_path):
    """No probe needed: a run since usage capture carries tokens and calls."""
    _bench_run(
        tmp_path,
        "gpt-5.5-2026-04-23_scaffolding_rigor_tutormoments-preview_20260929",
        tutor="gpt-5.5-2026-04-23",
    )
    figures, _ = refresh.measured_cost(REPO, tmp_path, "scaffolding_rigor")
    row = figures["gpt-5.5-2026-04-23"]
    assert row["n_calls"] == 1560
    assert row["prompt_tokens_per_response"] == 6500
    assert row["source"] == {
        "kind": "run",
        "run_id": "gpt-5.5-2026-04-23_scaffolding_rigor_tutormoments-preview_20260929",
        "n_conversations": 520,
    }


def test_measured_cost_prefers_the_run_over_a_probe(refresh, tmp_path):
    """The run covers every moment; the probe only the 112-moment subsample."""
    _bench_run(
        tmp_path,
        "mini_scaffolding_rigor_tutormoments-preview_20260929",
        tutor="gpt-5.4-mini-2026-03-17",
        output=100,
    )
    _probe_dir(
        tmp_path,
        "mini_scaffolding_rigor_latency_20261001",
        tutor="gpt-5.4-mini-2026-03-17",
        usage=_usage(),
    )
    figures, _ = refresh.measured_cost(REPO, tmp_path, "scaffolding_rigor")
    row = figures["gpt-5.4-mini-2026-03-17"]
    assert row["source"]["kind"] == "run"
    assert row["output_tokens_per_response"] == 100


def test_measured_cost_skips_a_probe_without_usage(refresh, tmp_path):
    """The August probes recorded timings only: not measured, never 0."""
    _probe_dir(
        tmp_path, "a_scaffolding_rigor_latency_20260818", tutor="claude-opus-4-8"
    )
    figures, _ = refresh.measured_cost(REPO, tmp_path, "scaffolding_rigor")
    assert figures == {}


def test_measured_cost_ignores_other_prompt_modes(refresh, tmp_path):
    _bench_run(tmp_path, "a_plain_x_20260929", tutor="gpt-5.5-2026-04-23", mode="plain")
    figures, _ = refresh.measured_cost(REPO, tmp_path, "scaffolding_rigor")
    assert figures == {}


def test_cost_rows_follow_the_latency_roster_and_name_the_omitted(refresh):
    rows = [
        {"id": "a", "name": "Model A", "score": 0.8, "ttft_s": 9.0},
        {"id": "b", "name": "Model B", "score": 0.7},
    ]
    figures = {
        "a": {"uncached_cost_per_response_usd": 0.01},
        "not-on-site": {"uncached_cost_per_response_usd": 0.02},
    }
    models, omitted = refresh.cost_rows(rows, figures)
    assert models == [
        {
            "id": "a",
            "name": "Model A",
            "score": 0.8,
            "uncached_cost_per_response_usd": 0.01,
        }
    ]
    assert omitted == ["Model B"]


def test_refresh_ttft_only_also_writes_cost_json(refresh, tmp_path, monkeypatch):
    out = tmp_path / "data"
    out.mkdir()
    (out / "latency.json").write_text(
        json.dumps(
            {
                "source": "paper",
                "models": [
                    {
                        "id": "gpt-5.4-mini-2026-03-17",
                        "name": "GPT 5.4 mini",
                        "score": 0.7,
                    },
                    {"id": "claude-opus-4-8", "name": "Claude Opus 4.8", "score": 0.84},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(refresh, "OUT_DIR", out)
    probes = tmp_path / "results"
    _probe_dir(
        probes,
        "mini_scaffolding_rigor_latency_20261001",
        tutor="gpt-5.4-mini-2026-03-17",
        usage=_usage(),
    )
    _probe_dir(
        probes, "opus_scaffolding_rigor_latency_20260818", tutor="claude-opus-4-8"
    )

    refresh.refresh_ttft_only(REPO, probes)

    cost = json.loads((out / "cost.json").read_text("utf-8"))
    assert [m["id"] for m in cost["models"]] == ["gpt-5.4-mini-2026-03-17"]
    assert cost["models"][0]["score"] == 0.7, "the latency chart's y value"
    assert cost["omitted"] == ["Claude Opus 4.8"]


def test_write_cost_without_costable_runs_leaves_cost_json_alone(
    refresh, tmp_path, monkeypatch
):
    out = tmp_path / "data"
    out.mkdir()
    (out / "cost.json").write_text('{"models": ["kept"]}', encoding="utf-8")
    monkeypatch.setattr(refresh, "OUT_DIR", out)

    refresh.write_cost(REPO, tmp_path / "empty", [{"id": "a", "name": "A", "score": 1}])

    assert json.loads((out / "cost.json").read_text("utf-8")) == {"models": ["kept"]}
