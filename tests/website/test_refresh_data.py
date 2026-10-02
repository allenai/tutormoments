"""Unit tests for website/scripts/refresh-data.py.

The script previously carried its own copy of the probe's publishability rules
and got them wrong -- it gated on cache hit *rate*, which the runtime refuses to
do, and would have published a "warm" figure for a provider whose hits read back
one shared block of system prompt. These tests pin the corrected behaviour:
figures come from `tutormoments.latency`, the site's model ids are matched to the
probe's, and a stale figure is never left standing next to a fresh one.

The paper models' score and end-to-end-latency assembly is thin I/O over an
analysis export and is covered by tests/analysis/test_benchmark_perf_cost.py.
The tests here cover the rest: later models scored from their full runs, and
every figure the checkout cannot rebuild carried forward rather than dropped.
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
# apply_ttft / build_benchmark_json
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


def _write_site(out: Path, leaderboard: list, latency: list, **extra) -> None:
    """The committed static/data JSON a refresh starts from."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(
        json.dumps({"source": "paper", "n_moments": 520, "models": leaderboard}),
        encoding="utf-8",
    )
    (out / "latency.json").write_text(
        json.dumps({"source": "Figure 7, paper", "models": latency, **extra}),
        encoding="utf-8",
    )


PAPER_OPUS = {
    "id": "claude-opus-4-8",
    "name": "Claude Opus 4.8",
    "plain": {"scaffolding": 0.615, "rigor": 0.208, "avoids_over": 0.462},
    "eval_aware": {"scaffolding": 0.858, "rigor": 0.831, "avoids_over": 0.896},
}
PAPER_OPUS_LAT = {
    "id": "claude-opus-4-8",
    "name": "Claude Opus 4.8",
    "latency_s": 12.5,
    "latency_estimated": True,
    "score": 0.8445,
}


@pytest.fixture
def site(refresh, tmp_path, monkeypatch):
    """A checkout without the paper's results/benchmark, and its site data."""
    out = tmp_path / "data"
    monkeypatch.setattr(refresh, "OUT_DIR", out)
    return {"repo": tmp_path / "repo", "results": tmp_path / "results", "out": out}


def _read(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text("utf-8"))


def test_build_keeps_the_paper_rows_it_cannot_rebuild(refresh, site):
    """Most checkouts have probe runs but not the paper's scored sweep.
    Rebuilding there must keep the paper's scores and refresh only TTFAT."""
    _write_site(site["out"], [PAPER_OPUS], [PAPER_OPUS_LAT])
    _probe_dir(
        site["results"], "a_scaffolding_rigor_latency_20260818", tutor="claude-opus-4-8"
    )

    refresh.build_benchmark_json(site["repo"], site["results"])

    lb = _read(site["out"], "leaderboard.json")["models"][0]
    assert lb["eval_aware"] == PAPER_OPUS["eval_aware"]
    assert lb["reasoning"] == "thinking: adaptive, effort: xhigh", "as configured"
    assert lb["source"] == "paper"
    lat = _read(site["out"], "latency.json")
    row = lat["models"][0]
    assert row["score"] == 0.8445
    assert row["latency_s"] == 12.5
    assert row["ttft_s"] == 9.0
    assert lat["ttft"]["subsample_id"] == "589e8acf8ac761f2"


def test_build_without_probe_runs_keeps_the_published_ttfat(refresh, site):
    """No probe runs at all is a checkout gap, not a measurement that every
    model lost its figure."""
    ttft = {"mode": "scaffolding_rigor", "subsample_id": "589e8acf8ac761f2"}
    _write_site(
        site["out"],
        [PAPER_OPUS],
        [{**PAPER_OPUS_LAT, "ttft_s": 9.04, "ttlt_s": 10.52}],
        ttft=ttft,
    )

    refresh.build_benchmark_json(site["repo"], site["results"])

    lat = _read(site["out"], "latency.json")
    assert lat["models"][0]["ttft_s"] == 9.04
    assert lat["models"][0]["ttlt_s"] == 10.52
    assert lat["ttft"] == ttft


def test_build_scores_a_later_model_from_its_full_runs(refresh, site):
    _write_site(site["out"], [PAPER_OPUS], [PAPER_OPUS_LAT])
    for mode, scores in (
        ("plain", (0.85, 0.2, 0.425)),
        ("scaffolding_rigor", (0.95, 0.8, 0.096)),
    ):
        _bench_run(
            site["results"],
            f"gpt-6-sol-none_{mode}_tutormoments-preview_20261001",
            tutor="gpt-6-sol-none",
            mode=mode,
            scores=scores,
            thinking={"reasoning": "none"},
        )

    refresh.build_benchmark_json(site["repo"], site["results"])

    rows = {m["id"]: m for m in _read(site["out"], "leaderboard.json")["models"]}
    sol = rows["gpt-6-sol-none"]
    assert sol["name"] == "GPT-6 Sol"
    assert sol["reasoning"] == "reasoning: none"
    assert sol["plain"] == {"scaffolding": 0.85, "rigor": 0.2, "avoids_over": 0.575}
    assert sol["eval_aware"] == {
        "scaffolding": 0.95,
        "rigor": 0.8,
        "avoids_over": 0.904,
    }
    assert sol["source"] == {
        "plain": "gpt-6-sol-none_plain_tutormoments-preview_20261001",
        "eval_aware": "gpt-6-sol-none_scaffolding_rigor_tutormoments-preview_20261001",
    }
    assert list(rows) == ["claude-opus-4-8", "gpt-6-sol-none"], "MODELS order"

    lat = {m["id"]: m for m in _read(site["out"], "latency.json")["models"]}
    assert lat["gpt-6-sol-none"]["score"] == 0.875
    assert lat["gpt-6-sol-none"]["latency_s"] == 1.47
    assert "ttft_s" not in lat["gpt-6-sol-none"], "no probe: not measured, never 0"

    cost = _read(site["out"], "cost.json")
    assert [m["id"] for m in cost["models"]] == ["gpt-6-sol-none"]
    assert cost["models"][0]["score"] == 0.875, "the latency chart's y value"
    assert cost["omitted"] == ["Claude Opus 4.8"]


def test_build_needs_both_prompts_before_replacing_a_carried_row(refresh, site):
    """A half-finished sweep (one prompt done) must not knock out the row."""
    prior = {**PAPER_OPUS, "id": "gpt-6-sol-none", "name": "GPT-6 Sol"}
    _write_site(site["out"], [prior], [])
    _bench_run(
        site["results"],
        "gpt-6-sol-none_plain_tutormoments-preview_20261101",
        tutor="gpt-6-sol-none",
        mode="plain",
    )

    refresh.build_benchmark_json(site["repo"], site["results"])

    row = _read(site["out"], "leaderboard.json")["models"][0]
    assert row["eval_aware"] == PAPER_OPUS["eval_aware"]


def test_full_runs_takes_the_newest_complete_unsampled_run(refresh, tmp_path):
    kw = {"tutor": "deepseek-ai/DeepSeek-V4-Pro-0813"}
    _bench_run(
        tmp_path, "ds_scaffolding_rigor_x_20260901", scores=(0.5, 0.5, 0.5), **kw
    )
    _bench_run(
        tmp_path, "ds_scaffolding_rigor_x_20260929", scores=(0.6, 0.6, 0.5), **kw
    )
    _bench_run(tmp_path, "ds_scaffolding_rigor_x_20261001", sample=10, **kw)
    _bench_run(tmp_path, "ds_scaffolding_rigor_x_20261002", failed=3, **kw)
    _probe_dir(tmp_path, "ds_scaffolding_rigor_latency_20261003", tutor=kw["tutor"])

    runs = refresh.full_runs(tmp_path)

    assert list(runs) == [("deepseek-ai_DeepSeek-V4-Pro-0813", "scaffolding_rigor")]
    cell = runs[("deepseek-ai_DeepSeek-V4-Pro-0813", "scaffolding_rigor")]
    assert cell["run_id"] == "ds_scaffolding_rigor_x_20260929"


@pytest.mark.parametrize(
    "thinking, label",
    [
        (
            {"effort": "high", "thinking": {"type": "adaptive"}},
            "thinking: adaptive, effort: high",
        ),
        ({"thinking": {"type": "disabled"}}, "thinking: disabled"),
        (
            {"thinking_level": "minimal", "include_thoughts": True},
            "thinking_level: minimal",
        ),
        ({"thinking_budget": -1, "include_thoughts": True}, "thinking_budget: -1"),
        ({"reasoning_effort": "max"}, "reasoning_effort: max"),
        ({"reasoning": "none"}, "reasoning: none"),
        ({}, "none sent (model default)"),
    ],
)
def test_reasoning_label_states_the_configured_parameters(refresh, thinking, label):
    assert refresh.reasoning_label(thinking) == label


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
    scores: tuple = (0.9, 0.8, 0.1),
    thinking: dict | None = None,
    sample: int | None = None,
    failed: int = 0,
) -> None:
    """A benchmark run directory: summary.json + config.json, as cli.py writes.
    ``scores`` is (scaffold score, rigor score, overscaffold rate)."""
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
        "n_scenarios": 520,
        "run_counts": {"attempted": 520, "succeeded": 520 - failed, "failed": failed},
        "scaffold_calibrated": {"score": scores[0]},
        "rigor_calibrated": {"score": scores[1]},
        "overscaffold": {"rate": scores[2]},
        "latency": {"source": "run", "tutor": {"n": n_calls, "mean_seconds": 1.469}},
        "tokens": {"tutor": usage},
        "cost": {"n_conversations": 520},
    }
    config = {
        "sample": sample,
        "arm": tutor,
        "resolved_tutors": {tutor: {"thinking": thinking or {}}},
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")


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


def test_write_cost_without_costable_runs_leaves_cost_json_alone(
    refresh, tmp_path, monkeypatch
):
    out = tmp_path / "data"
    out.mkdir()
    (out / "cost.json").write_text('{"models": ["kept"]}', encoding="utf-8")
    monkeypatch.setattr(refresh, "OUT_DIR", out)

    refresh.write_cost(REPO, tmp_path / "empty", [{"id": "a", "name": "A", "score": 1}])

    assert json.loads((out / "cost.json").read_text("utf-8")) == {"models": ["kept"]}


# ---------------------------------------------------------------------------
# build_action_distribution
# ---------------------------------------------------------------------------


def test_action_distribution_without_runs_shows_the_paper_models(
    refresh, tmp_path, monkeypatch
):
    """The paper's export has columns for its seven models only; with no runs
    and nothing committed, later rows of MODELS are left off, not a crash."""
    monkeypatch.setattr(refresh, "OUT_DIR", tmp_path / "data")

    refresh.build_action_distribution(_paper_csv(refresh, tmp_path / "v1.csv"), "test")

    out = json.loads(
        (tmp_path / "data" / "action_distribution.json").read_text("utf-8")
    )
    assert [m["id"] for m in out["models"]] == [
        m for m, _ in refresh.MODELS if m in refresh.ACTION_CSV_MODELS
    ]


def _paper_csv(refresh, path: Path) -> Path:
    """A v1_action_taxonomy_distribution.csv with every column the paper's
    export carries; per-letter values are arbitrary."""
    import csv

    prefixes = ["human"] + [
        f"{col}__{p}"
        for col in refresh.ACTION_CSV_MODELS.values()
        for p in ("plain", "SR")
    ]
    fields = ["letter", "name", "orientation"] + [
        f"{pre}__{k}"
        for pre in prefixes
        for k in ("n_moments", "macro_mean_pct", "ci_low", "ci_high")
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, letter in enumerate("ABCDEFGHIJKLM"):
            row = {"letter": letter, "name": letter, "orientation": "neutral"}
            for f in fields[3:]:
                row[f] = 100 if f.endswith("n_moments") else float(i)
            w.writerow(row)
    return path


def test_action_distribution_scores_a_later_model_from_its_runs(
    refresh, tmp_path, monkeypatch
):
    """A later model's cell is the macro mean over its run's own
    classifications: per moment, the share of facets in each letter."""
    pytest.importorskip("pandas")
    from tutormoments import taxonomy

    results = tmp_path / "results"
    for mode, letters in (("plain", "AAB"), ("scaffolding_rigor", "GG")):
        run_id = f"gpt-6-sol-none_{mode}_tutormoments-preview_20261001"
        _bench_run(results, run_id, tutor="gpt-6-sol-none", mode=mode)
        facets = [
            taxonomy.Facet(
                moment_id=f"m{j}",
                transcript_id="t",
                turn_start=0,
                turn_end=1,
                statement_index=i,
                statement="s",
                annotation_type="scaffolding",
                situation_label="scaffolding",
                category=letter,
            )
            for j in range(2)
            for i, letter in enumerate(letters if j == 0 else "C")
        ]
        (results / run_id / "taxonomy").mkdir()
        taxonomy.write_classified_csv(
            facets, results / run_id / "taxonomy" / "classified.csv"
        )
    fp = _paper_csv(refresh, tmp_path / "v1.csv")
    monkeypatch.setattr(refresh, "OUT_DIR", tmp_path / "data")

    refresh.build_action_distribution(fp, "test", repo=REPO, results_root=results)

    out = json.loads(
        (tmp_path / "data" / "action_distribution.json").read_text("utf-8")
    )
    sol = {m["id"]: m for m in out["models"]}["gpt-6-sol-none"]
    # moment 0 is 2/3 A, moment 1 is all C: macro A = (2/3 + 0) / 2.
    assert sol["plain"]["A"]["pct"] == pytest.approx(33.33, abs=0.01)
    assert sol["plain"]["C"]["pct"] == pytest.approx(50.0)
    assert sol["eval_aware"]["G"]["pct"] == pytest.approx(50.0)
    assert sol["n_moments"] == {"plain": 2, "eval_aware": 2}
    assert sol["source"]["eval_aware"].startswith("gpt-6-sol-none_scaffolding_rigor")
    paper = {m["id"]: m for m in out["models"]}["claude-opus-4-8"]
    assert paper["source"] == "paper"
    assert paper["n_moments"] == {"plain": 100, "eval_aware": 100}


def test_action_distribution_carries_a_later_model_it_cannot_rebuild(
    refresh, tmp_path, monkeypatch
):
    out_dir = tmp_path / "data"
    out_dir.mkdir()
    kept = {"id": "gpt-6-sol-none", "name": "old", "source": {"plain": "r"}}
    (out_dir / "action_distribution.json").write_text(
        json.dumps({"models": [kept]}), encoding="utf-8"
    )
    monkeypatch.setattr(refresh, "OUT_DIR", out_dir)

    refresh.build_action_distribution(_paper_csv(refresh, tmp_path / "v1.csv"), "test")

    out = json.loads((out_dir / "action_distribution.json").read_text("utf-8"))
    sol = {m["id"]: m for m in out["models"]}["gpt-6-sol-none"]
    assert sol == {**kept, "name": "GPT-6 Sol"}


# ---------------------------------------------------------------------------
# build_kl
# ---------------------------------------------------------------------------


def _facet_kwargs(moment, situation, letter, i=0):
    return dict(
        moment_id=moment,
        transcript_id="t",
        turn_start=0,
        turn_end=1,
        statement_index=i,
        statement="s",
        annotation_type="scaffolding",
        situation_label=situation,
        category=letter,
    )


# Per situation: two moments' letters. S leans A, R leans G.
KL_CELL = {"scaffolding": ("AAB", "AC"), "rigor": ("GG", "GA")}


def _kl_facets(taxonomy, prefix=""):
    return [
        taxonomy.Facet(**_facet_kwargs(f"{prefix}{sit}{j}", sit, letter, i))
        for sit, moments in KL_CELL.items()
        for j, letters in enumerate(moments)
        for i, letter in enumerate(letters)
    ]


@pytest.fixture
def kl_env(refresh, tmp_path, monkeypatch):
    """The runtime's taxonomy with its two network fetches stubbed: the human
    reference and the release's action_taxonomy.jsonl."""
    from tutormoments import taxonomy

    release = tmp_path / "action_taxonomy.jsonl"
    with release.open("w", encoding="utf-8") as fh:
        for mode in ("plain", "scaffolding_rigor"):
            for f in _kl_facets(taxonomy):
                fh.write(
                    json.dumps(
                        {
                            **f.to_dict(),
                            "source": "benchmark_520",
                            "tutor_model": "claude-opus-4-8",
                            "prompt_mode": mode,
                        }
                    )
                    + "\n"
                )
            # an excluded facet (no category) must not count
            fh.write(
                json.dumps(
                    {
                        **_facet_kwargs("x", "rigor", None),
                        "source": "benchmark_520",
                        "tutor_model": "claude-opus-4-8",
                        "prompt_mode": mode,
                    }
                )
                + "\n"
            )
    human = {"s_r": 0.5, "r_s": 0.6, "n_scaffolding": 260, "n_rigor": 258}
    monkeypatch.setattr(
        taxonomy,
        "human_reference",
        lambda: {"label": "h", "dataset": "d", "revision": "r", "kl": human},
    )
    monkeypatch.setattr(taxonomy, "_hf_download", lambda *a: release)
    monkeypatch.setattr(refresh, "OUT_DIR", tmp_path / "data")
    return taxonomy, tmp_path / "results"


def test_build_kl_scores_paper_and_later_models_with_the_runtime(refresh, kl_env):
    taxonomy, results = kl_env
    for mode in ("plain", "scaffolding_rigor"):
        run_id = f"gpt-6-sol-none_{mode}_tutormoments-preview_20261001"
        _bench_run(results, run_id, tutor="gpt-6-sol-none", mode=mode)
        (results / run_id / "taxonomy").mkdir()
        taxonomy.write_classified_csv(
            _kl_facets(taxonomy), results / run_id / "taxonomy" / "classified.csv"
        )

    refresh.build_kl(REPO, results)

    out = json.loads((refresh.OUT_DIR / "kl.json").read_text("utf-8"))
    want = taxonomy.kl_situation(_kl_facets(taxonomy))
    rows = {m["id"]: m for m in out["models"]}
    for model in ("claude-opus-4-8", "gpt-6-sol-none"):
        cell = rows[model]["eval_aware"]
        assert cell["s_r"] == pytest.approx(want["s_r"], abs=1e-4)
        assert cell["r_s"] == pytest.approx(want["r_s"], abs=1e-4)
        assert cell["mean"] == pytest.approx((want["s_r"] + want["r_s"]) / 2, abs=1e-4)
        assert (cell["n_scaffolding"], cell["n_rigor"]) == (2, 2)
    dataset, revision = taxonomy.HUMAN_REFERENCE
    assert rows["claude-opus-4-8"]["source"] == {"release": f"{dataset}@{revision}"}
    assert rows["gpt-6-sol-none"]["source"]["plain"].startswith("gpt-6-sol-none_plain")
    assert out["human"]["mean"] == pytest.approx(0.55)


def test_build_kl_carries_rows_it_cannot_rebuild(refresh, kl_env):
    _, results = kl_env
    refresh.OUT_DIR.mkdir(parents=True)
    kept = {"id": "gpt-6-sol-none", "name": "old", "plain": {"mean": 0.2}}
    (refresh.OUT_DIR / "kl.json").write_text(
        json.dumps({"human": {"mean": 0.55}, "models": [kept]}), encoding="utf-8"
    )

    refresh.build_kl(REPO, results)  # no runs for gpt-6-sol-none

    out = json.loads((refresh.OUT_DIR / "kl.json").read_text("utf-8"))
    rows = {m["id"]: m for m in out["models"]}
    assert rows["gpt-6-sol-none"] == {**kept, "name": "GPT-6 Sol"}
    assert "claude-opus-4-8" in rows, "rebuilt from the release"
