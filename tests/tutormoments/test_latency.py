"""The latency probe: percentiles, cache-split aggregation, subsample resolution.

The load-bearing behaviours here are the ones that stop a latency number from
lying: never inferring cache state, never publishing a "warm" figure computed
from a sample that mostly missed, and never letting a derived subsample be
mistaken for the frozen one that makes measurements comparable over time.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tutormoments.latency import (
    MIN_CACHE_HIT_SAMPLES,
    MIN_SESSION_CACHE_READ_TOKENS,
    aggregate_timings,
    format_probe_summary,
    latency_stats,
    probe_figures,
    probe_runs,
    probe_subsample_ids,
    resolve_subsample,
    run_probe,
    warm_figure_is_publishable,
    withheld_reason,
)
from tutormoments.moments import PROBE_IDS_FILENAME, subsample_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timing(ttft=1.0, ttlt=3.0, cache_state="hit", tps=20.0):
    return {
        "ttfc_seconds": ttft,
        "ttft_seconds": ttft,
        "ttlt_seconds": ttlt,
        "output_tokens": 40,
        "cache_read_input_tokens": 900 if cache_state == "hit" else 0,
        "output_tps": tps,
        "turn_index": 0,
        "cache_state": cache_state,
    }


def _moment(mid):
    return SimpleNamespace(id=mid)


# ---------------------------------------------------------------------------
# latency_stats
# ---------------------------------------------------------------------------


def test_latency_stats_none_on_empty():
    assert latency_stats([]) is None


def test_latency_stats_reports_the_aa_percentile_set():
    """P5/P50/P95 -- the set Artificial Analysis publishes."""
    stats = latency_stats([float(i) for i in range(1, 101)])
    assert set(stats) >= {"p5_seconds", "p50_seconds", "p95_seconds"}
    assert stats["p5_seconds"] < stats["p50_seconds"] < stats["p95_seconds"]


def test_latency_stats_p50_p95_rule_unchanged_from_pre_streaming():
    """The historical sorted-index rule is preserved so previously published
    p50/p95 figures stay reproducible."""
    samples = [1.0, 2.0, 3.0, 4.0]
    s = sorted(samples)
    expected_p50 = s[len(s) // 2]
    expected_p95 = s[max(0, min(len(s) - 1, int(round(0.95 * len(s))) - 1))]
    stats = latency_stats(samples)
    assert stats["p50_seconds"] == expected_p50
    assert stats["p95_seconds"] == expected_p95


def test_latency_stats_single_sample():
    stats = latency_stats([2.5])
    assert stats["n"] == 1
    assert stats["p5_seconds"] == stats["p50_seconds"] == stats["p95_seconds"] == 2.5


# ---------------------------------------------------------------------------
# aggregate_timings
# ---------------------------------------------------------------------------


def test_aggregate_splits_by_cache_state():
    """Hit and miss are different student experiences and must not be pooled."""
    timings = [
        _timing(ttft=0.4, cache_state="hit"),
        _timing(ttft=0.5, cache_state="hit"),
        _timing(ttft=2.0, cache_state="miss"),
    ]
    out = aggregate_timings(timings)
    assert out["ttft"]["hit"]["n"] == 2
    assert out["ttft"]["miss"]["n"] == 1
    assert out["ttft"]["hit"]["p50_seconds"] < out["ttft"]["miss"]["p50_seconds"]


def test_aggregate_reports_cache_hit_rate():
    timings = [_timing(cache_state="hit")] * 3 + [_timing(cache_state="miss")]
    assert aggregate_timings(timings)["cache_hit_rate"] == pytest.approx(0.75)


def test_aggregate_hit_rate_none_when_provider_reports_nothing():
    """Gemini/Together report no cache tokens -- rate is unknown, not zero."""
    timings = [_timing(cache_state="unknown") for _ in range(3)]
    out = aggregate_timings(timings)
    assert out["cache_hit_rate"] is None
    assert out["cache_state_known"] == 0
    assert out["ttft"]["all"]["n"] == 3, "samples still counted overall"


def test_aggregate_empty_is_safe():
    out = aggregate_timings([])
    assert out["n_samples"] == 0
    assert out["ttft"]["all"] is None
    assert out["cache_hit_rate"] is None


def test_aggregate_skips_samples_missing_the_metric():
    """A stream that produced no visible token has ttft=None; don't count it."""
    timings = [_timing(ttft=1.0), {**_timing(), "ttft_seconds": None}]
    assert aggregate_timings(timings)["ttft"]["all"]["n"] == 1


def test_aggregate_keeps_throughput_as_diagnostic_only():
    out = aggregate_timings([_timing(tps=10.0), _timing(tps=30.0)])
    assert out["output_tps_mean"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# warm_figure_is_publishable
# ---------------------------------------------------------------------------


def test_warm_figure_withheld_when_cache_state_unknown():
    """Providers with no real prompt cache must not get a 'warm' number --
    it would make them look slower for a harness reason, not a model one."""
    assert not warm_figure_is_publishable({"cache_hit_rate": None})


def test_warm_figure_withheld_below_hit_sample_floor():
    block = {
        "cache_hit_rate": 1.0,
        "cache_read_p50_on_hits": 8000,
        "ttft": {"hit": {"n": MIN_CACHE_HIT_SAMPLES - 1}},
    }
    assert not warm_figure_is_publishable(block)


def test_warm_figure_published_at_or_above_floor():
    block = {
        "cache_hit_rate": 1.0,
        "cache_read_p50_on_hits": 8000,
        "ttft": {"hit": {"n": MIN_CACHE_HIT_SAMPLES}},
    }
    assert warm_figure_is_publishable(block)


def test_warm_figure_gate_does_not_depend_on_max_turns():
    """max_turns fixes the hit *rate* (0.5 at 3 turns, 0.67 at 5), so gating
    on the rate would let a run knob decide whether a model is publishable."""
    at_3_turns = {
        "cache_hit_rate": 0.5,
        "cache_read_p50_on_hits": 8000,
        "ttft": {"hit": {"n": 40}},
    }
    at_5_turns = {
        "cache_hit_rate": 0.667,
        "cache_read_p50_on_hits": 8000,
        "ttft": {"hit": {"n": 80}},
    }
    assert warm_figure_is_publishable(at_3_turns)
    assert warm_figure_is_publishable(at_5_turns)


def test_aggregate_counts_turns_that_produced_no_visible_token():
    """A model can look fast by not answering; the conditioning must show."""
    timings = [_timing(ttft=1.0)] * 3 + [{**_timing(), "ttft_seconds": None}]
    out = aggregate_timings(timings)
    assert out["n_no_visible_output"] == 1
    assert out["no_visible_output_rate"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# resolve_subsample
# ---------------------------------------------------------------------------


def _write_frozen(tmp_path: Path, ids: list[str]) -> str:
    (tmp_path / PROBE_IDS_FILENAME).write_text(json.dumps(ids), encoding="utf-8")
    return str(tmp_path)


def test_resolve_prefers_the_frozen_list(tmp_path):
    moments = [_moment(f"m{i}") for i in range(10)]
    data_path = _write_frozen(tmp_path, ["m1", "m4", "m7"])

    picked, prov = resolve_subsample(moments, n=40, data_path=data_path)

    assert [m.id for m in picked] == ["m1", "m4", "m7"]
    assert prov["subsample_source"] == "frozen_release"
    assert prov["subsample_complete"] is True
    assert prov["subsample_id"] == subsample_id(["m1", "m4", "m7"])


def test_resolve_frozen_list_beats_n(tmp_path):
    """--n must not silently shrink a frozen sample; the list is the contract."""
    moments = [_moment(f"m{i}") for i in range(10)]
    data_path = _write_frozen(tmp_path, ["m1", "m4", "m7"])

    picked, _ = resolve_subsample(moments, n=1, data_path=data_path)

    assert len(picked) == 3


def test_resolve_superset_release_keeps_the_series_intact(tmp_path):
    """A later release that adds moments still resolves every frozen id, so
    the subsample_id -- and therefore comparability -- is unchanged."""
    frozen = ["m1", "m4", "m7"]
    data_path = _write_frozen(tmp_path, frozen)
    original = [_moment(f"m{i}") for i in range(10)]
    superset = original + [_moment(f"new{i}") for i in range(50)]

    a_moments, a_prov = resolve_subsample(original, n=40, data_path=data_path)
    b_moments, b_prov = resolve_subsample(superset, n=40, data_path=data_path)

    assert [m.id for m in a_moments] == [m.id for m in b_moments]
    assert a_prov["subsample_id"] == b_prov["subsample_id"]
    assert b_prov["subsample_complete"] is True


def test_resolve_flags_dropped_ids_as_incomplete(tmp_path):
    """A release that drops a frozen moment breaks the series -- say so."""
    data_path = _write_frozen(tmp_path, ["m1", "gone", "m7"])
    moments = [_moment("m1"), _moment("m7")]

    picked, prov = resolve_subsample(moments, n=40, data_path=data_path)

    assert [m.id for m in picked] == ["m1", "m7"]
    assert prov["subsample_complete"] is False
    assert prov["missing_ids"] == ["gone"]


@pytest.fixture
def no_packaged_list(monkeypatch):
    """Suppress the packaged id list to reach the derivation tier."""
    monkeypatch.setattr("tutormoments.latency.packaged_probe_ids", lambda: None)


def test_resolve_uses_the_packaged_list_when_the_release_has_none(tmp_path):
    """This is the path most runs take.

    The default config loads moments from the published Hugging Face dataset,
    where there is no local release directory -- so without the packaged list
    the frozen subsample would never apply on the default code path and
    latency would silently stop being a time series.
    """
    monkeypatched = ["m2", "m5"]
    import tutormoments.latency as lat_mod

    original = lat_mod.packaged_probe_ids
    lat_mod.packaged_probe_ids = lambda: monkeypatched
    try:
        moments = [_moment(f"m{i}") for i in range(10)]
        picked, prov = resolve_subsample(moments, n=3, data_path=None)
    finally:
        lat_mod.packaged_probe_ids = original

    assert [m.id for m in picked] == ["m2", "m5"]
    assert prov["subsample_source"] == "frozen_packaged"


def test_release_list_outranks_the_packaged_one(tmp_path):
    """A dataset's own statement about itself wins over the shipped default."""
    data_path = _write_frozen(tmp_path, ["m1", "m3"])
    moments = [_moment(f"m{i}") for i in range(10)]

    picked, prov = resolve_subsample(moments, n=3, data_path=data_path)

    assert [m.id for m in picked] == ["m1", "m3"]
    assert prov["subsample_source"] == "frozen_release"


def test_resolve_falls_back_and_marks_it_derived(tmp_path, no_packaged_list):
    """With neither source available, derivation still works -- but is never
    mistakable for a frozen run."""
    moments = [_moment(f"m{i}") for i in range(10)]

    picked, prov = resolve_subsample(moments, n=3, data_path=str(tmp_path))

    assert len(picked) == 3
    assert prov["subsample_source"] == "derived"


def test_derived_and_frozen_have_different_subsample_ids(tmp_path, no_packaged_list):
    """So a derived run can never be silently appended to a frozen series."""
    moments = [_moment(f"m{i}") for i in range(10)]
    _, derived = resolve_subsample(moments, n=3, data_path=str(tmp_path))
    data_path = _write_frozen(tmp_path, ["m5", "m6", "m7"])
    _, frozen = resolve_subsample(moments, n=3, data_path=data_path)

    assert derived["subsample_id"] != frozen["subsample_id"]


def test_resolve_without_data_path_derives(no_packaged_list):
    """HF-loaded datasets have no local dir to read the frozen list from."""
    moments = [_moment(f"m{i}") for i in range(5)]
    _, prov = resolve_subsample(moments, n=2, data_path=None)
    assert prov["subsample_source"] == "derived"


def test_packaged_list_is_present_and_well_formed():
    """The shipped list must actually be shipped -- a packaging slip would
    silently demote every default-path run to a derived subsample."""
    from tutormoments.moments import packaged_probe_ids as real_packaged

    ids = real_packaged()
    assert ids, "latency_probe_ids.json is missing from the package"
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert all(i.startswith("balanced_520:") for i in ids)


# ---------------------------------------------------------------------------
# run_probe
# ---------------------------------------------------------------------------


class _Cfg:
    dataset = None
    dataset_revision = None
    dataset_config = "moments"
    max_turns = 4
    student = {"model": "claude-haiku-4-5", "mode": "oracle"}

    def __init__(self, data_path):
        self.data_path = data_path


def _fake_transcript(ttft):
    return SimpleNamespace(
        tutor_timings=[_timing(ttft=ttft)],
        student_timings=[_timing(ttft=0.2)],
    )


def _patch_load(monkeypatch, moments):
    monkeypatch.setattr(
        "tutormoments.moments.load_moments",
        lambda **kw: (moments, {"record_count": len(moments)}),
    )


def test_run_probe_writes_latency_json(tmp_path, monkeypatch):
    moments = [_moment("m1"), _moment("m2")]
    _patch_load(monkeypatch, moments)
    (tmp_path / "release").mkdir()
    data_path = _write_frozen(tmp_path / "release", ["m1", "m2"])

    run_id, block = run_probe(
        "claude-opus-4-8",
        "plain",
        cfg=_Cfg(data_path),
        n=40,
        results_root=str(tmp_path / "results"),
        date="20260817",
        _run_conversation=lambda *a, **k: _fake_transcript(0.5),
    )

    written = json.loads(
        (tmp_path / "results" / run_id / "latency.json").read_text(encoding="utf-8")
    )
    assert written["source"] == "probe"
    assert written["tutor"]["n_samples"] == 2
    assert block["subsample"]["subsample_source"] == "frozen_release"


def test_run_probe_measures_each_moment_exactly_once(tmp_path, monkeypatch):
    """No repeat pass, deliberately.

    Re-sending byte-identical requests would be served from cache by any
    provider with automatic prefix caching, so a repeated "cold" sample is
    not cold. Distinct moments are independent measurements; repeats are not.
    To check whether the environment is stable over time, run the probe twice
    and compare the two blocks -- each carries its own `measured_at`, which a
    pooled repeat would have thrown away.
    """
    (tmp_path / "release").mkdir()
    data_path = _write_frozen(tmp_path / "release", ["m1", "m2"])
    _patch_load(monkeypatch, [_moment("m1"), _moment("m2")])
    calls = []

    def _conv(moment, **kwargs):
        calls.append(moment.id)
        return _fake_transcript(0.5)

    _, block = run_probe(
        "claude-opus-4-8",
        "plain",
        cfg=_Cfg(data_path),
        n=40,
        results_root=str(tmp_path / "results"),
        date="20260817",
        _run_conversation=_conv,
    )

    assert calls == ["m1", "m2"], "each moment measured once, in order"
    assert block["tutor"]["n_samples"] == 2
    assert "repeats" not in block["measurement_environment"]


def test_run_probe_survives_one_failed_moment(tmp_path, monkeypatch):
    """One bad moment must not discard an otherwise complete measurement."""
    (tmp_path / "release").mkdir()
    data_path = _write_frozen(tmp_path / "release", ["m1", "m2"])
    _patch_load(monkeypatch, [_moment("m1"), _moment("m2")])

    def _conv(moment, **kwargs):
        if moment.id == "m1":
            raise RuntimeError("api exploded")
        return _fake_transcript(0.5)

    _, block = run_probe(
        "claude-opus-4-8",
        "plain",
        cfg=_Cfg(data_path),
        n=40,
        results_root=str(tmp_path / "results"),
        date="20260817",
        _run_conversation=_conv,
    )

    assert block["failed_moments"] == ["m1"]
    assert block["tutor"]["n_samples"] == 1


def test_run_probe_records_serial_concurrency(tmp_path, monkeypatch):
    """The whole point of the probe: concurrency 1, recorded as such."""
    (tmp_path / "release").mkdir()
    data_path = _write_frozen(tmp_path / "release", ["m1"])
    _patch_load(monkeypatch, [_moment("m1")])

    _, block = run_probe(
        "claude-opus-4-8",
        "plain",
        cfg=_Cfg(data_path),
        n=40,
        results_root=str(tmp_path / "results"),
        date="20260817",
        _run_conversation=lambda *a, **k: _fake_transcript(0.5),
    )

    assert block["measurement_environment"]["concurrency"] == 1


def test_run_probe_records_location_label(tmp_path, monkeypatch):
    """We can't pin an egress region, so we record the one we had."""
    (tmp_path / "release").mkdir()
    data_path = _write_frozen(tmp_path / "release", ["m1"])
    _patch_load(monkeypatch, [_moment("m1")])
    monkeypatch.setenv("TUTORMOMENTS_LATENCY_LOCATION", "gcp-us-central1")

    _, block = run_probe(
        "claude-opus-4-8",
        "plain",
        cfg=_Cfg(data_path),
        n=40,
        results_root=str(tmp_path / "results"),
        date="20260817",
        _run_conversation=lambda *a, **k: _fake_transcript(0.5),
    )

    assert block["measurement_environment"]["location"] == "gcp-us-central1"


def test_run_probe_rejects_empty_subsample(tmp_path, monkeypatch):
    (tmp_path / "release").mkdir()
    data_path = _write_frozen(tmp_path / "release", ["nope"])
    _patch_load(monkeypatch, [_moment("m1")])

    with pytest.raises(RuntimeError, match="zero moments"):
        run_probe(
            "claude-opus-4-8",
            "plain",
            cfg=_Cfg(data_path),
            n=40,
            results_root=str(tmp_path / "results"),
            date="20260817",
            _run_conversation=lambda *a, **k: _fake_transcript(0.5),
        )


# ---------------------------------------------------------------------------
# format_probe_summary
# ---------------------------------------------------------------------------


def test_summary_dashes_warm_figures_when_cache_unknown():
    block = {
        "tutor_model": "deepseek-ai/DeepSeek-V4-Pro",
        "mode": "plain",
        "tutor": aggregate_timings([_timing(cache_state="unknown")]),
        "subsample": {"subsample_source": "frozen_release", "subsample_id": "abc"},
        "measurement_environment": {},
    }
    text = format_probe_summary(block)
    assert "warm figure withheld" in text


def test_summary_warns_on_incomplete_subsample():
    block = {
        "tutor_model": "claude-opus-4-8",
        "mode": "plain",
        "tutor": aggregate_timings([_timing()]),
        "subsample": {
            "subsample_source": "frozen_release",
            "subsample_id": "abc",
            "subsample_complete": False,
            "missing_ids": ["gone"],
        },
        "measurement_environment": {},
    }
    assert "not comparable to earlier runs" in format_probe_summary(block)


def test_warm_figure_withheld_for_incidental_prefix_cache_hits():
    """Together reports cached_tokens, but a median hit reads 256 tokens -- one
    quantised block of the shared system prompt, not this conversation's
    transcript. Its hits also land on first turns and misses on later ones,
    tracking run warmup rather than session position. Calling that "warm"
    would compare it against Anthropic's genuine 8k-token session cache."""
    together_like = {
        "cache_hit_rate": 0.91,
        "cache_read_p50_on_hits": 256,
        "ttft": {"hit": {"n": 73}},
    }
    assert not warm_figure_is_publishable(together_like)


def test_warm_figure_published_for_real_session_cache():
    anthropic_like = {
        "cache_hit_rate": 0.5,
        "cache_read_p50_on_hits": 10028,
        "ttft": {"hit": {"n": 40}},
    }
    assert warm_figure_is_publishable(anthropic_like)


def test_session_cache_threshold_sits_between_the_observed_regimes():
    """256 (incidental block) < threshold <= 1180 (smallest real head seen)."""
    assert 256 < MIN_SESSION_CACHE_READ_TOKENS <= 1180


def test_aggregate_reports_median_cache_read_on_hits():
    timings = [
        {**_timing(cache_state="hit"), "cache_read_input_tokens": 256},
        {**_timing(cache_state="hit"), "cache_read_input_tokens": 9000},
        {**_timing(cache_state="hit"), "cache_read_input_tokens": 8000},
    ]
    assert aggregate_timings(timings)["cache_read_p50_on_hits"] == 8000


def test_summary_reports_pooled_figures_when_cache_state_is_unknown():
    """Gemini reports no cache tokens, so every sample is `unknown` and both
    the miss and hit splits are empty. The pooled figure must still show --
    80 valid measurements are not nothing."""
    block = {
        "tutor_model": "gemini-2.5-pro",
        "mode": "plain",
        "tutor": aggregate_timings(
            [{**_timing(ttft=2.0), "cache_state": "unknown"} for _ in range(80)]
        ),
        "subsample": {"subsample_source": "frozen_packaged", "subsample_id": "x"},
        "measurement_environment": {},
    }
    text = format_probe_summary(block)
    assert "TTFT p50 all" in text
    assert "2.000" in text, "pooled TTFT must be visible"
    assert "provider reports no cache tokens" in text


def test_summary_explains_why_a_warm_figure_was_withheld():
    """The reason differs and matters: no cache at all vs incidental hits."""
    incidental = aggregate_timings(
        [
            {**_timing(cache_state="hit"), "cache_read_input_tokens": 256}
            for _ in range(40)
        ]
    )
    block = {
        "tutor_model": "deepseek-ai/DeepSeek-V4-Pro",
        "mode": "plain",
        "tutor": incidental,
        "subsample": {"subsample_source": "frozen_packaged", "subsample_id": "x"},
        "measurement_environment": {},
    }
    assert "incidental shared prefix" in format_probe_summary(block)


def test_no_hits_is_reported_as_no_hits_not_as_a_fidelity_problem():
    """A provider that reports cache tokens but recorded zero hits has no
    `cache_read_p50_on_hits` to judge, so the read-size branch must not claim
    the hits were incidental -- there were none. Reporting a fidelity problem
    here would send a reader hunting for a caching bug that isn't there.
    """
    all_missed = aggregate_timings(
        [
            {**_timing(cache_state="miss"), "cache_read_input_tokens": 0}
            for _ in range(40)
        ]
    )
    assert all_missed["cache_hit_rate"] == 0.0, "hit rate known, and it is zero"
    assert all_missed["cache_read_p50_on_hits"] is None

    reason = withheld_reason(all_missed)
    assert "only 0 cache hit(s)" in reason
    assert "incidental" not in reason
    assert "None tokens" not in reason


def test_ttfc_is_aggregated_alongside_the_headline_metrics():
    """docs/latency.md tells a reader to check `ttfc` well below `ttft` on
    thinking models. That has to be readable off the block: recomputing it
    from `samples` is a different (and easily skipped) piece of work.
    """
    block = aggregate_timings(
        [
            {**_timing(cache_state="hit"), "ttfc_seconds": 0.1, "ttft_seconds": 4.0},
            {**_timing(cache_state="miss"), "ttfc_seconds": 0.2, "ttft_seconds": 5.0},
        ]
    )
    assert block["ttfc"]["all"]["n"] == 2
    assert block["ttfc"]["hit"]["p50_seconds"] == 0.1
    assert block["ttfc"]["miss"]["p50_seconds"] == 0.2
    assert block["ttfc"]["all"]["p50_seconds"] < block["ttft"]["all"]["p50_seconds"]


def test_withheld_reason_follows_the_gate_order():
    """Each reason corresponds to the condition that actually failed."""
    unknown = aggregate_timings([_timing(cache_state="unknown")])
    assert "reports no cache tokens" in withheld_reason(unknown)

    too_few = aggregate_timings(
        [{**_timing(cache_state="hit"), "cache_read_input_tokens": 9000}]
        + [{**_timing(cache_state="miss"), "cache_read_input_tokens": 0}] * 5
    )
    assert "cache hit(s)" in withheld_reason(too_few)

    incidental = aggregate_timings(
        [
            {**_timing(cache_state="hit"), "cache_read_input_tokens": 256}
            for _ in range(40)
        ]
    )
    assert "incidental shared prefix" in withheld_reason(incidental)


# ---------------------------------------------------------------------------
# Reading probe results back: probe_figures / probe_runs
# ---------------------------------------------------------------------------


def _probe_tutor_block(
    *,
    p50_all=9.0,
    p50_miss=12.0,
    p50_hit=8.0,
    cache_hit_rate=0.5,
    cache_read=8000,
    n_hits=40,
):
    return {
        "cache_hit_rate": cache_hit_rate,
        "cache_read_p50_on_hits": cache_read,
        "ttft": {
            "all": {"n": 112, "p50_seconds": p50_all},
            "miss": {"n": 40, "p50_seconds": p50_miss},
            "hit": {"n": n_hits, "p50_seconds": p50_hit},
        },
        "ttlt": {"all": {"n": 112, "p50_seconds": p50_all + 1.0}},
    }


def _probe_samples(first=(12.0, 13.0, 14.0), later=(7.0, 8.0, 9.0)):
    """Tutor samples spanning both turn positions."""
    rows = [{"ttft_seconds": v, "turn_index": 0} for v in first]
    rows += [{"ttft_seconds": v, "turn_index": 1 + i % 2} for i, v in enumerate(later)]
    return rows


def _probe_block(tutor_block=None, samples=None):
    """A whole latency.json dict, as probe_figures takes it."""
    return {
        "tutor": tutor_block or _probe_tutor_block(),
        "samples": _probe_samples() if samples is None else samples,
    }


def _write_probe(
    root: Path,
    run_id: str,
    *,
    tutor="model-a",
    mode="scaffolding_rigor",
    measured_at="2026-08-18T10:00:00",
    source="frozen_packaged",
    complete=True,
    sub_id="589e8acf8ac761f2",
    tutor_block=None,
):
    run = root / run_id
    run.mkdir(parents=True)
    (run / "latency.json").write_text(
        json.dumps(
            {
                "source": "probe",
                "tutor_model": tutor,
                "mode": mode,
                "tutor": tutor_block or _probe_tutor_block(),
                "samples": _probe_samples(),
                "subsample": {
                    "subsample_source": source,
                    "subsample_id": sub_id,
                    "subsample_complete": complete,
                },
                "measurement_environment": {"measured_at": measured_at},
            }
        ),
        encoding="utf-8",
    )
    return run


def test_probe_figures_always_publishes_the_pooled_number():
    """Pooled TTFT is measured identically on every provider, so it is the one
    figure that can rank the whole roster."""
    figs = probe_figures(
        _probe_block(_probe_tutor_block(cache_hit_rate=None, cache_read=None))
    )
    assert figs["ttft_p50"] == 9.0
    assert figs["ttlt_p50"] == 10.0


def test_probe_figures_splits_on_turn_position_not_cache_state():
    """First/later keys on turn_index, which the probe recorded itself. A
    cache-state split needs a publishability gate and, even where it passes,
    dilutes the first-message bucket with later-turn calls whose cache
    silently missed (measured on gpt-5.5: cache-based cold p50 6.91s against
    an actual first-message p50 of 7.53s)."""
    figs = probe_figures(_probe_block())
    assert figs["ttft_first_p50"] == 13.0  # p50 of (12, 13, 14) -- turn 0
    assert figs["ttft_later_p50"] == 8.0  # p50 of (7, 8, 9) -- turns 1 and 2


def test_probe_figures_splits_for_a_provider_reporting_no_cache_tokens():
    """The point of the turn split: Gemini reports no cache tokens, so a
    cache-state split cannot exist for it -- but its first/later figures are
    as real as anyone's."""
    figs = probe_figures(
        _probe_block(_probe_tutor_block(cache_hit_rate=None, cache_read=None))
    )
    assert figs["ttft_first_p50"] == 13.0
    assert figs["ttft_later_p50"] == 8.0


def test_probe_figures_excludes_samples_with_no_visible_output():
    """A call that produced no visible token has no TTFT and must not enter a
    position bucket."""
    samples = _probe_samples() + [{"ttft_seconds": None, "turn_index": 0}]
    figs = probe_figures(_probe_block(samples=samples))
    assert figs["ttft_first_p50"] == 13.0


def test_probe_figures_reads_old_files_without_a_stored_split():
    """The split is computed from `samples`, which every probe has always
    written -- a latency.json from before the split existed still yields one."""
    block = _probe_block()
    assert "ttft_first_p50" not in json.dumps(block)  # nothing precomputed
    assert probe_figures(block)["ttft_first_p50"] == 13.0


def test_probe_figures_tolerates_an_absent_probe():
    assert probe_figures({}) == {
        "ttft_p50": None,
        "ttlt_p50": None,
        "ttft_first_p50": None,
        "ttft_later_p50": None,
    }


def test_probe_runs_keys_by_model_and_mode(tmp_path):
    _write_probe(tmp_path, "model-a_scaffolding_rigor_latency_20260818")
    _write_probe(
        tmp_path, "model-b_plain_latency_20260818", tutor="model-b", mode="plain"
    )
    found = probe_runs(str(tmp_path))
    assert set(found) == {("model-a", "scaffolding_rigor"), ("model-b", "plain")}


def test_probe_runs_prefers_the_newest_measurement(tmp_path):
    """Re-measuring a model supersedes its earlier figure without anyone
    having to delete the old run directory."""
    _write_probe(
        tmp_path,
        "model-a_scaffolding_rigor_latency_20260817",
        measured_at="2026-08-17T14:50:00",
        tutor_block=_probe_tutor_block(p50_all=99.0),
    )
    _write_probe(
        tmp_path,
        "model-a_scaffolding_rigor_latency_20260818",
        measured_at="2026-08-18T10:00:00",
        tutor_block=_probe_tutor_block(p50_all=9.0),
    )
    block = probe_runs(str(tmp_path))[("model-a", "scaffolding_rigor")]
    assert block["tutor"]["ttft"]["all"]["p50_seconds"] == 9.0


def test_probe_runs_ignores_a_derived_subsample(tmp_path):
    """A derived sample spans no particular prompt-length distribution, so it
    is comparable to nothing -- least of all to another model's frozen run."""
    _write_probe(
        tmp_path, "model-a_scaffolding_rigor_latency_20260818", source="derived"
    )
    assert probe_runs(str(tmp_path)) == {}


def test_probe_runs_ignores_an_incomplete_frozen_subsample(tmp_path):
    """Dropped ids mean this is not the same sample as the run before it."""
    _write_probe(tmp_path, "model-a_scaffolding_rigor_latency_20260818", complete=False)
    assert probe_runs(str(tmp_path)) == {}


def test_probe_runs_ignores_benchmark_run_directories(tmp_path):
    """Benchmark runs write summary.json, not latency.json, and their latency
    was gathered under --concurrency."""
    (tmp_path / "model-a_scaffolding_rigor_tutormoments-preview_20260807").mkdir()
    assert probe_runs(str(tmp_path)) == {}


def test_probe_runs_on_a_missing_results_root(tmp_path):
    assert probe_runs(str(tmp_path / "nope")) == {}


def test_probe_subsample_ids_exposes_a_mixed_set(tmp_path):
    """Eligibility is per-probe, so two frozen-but-different samples each pass
    it. Callers publishing several cells need to see that."""
    _write_probe(
        tmp_path,
        "model-a_scaffolding_rigor_latency_20260818",
        sub_id="589e8acf8ac761f2",
    )
    _write_probe(
        tmp_path,
        "model-b_scaffolding_rigor_latency_20260817",
        tutor="model-b",
        sub_id="84b4ad5615876a3e",
    )
    assert probe_subsample_ids(probe_runs(str(tmp_path))) == {
        "589e8acf8ac761f2",
        "84b4ad5615876a3e",
    }
