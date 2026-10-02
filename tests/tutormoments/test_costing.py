"""Tests: costing (cost computation over canonical usage vectors)."""

import json
import logging

import pytest

from tutormoments.costing import (
    billed_cost_estimate,
    cost_usd,
    costed_runs,
    list_cost,
    role_cost_usd,
    role_uncached_cost_usd,
    run_uncached_cost_figures,
    summary_cost_block,
    uncached_cost_figures,
    uncached_cost_usd,
)
from tutormoments.models import get_pricing, pricing_version

# ---------------------------------------------------------------------------
# The pricing table itself.
# ---------------------------------------------------------------------------

# Every model that has been run through the benchmark (tutor arms in results/
# plus the default-config role models) must be priced; a roster model without
# rates would silently null the run's cost figures.
ROSTER = [
    "claude-opus-4-8",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-5-5",
    "claude-fable-5-1",
    "claude-sonnet-5-5",
    "gemini-2.5-pro",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.5-2026-04-23",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
]


@pytest.mark.parametrize("model", ROSTER)
def test_roster_models_are_priced(model):
    rates = get_pricing(model)
    assert rates, f"benchmark roster model '{model}' has no pricing entry"
    for key in ("input", "output", "cache_read", "cache_write"):
        assert isinstance(rates[key], (int, float))
    assert rates["as_of"] and rates["source"]


def test_pricing_version_is_set():
    assert pricing_version()


# ---------------------------------------------------------------------------
# cost_usd: exact arithmetic over one vector.
# ---------------------------------------------------------------------------

RATES = {
    "input": 5.00,
    "output": 25.00,
    "cache_read": 0.50,
    "cache_write": 6.25,
    "batch_multiplier": 0.5,
}


def test_cost_usd_prices_each_bucket_at_its_rate():
    usage = {
        "input_uncached": 1_000_000,
        "cache_read": 1_000_000,
        "cache_write": 1_000_000,
        "output": 1_000_000,
        "reasoning": 1_000_000,
    }
    # 5 + 0.50 + 6.25 + (1M + 1M reasoning) at 25 = 61.75
    assert cost_usd(usage, RATES) == pytest.approx(61.75)


def test_cost_usd_batch_multiplier_applies_to_everything():
    usage = {"input_uncached": 2_000_000, "output": 1_000_000}
    assert cost_usd(usage, RATES, batch=True) == pytest.approx((10 + 25) * 0.5)


def test_cost_usd_missing_keys_are_zero():
    assert cost_usd({}, RATES) == 0.0


def test_cost_usd_anthropic_buckets_are_never_double_counted():
    # The double-count trap: Anthropic's cache buckets are disjoint from
    # input_tokens (and thus input_uncached) at the capture boundary. This
    # realistic vector (the 2026-08-24 live smoke's second call) must price
    # the 7,924 cached tokens at the cache-read rate ONLY -- adding them back
    # at the base input rate would multiply the input cost by ~40x.
    usage = {
        "input_uncached": 20,
        "cache_read": 7924,
        "cache_write": 0,
        "output": 150,
        "reasoning": 0,
        "total": 8094,
        # legacy keys ride along; costing must ignore them
        "input_tokens": 20,
        "output_tokens": 150,
        "total_tokens": 170,
    }
    expected = (20 * 5.00 + 7924 * 0.50 + 150 * 25.00) / 1_000_000
    assert cost_usd(usage, RATES) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# role_cost_usd: provenance-driven pricing of one role aggregate.
# ---------------------------------------------------------------------------


def _tutor_usage(**overrides):
    usage = {
        "input_uncached": 100_000,
        "cache_read": 0,
        "cache_write": 0,
        "output": 50_000,
        "reasoning": 0,
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "endpoint": "sync",
    }
    usage.update(overrides)
    return usage


def test_role_cost_prices_sync_usage_at_list_rates():
    expected = (100_000 * 5.00 + 50_000 * 25.00) / 1_000_000
    assert role_cost_usd(_tutor_usage()) == pytest.approx(expected)


def test_role_cost_applies_batch_discount_to_batch_endpoint():
    sync = role_cost_usd(_tutor_usage())
    batch = role_cost_usd(_tutor_usage(endpoint="batch"))
    assert batch == pytest.approx(sync * 0.5)


def test_role_cost_resolves_dated_snapshots_via_prefix():
    usage = _tutor_usage(provider="openai", model="gpt-5.5-2026-04-23", endpoint="sync")
    expected = (100_000 * 5.00 + 50_000 * 30.00) / 1_000_000
    assert role_cost_usd(usage) == pytest.approx(expected)


def test_role_cost_stream_and_sync_mix_is_still_priceable():
    # stream+sync is a benign mix -- both bill at list rates.
    expected = (100_000 * 5.00 + 50_000 * 25.00) / 1_000_000
    assert role_cost_usd(_tutor_usage(endpoint="stream+sync")) == pytest.approx(
        expected
    )


def test_role_cost_pre_vector_contamination_is_uncosted(caplog):
    # A resume over an old run's transcripts sums legacy counters that carry
    # no canonical vector: the aggregate's legacy total_tokens then exceeds
    # its canonical total (which is >= total_tokens on every provider when
    # everything was captured). Pricing just the vector-bearing part would
    # understate, so the cost is null.
    usage = _tutor_usage(total=150_000, total_tokens=450_000)
    with caplog.at_level(logging.WARNING, logger="tutormoments.costing"):
        assert role_cost_usd(usage) is None
    assert "pre-vector" in caplog.text


def test_role_cost_anthropic_cache_exceeding_legacy_total_still_priced():
    # The legitimate inequality direction: Anthropic's canonical total exceeds
    # legacy total_tokens by the cache buckets. Must NOT trip the guard.
    usage = _tutor_usage(cache_read=7924, total=157_924, total_tokens=150_000)
    assert role_cost_usd(usage) is not None


@pytest.mark.parametrize(
    ("overrides", "warning"),
    [
        ({"model": None}, "no model provenance"),
        ({"model": "claude-opus-4-8+gpt-5.5"}, "mixes models"),
        ({"model": "claude-sonnet-5"}, "no pricing entry"),
        ({"endpoint": "batch+sync"}, "cannot be apportioned"),
    ],
)
def test_role_cost_null_cases_warn(overrides, warning, caplog):
    usage = _tutor_usage(**overrides)
    if "model" in overrides and overrides["model"] is None:
        del usage["model"]
    with caplog.at_level(logging.WARNING, logger="tutormoments.costing"):
        assert role_cost_usd(usage) is None
    assert warning in caplog.text


# ---------------------------------------------------------------------------
# The two run-level figures.
# ---------------------------------------------------------------------------


def _tokens_block():
    return {
        "tutor": _tutor_usage(),
        "student": _tutor_usage(model="claude-opus-4-6"),
        "scorer": _tutor_usage(model="claude-opus-4-6", endpoint="batch"),
        "total": {},  # derived; costing must never price it
    }


def test_list_cost_is_tutor_only_no_discount():
    assert list_cost(_tokens_block()) == pytest.approx(role_cost_usd(_tutor_usage()))


def test_list_cost_refuses_batch_tagged_tutor_usage():
    tokens = _tokens_block()
    tokens["tutor"]["endpoint"] = "batch"
    with pytest.raises(ValueError, match="batch-tagged"):
        list_cost(tokens)


def test_list_cost_none_without_tutor_block():
    assert list_cost({}) is None


def test_billed_estimate_sums_roles_with_batch_discount():
    tokens = _tokens_block()
    expected = (
        role_cost_usd(tokens["tutor"])
        + role_cost_usd(tokens["student"])
        + role_cost_usd(tokens["scorer"])  # 0.5x inside
    )
    assert billed_cost_estimate(tokens) == pytest.approx(expected)
    # and the scorer really was discounted relative to its sync price
    assert role_cost_usd(tokens["scorer"]) == pytest.approx(
        role_cost_usd(tokens["student"]) * 0.5
    )


def test_billed_estimate_null_when_any_role_unpriceable():
    tokens = _tokens_block()
    tokens["scorer"]["model"] = "claude-sonnet-5"  # registered, unpriced
    assert billed_cost_estimate(tokens) is None


def test_billed_estimate_null_when_no_roles_present():
    assert billed_cost_estimate({"total": {}}) is None


# ---------------------------------------------------------------------------
# summary_cost_block: the summary.json `cost` block.
# ---------------------------------------------------------------------------


def test_summary_cost_block_shape_and_figures():
    tokens = _tokens_block()
    block = summary_cost_block(tokens, n_conversations=4)
    expected_list = role_cost_usd(_tutor_usage())
    assert block["tutor_list_cost_usd"] == pytest.approx(expected_list)
    assert block["tutor_cost_per_conversation_usd"] == pytest.approx(expected_list / 4)
    assert block["n_conversations"] == 4
    assert block["run_billed_cost_estimate_usd"] == pytest.approx(
        billed_cost_estimate(_tokens_block())
    )
    assert block["pricing_version"] == pricing_version()


def test_summary_cost_block_snapshots_resolved_rates():
    # Reproducibility: the block records the rates each role's model resolved
    # to, so the figures can be recomputed after the registry moves on.
    block = summary_cost_block(_tokens_block(), n_conversations=1)
    assert set(block["rates"]) == {"claude-opus-4-8", "claude-opus-4-6"}
    assert block["rates"]["claude-opus-4-8"] == get_pricing("claude-opus-4-8")


def test_summary_cost_block_zero_conversations_nulls_per_conversation():
    block = summary_cost_block(_tokens_block(), n_conversations=0)
    assert block["tutor_list_cost_usd"] is not None
    assert block["tutor_cost_per_conversation_usd"] is None


def test_summary_cost_block_unpriceable_roles_null_figures_not_block():
    # An unpriced scorer nulls the billed estimate; tutor figures survive.
    tokens = _tokens_block()
    tokens["scorer"]["model"] = "claude-sonnet-5"
    block = summary_cost_block(tokens, n_conversations=2)
    assert block["run_billed_cost_estimate_usd"] is None
    assert block["tutor_list_cost_usd"] is not None
    assert "claude-sonnet-5" not in block["rates"]

    # And a fully uncostable run still writes the block (all-None figures),
    # so readers can tell "uncosted" from "pre-cost run".
    empty = summary_cost_block({}, n_conversations=0)
    assert empty["tutor_list_cost_usd"] is None
    assert empty["run_billed_cost_estimate_usd"] is None
    assert empty["rates"] == {}


# ---------------------------------------------------------------------------
# Figure (3): uncached cost -- the website cost chart's deployment ceiling.
# ---------------------------------------------------------------------------


def test_uncached_cost_prices_every_prompt_bucket_at_the_input_rate():
    # Anthropic shape: the cache buckets are disjoint from input_uncached, so
    # the full prompt is their sum -- each token priced once, at `input`.
    usage = {
        "input_uncached": 20,
        "cache_read": 7924,
        "cache_write": 1000,
        "output": 150,
        "reasoning": 0,
    }
    expected = ((20 + 7924 + 1000) * 5.00 + 150 * 25.00) / 1_000_000
    assert uncached_cost_usd(usage, RATES) == pytest.approx(expected)


def test_uncached_cost_bills_reasoning_at_the_output_rate():
    # Where thinking models pay: reasoning is billed as output everywhere.
    usage = {"input_uncached": 6000, "output": 60, "reasoning": 1500}
    expected = (6000 * 5.00 + (60 + 1500) * 25.00) / 1_000_000
    assert uncached_cost_usd(usage, RATES) == pytest.approx(expected)


def test_uncached_cost_does_not_depend_on_the_cache_split():
    # The point of the figure: the same tokens cost the same whatever cache
    # mix the harness happened to get -- whereas the as-run figure moves.
    cold = {"input_uncached": 9000, "output": 400}
    warm = {
        "input_uncached": 100,
        "cache_read": 8000,
        "cache_write": 900,
        "output": 400,
    }
    assert uncached_cost_usd(cold, RATES) == pytest.approx(
        uncached_cost_usd(warm, RATES)
    )
    assert cost_usd(cold, RATES) != pytest.approx(cost_usd(warm, RATES))


def test_role_uncached_cost_prices_from_provenance():
    usage = _tutor_usage(cache_read=40_000)
    expected = (140_000 * 5.00 + 50_000 * 25.00) / 1_000_000
    assert role_uncached_cost_usd(usage) == pytest.approx(expected)


def test_role_uncached_cost_applies_no_batch_discount():
    # A list-price figure: the endpoint neither discounts nor nulls it.
    sync = role_uncached_cost_usd(_tutor_usage())
    assert role_uncached_cost_usd(_tutor_usage(endpoint="batch")) == sync
    assert role_uncached_cost_usd(_tutor_usage(endpoint="batch+sync")) == sync


@pytest.mark.parametrize(
    ("overrides", "warning"),
    [
        ({"total": 150_000, "total_tokens": 450_000}, "pre-vector"),
        ({"model": None}, "no model provenance"),
        ({"model": "claude-opus-4-8+gpt-5.5"}, "mixes models"),
        ({"model": "claude-sonnet-5"}, "no pricing entry"),
    ],
)
def test_role_uncached_cost_null_cases_match_role_cost(overrides, warning, caplog):
    usage = _tutor_usage(**overrides)
    if "model" in overrides and overrides["model"] is None:
        del usage["model"]
    with caplog.at_level(logging.WARNING, logger="tutormoments.costing"):
        assert role_uncached_cost_usd(usage) is None
    assert warning in caplog.text


# ---------------------------------------------------------------------------
# Figure (3) per call, and its benchmark-run source.
# ---------------------------------------------------------------------------


def test_uncached_cost_figures_divide_by_counted_calls():
    usage = _tutor_usage(cache_read=20_000, reasoning=10_000)
    figs = uncached_cost_figures(usage, n_calls=40)
    assert figs["uncached_cost_per_call_usd"] == pytest.approx(
        role_uncached_cost_usd(usage) / 40
    )
    assert figs["prompt_tokens_per_call"] == 3000  # (100k + 20k) / 40
    assert figs["output_tokens_per_call"] == 1500  # (50k + 10k reasoning) / 40
    assert figs["model"] == "claude-opus-4-8"
    assert figs["rates"] == {"claude-opus-4-8": get_pricing("claude-opus-4-8")}
    assert figs["pricing_version"] == pricing_version()


def test_uncached_cost_figures_none_without_calls():
    assert uncached_cost_figures(_tutor_usage(), n_calls=0) is None


def test_uncached_cost_figures_keep_tokens_when_unpriceable():
    figs = uncached_cost_figures(_tutor_usage(model="claude-sonnet-5"), n_calls=10)
    assert figs["uncached_cost_per_call_usd"] is None
    assert figs["prompt_tokens_per_call"] == 10_000
    assert figs["rates"] == {}


def _summary(**overrides):
    summary = {
        "tutor_model": "claude-opus-4-8",
        "mode": "scaffolding_rigor",
        "run_counts": {"attempted": 520, "succeeded": 520, "failed": 0},
        "latency": {"source": "run", "tutor": {"n": 1560}},
        "tokens": {"tutor": _tutor_usage()},
        "cost": {"n_conversations": 520},
    }
    summary.update(overrides)
    return summary


def test_run_figures_take_the_call_count_from_the_run_latency_block():
    # tokens.tutor and latency.tutor.n come from the same transcripts, so
    # the latter is the exact call count behind the former.
    figs = run_uncached_cost_figures(_summary())
    assert figs["n_calls"] == 1560
    assert figs["uncached_cost_per_call_usd"] == pytest.approx(
        role_uncached_cost_usd(_tutor_usage()) / 1560
    )


def test_run_figures_null_on_a_pre_vector_run(caplog):
    # A run (or resume) over transcripts from before usage capture.
    tokens = {"tutor": _tutor_usage(total=0, total_tokens=450_000)}
    with caplog.at_level(logging.WARNING, logger="tutormoments.costing"):
        figs = run_uncached_cost_figures(_summary(tokens=tokens))
    assert figs["uncached_cost_per_call_usd"] is None


@pytest.mark.parametrize("summary", [{}, _summary(latency=None), _summary(tokens={})])
def test_run_figures_none_without_the_blocks(summary):
    assert run_uncached_cost_figures(summary) is None


def _write_run(root, run_id, summary, sample=None):
    run = root / run_id
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"sample": sample}), encoding="utf-8")


def test_costed_runs_keys_by_arm_and_mode(tmp_path):
    _write_run(tmp_path, "opus_scaffolding_rigor_x_20260929", _summary())
    runs = costed_runs(str(tmp_path))
    cell = runs[("claude-opus-4-8", "scaffolding_rigor")]
    assert cell["run_id"] == "opus_scaffolding_rigor_x_20260929"
    assert cell["n_conversations"] == 520
    assert cell["n_calls"] == 1560


def test_costed_runs_prefers_the_newest_run(tmp_path):
    _write_run(tmp_path, "opus_scaffolding_rigor_x_20260929", _summary())
    _write_run(tmp_path, "opus_scaffolding_rigor_x_20261015", _summary())
    cell = costed_runs(str(tmp_path))[("claude-opus-4-8", "scaffolding_rigor")]
    assert cell["run_id"] == "opus_scaffolding_rigor_x_20261015"


@pytest.mark.parametrize(
    ("summary", "sample"),
    [
        # --sample replayed only part of the dataset.
        (_summary(), 10),
        # Failed moments: not the same set of conversations as a full run.
        (_summary(run_counts={"attempted": 520, "succeeded": 519, "failed": 1}), None),
        # Pre-vector usage cannot be priced.
        (_summary(tokens={"tutor": _tutor_usage(total=0, total_tokens=9)}), None),
        # A latency probe directory has no summary.json at all.
        (None, None),
    ],
)
def test_costed_runs_skips_ineligible_runs(tmp_path, summary, sample):
    if summary is None:
        (tmp_path / "probe_scaffolding_rigor_latency_20261001").mkdir()
    else:
        _write_run(tmp_path, "opus_scaffolding_rigor_x_20260929", summary, sample)
    assert costed_runs(str(tmp_path)) == {}


def test_costed_runs_on_a_missing_results_root(tmp_path):
    assert costed_runs(str(tmp_path / "nope")) == {}
