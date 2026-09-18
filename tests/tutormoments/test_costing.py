"""Tests: costing (cost computation over canonical usage vectors)."""

import logging

import pytest

from tutormoments.costing import (
    billed_cost_estimate,
    cost_usd,
    list_cost,
    role_cost_usd,
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
    "gemini-2.5-pro",
    "gemini-3.5-flash",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.5-2026-04-23",
    "deepseek-ai/DeepSeek-V4-Pro",
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
