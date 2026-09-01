"""Tests for tutormoments.usage: canonical usage-vector accumulation.

Phase 2 of the cost-tracking plan: aggregation must stop destroying the
canonical vector captured at the client boundary. These tests pin the
accumulator's contract -- every integer key survives, provenance strings are
kept when uniform and "+"-merged when mixed, and non-usage strings are
dropped.
"""

from tutormoments.usage import EMPTY_USAGE, add_usage, sum_usage


def _vector(**overrides) -> dict:
    """A realistic post-Phase-1 usage dict (legacy + canonical + provenance)."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
        "input_uncached": 20,
        "cache_read": 80,
        "cache_write": 0,
        "output": 40,
        "reasoning": 0,
        "total": 140,
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "endpoint": "sync",
    }
    usage.update(overrides)
    return usage


def test_add_usage_sums_every_integer_key():
    """No allowlist: canonical keys and provider extras all accumulate."""
    total = dict(EMPTY_USAGE)
    add_usage(total, _vector())
    add_usage(total, _vector(cache_write=50, cache_read_input_tokens=7924))

    assert total["input_uncached"] == 40
    assert total["cache_read"] == 160
    assert total["cache_write"] == 50
    assert total["output"] == 80
    assert total["total"] == 280
    assert total["input_tokens"] == 200
    # A provider extra absent from EMPTY_USAGE still survives.
    assert total["cache_read_input_tokens"] == 7924


def test_add_usage_keeps_uniform_provenance():
    total = dict(EMPTY_USAGE)
    add_usage(total, _vector())
    add_usage(total, _vector())
    assert total["provider"] == "anthropic"
    assert total["model"] == "claude-sonnet-5"
    assert total["endpoint"] == "sync"


def test_add_usage_merges_mixed_provenance_deterministically():
    """Mixed provenance becomes a sorted "+"-join, so whether "batch"
    contributed stays recoverable from an aggregate."""
    total = dict(EMPTY_USAGE)
    add_usage(total, _vector(endpoint="sync"))
    add_usage(total, _vector(endpoint="stream"))
    add_usage(total, _vector(endpoint="sync"))
    assert total["endpoint"] == "stream+sync"

    # Merging an already-merged aggregate keeps the union stable.
    combined = sum_usage(total, _vector(endpoint="batch"))
    assert combined["endpoint"] == "batch+stream+sync"


def test_add_usage_ignores_bools_and_foreign_strings():
    """Bools are not counters; non-provenance strings (e.g. the taxonomy
    usage log's per-batch "label") have no aggregate meaning."""
    total = dict(EMPTY_USAGE)
    add_usage(total, {**_vector(), "label": "batch_003", "truncated": True})
    assert "label" not in total
    assert "truncated" not in total


def test_add_usage_never_clobbers_non_numeric_totals():
    """A numeric value under a provenance key in `new` must not overwrite
    an accumulated string (and vice versa)."""
    total = {"model": "claude-sonnet-5"}
    add_usage(total, {"model": 3})
    assert total["model"] == "claude-sonnet-5"


def test_sum_usage_seeds_stable_schema_and_skips_non_dicts():
    total = sum_usage(None, {}, _vector())
    for key in EMPTY_USAGE:
        assert key in total
    assert total["total"] == 140
    # Empty sum still has the full zero schema.
    assert sum_usage() == EMPTY_USAGE


def test_sum_usage_returns_fresh_dict():
    a = _vector()
    total = sum_usage(a)
    total["output"] += 1
    assert a["output"] == 40
    assert EMPTY_USAGE["output"] == 0
