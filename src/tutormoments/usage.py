"""Canonical usage-vector accumulation.

Every LLM call is normalized at the client boundary
(``client.normalize_usage``) into the canonical cost vector —
``input_uncached / cache_read / cache_write / output / reasoning / total`` —
plus provenance strings (``provider``, ``model``, ``endpoint``) and the
legacy per-provider counters (``input_tokens`` / ``output_tokens`` /
``total_tokens``). This module is where those vectors are summed without
being destroyed: accumulators sum *every* integer-valued key rather than an
allowlist, so a new field captured at the boundary survives to transcripts
and run summaries without touching aggregation code again.

Deliberately lightweight (stdlib only): conversation.py, scoring.py,
taxonomy.py and cli.py aggregate usage without importing the
provider-SDK-heavy client module.
"""

__all__ = ["EMPTY_USAGE", "add_usage", "sum_usage"]

# Zero legacy counters plus the canonical cost vector. Accumulators start
# from a copy so aggregated usage has a stable schema even when every
# contribution is empty (e.g. registered/callable tutors, error rows).
EMPTY_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "input_uncached": 0,
    "cache_read": 0,
    "cache_write": 0,
    "output": 0,
    "reasoning": 0,
    "total": 0,
}

# Provenance strings carried inside usage dicts (normalize_usage). These are
# the only non-numeric keys an accumulator preserves; anything else
# non-numeric (e.g. the taxonomy usage log's per-batch "label") is
# call-level detail that has no aggregate meaning and is dropped.
_PROVENANCE_KEYS = ("provider", "model", "endpoint")


def _merge_provenance(prior: str, value: str) -> str:
    """Combine two provenance values into a deterministic union.

    Uniform values stay as-is; a genuine mix becomes a sorted "+"-join
    (e.g. "stream+sync"), which keeps the one fact costing needs — whether
    "batch" contributed — recoverable from an aggregate.
    """
    parts = set(prior.split("+")) | set(value.split("+"))
    return "+".join(sorted(parts))


def add_usage(total: dict, new: dict) -> dict:
    """Accumulate one usage dict into ``total`` in place; returns ``total``.

    Sums every integer-valued key (bools excluded) so the canonical vector,
    the legacy counters, and any provider extras all survive. Provenance
    strings are kept when uniform and "+"-merged when mixed.
    """
    for key, value in new.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        base = total.get(key, 0)
        if isinstance(base, bool) or not isinstance(base, int):
            continue
        total[key] = base + value
    for key in _PROVENANCE_KEYS:
        value = new.get(key)
        if not isinstance(value, str) or not value:
            continue
        prior = total.get(key)
        if isinstance(prior, str) and prior:
            total[key] = _merge_provenance(prior, value)
        else:
            total[key] = value
    return total


def sum_usage(*usages: dict) -> dict:
    """Sum N usage dicts into a fresh dict seeded with ``EMPTY_USAGE``.

    Non-dict entries (None, error placeholders) are skipped.
    """
    total = dict(EMPTY_USAGE)
    for usage in usages:
        if isinstance(usage, dict):
            add_usage(total, usage)
    return total
