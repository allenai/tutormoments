"""Cost computation over canonical usage vectors.

Turns the usage vectors that aggregation preserves (see ``usage.py``) into
the two dollar figures the cost plan defines:

1. **Tutor list cost** (``list_cost``) -- the leaderboard/deployment figure:
   tutor tokens only, at list rates with the actual cache mix, never
   batch-discounted (post Phase 0, tutor usage is structurally sync, and
   ``list_cost`` asserts that rather than special-casing it).
2. **Billed run cost estimate** (``billed_cost_estimate``) -- the
   replication figure: every role, with the batch discount applied to
   batch-tagged usage. This is the number that should reconcile against
   provider invoices.

Rates come from the packaged model registry (``models.yaml``); each entry
carries an ``as_of`` lookup date and ``source`` URL, and the registry's
``pricing_version`` is recorded into run summaries so published costs stay
interpretable after rates change. An unpriced or ambiguous aggregate yields
``None`` plus a logged warning -- never a crash, never a silent 0.

Cost is exact arithmetic over recorded usage: providers report the
cached/uncached split on every response, so no cache-hit-rate estimation
appears anywhere. Old runs (pre usage-vector capture) discarded that split
and cannot be back-costed; their canonical buckets are all zero, which
surfaces here as a zero-dollar cost on a nonzero legacy token count --
report layers must treat pre-vector runs as uncosted rather than free.
"""

import logging

from tutormoments.models import get_pricing

__all__ = ["cost_usd", "role_cost_usd", "list_cost", "billed_cost_estimate"]

logger = logging.getLogger(__name__)

_MTOK = 1_000_000

# Roles whose usage the run-level summary aggregates (cli.py's tokens block),
# in reporting order. "total" is derived there and never costed directly --
# costing role-by-role is what lets the batch discount apply to exactly the
# batch-tagged usage.
_SUMMARY_ROLES = ("tutor", "student", "scorer", "taxonomy")


def cost_usd(usage: dict, rates: dict, batch: bool = False) -> float:
    """Price one usage vector at the given per-MTok rates.

    Reasoning tokens bill at the output rate on every provider that reports
    them (tracked separately in the vector so no information is lost, folded
    together here because no provider prices them apart). The cache buckets
    are disjoint from ``input_uncached`` by construction at the capture
    boundary -- in particular Anthropic's ``input_tokens`` already excludes
    them -- so each bucket is priced exactly once, at its own rate.
    """
    cost = (
        usage.get("input_uncached", 0) * rates["input"]
        + usage.get("cache_read", 0) * rates["cache_read"]
        + usage.get("cache_write", 0) * rates["cache_write"]
        + (usage.get("output", 0) + usage.get("reasoning", 0)) * rates["output"]
    ) / _MTOK
    if batch:
        cost *= rates["batch_multiplier"]
    return cost


def _endpoints(usage: dict) -> set[str]:
    """The endpoint provenance of an aggregate, as a set ("+" splits mixes)."""
    endpoint = usage.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        return set()
    return set(endpoint.split("+"))


def role_cost_usd(usage: dict) -> float | None:
    """Price one role aggregate from its embedded provenance.

    Returns None (with a logged warning) when the aggregate cannot be priced
    exactly: no model provenance, mixed models (their rates differ), an
    unpriced model, or an endpoint mix where batch-tagged usage cannot be
    separated from sync usage. Registered/callable tutors synthesize empty
    usage and land in the no-provenance case -- their cost is legitimately
    null.
    """
    model = usage.get("model")
    if not isinstance(model, str) or not model:
        logger.warning("usage aggregate has no model provenance; cost is null")
        return None
    if "+" in model:
        logger.warning("usage aggregate mixes models (%s); cost is null", model)
        return None
    rates = get_pricing(model)
    if not rates:
        logger.warning("model '%s' has no pricing entry; cost is null", model)
        return None
    endpoints = _endpoints(usage)
    if "batch" in endpoints and endpoints != {"batch"}:
        logger.warning(
            "usage aggregate mixes batch and non-batch endpoints (%s); "
            "the batch discount cannot be apportioned, cost is null",
            usage.get("endpoint"),
        )
        return None
    return cost_usd(usage, rates, batch=endpoints == {"batch"})


def list_cost(tokens: dict) -> float | None:
    """Figure (1): tutor cost at list rates -- no batch discount, ever.

    ``tokens`` is a run summary's per-role token block. Batching is a harness
    throughput choice, not a model property, so a batch-tagged tutor
    aggregate is a structural invariant violation (conversations are sync by
    construction since Phase 0) and raises rather than silently discounting
    the leaderboard figure.
    """
    tutor = tokens.get("tutor")
    if not isinstance(tutor, dict):
        return None
    if "batch" in _endpoints(tutor):
        raise ValueError(
            "tutor usage is batch-tagged, which run_conversation cannot "
            f"produce (endpoint={tutor.get('endpoint')!r}); refusing to "
            "compute a list cost from it"
        )
    return role_cost_usd(tutor)


def billed_cost_estimate(tokens: dict) -> float | None:
    """Figure (2): the whole run at billed rates, batch discount included.

    Sums every role present in the summary token block. Any unpriceable role
    makes the whole estimate None: a partial sum would silently understate
    the replication cost, which is worse than no number.
    """
    total = 0.0
    priced_any = False
    for role in _SUMMARY_ROLES:
        usage = tokens.get(role)
        if not isinstance(usage, dict):
            continue
        cost = role_cost_usd(usage)
        if cost is None:
            return None
        total += cost
        priced_any = True
    return total if priced_any else None
