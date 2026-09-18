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
and cannot be back-costed; ``role_cost_usd`` detects their contributions
(legacy token counts exceeding the canonical vector) and returns None
rather than pricing the fraction of the usage that carried a vector.

``summary_cost_block`` packages the figures for ``summary.json`` together
with the registry's ``pricing_version`` and a snapshot of the resolved
per-model rates, so a published cost stays interpretable and reproducible
after later rate updates. See docs/cost.md.
"""

import logging

from tutormoments.models import get_pricing, pricing_version

__all__ = [
    "cost_usd",
    "role_cost_usd",
    "list_cost",
    "billed_cost_estimate",
    "summary_cost_block",
]

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
    # Pre-vector contamination check. At the capture boundary the canonical
    # total is >= the legacy total_tokens on every provider (equal on
    # OpenAI/Gemini/Together; Anthropic exceeds it by the cache buckets its
    # legacy input_tokens excludes), so an aggregate where total_tokens wins
    # must include contributions whose cache split was discarded before
    # capture existed -- e.g. a resume over an old run's transcripts. Pricing
    # only the vector-bearing part would silently understate.
    if usage.get("total", 0) < usage.get("total_tokens", 0):
        logger.warning(
            "usage aggregate includes pre-vector contributions "
            "(canonical total %s < legacy total_tokens %s), which cannot be "
            "back-costed; cost is null",
            usage.get("total", 0),
            usage.get("total_tokens", 0),
        )
        return None
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


def _resolved_rates(tokens: dict) -> dict:
    """Snapshot the pricing entries the roles resolved to, keyed by model id.

    Recorded into the summary so a published cost can be recomputed after the
    registry's rates move on. Only single-model, priced aggregates contribute;
    an unpriceable role already nulled its figure and has nothing to snapshot.
    """
    rates: dict = {}
    for role in _SUMMARY_ROLES:
        usage = tokens.get(role)
        if not isinstance(usage, dict):
            continue
        model = usage.get("model")
        if not isinstance(model, str) or not model or "+" in model:
            continue
        entry = get_pricing(model)
        if entry:
            rates[model] = entry
    return rates


def summary_cost_block(tokens: dict, n_conversations: int) -> dict:
    """Build the run summary's ``cost`` block from its ``tokens`` block.

    ``n_conversations`` is the number of conversations whose tutor usage the
    token block aggregated (all trials pooled) -- the denominator of the
    headline ``tutor_cost_per_conversation_usd``. Figures are None whenever
    they cannot be computed exactly (see ``role_cost_usd``); the block itself
    is always present so readers can tell "uncosted" from "pre-cost run".
    """
    tutor_list_cost = list_cost(tokens)
    per_conversation = (
        tutor_list_cost / n_conversations
        if tutor_list_cost is not None and n_conversations > 0
        else None
    )
    return {
        # Figure (1) and its per-conversation headline.
        "tutor_list_cost_usd": tutor_list_cost,
        "tutor_cost_per_conversation_usd": per_conversation,
        "n_conversations": n_conversations,
        # Figure (2).
        "run_billed_cost_estimate_usd": billed_cost_estimate(tokens),
        # Reproducibility: which rate table produced these figures.
        "pricing_version": pricing_version(),
        "rates": _resolved_rates(tokens),
    }
