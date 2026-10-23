"""Cost computation over canonical usage vectors.

Turns the usage vectors that aggregation preserves (see ``usage.py``) into
three dollar figures:

1. **Tutor list cost** (``list_cost``) -- the leaderboard figure: tutor
   tokens only, at list rates with the actual cache mix, never
   batch-discounted (post Phase 0, tutor usage is structurally sync, and
   ``list_cost`` asserts that rather than special-casing it).
2. **Billed run cost estimate** (``billed_cost_estimate``) -- the
   replication figure: every role, with the batch discount applied to
   batch-tagged usage. This is the number that should reconcile against
   provider invoices.
3. **Uncached cost** (``uncached_cost_usd`` / ``role_uncached_cost_usd``) --
   the deployment ceiling the website's cost chart plots: every prompt token
   priced at the full input rate, as if nothing were cached. Figure (1)
   swings by up to ~3x with the cache mix our harness happened to get
   (client breakpoints, provider mechanics, routing, TTLs), which describes
   the harness rather than the model. A tutoring provider pays the uncached
   cost at every session start and every cache expiry, and real students
   pause longer than cache TTLs. Because every prompt bucket is priced at the
   same rate, this figure does not depend on the recorded cache split.
   ``uncached_cost_figures`` turns it into per-call figures; its two sources
   are benchmark runs (``run_uncached_cost_figures`` / ``costed_runs``) and,
   for models whose runs predate usage capture, the latency probe
   (``latency.probe_cost_figures``).

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

from tutormoments import results
from tutormoments.models import get_pricing, pricing_version

__all__ = [
    "cost_usd",
    "uncached_cost_usd",
    "role_cost_usd",
    "role_uncached_cost_usd",
    "uncached_cost_figures",
    "run_uncached_cost_figures",
    "costed_runs",
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


def uncached_cost_usd(usage: dict, rates: dict) -> float:
    """Figure (3) arithmetic: price one vector as if nothing were cached.

    Every prompt bucket -- uncached, cache read and cache write -- is priced
    at the base input rate, and reasoning at the output rate as in
    ``cost_usd``. The buckets are disjoint at the capture boundary, so their
    sum counts the full prompt exactly once on every provider. No batch
    multiplier: this is a list-price figure.
    """
    prompt = (
        usage.get("input_uncached", 0)
        + usage.get("cache_read", 0)
        + usage.get("cache_write", 0)
    )
    output = usage.get("output", 0) + usage.get("reasoning", 0)
    return (prompt * rates["input"] + output * rates["output"]) / _MTOK


def _endpoints(usage: dict) -> set[str]:
    """The endpoint provenance of an aggregate, as a set ("+" splits mixes)."""
    endpoint = usage.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        return set()
    return set(endpoint.split("+"))


def _priceable_rates(usage: dict) -> dict | None:
    """The pricing entry one aggregate resolves to, or None.

    None (with a logged warning) when the aggregate cannot be priced
    exactly: pre-vector contributions, no model provenance, mixed models
    (their rates differ), or an unpriced model. Shared by every figure so a
    null means the same thing whichever figure was asked for.
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
    return rates


def role_cost_usd(usage: dict) -> float | None:
    """Price one role aggregate from its embedded provenance.

    Returns None (with a logged warning) when the aggregate cannot be priced
    exactly: any ``_priceable_rates`` case, or an endpoint mix where
    batch-tagged usage cannot be separated from sync usage. Registered/
    callable tutors synthesize empty usage and land in the no-provenance
    case -- their cost is legitimately null.
    """
    rates = _priceable_rates(usage)
    if rates is None:
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


def role_uncached_cost_usd(usage: dict) -> float | None:
    """Figure (3) for one role aggregate, priced from its provenance.

    Same null rules as ``role_cost_usd`` minus the endpoint one: a list-price
    figure applies no batch discount, so which endpoint served the usage
    does not enter into it.
    """
    rates = _priceable_rates(usage)
    if rates is None:
        return None
    return uncached_cost_usd(usage, rates)


def uncached_cost_figures(usage: dict, n_calls: int) -> dict | None:
    """Figure (3) per call, for one tutor aggregate over ``n_calls`` calls.

    ``n_calls`` must be the counted calls behind ``usage``: dividing by
    conversations times an assumed turn count breaks as soon as an ``[END]``
    or ``[PROBLEM_CHANGE]`` ends a conversation early. Priced at the
    registry's *current* rates -- tokens are the measurement, prices a
    lookup. None when there are no calls; ``uncached_cost_per_call_usd`` is
    None, with the token means kept, when the usage cannot be priced.

    Output is visible output plus reasoning. On OpenAI and Together reasoning
    already sits inside ``output`` (their ``reasoning`` bucket stays 0), so
    the sum is the billed completion either way; a separate reasoning figure
    would not be comparable across providers and is not reported.
    """
    if n_calls <= 0:
        return None
    cost = role_uncached_cost_usd(usage)
    model = usage.get("model")
    rates = get_pricing(model) if cost is not None else None
    prompt = (
        usage.get("input_uncached", 0)
        + usage.get("cache_read", 0)
        + usage.get("cache_write", 0)
    )
    output = usage.get("output", 0) + usage.get("reasoning", 0)
    return {
        "n_calls": n_calls,
        "model": model,
        "prompt_tokens_per_call": round(prompt / n_calls, 1),
        "output_tokens_per_call": round(output / n_calls, 1),
        "uncached_cost_per_call_usd": cost / n_calls if cost is not None else None,
        "pricing_version": pricing_version(),
        "rates": {model: rates} if rates else {},
    }


def run_uncached_cost_figures(summary: dict) -> dict | None:
    """Figure (3) per tutor call from one run's ``summary.json``.

    The summary's tutor token block and its tutor latency count are built
    from the same completed transcripts (resumed ones included), so the
    count is the exact number of hosted tutor calls behind the tokens. A run
    whose transcripts predate usage-vector capture is caught by the
    pre-vector check and yields a None cost.
    """
    tutor = (summary.get("tokens") or {}).get("tutor")
    n_calls = ((summary.get("latency") or {}).get("tutor") or {}).get("n") or 0
    if not isinstance(tutor, dict):
        return None
    return uncached_cost_figures(tutor, n_calls)


def costed_runs(results_root: str = "results") -> dict:
    """Latest costable full benchmark run per ``(tutor_model, mode)``.

    Returns ``{(tutor_model, mode): {"run_id", "n_conversations", **figures}}``
    with ``figures`` from ``run_uncached_cost_figures``. Eligible runs replayed
    the whole dataset (no ``--sample``) with no failed moments, and their
    usage prices exactly; among those the newest run id (date suffix) wins,
    so a re-run supersedes an earlier one. Mirrors ``latency.probe_runs``,
    which is the fallback source for models with no eligible run.
    """
    out: dict[tuple[str, str], dict] = {}
    best: dict[tuple[str, str], tuple] = {}
    for run_id in results.list_runs(results_root):
        summary = results.read_summary(run_id, results_root=results_root)
        if not summary or not summary.get("tokens"):
            continue
        config = results.read_config(run_id, results_root=results_root) or {}
        if config.get("sample") is not None:
            continue
        if (summary.get("run_counts") or {}).get("failed", 0):
            continue
        figures = run_uncached_cost_figures(summary)
        if not figures or figures["uncached_cost_per_call_usd"] is None:
            continue
        cell = (summary.get("tutor_model", ""), summary.get("mode", ""))
        rank = (run_id.rsplit("_", 1)[-1], run_id)
        if cell not in best or rank > best[cell]:
            best[cell] = rank
            out[cell] = {
                "run_id": run_id,
                "n_conversations": (summary.get("cost") or {}).get("n_conversations"),
                **figures,
            }
    return out


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
