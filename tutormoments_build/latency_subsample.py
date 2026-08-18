"""Frozen moment subsample used by the `tutormoments latency` probe.

Measuring latency over all 520 moments on every model would be needlessly
expensive, so the probe runs a subsample. That subsample is **frozen**: the
ids are selected once, committed to `latency_probe_ids.json`, and shipped in
the release. Selecting at runtime instead would silently re-pick the sample
whenever the dataset changed, so a later measurement would be over different
prompts than an earlier one -- the numbers would drift for reasons that have
nothing to do with the model. A frozen list keeps latency a time series: a
future release that is a *superset* of the current one still resolves every
frozen id, so old and new measurements stay comparable.

Deliberately no CLI subcommand, mirroring `balanced_520_ids.json` (also a
committed artifact with no generator). A convenient regeneration command
would invite re-running it against a newer release, which is precisely the
failure the freeze exists to prevent. The selection rule lives here as a
tested function so it stays auditable --
`tests/tutormoments_build/test_latency_subsample.py` recomputes it against a
local release and asserts it reproduces the committed list.
"""

import json
from pathlib import Path

# The filename, the hash, and the reader live runtime-side so the runtime can
# consume the released list without importing build code. Only the *selection
# rule* is build-side -- choosing which moments constitute the sample is
# benchmark-defining, and the runtime never makes that choice.
from tutormoments.moments import (
    PROBE_IDS_FILENAME,
    packaged_probe_ids,
    read_probe_ids,
    subsample_id,
)

__all__ = [
    "PROBE_IDS_FILENAME",
    "DEFAULT_SUBSAMPLE_SIZE",
    "select_latency_subsample",
    "subsample_id",
    "write_probe_ids",
    "read_probe_ids",
    "packaged_probe_ids",
]

# One moment per conversation, and the release has 112 conversations -- so the
# subsample is "every conversation, once" rather than an arbitrary count. That
# is the largest sample the de-duplication rule permits, and it sets the
# resolution of a paired model comparison: ~+/-6.5 points on a warm win rate,
# ~+/-9.3 on cold, at roughly 56 minutes per model.
DEFAULT_SUBSAMPLE_SIZE = 112

# Backstop on the 2-opt refinement in _assign_targets. Each pass must strictly
# lower total error or the loop exits, so this only bounds pathological input;
# the current release converges in 3.
_MAX_REFINE_PASSES = 50


def _context_length(moment) -> int:
    """Total characters of pre-cut transcript context for a moment.

    A proxy for prompt length, which is the dominant driver of TTFT -- longer
    prompts mean more prefill before the first token appears.
    """
    return sum(len(turn.get("text") or "") for turn in (moment.context or []))


def _by_conversation(moments: list) -> dict[str, list]:
    """Group moments by source conversation, each group sorted by length.

    The grouping exists because moments cut from the same conversation share a
    long transcript prefix, and providers with automatic prefix caching serve
    the second one from cache. Measured on the previous 40-moment subsample:
    every `gpt-5.5` turn-1 cache hit came from a conversation contributing
    more than one moment, reading back 4.9k-9.0k tokens. Those samples are
    labelled cold and are not cold, which biases the cold figure downward on
    exactly the providers that cache silently. Taking one moment per
    conversation removes the shared prefix, so each sample is independent.
    """
    by_conv: dict[str, list] = {}
    for m in moments:
        by_conv.setdefault((m.provenance or {}).get("conv_id") or m.id, []).append(m)
    return {
        conv: sorted(group, key=lambda m: (_context_length(m), m.id))
        for conv, group in sorted(by_conv.items())
    }


def _length_targets(population: list[int], n: int) -> list[int]:
    """`n` quantiles of the population's context-length distribution.

    These are what the sample aims at. Targeting the *population's* quantiles
    rather than striding over whatever the de-duplication happens to leave is
    the whole point: the sample should look like the benchmark it is drawn
    from, including its short and long tails, since prompt length is the
    dominant driver of TTFT.
    """
    if n == 1:
        return [population[len(population) // 2]]
    last = len(population) - 1
    return [population[round(i * last / (n - 1))] for i in range(n)]


def _nearest(group: list, target: int):
    """The moment in `group` closest to `target` length; ties to shorter, then id."""
    return min(
        group,
        key=lambda m: (abs(_context_length(m) - target), _context_length(m), m.id),
    )


def _assign_targets(groups: dict[str, list], targets: list[int]) -> dict[int, str]:
    """Match each length target to a distinct conversation that can serve it.

    One moment per conversation is a hard constraint, so this is an assignment
    problem: which conversation covers which part of the length distribution.
    Solved in two deterministic stages.

    **Greedy, most-constrained-first.** Extreme targets are reachable by only
    a handful of conversations, so they are assigned before a central target
    consumes the one conversation that held the release's longest moment.
    Distance from the population median orders targets by that scarcity
    without needing a tolerance parameter.

    **Then 2-opt.** The greedy pass alone leaves central targets taking
    whatever is left, which skews the sample median high. Repeatedly swapping
    any pair of assignments that strictly lowers total error fixes that.
    Measured on the current release: median error 9.5% -> 0.6%, and mean
    deviation from the population's quantile curve 902 -> 203 characters,
    converging in 3 passes. Both stages reach the population's endpoints;
    refinement buys the shape in between. Strict improvement guarantees
    termination, so the pass cap is only a backstop.

    Both stages break ties on (distance, length, id), so the result does not
    depend on the order moments arrived in.
    """
    median = targets[len(targets) // 2]
    order = sorted(
        range(len(targets)), key=lambda i: (-abs(targets[i] - median), targets[i], i)
    )

    cache: dict[tuple[str, int], tuple] = {}

    def bid(conv: str, i: int) -> tuple:
        """(error, length, id) for the best `conv` can do on target `i`."""
        key = (conv, i)
        if key not in cache:
            m = _nearest(groups[conv], targets[i])
            cache[key] = (
                abs(_context_length(m) - targets[i]),
                _context_length(m),
                m.id,
            )
        return cache[key]

    remaining = set(groups)
    conv_of: dict[int, str] = {}
    for i in order:
        conv = min(remaining, key=lambda c: bid(c, i))
        conv_of[i] = conv
        remaining.discard(conv)

    for _pass in range(_MAX_REFINE_PASSES):
        improved = False
        for a in range(len(targets)):
            for b in range(a + 1, len(targets)):
                ca, cb = conv_of[a], conv_of[b]
                if bid(cb, a)[0] + bid(ca, b)[0] < bid(ca, a)[0] + bid(cb, b)[0]:
                    conv_of[a], conv_of[b] = cb, ca
                    improved = True
        if not improved:
            break
    return conv_of


def select_latency_subsample(
    moments: list, n: int = DEFAULT_SUBSAMPLE_SIZE
) -> list[str]:
    """Pick `n` moment ids reproducing the release's context-length distribution.

    Two constraints:

    1. **One moment per source conversation** (see _by_conversation), so no two
       samples share a transcript prefix a provider could serve from cache.
       This caps the sample at the release's conversation count.
    2. **Match the population's length distribution**, since length drives
       TTFT. Each of `n` quantiles of the full release's context lengths is
       assigned a distinct conversation, which contributes the moment nearest
       that quantile (see _assign_targets).

    Constraint 1 leaves *which* moment each conversation contributes free, and
    constraint 2 spends that freedom on coverage. An earlier rule took each
    conversation's median moment instead; that satisfied 1 but discarded every
    conversation's shortest and longest moment, so the sample spanned only the
    9th-98th percentile of prompt length (5,223-37,695 characters against a
    population of 984-55,681) -- it never measured the tails of the axis TTFT
    depends on most. Targeting population quantiles costs nothing: same
    sample size, same one-per-conversation guarantee, same runtime.

    Ties are broken by id so the result is deterministic regardless of the
    input ordering. Returns ids in the released (id-sorted) order rather than
    length order, so the committed file reads as a stable set.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not moments:
        return []

    groups = _by_conversation(moments)
    # One per conversation is the hard cap, however large n is.
    n = min(n, len(groups))
    targets = _length_targets(sorted(_context_length(m) for m in moments), n)
    conv_of = _assign_targets(groups, targets)
    return sorted(_nearest(groups[conv_of[i]], targets[i]).id for i in range(n))


def write_probe_ids(ids: list[str], path: str | Path) -> Path:
    """Write the frozen id list as sorted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(ids), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
