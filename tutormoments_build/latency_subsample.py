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


def _context_length(moment) -> int:
    """Total characters of pre-cut transcript context for a moment.

    A proxy for prompt length, which is the dominant driver of TTFT -- longer
    prompts mean more prefill before the first token appears.
    """
    return sum(len(turn.get("text") or "") for turn in (moment.context or []))


def _one_per_conversation(moments: list) -> list:
    """Keep one moment per source conversation, nearest that conversation's
    median context length.

    Moments cut from the same conversation share a long transcript prefix, and
    providers with automatic prefix caching serve the second one from cache.
    Measured on the previous 40-moment subsample: every `gpt-5.5` turn-1 cache
    hit came from a conversation contributing more than one moment, reading
    back 4.9k-9.0k tokens. Those samples are labelled cold and are not cold,
    which biases the cold figure downward on exactly the providers that cache
    silently. De-duplicating removes the shared prefix, so each sample is an
    independent measurement.

    The median-length representative keeps the length distribution honest --
    taking the first or shortest moment per conversation would skew the sample.
    Ties break by id for determinism.
    """
    by_conv: dict[str, list] = {}
    for m in moments:
        by_conv.setdefault((m.provenance or {}).get("conv_id") or m.id, []).append(m)

    reps = []
    for _conv, group in sorted(by_conv.items()):
        ordered = sorted(group, key=lambda m: (_context_length(m), m.id))
        reps.append(ordered[len(ordered) // 2])
    return reps


def select_latency_subsample(
    moments: list, n: int = DEFAULT_SUBSAMPLE_SIZE
) -> list[str]:
    """Pick `n` moment ids spanning the context-length distribution.

    Two constraints, in order:

    1. **One moment per source conversation** (see _one_per_conversation), so
       no two samples share a transcript prefix that a provider could serve
       from cache. This caps the sample at the release's conversation count.
    2. **Stride across context length**, since length drives TTFT -- so the
       sample covers short, middling and long prompts rather than clustering
       wherever an arbitrary id ordering happened to land.

    Ties are broken by id so the result is deterministic regardless of the
    input ordering. Returns ids in the released (id-sorted) order rather than
    length order, so the committed file reads as a stable set.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not moments:
        return []

    ordered = sorted(
        _one_per_conversation(moments), key=lambda m: (_context_length(m), m.id)
    )
    if n >= len(ordered):
        return sorted(m.id for m in ordered)

    # Evenly-spaced indices across the full sorted range, endpoints included,
    # so the shortest and longest prompts are always represented.
    step = (len(ordered) - 1) / (n - 1) if n > 1 else 0
    picked = {ordered[round(i * step)].id for i in range(n)}
    return sorted(picked)


def write_probe_ids(ids: list[str], path: str | Path) -> Path:
    """Write the frozen id list as sorted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sorted(ids), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
