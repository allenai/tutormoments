"""Selection rule for the frozen latency-probe subsample.

The subsample is committed, not computed at run time, so these tests are what
keep it auditable: the rule is exercised directly on synthetic moments, and
the committed list is recomputed against a local release to prove it is what
the rule produces. That gives full reproducibility without shipping a
regeneration command that would invite re-picking the sample against a newer
release and silently breaking the latency time series.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tutormoments.moments import PROBE_IDS_FILENAME, read_probe_ids, subsample_id

pytest.importorskip("tutormoments_build")

from tutormoments_build.latency_subsample import (  # noqa: E402
    select_latency_subsample,
    write_probe_ids,
)

_REPO_ROOT = Path(__file__).parent.parent.parent
_RELEASE_DIR = _REPO_ROOT / "data" / "balanced_520_release"
# Canonical location is the runtime package: the probe list is an output
# the runtime reads, not an input to the build (see packaged_probe_ids).
_COMMITTED_IDS = _REPO_ROOT / "src" / "tutormoments" / PROBE_IDS_FILENAME


def _moment(mid: str, context_chars: int, conv: str | None = None):
    """A stand-in carrying only what the selection rule reads."""
    return SimpleNamespace(
        id=mid,
        context=[{"text": "x" * context_chars}],
        provenance={"conv_id": conv or mid},
    )


def _moments(lengths: list[int]):
    """One moment per conversation, so length striding is what's exercised."""
    return [_moment(f"m{i:03d}", n) for i, n in enumerate(lengths)]


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def test_selection_is_deterministic():
    moments = _moments([100, 5, 900, 42, 7, 613, 88, 3])
    assert select_latency_subsample(moments, 4) == select_latency_subsample(moments, 4)


def test_selection_is_independent_of_input_order():
    """Ties break by id, so a shuffled release yields the same sample."""
    lengths = [100, 5, 900, 42, 7, 613, 88, 3, 42, 42]
    moments = _moments(lengths)
    shuffled = [moments[i] for i in (7, 2, 9, 0, 4, 1, 8, 3, 6, 5)]
    assert select_latency_subsample(moments, 5) == select_latency_subsample(shuffled, 5)


def test_selection_spans_the_length_distribution():
    """TTFT is length-driven, so the sample must cover short and long prompts
    rather than cluster in the middle. With one moment per conversation the
    endpoints are reachable, so they are asserted exactly here."""
    moments = _moments(list(range(1, 101)))  # ids m000..m099, lengths 1..100
    picked = set(select_latency_subsample(moments, 5))
    by_id = {m.id: len(m.context[0]["text"]) for m in moments}
    lengths = sorted(by_id[i] for i in picked)
    assert lengths[0] == 1, "shortest prompt must be represented"
    assert lengths[-1] == 100, "longest prompt must be represented"
    # Evenly spread rather than bunched anywhere.
    gaps = [b - a for a, b in zip(lengths, lengths[1:])]
    assert max(gaps) - min(gaps) <= 1


def test_selection_takes_one_moment_per_conversation():
    """Moments from one conversation share a transcript prefix, which
    providers with automatic prefix caching serve from cache -- so a second
    moment from the same conversation is not an independent measurement, and
    a sample labelled cold would not be cold."""
    moments = [
        _moment("a1", 100, conv="A"),
        _moment("a2", 200, conv="A"),
        _moment("a3", 300, conv="A"),
        _moment("b1", 150, conv="B"),
        _moment("c1", 250, conv="C"),
    ]
    picked = select_latency_subsample(moments, 10)
    convs = {m.provenance["conv_id"] for m in moments if m.id in set(picked)}
    assert len(picked) == 3, "one per conversation, not one per moment"
    assert convs == {"A", "B", "C"}


def test_conversation_representative_is_median_length():
    """Taking the first or shortest moment per conversation would skew the
    sample's length distribution; the median keeps it honest."""
    moments = [
        _moment("a1", 10, conv="A"),
        _moment("a2", 500, conv="A"),
        _moment("a3", 9000, conv="A"),
    ]
    assert select_latency_subsample(moments, 10) == ["a2"]


def test_dedup_caps_the_sample_at_the_conversation_count():
    """n cannot exceed the number of conversations, however large it is."""
    moments = [_moment(f"m{i}", 100 + i, conv="only-one") for i in range(20)]
    assert len(select_latency_subsample(moments, 20)) == 1


def test_selection_returns_exactly_n():
    moments = _moments(list(range(1, 201)))
    assert len(select_latency_subsample(moments, 40)) == 40


def test_selection_returns_all_when_n_exceeds_population():
    moments = _moments([10, 20, 30])
    assert select_latency_subsample(moments, 99) == ["m000", "m001", "m002"]


def test_selection_handles_empty_release():
    assert select_latency_subsample([], 10) == []


def test_selection_rejects_non_positive_n():
    with pytest.raises(ValueError, match="must be positive"):
        select_latency_subsample(_moments([1, 2]), 0)


def test_selection_output_is_sorted_by_id():
    """The committed file reads as a stable set, not in length order."""
    ids = select_latency_subsample(_moments([900, 5, 300, 42, 7]), 3)
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# subsample_id
# ---------------------------------------------------------------------------


def test_subsample_id_ignores_ordering():
    assert subsample_id(["b", "a", "c"]) == subsample_id(["a", "b", "c"])


def test_subsample_id_changes_when_the_sample_changes():
    """A drifted sample must be visibly incomparable, not quietly different."""
    assert subsample_id(["a", "b"]) != subsample_id(["a", "c"])


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path):
    ids = ["m003", "m001", "m002"]
    write_probe_ids(ids, tmp_path / PROBE_IDS_FILENAME)
    assert read_probe_ids(tmp_path) == sorted(ids)


def test_read_probe_ids_returns_none_when_absent(tmp_path):
    """Releases predating the probe carry no list; callers must fall back."""
    assert read_probe_ids(tmp_path) is None


# ---------------------------------------------------------------------------
# The committed artifact matches the rule
# ---------------------------------------------------------------------------


def _released_moments():
    """Load the released moments from a local release dir or the HF cache.

    Tries the local release first (offline, exact), then the published
    dataset, which is what most checkouts actually have. Returns None when
    neither is reachable so the audit skips rather than fails on a machine
    with no dataset.
    """
    from tutormoments.moments import load_moments

    if (_RELEASE_DIR / "moments.jsonl").exists():
        return load_moments(data_path=_RELEASE_DIR)[0]
    try:
        return load_moments(dataset="allenai/tutormoments-preview", config="moments")[0]
    except Exception:  # noqa: BLE001 -- no network / not cached / HF down
        return None


@pytest.mark.skipif(
    not _COMMITTED_IDS.exists(), reason="committed probe id list not present"
)
def test_committed_ids_match_the_selection_rule():
    """The committed list is exactly what the rule produces on the release.

    This is the audit that replaces a regeneration command: the frozen sample
    stays verifiable without providing a button that would re-pick it. If this
    fails, either the rule changed or the list was hand-edited -- both break
    comparability with every prior latency measurement.
    """
    moments = _released_moments()
    if moments is None:
        pytest.skip("no local release and the published dataset is unreachable")

    committed = json.loads(_COMMITTED_IDS.read_text(encoding="utf-8"))
    assert select_latency_subsample(moments, len(committed)) == sorted(committed)


@pytest.mark.skipif(
    not _COMMITTED_IDS.exists(), reason="committed probe id list not present"
)
def test_committed_ids_span_the_length_distribution():
    """The committed sample must actually cover short and long prompts.

    Context length in this release spans roughly 1k to 56k characters, and
    prompt length drives TTFT -- a sample clustered at one end would report a
    latency that no representative moment produces.
    """
    moments = _released_moments()
    if moments is None:
        pytest.skip("no local release and the published dataset is unreachable")

    committed = set(json.loads(_COMMITTED_IDS.read_text(encoding="utf-8")))
    lengths = {m.id: sum(len(t.get("text") or "") for t in m.context) for m in moments}
    assert committed <= set(lengths), "committed ids must all exist in the release"

    picked = sorted(lengths[i] for i in committed)
    population = sorted(lengths.values())

    def q(a, f):
        return a[min(len(a) - 1, int(len(a) * f))]

    # Not the exact population extremes: one-moment-per-conversation drops
    # atypical moments whose conversation's median sits elsewhere. What must
    # hold is that the sample is representative and spans the bulk of the
    # distribution -- measured, the median lands within ~2% of the
    # population's.
    assert abs(q(picked, 0.5) - q(population, 0.5)) < 0.10 * q(population, 0.5)
    assert picked[0] <= q(population, 0.25)
    assert picked[-1] >= q(population, 0.75)
