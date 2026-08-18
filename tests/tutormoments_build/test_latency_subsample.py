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
    """One moment per conversation, so length coverage is what's exercised."""
    return [_moment(f"m{i:03d}", n) for i, n in enumerate(lengths)]


def _len(moment) -> int:
    """Context length of a stand-in moment, as the selection rule sees it."""
    return sum(len(t["text"]) for t in moment.context)


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


def test_lone_conversation_contributes_its_most_typical_moment():
    """With room for one sample, take the moment nearest the population's
    median length rather than an arbitrary or extreme one."""
    moments = [
        _moment("a1", 10, conv="A"),
        _moment("a2", 500, conv="A"),
        _moment("a3", 9000, conv="A"),
    ]
    assert select_latency_subsample(moments, 10) == ["a2"]


def test_sample_reaches_the_population_tails():
    """The de-duplication fixes *how many* moments each conversation gives, not
    *which*. Spending that freedom on coverage is what keeps the tails in.

    Every conversation here has a short, a middling and a long moment. Taking
    each conversation's median (the previous rule) would return 50/51/52 and
    never measure a prompt outside that band, even though the release spans
    1-100. Prompt length is the dominant driver of TTFT, so the untouched tails
    were the part worth measuring most.
    """
    moments = [
        _moment("a_short", 1, conv="A"),
        _moment("a_mid", 50, conv="A"),
        _moment("a_long", 99, conv="A"),
        _moment("b_short", 2, conv="B"),
        _moment("b_mid", 51, conv="B"),
        _moment("b_long", 100, conv="B"),
        _moment("c_short", 3, conv="C"),
        _moment("c_mid", 52, conv="C"),
        _moment("c_long", 98, conv="C"),
    ]
    picked = select_latency_subsample(moments, 3)
    lengths = sorted({m.id: _len(m) for m in moments}[i] for i in picked)

    assert lengths[0] == 1, "the release's shortest prompt must be measured"
    assert lengths[-1] == 100, "the release's longest prompt must be measured"
    assert 40 <= lengths[1] <= 60, "and the middle must still be represented"


def test_tail_coverage_does_not_reuse_a_conversation():
    """Reaching the tails must not come at the cost of prefix independence:
    the shortest and longest moments here live in the same conversation, so
    only one of them can be taken."""
    moments = [
        _moment("a_short", 1, conv="A"),
        _moment("a_long", 100, conv="A"),
        _moment("b_mid", 50, conv="B"),
    ]
    picked = select_latency_subsample(moments, 2)
    assert len(picked) == 2
    assert not {"a_short", "a_long"} <= set(picked), "one moment per conversation"
    assert "b_mid" in picked


def test_order_independent_with_multi_moment_conversations():
    """Determinism has to survive the assignment step, not just the grouping:
    ties break on (distance, length, id) at every stage."""
    moments = [
        _moment(f"{c}{i}", n, conv=c)
        for c, lens in (("A", (5, 60, 90)), ("B", (5, 61, 90)), ("C", (7, 62, 91)))
        for i, n in enumerate(lens)
    ]
    shuffled = [moments[i] for i in (4, 0, 8, 3, 6, 1, 7, 2, 5)]
    assert select_latency_subsample(moments, 3) == select_latency_subsample(shuffled, 3)


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
    """The committed sample must reproduce the release's length distribution.

    Context length in this release spans 984 to 55,681 characters, and prompt
    length drives TTFT -- a sample clustered in the middle would report a
    latency that says nothing about short or long prompts. One moment per
    conversation constrains how many samples there are, not which, so the
    exact endpoints are reachable and are asserted here rather than conceded.
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
        return a[min(len(a) - 1, round(f * (len(a) - 1)))]

    assert picked[0] == population[0], "the shortest prompt in the release"
    assert picked[-1] == population[-1], "the longest prompt in the release"
    assert abs(q(picked, 0.5) - q(population, 0.5)) < 0.02 * q(population, 0.5)

    # Not just the endpoints: the sample must track the population's shape
    # across the whole range, or it would span the right interval while
    # clustering inside it. Deviations are measured against the median prompt
    # rather than against each quantile's own value -- a few hundred
    # characters is a rounding error at the long tail but a large *fraction*
    # at the short one, which would make a relative bound meaninglessly tight
    # there. Measured on this release: mean 203 characters, worst 797, against
    # a median prompt of ~15k.
    worst = max(abs(q(picked, i / 20) - q(population, i / 20)) for i in range(21))
    assert worst < 0.10 * q(population, 0.5), "sample must track the population's shape"
