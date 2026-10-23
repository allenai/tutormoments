"""Tests for the scaffolding-vs-rigor KL divergence and its human reference.

The KL method must stay the paper's (kl_divergence_table.ipynb, now deleted;
its locked .tex is the paper's record): per situation, macro % -> pseudo-counts
over that situation's moments, add-one smoothing, KL in nats.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tutormoments import taxonomy as tx

SHA = "a" * 40


def _f(moment_id, situation, category, **kw):
    return tx.Facet(
        moment_id=moment_id,
        transcript_id=kw.pop("transcript_id", "t"),
        turn_start=kw.pop("turn_start", 0),
        turn_end=kw.pop("turn_end", 1),
        statement_index=kw.pop("statement_index", 0),
        statement=kw.pop("statement", "The tutor acts."),
        annotation_type="scaffolding",
        situation_label=situation,
        category=category,
        **kw,
    )


# ---------------------------------------------------------------------------
# kl_situation
# ---------------------------------------------------------------------------


def _hand_kl(s_counts: dict, r_counts: dict) -> tuple[float, float]:
    """The notebook's arithmetic, spelled out: smoothed pseudo-counts -> KL."""
    k = len(tx.CATEGORY_LETTERS)
    s = [1.0] * k
    r = [1.0] * k
    for letter, c in s_counts.items():
        s[tx.CATEGORY_LETTERS.index(letter)] += c
    for letter, c in r_counts.items():
        r[tx.CATEGORY_LETTERS.index(letter)] += c
    s = [v / sum(s) for v in s]
    r = [v / sum(r) for v in r]
    return (
        sum(a * math.log(a / b) for a, b in zip(s, r)),
        sum(b * math.log(b / a) for a, b in zip(s, r)),
    )


def test_kl_matches_the_paper_formula():
    # S: two moments, all A -> pseudo-count A=2. R: r1 half A half G, r2 all
    # G -> macro A=25%, G=75% over 2 moments -> A=0.5, G=1.5.
    facets = [
        _f("s1", "scaffolding", "A"),
        _f("s2", "scaffolding", "A"),
        _f("r1", "rigor", "A"),
        _f("r1", "rigor", "G"),
        _f("r2", "rigor", "G"),
    ]
    kl = tx.kl_situation(facets)
    s_r, r_s = _hand_kl({"A": 2}, {"A": 0.5, "G": 1.5})
    assert kl["s_r"] == pytest.approx(s_r)
    assert kl["r_s"] == pytest.approx(r_s)
    assert (kl["n_scaffolding"], kl["n_rigor"]) == (2, 2)


def test_kl_matches_the_pandas_macro_distribution_method():
    """Parity with the notebook's own pipeline (macro_distribution -> KL)."""
    pytest.importorskip("pandas")
    letters = tx.CATEGORY_LETTERS
    facets = []
    for i in range(12):
        sit = "scaffolding" if i % 2 else "rigor"
        for j in range(1 + i % 4):
            facets.append(_f(f"m{i}", sit, letters[(i * 3 + j * 5) % len(letters)]))
    facets.append(_f("mx", "mixed", "A"))  # not in either side

    dists = tx.macro_distribution(
        tx.facets_to_dataframe(facets), group_keys=("situation_label",)
    )
    probs = {}
    for sit in ("scaffolding", "rigor"):
        sub = dists[dists["situation_label"] == sit].set_index("letter")
        pcts = sub["mean_pct"].reindex(letters, fill_value=0.0).tolist()
        probs[sit] = tx.smoothed_probs(pcts, int(sub["n_moments"].iloc[0]))
    s, r = probs["scaffolding"], probs["rigor"]

    kl = tx.kl_situation(facets)
    assert kl["s_r"] == pytest.approx(tx.kl_nats(s, r), abs=1e-12)
    assert kl["r_s"] == pytest.approx(tx.kl_nats(r, s), abs=1e-12)
    assert (kl["n_scaffolding"], kl["n_rigor"]) == (6, 6)


def test_kl_is_asymmetric_and_zero_for_identical_sides():
    facets = [_f("s1", "scaffolding", "A"), _f("s2", "scaffolding", "B")] + [
        _f(f"r{i}", "rigor", "G") for i in range(2)
    ]
    kl = tx.kl_situation(facets)
    assert kl["s_r"] > 0 and kl["r_s"] > 0
    assert kl["s_r"] != pytest.approx(kl["r_s"])

    same = [_f("s1", "scaffolding", "A"), _f("r1", "rigor", "A")]
    kl = tx.kl_situation(same)
    assert kl["s_r"] == pytest.approx(0.0, abs=1e-12)
    assert kl["r_s"] == pytest.approx(0.0, abs=1e-12)


def test_kl_is_none_when_a_situation_is_missing():
    kl = tx.kl_situation([_f("s1", "scaffolding", "A"), _f("x", "mixed", "G")])
    assert kl == {"s_r": None, "r_s": None, "n_scaffolding": 1, "n_rigor": 0}
    assert tx.kl_situation([]) == {
        "s_r": None,
        "r_s": None,
        "n_scaffolding": 0,
        "n_rigor": 0,
    }


def test_kl_counts_a_moment_once_across_trials_and_skips_unclassified():
    """Trials pool under one moment id, so n is unique moments; facets with no
    category (unclassified) neither add weight nor make a moment."""
    pooled = [
        _f("s1", "scaffolding", "A"),
        _f("s1", "scaffolding", "A", statement_index=1),  # trial 2
        _f("s2", "scaffolding", None),
        _f("r1", "rigor", "G"),
    ]
    kl = tx.kl_situation(pooled)
    assert (kl["n_scaffolding"], kl["n_rigor"]) == (1, 1)
    single = tx.kl_situation([_f("s1", "scaffolding", "A"), _f("r1", "rigor", "G")])
    assert kl["s_r"] == pytest.approx(single["s_r"])


# ---------------------------------------------------------------------------
# human_reference
# ---------------------------------------------------------------------------


def _moment(mid, conv_tail, ts, te, dimension):
    return {
        "id": f"balanced_520:{mid}",
        "dimension": dimension,
        "provenance": {
            "conv_id": f"tutor_student_{conv_tail}",
            "turn_start": ts,
            "turn_end": te,
        },
    }


def _row(transcript, ts, te, category, *, source="human", sit="mixed", idx=0):
    return {
        "source": source,
        "tutor_model": None if source == "human" else "m",
        "prompt_mode": None if source == "human" else "plain",
        "moment_id": f"{transcript}__{ts}_{te}__ann",
        "transcript_id": transcript,
        "turn_start": ts,
        "turn_end": te,
        "statement_index": idx,
        "statement": f"The tutor does {category}.",
        "situation_label": sit,
        "action_label": None,
        "result_label": None,
        "stance_prefixed": False,
        "category": category,
        "excluded_reason": None if category else "non_action",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def release(tmp_path):
    moments = _write_jsonl(
        tmp_path / "moments.jsonl",
        [
            _moment("m1", "t1", 10, 20, "scaffolding"),
            _moment("m2", "t2", 5, 9, "rigor"),
            _moment("m3", "t3", 0, 4, "rigor"),  # no human rows at its span
        ],
    )
    rows = [
        # m1: two annotators at the same span -> one pooled moment. Their own
        # situation labels disagree with the moment's; the moment wins.
        _row("t1", 10, 20, "A", sit="rigor"),
        _row("t1", 10, 20, "B", sit="mixed", idx=1),
        _row("t1", 10, 20, "C", sit="rigor"),
        # m2
        _row("t2", 5, 9, "G", sit="rigor"),
        # Ignored: excluded facet, an LM row at a moment span, another span.
        _row("t2", 5, 9, None, sit="rigor"),
        _row("t2", 5, 9, "A", source="benchmark_520", sit="rigor"),
        _row("t2", 5, 10, "A", sit="rigor"),
    ]
    taxonomy = _write_jsonl(tmp_path / "action_taxonomy.jsonl", rows)
    return {"moments.jsonl": moments, "action_taxonomy.jsonl": taxonomy}


def test_human_reference_facets_join_by_span_pool_annotators_and_use_dimension(
    release,
):
    facets = tx.human_reference_facets(
        release["moments.jsonl"], release["action_taxonomy.jsonl"]
    )
    by_moment = {}
    for f in facets:
        by_moment.setdefault(f.moment_id, []).append(f)
    assert set(by_moment) == {"balanced_520:m1", "balanced_520:m2"}
    assert sorted(f.category for f in by_moment["balanced_520:m1"]) == ["A", "B", "C"]
    assert {f.situation_label for f in by_moment["balanced_520:m1"]} == {"scaffolding"}
    assert [f.category for f in by_moment["balanced_520:m2"]] == ["G"]


def test_human_reference_downloads_at_the_revision_and_computes_kl(release):
    calls = []

    def download(dataset, filename, revision):
        calls.append((dataset, filename, revision))
        return release[filename]

    ref = tx.human_reference("org/ds", SHA, download=download)
    assert sorted(calls) == [
        ("org/ds", "action_taxonomy.jsonl", SHA),
        ("org/ds", "moments.jsonl", SHA),
    ]
    assert ref["label"] == tx.HUMAN_REFERENCE_LABEL
    assert (ref["dataset"], ref["revision"]) == ("org/ds", SHA)
    assert ref["kl"]["n_scaffolding"] == 1 and ref["kl"]["n_rigor"] == 1
    # m1 is a third each A/B/C; m2 is all G.
    s_r, r_s = _hand_kl({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}, {"G": 1})
    assert ref["kl"]["s_r"] == pytest.approx(s_r)
    assert ref["kl"]["r_s"] == pytest.approx(r_s)


@pytest.mark.parametrize("revision", ["main", "3bb0c104", "A" * 40, "", None])
def test_human_reference_rejects_a_revision_that_is_not_a_full_sha(revision):
    def download(*_a):
        raise AssertionError("must not download at a floating revision")

    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        tx.human_reference("org/ds", revision, download=download)


def test_human_reference_pin_is_a_full_sha():
    dataset, revision = tx.HUMAN_REFERENCE
    assert dataset == "allenai/tutormoments-preview"
    assert tx._FULL_SHA_RE.match(revision)


# ---------------------------------------------------------------------------
# Online audit: the pinned reference reproduces the documented numbers.
# Skips when the Hub is unreachable and nothing is cached.
# ---------------------------------------------------------------------------


def _real_download(dataset, filename, revision):
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(dataset, filename, repo_type="dataset", revision=revision)
        )
    except Exception as e:  # noqa: BLE001 -- no network / not cached / HF down
        pytest.skip(f"pinned release unreachable: {e}")


def test_pinned_human_reference_matches_the_release_and_documented_values():
    from tutormoments.moments import file_sha256

    dataset, revision = tx.HUMAN_REFERENCE
    manifest = json.loads(
        _real_download(dataset, "action_taxonomy.manifest.json", revision).read_text(
            encoding="utf-8"
        )
    )
    taxonomy_path = _real_download(dataset, "action_taxonomy.jsonl", revision)
    assert file_sha256(taxonomy_path) == manifest["file_sha256"]

    ref = tx.human_reference(dataset, revision, download=_real_download)
    kl = ref["kl"]
    assert (kl["n_scaffolding"], kl["n_rigor"]) == (260, 258)
    assert kl["s_r"] == pytest.approx(0.524, abs=5e-4)
    assert kl["r_s"] == pytest.approx(0.591, abs=5e-4)
