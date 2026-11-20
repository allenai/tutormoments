import json

import pytest

from tutormoments_build.v2 import classify_replays as R
from tutormoments_build.v2 import scoring
from tutormoments_build.v2 import taxonomy as T
from tutormoments_build.v2.taxonomy_replays import PREDICTED_NEITHER


def record(moment_id, scaffolding, rigor, over=None, *, situation=None, failed=False):
    """A classify_replays record with just the fields scoring reads."""
    if failed:
        action, routing = (
            {"scaffolding": None, "rigor": None},
            {"routed": False, "reason": R.NOT_ROUTED_FAILED},
        )
    elif scaffolding:
        action, routing = (
            {"scaffolding": True, "rigor": rigor},
            {"routed": True, "reason": None},
        )
    else:
        action, routing = (
            {"scaffolding": False, "rigor": rigor},
            {"routed": False, "reason": R.NOT_ROUTED_NO_SCAFFOLDING},
        )
    return {
        "moment_id": moment_id,
        "situation_type": situation,
        "action_direction": action,
        "over_scaffolding_routing": routing,
        "over_scaffolding": {"over_scaffolding": over} if routing["routed"] else None,
    }


SITUATIONS = {
    "so1": "scaffold_only",
    "so2": "scaffold_only",
    "sm1": "scaffold_maj",
    "ff1": "fifty_fifty",
    "rm1": "rigor_maj",
    "ro1": "rigor_only",
    "ro2": "rigor_only",
}


def test_metrics_on_known_records():
    records = [
        record("so1", True, False, over=False),  # pure scaffold
        record("so2", True, True, over=True),  # both, over-scaffolded
        record("sm1", False, True),  # missed scaffolding
        record("ff1", True, False, over=True),
        record("rm1", False, True),
        record("ro1", False, True),  # pure rigor
        record("ro2", True, True, over=False),  # rigor but also scaffolds
    ]
    s = scoring.score(records, SITUATIONS)
    assert (s["scaffold_only"]["hits"], s["scaffold_only"]["n"]) == (1, 2)
    assert (s["rigor_only"]["hits"], s["rigor_only"]["n"]) == (1, 2)
    assert (s["rigor_present"]["hits"], s["rigor_present"]["n"]) == (3, 3)
    assert (s["scaffolding_present"]["hits"], s["scaffolding_present"]["n"]) == (2, 3)
    # No-scaffolding replays count as not over-scaffolding.
    assert (s["no_over_scaffolding"]["hits"], s["no_over_scaffolding"]["n"]) == (5, 7)
    assert s["rigor_present"]["rate"] == 1.0


def test_unknown_predictions_are_excluded_not_scored():
    records = [
        record("so1", None, None, failed=True),
        record("so2", True, False, over=None),  # routed, over-scaffolding unparsed
    ]
    s = scoring.score(records, SITUATIONS)
    assert s["scaffold_only"] == {
        "situations": ["scaffold_only"],
        "hits": 1,
        "n": 1,
        "excluded": 1,
        "rate": 1.0,
    }
    assert s["no_over_scaffolding"]["n"] == 0
    assert s["no_over_scaffolding"]["excluded"] == 2
    assert s["no_over_scaffolding"]["rate"] is None


def test_gold_comes_from_ground_truth_and_must_agree():
    with pytest.raises(KeyError):
        scoring.score([record("missing", True, False)], SITUATIONS)
    with pytest.raises(ValueError):
        scoring.score([record("so1", True, False, situation="rigor_only")], SITUATIONS)


def test_main_reads_ground_truth_splits(tmp_path, capsys):
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "iteration.jsonl").write_text(
        json.dumps({"moment_id": "a", "situation_type": "rigor_only"}) + "\n"
    )
    (gt / "test.jsonl").write_text(
        json.dumps({"moment_id": "b", "situation_type": "scaffold_only"}) + "\n"
    )
    classified = tmp_path / "c.jsonl"
    classified.write_text(
        "\n".join(
            json.dumps(r)
            for r in (record("a", False, True), record("b", True, False, over=False))
        )
    )
    out = tmp_path / "scores.json"
    assert (
        scoring.main([str(classified), "--ground-truth", str(gt), "--json", str(out)])
        == 0
    )
    scores = json.loads(out.read_text())["decisions"][str(classified)]
    assert scores["rigor_only"]["rate"] == 1.0
    assert scores["scaffold_only"]["rate"] == 1.0
    assert "rigor_only" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Action divergence
# ---------------------------------------------------------------------------

DIV_SITUATIONS = {
    "s1": "scaffold_only",
    "s2": "scaffold_maj",
    "s3": "scaffold_only",
    "r1": "rigor_only",
    "r2": "rigor_maj",
    "r3": "rigor_only",
    "f1": "fifty_fifty",
}


def tax(moment_id, categories=(), *, skipped=None, actions=True, failed=False):
    taxonomy = None
    if skipped is None:
        taxonomy = {
            "actions": [{"category": c} for c in categories] if actions else None,
            "error": "overloaded" if failed else None,
            "raw": "" if failed else "x",
        }
    return {
        "moment_id": moment_id,
        "prompt_version": "2",
        "asked": skipped is None,
        "skipped": skipped,
        "taxonomy": taxonomy,
    }


def test_each_moment_weighs_the_same_and_types_count_once():
    records = [
        tax("s1", ["hint", "hint", "co_solve"]),  # hint twice still counts once
        tax("s2", ["co_solve"]),
        tax("r1", ["justify_explain", "made_up"]),  # made_up is dropped
        tax("r2", ["justify_explain"]),
        tax("f1", ["define_term"]),  # fifty_fifty is in neither group
    ]
    d = scoring.action_divergence(records, DIV_SITUATIONS)
    s, r = d["distributions"]["scaffolding"], d["distributions"]["rigor"]
    # s1 = 1/2 hint + 1/2 co_solve, s2 = co_solve; mean over moments.
    assert s["hint"] == pytest.approx(1 / 4)
    assert s["co_solve"] == pytest.approx(3 / 4)
    assert r["justify_explain"] == pytest.approx(1)
    assert sum(s.values()) == pytest.approx(1) and s["define_term"] == 0
    assert d["n"] == {"scaffolding": 2, "rigor": 2}
    assert 0 < d["jsd"] < 1


def test_moments_without_listed_types_count_as_none():
    records = [
        tax("s1", ["hint"]),
        tax("s2", skipped=PREDICTED_NEITHER),
        tax("r1", []),
        tax("r2", ["justify_explain"]),
        tax("r3", ["made_up"]),  # only unlisted labels
    ]
    d = scoring.action_divergence(records, DIV_SITUATIONS)
    assert d["n"] == {"scaffolding": 2, "rigor": 3}
    assert d["distributions"]["scaffolding"][scoring.NONE] == pytest.approx(1 / 2)
    assert d["distributions"]["rigor"][scoring.NONE] == pytest.approx(2 / 3)
    assert d["excluded"] == {}


def test_unknown_answers_are_excluded():
    records = [
        tax("s1", ["hint"]),
        tax("s2", skipped="action_direction unparseable"),
        tax("s3", actions=False),
        tax("r1", failed=True),
        tax("r2", ["justify_explain"]),
    ]
    d = scoring.action_divergence(records, DIV_SITUATIONS)
    assert d["n"] == {"scaffolding": 1, "rigor": 1}
    assert d["excluded"] == {
        "not asked: action_direction unparseable": 1,
        "request failed": 1,
        "unparseable": 1,
    }


def test_divergence_bounds():
    same = [tax("s1", ["hint"]), tax("r1", ["hint"])]
    assert scoring.action_divergence(same, DIV_SITUATIONS)["jsd"] == pytest.approx(0)
    disjoint = [tax("s1", ["hint"]), tax("r1", skipped=PREDICTED_NEITHER)]
    jsd = scoring.action_divergence(disjoint, DIV_SITUATIONS)["jsd"]
    assert jsd == pytest.approx(1)
    empty_side = [tax("s1", ["hint"]), tax("r1", failed=True)]
    assert scoring.action_divergence(empty_side, DIV_SITUATIONS)["jsd"] is None


def test_divergence_rejects_mixed_prompt_versions():
    records = [tax("s1", ["hint"]), {**tax("r1", ["hint"]), "prompt_version": "1"}]
    with pytest.raises(ValueError):
        scoring.action_divergence(records, DIV_SITUATIONS)


# ---------------------------------------------------------------------------
# Action diversity
# ---------------------------------------------------------------------------


def test_diversity_counts_only_the_called_for_family_across_moments():
    records = [
        tax("s1", ["hint", "justify_explain"]),  # rigor type ignored here
        tax("s2", ["co_solve"]),
        tax("s3", skipped=PREDICTED_NEITHER),  # no scaffolding type
        tax("r1", ["justify_explain"]),
        tax("r2", ["justify_explain", "hint"]),
        tax("f1", ["define_term"]),  # fifty_fifty ignored
    ]
    d = scoring.action_diversity(records, DIV_SITUATIONS)
    s, r = d["scaffolding"], d["rigor"]
    # Two scaffolding types used equally across moments (s3 left out): 1 bit.
    assert s["entropy_bits"] == pytest.approx(1)
    assert set(s["distribution"]) == {
        c
        for c, f in T.parse_categories(T.load_template("2")).items()
        if f == "scaffolding"
    }
    # Every rigor moment used the same one rigor type: no diversity.
    assert r["entropy_bits"] == pytest.approx(0)


def test_diversity_moments_weigh_the_same():
    records = [tax("s1", ["hint", "co_solve"]), tax("s2", ["hint"])]
    s = scoring.action_diversity(records, DIV_SITUATIONS)["scaffolding"]
    assert s["distribution"]["hint"] == pytest.approx(3 / 4)
    assert s["distribution"]["co_solve"] == pytest.approx(1 / 4)


def test_diversity_without_any_family_type_is_none():
    records = [tax("r1", []), tax("r2", failed=True)]
    d = scoring.action_diversity(records, DIV_SITUATIONS)
    assert d["rigor"]["entropy_bits"] is None and d["rigor"]["distribution"] is None
    assert d["scaffolding"]["entropy_bits"] is None
    assert d["excluded"] == {"request failed": 1}
