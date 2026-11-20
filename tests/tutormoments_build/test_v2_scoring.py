import json

import pytest

from tutormoments_build.v2 import classify_replays as R
from tutormoments_build.v2 import scoring


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
    scores = json.loads(out.read_text())[str(classified)]
    assert scores["rigor_only"]["rate"] == 1.0
    assert scores["scaffold_only"]["rate"] == 1.0
    assert "rigor_only" in capsys.readouterr().out
