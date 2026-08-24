"""Tests for tutormoments_build.v2.evaluate."""

import json

import pytest

E = pytest.importorskip("tutormoments_build.v2.evaluate")


def record(moment_id, *, gold, scaffolding=None, rigor=None, over=None, asked=False):
    """One prediction record, trimmed to the fields the scorer reads."""
    return {
        "moment_id": moment_id,
        "labels": gold,
        "action_direction": (
            None
            if scaffolding is None and rigor is None
            else {"scaffolding": scaffolding, "rigor": rigor}
        ),
        "over_scaffolding": {"over_scaffolding": over} if asked else None,
    }


GOLD_YES = {
    "scaffolding_present": True,
    "rigor_present": True,
    "over_scaffolding_present": True,
}
GOLD_NO = {
    "scaffolding_present": False,
    "rigor_present": False,
    "over_scaffolding_present": False,
}


# ===========================================================================
# Metrics
# ===========================================================================


def test_score_counts_the_confusion_matrix_and_f1():
    # 2 TP, 1 FP, 1 FN, 1 TN -> precision 2/3, recall 2/3, F1 2/3
    pairs = [(True, True), (True, True), (False, True), (True, False), (False, False)]
    out = E.score(pairs)

    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (2, 1, 1, 1)
    assert out["support"] == 3
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)
    assert out["accuracy"] == pytest.approx(3 / 5)


def test_score_is_zero_rather_than_undefined_with_no_positives():
    out = E.score([(False, False), (False, False)])

    assert (out["precision"], out["recall"], out["f1"]) == (0.0, 0.0, 0.0)
    assert out["accuracy"] == 1.0


def test_score_of_nothing_does_not_divide_by_zero():
    assert E.score([])["f1"] == 0.0


# ===========================================================================
# Collecting pairs
# ===========================================================================


def test_unparsed_predictions_are_not_scored_as_negatives():
    """A None prediction is missing data, not a "no" answer."""
    records = [
        record("a", gold=GOLD_YES, scaffolding=True, rigor=True),
        record("b", gold=GOLD_YES, scaffolding=None, rigor=None),
    ]

    pairs, missing = E.collect(
        records, "action_direction", "scaffolding", "scaffolding_present"
    )

    assert pairs == [(True, True)]
    assert missing == 1


def test_over_scaffolding_is_scored_only_where_it_was_asked():
    records = [
        record("a", gold=GOLD_YES, over=True, asked=True),
        record("b", gold=GOLD_NO, asked=False),
    ]

    pairs, missing = E.collect(
        records, "over_scaffolding", "over_scaffolding", "over_scaffolding_present"
    )

    assert pairs == [(True, True)]
    assert missing == 1


def test_evaluate_scores_every_task():
    rows = E.evaluate(
        [
            record(
                "a", gold=GOLD_YES, scaffolding=True, rigor=False, over=True, asked=True
            )
        ]
    )

    assert [row["task"] for row in rows] == ["scaffolding", "rigor", "over-scaffolding"]
    by_task = {row["task"]: row for row in rows}
    assert by_task["scaffolding"]["tp"] == 1
    assert by_task["rigor"]["fn"] == 1
    assert by_task["over-scaffolding"]["tp"] == 1


# ===========================================================================
# CLI
# ===========================================================================


def _write_predictions(tmp_path, model="claude-opus-5", split="iteration"):
    model_dir = tmp_path / model
    model_dir.mkdir(parents=True)
    records = [
        record("a", gold=GOLD_YES, scaffolding=True, rigor=True, over=True, asked=True),
        record("b", gold=GOLD_NO, scaffolding=True, rigor=False),
    ]
    (model_dir / f"{split}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return str(tmp_path)


def test_cli_reports_and_writes_json(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path)
    out = tmp_path / "scores.json"

    assert E.main(["--pred-dir", pred_dir, "--json", str(out)]) == 0

    assert "F1" in capsys.readouterr().out
    scores = json.loads(out.read_text(encoding="utf-8"))
    assert scores[0]["model"] == "claude-opus-5"
    assert scores[0]["split"] == "iteration"
    scaffolding = scores[0]["tasks"][0]
    assert (scaffolding["tp"], scaffolding["fp"]) == (1, 1)
    assert scaffolding["f1"] == pytest.approx(2 / 3)


def test_cli_fails_when_there_is_nothing_to_score(tmp_path):
    (tmp_path / "claude-opus-5").mkdir()

    assert E.main(["--pred-dir", str(tmp_path)]) == 1


def test_cli_scores_only_the_model_asked_for(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path)
    _write_predictions(tmp_path, model="gemini-3.5-flash")

    assert E.main(["--pred-dir", pred_dir]) == 0
    assert "gemini-3.5-flash" in capsys.readouterr().out

    assert E.main(["--pred-dir", pred_dir, "--model", "claude-opus-5"]) == 0
    assert "gemini-3.5-flash" not in capsys.readouterr().out
