"""Tests for tutormoments_build.v2.evaluate."""

import json

import pytest

E = pytest.importorskip("tutormoments_build.v2.evaluate")


def record(
    moment_id,
    *,
    gold,
    scaffolding=None,
    rigor=None,
    over=None,
    asked=False,
    turns=None,
):
    """One prediction record, trimmed to the fields the scorer reads."""
    return {
        "moment_id": moment_id,
        "labels": gold,
        "post_cut_dialogue_rows": turns,
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
    assert by_task["scaffolding"]["all"]["tp"] == 1
    assert by_task["rigor"]["all"]["fn"] == 1
    assert by_task["over-scaffolding"]["all"]["tp"] == 1


def test_evaluate_has_no_agreed_block_without_the_ground_truth():
    """The full numbers need only the predictions; the subset needs the join."""
    rows = E.evaluate([record("a", gold=GOLD_YES, scaffolding=True, rigor=True)])

    assert all(row["agreed"] is None for row in rows)


# ===========================================================================
# The annotator-agreement subset
# ===========================================================================

AGREED = dict.fromkeys(GOLD_YES, True)
CONTESTED = dict.fromkeys(GOLD_YES, False)


def test_agreed_records_keeps_only_moments_both_passes_called_the_same():
    records = [record("a", gold=GOLD_YES), record("b", gold=GOLD_YES)]
    agreement = {"a": AGREED, "b": CONTESTED}

    kept, unknown = E.agreed_records(records, agreement, "scaffolding_present")

    assert [r["moment_id"] for r in kept] == ["a"]
    assert unknown == 0


def test_agreement_is_per_construct():
    """A moment can be agreed for one construct and contested for another."""
    records = [record("a", gold=GOLD_YES)]
    agreement = {"a": {"scaffolding_present": True, "rigor_present": False}}

    assert E.agreed_records(records, agreement, "scaffolding_present")[0] != []
    assert E.agreed_records(records, agreement, "rigor_present")[0] == []


def test_moments_missing_from_the_ground_truth_land_in_neither_subset():
    records = [record("a", gold=GOLD_YES), record("gone", gold=GOLD_YES)]

    kept, unknown = E.agreed_records(records, {"a": AGREED}, "scaffolding_present")

    assert [r["moment_id"] for r in kept] == ["a"]
    assert unknown == 1


def test_evaluate_scores_the_agreed_subset_separately():
    """The contested moment counts in full but not in the agreed block."""
    records = [
        record("a", gold=GOLD_YES, scaffolding=True, rigor=True),
        record("b", gold=GOLD_YES, scaffolding=True, rigor=True),
    ]
    agreement = {"a": AGREED, "b": CONTESTED}

    by_task = {row["task"]: row for row in E.evaluate(records, agreement)}
    scaffolding = by_task["scaffolding"]

    assert scaffolding["all"]["tp"] == 2
    assert scaffolding["agreed"]["tp"] == 1
    assert scaffolding["agreed"]["contested"] == 1
    assert scaffolding["agreed"]["unknown"] == 0


def test_agreed_subset_keeps_the_false_positives_it_inherits():
    """Gold is a union, so a gold negative is agreed and its FP stays scored."""
    records = [
        record("a", gold=GOLD_YES, scaffolding=True, rigor=True),  # contested TP
        record("b", gold=GOLD_NO, scaffolding=True, rigor=True),  # agreed FP
    ]
    agreement = {"a": CONTESTED, "b": AGREED}

    scaffolding = E.evaluate(records, agreement)[0]

    assert (scaffolding["all"]["tp"], scaffolding["all"]["fp"]) == (1, 1)
    assert (scaffolding["agreed"]["tp"], scaffolding["agreed"]["fp"]) == (0, 1)


def test_stale_predictions_are_detected():
    """Agreement is joined on moment_id, which only holds while gold matches."""
    predictions = [record("a", gold=GOLD_YES), record("b", gold=GOLD_NO)]
    ground_truth = [
        {"moment_id": "a", "labels": GOLD_YES},
        {"moment_id": "b", "labels": GOLD_YES},
    ]

    assert E.stale_moments(predictions, ground_truth) == ["b"]


# ===========================================================================
# The short-post-cut subset
# ===========================================================================


def test_short_records_keeps_only_the_moments_below_the_threshold():
    records = [
        record("short", gold=GOLD_YES, turns=9),
        record("boundary", gold=GOLD_YES, turns=10),
        record("long", gold=GOLD_YES, turns=40),
    ]

    kept, unknown = E.short_records(records, 10)

    assert [r["moment_id"] for r in kept] == ["short"]
    assert unknown == 0


def test_records_without_a_turn_count_are_left_out_of_the_short_subset():
    """A record predating post_cut_dialogue_rows cannot be placed either side."""
    records = [record("a", gold=GOLD_YES, turns=1), record("old", gold=GOLD_YES)]

    kept, unknown = E.short_records(records, 10)

    assert [r["moment_id"] for r in kept] == ["a"]
    assert unknown == 1


def test_evaluate_scores_the_short_subset_separately():
    records = [
        record("short", gold=GOLD_YES, scaffolding=True, rigor=True, turns=3),
        record("long", gold=GOLD_YES, scaffolding=True, rigor=True, turns=30),
    ]

    scaffolding = E.evaluate(records)[0]

    assert scaffolding["all"]["tp"] == 2
    assert scaffolding["short"]["tp"] == 1
    assert scaffolding["short"]["moments"] == 1
    assert scaffolding["short"]["of_moments"] == 2
    assert scaffolding["short"]["turns"] == 10


def test_the_short_subset_drops_gold_negatives_too():
    """Unlike the agreed subset, it slices the excerpt rather than the labels."""
    records = [
        record("short", gold=GOLD_YES, scaffolding=True, rigor=True, turns=2),
        record("long", gold=GOLD_NO, scaffolding=True, rigor=True, turns=30),
    ]

    scaffolding = E.evaluate(records)[0]

    assert (scaffolding["all"]["tp"], scaffolding["all"]["fp"]) == (1, 1)
    assert (scaffolding["short"]["tp"], scaffolding["short"]["fp"]) == (1, 0)


def test_the_short_threshold_is_configurable():
    records = [record("a", gold=GOLD_YES, scaffolding=True, rigor=True, turns=7)]

    assert E.evaluate(records, None, 5)[0]["short"]["n"] == 0
    assert E.evaluate(records, None, 8)[0]["short"]["n"] == 1


# ===========================================================================
# CLI
# ===========================================================================


def _write_predictions(
    tmp_path, model="claude-opus-5", split="iteration", prompt_version="1"
):
    # Predictions live under their own directory: every subdirectory of
    # --pred-dir is read as a model name, so the ground truth cannot sit beside
    # them.
    version_dir = tmp_path / "preds" / model / prompt_version
    version_dir.mkdir(parents=True)
    records = [
        record(
            "a",
            gold=GOLD_YES,
            scaffolding=True,
            rigor=True,
            over=True,
            asked=True,
            turns=2,
        ),
        record("b", gold=GOLD_NO, scaffolding=True, rigor=False, turns=30),
    ]
    (version_dir / f"{split}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return str(tmp_path / "preds")


def _write_ground_truth(tmp_path, split="iteration", agreement=None):
    """Ground truth for the two moments _write_predictions writes."""
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir(exist_ok=True)
    records = [
        {"moment_id": "a", "labels": GOLD_YES, "agreement": agreement or AGREED},
        {"moment_id": "b", "labels": GOLD_NO, "agreement": AGREED},
    ]
    (gt_dir / f"{split}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return str(gt_dir)


def test_cli_reports_and_writes_json(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path)
    gt_dir = _write_ground_truth(tmp_path)
    out = tmp_path / "scores.json"

    assert (
        E.main(
            ["--pred-dir", pred_dir, "--ground-truth-dir", gt_dir, "--json", str(out)]
        )
        == 0
    )

    assert "F1" in capsys.readouterr().out
    scores = json.loads(out.read_text(encoding="utf-8"))
    assert scores[0]["model"] == "claude-opus-5"
    assert scores[0]["prompt_version"] == "1"
    assert scores[0]["split"] == "iteration"
    scaffolding = scores[0]["tasks"][0]
    assert (scaffolding["all"]["tp"], scaffolding["all"]["fp"]) == (1, 1)
    assert scaffolding["all"]["f1"] == pytest.approx(2 / 3)
    assert scaffolding["agreed"]["tp"] == 1


def test_cli_reports_both_subsets(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path)
    gt_dir = _write_ground_truth(tmp_path, agreement=CONTESTED)

    assert E.main(["--pred-dir", pred_dir, "--ground-truth-dir", gt_dir]) == 0

    out = capsys.readouterr().out
    assert "all moments" in out
    assert "both annotation passes agreed on the construct" in out
    assert "contested moments left out" in out


def test_cli_reports_the_short_moments(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path)

    assert E.main(["--pred-dir", pred_dir, "--ground-truth-dir", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "fewer than 10 turns after the cut" in out
    assert "1 of 2 moment(s)" in out


def test_cli_short_turns_sets_the_threshold(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path)

    assert (
        E.main(
            [
                "--pred-dir",
                pred_dir,
                "--ground-truth-dir",
                str(tmp_path),
                "--short-turns",
                "40",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "fewer than 40 turns after the cut" in out
    assert "2 of 2 moment(s)" in out


def test_cli_reports_the_full_numbers_without_a_ground_truth(tmp_path, capsys):
    """A missing ground truth costs the subset, not the run."""
    pred_dir = _write_predictions(tmp_path)

    assert (
        E.main(["--pred-dir", pred_dir, "--ground-truth-dir", str(tmp_path / "nope")])
        == 0
    )

    out = capsys.readouterr().out
    assert "all moments" in out
    assert "agreement subset was skipped" in out


def test_cli_warns_when_the_predictions_are_stale(tmp_path, caplog):
    pred_dir = _write_predictions(tmp_path)
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()
    (gt_dir / "iteration.jsonl").write_text(
        json.dumps({"moment_id": "a", "labels": GOLD_NO, "agreement": AGREED}) + "\n",
        encoding="utf-8",
    )

    assert E.main(["--pred-dir", pred_dir, "--ground-truth-dir", str(gt_dir)]) == 0
    assert "differ from the current ground truth" in caplog.text


def test_cli_fails_when_there_is_nothing_to_score(tmp_path):
    (tmp_path / "claude-opus-5" / "1").mkdir(parents=True)

    assert E.main(["--pred-dir", str(tmp_path)]) == 1


def test_cli_scores_every_prompt_version_by_default(tmp_path, capsys):
    """A prompt edit is made to be compared, so both revisions report by default."""
    pred_dir = _write_predictions(tmp_path, prompt_version="1")
    _write_predictions(tmp_path, prompt_version="2")

    assert E.main(["--pred-dir", pred_dir]) == 0
    out = capsys.readouterr().out
    assert "prompts v1" in out
    assert "prompts v2" in out


def test_cli_scores_only_the_prompt_version_asked_for(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path, prompt_version="1")
    _write_predictions(tmp_path, prompt_version="2")

    assert E.main(["--pred-dir", pred_dir, "--prompt-version", "2"]) == 0
    out = capsys.readouterr().out
    assert "prompts v2" in out
    assert "prompts v1" not in out


def test_prompt_versions_sort_numerically(tmp_path):
    for version in ("1", "2", "10"):
        (tmp_path / version).mkdir()

    assert E.find_prompt_versions(str(tmp_path), None) == ["1", "2", "10"]


def test_cli_scores_only_the_model_asked_for(tmp_path, capsys):
    pred_dir = _write_predictions(tmp_path)
    _write_predictions(tmp_path, model="gemini-3.5-flash")

    assert E.main(["--pred-dir", pred_dir]) == 0
    assert "gemini-3.5-flash" in capsys.readouterr().out

    assert E.main(["--pred-dir", pred_dir, "--model", "claude-opus-5"]) == 0
    assert "gemini-3.5-flash" not in capsys.readouterr().out
