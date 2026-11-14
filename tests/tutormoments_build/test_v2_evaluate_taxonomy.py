"""Tests for reading taxonomy gold and predictions."""

import json

from tutormoments_build.v2 import evaluate_taxonomy as E
from tutormoments_build.v2 import taxonomy as T


def _adjudicator(
    annotator_id,
    scaffolding=(),
    rigor=(),
    *,
    throw_out=False,
    revision=1,
    created_at="2026-01-01",
    other="",
):
    final = {
        "situation": {},
        "action": {
            "scaffolding_strategies": list(scaffolding),
            "rigor_strategies": list(rigor),
            "scaffolding_strategy_other": other,
            "rigor_strategy_other": "",
        },
    }
    return {
        "annotator_id": annotator_id,
        "role": "adjudicator",
        "revision": revision,
        "created_at": created_at,
        "payload": {"meta": {"throw_out": throw_out}, "final": final},
    }


def _selector(annotator_id):
    payload = {"situation": {}, "action": {"scaffolding_strategies": ["hint"]}}
    return {"annotator_id": annotator_id, "role": "selector", "payload": payload}


def test_gold_aliases_collapse_onto_prompt_names():
    annotation = _adjudicator(
        "a",
        scaffolding=[
            "re_explain_procedure",
            "re_explain",
            "different_representation",
            "simpler_language",
        ],
    )
    assert E.annotation_actions(annotation)["actions"] == {"re_explain", "diff_rep"}


def test_predicted_simpler_language_is_not_remapped():
    record = {
        "taxonomy": {
            "actions": [
                {"category": "simpler_language", "family": "scaffolding"},
            ]
        }
    }
    assert E.prediction_actions(record)["actions"] == {"simpler_language"}


def test_gold_is_the_union_of_adjudicators_only():
    annotations = [
        _selector("s"),
        _adjudicator("a", scaffolding=["hint"], created_at="1"),
        _adjudicator("b", rigor=["justify_explain"], created_at="2"),
    ]
    gold = E.moment_gold(annotations, {"a": "A01", "b": "A02"})
    assert gold["actions"] == {"hint", "justify_explain"}
    assert gold["adjudicators"] == ["A01", "A02"]


def test_free_text_other_is_kept_out_of_the_compared_gold():
    annotations = [
        _adjudicator("a", scaffolding=["hint"], other="acted it out", created_at="1"),
        _adjudicator("b", other="sang a song", created_at="2"),
    ]
    gold = E.moment_gold(annotations, {"a": "A01", "b": "A02"})
    assert gold["actions"] == {"hint"}
    assert gold["other"] == {
        "A01": {"scaffolding": "acted it out"},
        "A02": {"scaffolding": "sang a song"},
    }


def test_thrown_out_adjudicator_contributes_nothing():
    annotations = [
        _adjudicator("a", scaffolding=["hint"]),
        _adjudicator("b", scaffolding=["co_solve"], throw_out=True),
    ]
    assert E.moment_gold(annotations, {})["actions"] == {"hint"}
    assert E.moment_gold([annotations[1]], {}) is None


def test_load_gold_drops_staff_and_superseded_resaves(tmp_path):
    rows = [
        {
            "source": "tutoring_provider_a",
            "moment": {"moment_id": "m1"},
            "annotations": [
                _adjudicator("a", scaffolding=["hint"], revision=1),
                _adjudicator("a", scaffolding=["co_solve"], revision=2),
                _adjudicator("staff", rigor=["define_term"]),
            ],
        }
    ]
    path = tmp_path / "tutoring_provider_a_annotations.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    excluded = tmp_path / "excluded.json"
    excluded.write_text(json.dumps(["staff"]))
    gold = E.load_gold([str(path)], str(excluded), str(tmp_path / "none.json"))
    assert gold["m1"]["actions"] == {"co_solve"}


def _pred(moment_id, actions, *, asked=True, parse_error=False, labels=None):
    taxonomy = None
    if asked:
        taxonomy = {
            "actions": None if parse_error else actions,
            "parse_error": parse_error,
        }
    return {
        "moment_id": moment_id,
        "split": "iteration",
        "asked": asked,
        "action": "scaffold" if asked else None,
        "labels": labels or {"scaffolding_present": True, "rigor_present": True},
        "taxonomy": taxonomy,
    }


def _gold(scaffolding=(), rigor=()):
    by_family = {"scaffolding": frozenset(scaffolding), "rigor": frozenset(rigor)}
    return {
        "actions": frozenset().union(*by_family.values()),
        "by_family": by_family,
        "adjudicators": ["A01"],
    }


def test_pair_separates_unknown_labels_and_counts_drops():
    hint = {"category": "hint", "family": "scaffolding", "evidence": "2"}
    praise = {"category": "praise", "family": None, "evidence": "3"}
    predictions = {
        "ok": _pred("ok", [hint, hint, praise]),
        "skipped": _pred("skipped", None, asked=False),
        "bad": _pred("bad", None, parse_error=True),
        "nogold": _pred("nogold", [hint]),
    }
    gold = {"ok": _gold(scaffolding=["hint", "co_solve"])}
    pairs, dropped = E.pair(gold, predictions)
    assert len(pairs) == 1
    assert pairs[0]["pred"] == {"hint"} and pairs[0]["pred_unknown"] == {"praise"}
    assert pairs[0]["gold"] == {"hint", "co_solve"}
    assert dropped == {E.NOT_ASKED: 1, E.UNPARSED: 1, E.NO_GOLD: 1}


def test_pair_keeps_only_families_the_gold_labels_mark_present():
    gold = {
        m: _gold(scaffolding=["hint"], rigor=["justify_explain"])
        for m in ("both", "scaffold", "rigor")
    }
    predictions = {
        "both": _pred("both", []),
        "scaffold": _pred(
            "scaffold", [], labels={"scaffolding_present": True, "rigor_present": False}
        ),
        "rigor": _pred(
            "rigor", [], labels={"scaffolding_present": False, "rigor_present": True}
        ),
    }
    pairs, _ = E.pair(gold, predictions)
    assert {p["moment_id"]: p["gold"] for p in pairs} == {
        "both": {"hint", "justify_explain"},
        "scaffold": {"hint"},
        "rigor": {"justify_explain"},
    }


def test_gold_keeps_each_familys_union_apart():
    annotations = [
        _adjudicator("a", scaffolding=["hint"], rigor=["define_term"], created_at="1"),
        _adjudicator("b", scaffolding=["co_solve"], created_at="2"),
    ]
    gold = E.moment_gold(annotations, {})
    assert gold["by_family"] == {
        "scaffolding": {"hint", "co_solve"},
        "rigor": {"define_term"},
    }


def test_mapped_gold_vocabulary_is_the_prompts():
    prompt = set(T.parse_categories(T.load_template()))
    gold_names = {
        "guiding_questions",
        "break_into_steps",
        "re_explain",
        "re_explain_procedure",
        "model_example",
        "co_solve",
        "simpler_language",
        "draw_diagram",
        "different_representation",
        "reduce_options",
        "hint",
        "remind_prior_problem",
        "highlight_problem_text",
        "give_answer",
        "state_correction",
        "justify_explain",
        "withdraw_support",
        "increase_complexity",
        "define_term",
        "find_fix_errors",
    }
    # Gold simpler_language is scored as re_explain, so the prompt's
    # simpler_language has no gold counterpart.
    assert {E.gold_label(n) for n in gold_names} == prompt - {"simpler_language"}


def test_jaccard_counts_two_empty_sets_as_agreement():
    assert E.jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == 1 / 3
    assert E.jaccard(frozenset(), frozenset()) == 1.0


def test_set_metrics_micro():
    sets = [
        (frozenset({"a", "b"}), frozenset({"a"})),  # a tp, b fn
        (frozenset({"a"}), frozenset({"a", "c"})),  # a tp, c fp
    ]
    m = E.set_metrics(sets)
    assert m["n"] == 2
    assert m["jaccard"] == 0.5
    assert m["micro_f1"] == 4 / 6  # tp 2, fp 1, fn 1
    assert m["micro_precision"] == 2 / 3 and m["micro_recall"] == 2 / 3
    assert m["per_label"]["b"]["precision"] is None
    assert m["mean_reference_labels"] == 1.5 and m["mean_compared_labels"] == 1.5


def test_confusion_pairs_misses_with_additions_else_none():
    matrix = E.confusion(
        [
            (frozenset({"a", "b"}), frozenset({"a", "c"})),
            (frozenset({"b"}), frozenset()),
            (frozenset(), frozenset({"c"})),
            (frozenset(), frozenset()),
        ]
    )
    assert matrix == {"a": {"a": 1}, "b": {"c": 1, E.NONE: 1}, E.NONE: {"c": 1}}


def _pass(role, scaffolding=(), rigor=(), *, sp=False, rp=False):
    action = {
        "scaffolding_strategies": list(scaffolding),
        "rigor_strategies": list(rigor),
        "scaffolding_present": sp,
        "rigor_present": rp,
    }
    payload = {"meta": {}, "situation": {}, "action": action}
    return {"annotator_id": role, "role": role, "payload": payload}


def test_pass_pairs_compare_only_families_both_marked_present():
    annotations = {
        "agree_scaffold": [
            _pass("selector", ["hint"], ["define_term"], sp=True, rp=True),
            _pass("reannotator", ["co_solve"], sp=True),
        ],
        "agree_neither": [
            _pass("selector", ["hint"], sp=True),
            _pass("reannotator", rigor=["define_term"], rp=True),
        ],
        "one_pass": [_pass("selector", ["hint"], sp=True)],
    }
    pairs, dropped = E.pass_pairs(annotations, list(annotations))
    assert pairs == [
        {
            "moment_id": "agree_scaffold",
            "action": "scaffold",
            "first": {"hint"},
            "second": {"co_solve"},
        }
    ]
    assert dropped == {
        "disagreed_rigor": 2,
        "disagreed_scaffolding": 1,
        E.NO_AGREED_FAMILY: 1,
        E.MISSING_PASS: 1,
    }


CATEGORIES = {"hint": "scaffolding", "co_solve": "scaffolding", "define_term": "rigor"}


def test_groups_are_all_then_each_family_cut_to_its_own_labels():
    pairs = [
        # Scaffolding only: the model's rigor label counts in "all" only.
        {
            "action": "scaffold",
            "gold": frozenset({"hint"}),
            "pred": frozenset({"co_solve", "define_term"}),
        },
        # Both: in both family rows; a scaffolding miss is not paired with a
        # rigor addition.
        {
            "action": "scaffold and push for rigor",
            "gold": frozenset({"hint"}),
            "pred": frozenset({"define_term"}),
        },
        {
            "action": "push for rigor",
            "gold": frozenset({"define_term"}),
            "pred": frozenset({"define_term"}),
        },
    ]
    groups = E.grouped_metrics(pairs, "gold", "pred", CATEGORIES)
    assert list(groups) == [E.ALL, "scaffolding_present", "rigor_present"]
    assert groups[E.ALL]["n"] == 3 and "confusion" not in groups[E.ALL]

    scaffolding = groups["scaffolding_present"]
    assert scaffolding["n"] == 2
    assert scaffolding["mean_compared_labels"] == 0.5  # define_term cut
    assert scaffolding["confusion"] == {"hint": {"co_solve": 1, E.NONE: 1}}

    rigor = groups["rigor_present"]
    assert rigor["n"] == 2
    assert rigor["jaccard"] == 0.5  # gold {} vs {define_term}, then a match
    assert rigor["confusion"] == {
        E.NONE: {"define_term": 1},
        "define_term": {"define_term": 1},
    }
    assert E.off_family_labels(pairs, "pred", CATEGORIES) == {"define_term": 1}


_TEMPLATES = {
    "1": "### Scaffolding\n`hint`: a\n`simpler_language`: b\n### Rigor\n`define_term`: c\n",
    "2": "### Scaffolding\n`hint`: a\n### Rigor\n`define_term`: c\n",
}


def _run_versions(tmp_path, monkeypatch, *extra):
    annotations = tmp_path / "tutoring_provider_a_annotations.jsonl"
    row = {
        "source": "tutoring_provider_a",
        "moment": {"moment_id": "m1"},
        "annotations": [_adjudicator("a", scaffolding=["hint"])],
    }
    annotations.write_text(json.dumps(row))
    record = _pred(
        "m1",
        [
            {"category": "hint", "family": "scaffolding"},
            {"category": "simpler_language", "family": "scaffolding"},
        ],
        labels={"scaffolding_present": True, "rigor_present": False},
    )
    for version in _TEMPLATES:
        width_dir = tmp_path / "preds" / "m" / version / "20"
        width_dir.mkdir(parents=True)
        (width_dir / "iteration.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(E, "load_template", lambda v=None: _TEMPLATES[v or "2"])
    out = tmp_path / "scores.json"
    argv = [
        "--pred-dir", str(tmp_path / "preds"),
        "--annotations", str(annotations),
        "--ground-truth-dir", str(tmp_path / "no_gt"),
        "--splits", "iteration",
        "--json", str(out),
        *extra,
    ]  # fmt: skip
    assert E.main(argv) == 0
    return {r["prompt_version"]: r for r in json.loads(out.read_text())}


def test_each_prompt_version_is_scored_with_its_own_vocabulary(tmp_path, monkeypatch):
    results = _run_versions(tmp_path, monkeypatch)
    assert sorted(results) == ["1", "2"]
    # v1 lists simpler_language, so it is a scaffolding false positive there;
    # v2 does not, so the same label falls outside every family.
    assert results["1"]["off_family_labels"] == {}
    assert results["2"]["off_family_labels"] == {"simpler_language": 1}


def test_prompt_version_flag_scores_only_that_version(tmp_path, monkeypatch):
    assert sorted(_run_versions(tmp_path, monkeypatch, "--prompt-version", "1")) == [
        "1"
    ]


def test_short_block_keeps_only_moments_at_or_under_the_turn_count():
    hint = {"category": "hint", "family": "scaffolding", "evidence": "2"}
    predictions = {}
    for moment_id, turns in (("short", 3), ("edge", 5), ("long", 9), ("old", None)):
        record = _pred(moment_id, [hint])
        if turns is not None:
            record["post_cut_dialogue_rows"] = turns
        predictions[moment_id] = record
    gold = {m: _gold(scaffolding=["hint"]) for m in predictions}
    cats = {"hint": "scaffolding", "define_term": "rigor"}

    result = E.evaluate_predictions(gold, predictions, cats, short_turns=5)

    short = result["short"]
    assert (short["moments"], short["of_moments"], short["unknown"]) == (2, 3, 1)
    assert short["groups"]["all"]["n"] == 2
    assert result["groups"]["all"]["n"] == 4
