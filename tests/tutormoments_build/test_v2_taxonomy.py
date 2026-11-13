"""Tests for the v2 fine-grained action taxonomy round."""

import pytest

from tutormoments_build.v2 import taxonomy as T
from tutormoments_build.v2.classify_excerpts import V2Spec

CFG = V2Spec(model="m", provider="anthropic", thinking={}, poll_interval=1)


def _excerpt(moment_id, *, scaffolding, rigor, post_cut_rows=2, split="iteration"):
    return {
        "moment_id": moment_id,
        "transcript_id": f"t-{moment_id}",
        "split": split,
        "post_cut_rows": post_cut_rows,
        "excerpts": {
            "20": {"excerpt": f"wide ({moment_id})"},
            "full": {"excerpt": f"whole ({moment_id})"},
        },
        "labels": {"scaffolding_present": scaffolding, "rigor_present": rigor},
        "situation_type": "scaffold_only",
    }


@pytest.fixture(scope="module")
def template():
    return T.load_template()


def test_categories_are_read_from_the_prompt_with_their_family(template):
    categories = T.parse_categories(template)
    assert categories["hint"] == "scaffolding"
    assert categories["justify_explain"] == "rigor"
    assert categories["give_answer"] == "scaffolding"
    assert categories["find_fix_errors"] == "rigor"


def test_parse_categories_raises_when_a_family_is_missing():
    with pytest.raises(ValueError, match="rigor"):
        T.parse_categories("### Scaffolding\n\n`hint`: a hint\n")


@pytest.mark.parametrize(
    "scaffolding, rigor, wording",
    [
        (True, True, "scaffold and push for rigor"),
        (True, False, "scaffold"),
        (False, True, "push for rigor"),
        (False, False, None),
    ],
)
def test_action_slot_comes_from_gold(scaffolding, rigor, wording):
    assert T.action_slot(_excerpt("a", scaffolding=scaffolding, rigor=rigor)) == wording


def test_entries_ask_rigor_only_moments_and_skip_neither_and_empty(template):
    records = [
        _excerpt("both", scaffolding=True, rigor=True),
        _excerpt("rigor", scaffolding=False, rigor=True),
        _excerpt("neither", scaffolding=False, rigor=False),
        _excerpt("empty", scaffolding=True, rigor=False, post_cut_rows=0),
    ]
    entries, counts = T.build_entries(records, template)
    keys = [e["key"] for e in entries]
    assert keys == ["taxonomy__both", "taxonomy__rigor"]
    assert counts["asked"] == 2
    assert counts[T.NO_GOLD_ACTION] == 1
    assert counts[T.NO_POST_CUT] == 1


def test_filled_prompt_carries_action_and_chosen_width(template):
    record = _excerpt("x", scaffolding=False, rigor=True)
    prompt = T.fill_prompt(template, record, "full")
    assert "decided to push for rigor" in prompt
    assert "whole (x)" in prompt and "wide (x)" not in prompt
    assert "{" + "action}" not in prompt and "{" + "excerpt}" not in prompt


CATS = {"hint": "scaffolding", "justify_explain": "rigor"}


@pytest.mark.parametrize(
    "text",
    [
        '[{"evidence": "3", "category": "hint"}]',
        'Here:\n```json\n[{"evidence": "3", "category": "Hint"}]\n```',
        '{"actions": [{"evidence": "3", "category": "hint"}]}',
        '{"evidence": "3", "category": "hint"}',
    ],
)
def test_parse_actions_accepts_common_shapes(text):
    fields, err = T.parse_actions(text, CATS)
    assert not err
    assert fields["actions"] == [
        {"category": "hint", "family": "scaffolding", "evidence": "3"}
    ]


def test_parse_actions_flags_unknown_labels_and_failures():
    fields, err = T.parse_actions(
        '[{"evidence": "4", "category": "praise"}, {"category": "justify_explain"}]',
        CATS,
    )
    assert not err
    assert fields["unknown_categories"] == ["praise"]
    assert [a["family"] for a in fields["actions"]] == [None, "rigor"]

    fields, err = T.parse_actions("no json here", CATS)
    assert err and fields["actions"] is None

    fields, err = T.parse_actions("[]", CATS)
    assert not err and fields["actions"] == []


def test_classify_writes_every_moment_and_records_provenance(monkeypatch, template):
    excerpts = {
        "iteration": [
            _excerpt("a", scaffolding=True, rigor=False),
            _excerpt("b", scaffolding=False, rigor=False),
        ]
    }
    raw = {
        "taxonomy__a": {
            "text": '[{"evidence": "2", "category": "hint"}]',
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        }
    }
    monkeypatch.setattr(T, "run_entries", lambda entries, cfg, **kw: raw)
    out, counts = T.classify(excerpts, CFG, template)
    a, b = out["iteration"]

    assert a["asked"] and a["action"] == "scaffold"
    assert a["taxonomy"]["actions"][0]["category"] == "hint"
    assert a["usage"]["total_tokens"] == 7
    assert a["prompt_sha256"] == T.prompt_sha256(template)

    assert not b["asked"] and b["skipped"] == T.NO_GOLD_ACTION
    assert b["taxonomy"] is None and b["action"] is None
    assert counts["unparsed"] == 0


def test_prediction_dir_flattens_vendor_prefix():
    assert T.prediction_dir("out", "org/Model", "20") == "out/org_Model/20"
