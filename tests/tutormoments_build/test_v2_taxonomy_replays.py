import json

import pytest

from tutormoments_build.v2 import taxonomy as T
from tutormoments_build.v2 import taxonomy_replays as TR
from tutormoments_build.v2.classify_excerpts import V2Spec

CFG = V2Spec(model="org/m", provider="anthropic", thinking={}, poll_interval=1)
SETTINGS = {"tutor": {"arm": "x"}, "turns": 1}


def replay(moment_id, gold=(True, False), post_cut_rows=1):
    return {
        "moment_id": moment_id,
        "transcript_id": f"t-{moment_id}",
        "settings": SETTINGS,
        "post_cut_rows": post_cut_rows,
        "post_cut_dialogue_rows": post_cut_rows,
        "excerpts": {"20": {"excerpt": f"replay of {moment_id}"}},
        "labels": {"scaffolding_present": gold[0], "rigor_present": gold[1]},
        "situation_type": "scaffold_only",
    }


def classified(moment_id, scaffolding, rigor, *, raw="{}", error=None, none=False):
    return {
        "moment_id": moment_id,
        "replay": {"file": "replays.jsonl", "settings": SETTINGS},
        "model": "scorer",
        "thinking": {},
        "prompt_version": "9",
        "context_turns": "20",
        "action_direction": None
        if none
        else {"scaffolding": scaffolding, "rigor": rigor, "raw": raw, "error": error},
    }


@pytest.fixture(scope="module")
def prompt():
    return T.load_prompt()


def joined(*pairs):
    replays = {r["moment_id"]: r for r, _ in pairs}
    return TR.join([c for _, c in pairs], replays)


def test_action_comes_from_the_prediction_not_gold(prompt):
    # Gold says scaffold; the LM tutor was predicted to push for rigor.
    records = joined((replay("a", gold=(True, False)), classified("a", False, True)))
    entries, counts = TR.entries_for(records, prompt.template, "20")
    assert counts["asked_push for rigor"] == 1 and counts["asked_scaffold"] == 0
    assert "decided to push for rigor" in json.dumps(entries)


def test_not_asked_reasons(prompt):
    records = joined(
        (replay("a", gold=(True, True)), classified("a", False, False)),
        (replay("b"), classified("b", None, None, raw="garbled")),
        (replay("c"), classified("c", None, None, raw="", error="overloaded")),
        (replay("d", post_cut_rows=0), classified("d", None, None, none=True)),
        (replay("e", gold=(False, False)), classified("e", True, False)),
    )
    entries, counts = TR.entries_for(records, prompt.template, "20")
    assert [TR.skip_reason(r) for r in records] == [
        TR.PREDICTED_NEITHER,
        TR.ACTION_UNPARSEABLE,
        TR.ACTION_FAILED,
        T.NO_POST_CUT,
        None,  # gold marks neither, but the replay scaffolded: asked
    ]
    assert len(entries) == counts["asked"] == 1


def test_join_rejects_missing_or_regenerated_replays():
    with pytest.raises(KeyError):
        TR.join([classified("a", True, False)], {})
    stale = {**replay("a"), "settings": {"turns": 2}}
    with pytest.raises(ValueError):
        TR.join([classified("a", True, False)], {"a": stale})


def test_classify_records_predicted_action_and_labels(monkeypatch, prompt):
    records = joined(
        (replay("a", gold=(False, True)), classified("a", True, False)),
        (replay("b"), classified("b", False, False)),
    )
    raw = {"taxonomy__a": {"text": '[{"evidence": "3", "category": "hint"}]'}}
    monkeypatch.setattr(
        TR.R.C,
        "run_entries",
        lambda entries, cfg, **kw: {k: raw[k] for k in (e["key"] for e in entries)},
    )
    out, counts = TR.classify(records, CFG, prompt, "c.jsonl", width="20")
    a, b = out
    assert a["asked"] and a["action"] == "scaffold"
    assert a["action_from"] == {
        "gated_on": "predicted",
        "scaffolding": True,
        "rigor": False,
    }
    assert a["taxonomy"]["actions"][0] == {
        "category": "hint",
        "family": "scaffolding",
        "evidence": "3",
    }
    assert a["attempts"] == 1 and a["classification"]["file"] == "c.jsonl"
    assert not b["asked"] and b["skipped"] == TR.PREDICTED_NEITHER
    assert b["taxonomy"] is None and b["attempts"] == 0


def test_output_path_sits_beside_the_classification():
    assert (
        TR.output_path("out/r__scorer-x_v9_ctx20.jsonl", "org/m", "2")
        == "out/r__scorer-x_v9_ctx20__taxonomy-org_m_v2.jsonl"
    )
