"""Tests for classifying replays: routing to stage 2, filling prompts, and retries."""

import json

import pytest

from tutormoments_build.v2 import classify_excerpts as C
from tutormoments_build.v2 import classify_replays as CR

PROMPTS = C.load_prompt_set(C.latest_prompt_version())
CFG = C.V2Spec("gemini-4-argon", "gemini", {"thinking_level": "high"}, 60)


def _replay(moment_id, *, gold=(False, True), situation="scaffold_maj"):
    return {
        "moment_id": moment_id,
        "source": "tutoring_provider_c",
        "transcript_id": f"t-{moment_id}",
        "conversation_id": f"c-{moment_id}",
        "settings": {"tutor": {"arm": "a"}, "turns": 1},
        "tutor_turns": 1,
        "ended_via": "TURNS",
        "post_cut_rows": 1,
        "excerpts": {
            "full": {
                "excerpt": f"whole\n>>> CUT POINT <<<\nTurn 3. TUTOR: {moment_id}"
            },
            "20": {"excerpt": f"window\n>>> CUT POINT <<<\nTurn 3. TUTOR: {moment_id}"},
        },
        "labels": {
            "scaffolding_present": gold[0],
            "rigor_present": gold[1],
            "over_scaffolding_present": False,
        },
        "situation_type": situation,
        C.DESCRIPTION_KEY: {
            "adjudicators": ["J01"],
            "situation_method": "single",
            "model": "m",
            "prompt": "p",
            "facets": [f"facet of {moment_id}"],
        },
    }


def _action(scaffolding, rigor):
    return {
        "text": json.dumps(
            {"description": "d", "scaffolding": scaffolding, "rigor": rigor}
        ),
        "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
    }


# Gold says rigor only for every replay. The predictions differ, so the tests
# can tell which one was used.
REPLAYS = [_replay("yes1"), _replay("no1"), _replay("bad1"), _replay("both")]
STAGE1 = {
    "action__yes1": _action("yes", "no"),
    "action__no1": _action("no", "yes"),
    "action__bad1": {"text": "garbled"},
    "action__both": _action("yes", "yes"),
}
STAGE2 = {
    "overscaffold__yes1": {"text": '{"description": "e", "over-scaffolding": "yes"}'},
    "overscaffold__both": {"text": '{"description": "e", "over-scaffolding": "no"}'},
}


@pytest.fixture
def batches(monkeypatch):
    """Serve STAGE1 then STAGE2, recording what each batch was sent."""
    sent = []

    def run_entries(entries, cfg, batch_id=None, display_name=""):
        sent.append({e["key"]: e for e in entries})
        return STAGE1 if len(sent) % 2 == 1 else STAGE2

    monkeypatch.setattr(C, "run_entries", run_entries)
    return sent


def _prompt(entry):
    return entry["request"]["contents"][0]["parts"][0]["text"]


def test_stage_two_is_routed_on_the_predicted_not_the_gold_scaffolding(batches):
    records, counts = CR.classify(REPLAYS, CFG, PROMPTS, "r.jsonl")

    assert len(batches) == 2
    assert set(batches[0]) == {f"action__{r['moment_id']}" for r in REPLAYS}
    # Gold marks no scaffolding anywhere; routing follows the prediction.
    assert set(batches[1]) == {"overscaffold__yes1", "overscaffold__both"}
    assert counts["over_scaffolding"] == 2

    by_id = {r["moment_id"]: r for r in records}
    assert by_id["yes1"]["over_scaffolding"]["over_scaffolding"] is True
    assert by_id["both"]["over_scaffolding"]["over_scaffolding"] is False
    assert by_id["yes1"]["over_scaffolding_routing"] == {
        "gated_on": "predicted",
        "routed": True,
        "reason": None,
        "from": {"scaffolding": True, "rigor": False},
    }


def test_not_routed_replays_record_why_and_are_never_a_no(batches):
    records, counts = CR.classify(REPLAYS, CFG, PROMPTS, "r.jsonl")
    by_id = {r["moment_id"]: r for r in records}

    for moment_id, reason in [
        ("no1", CR.NOT_ROUTED_NO_SCAFFOLDING),
        ("bad1", CR.NOT_ROUTED_UNPARSEABLE),
    ]:
        record = by_id[moment_id]
        assert record["over_scaffolding"] is None
        assert record["over_scaffolding_asked"] is False
        assert record["over_scaffolding_routing"]["reason"] == reason
    assert counts[CR.NOT_ROUTED_NO_SCAFFOLDING] == 1
    assert counts[CR.NOT_ROUTED_UNPARSEABLE] == 1
    assert counts["action_direction_unparsed"] == 1


def test_action_slot_is_predicted_and_the_others_gold(batches):
    records, _ = CR.classify(REPLAYS, CFG, PROMPTS, "r.jsonl")
    by_id = {r["moment_id"]: r for r in records}

    yes1 = _prompt(batches[1]["overscaffold__yes1"])
    both = _prompt(batches[1]["overscaffold__both"])
    # Gold action would be "pushing for rigor" for both.
    assert "tutor is scaffolding after the cut point" in yes1
    assert "scaffolding and pushing for rigor" in both
    assert "is appropriate for scaffolding" in yes1  # gold scaffold_maj
    assert "- facet of yes1" in yes1  # gold student-state facets
    assert by_id["yes1"]["prompt_slots"]["over_scaffolding"]["action"] == "scaffolding"
    assert by_id["yes1"]["description_from"]["adjudicators"] == ["J01"]


def test_stage_one_reads_the_replay_excerpt(batches):
    CR.classify(REPLAYS, CFG, PROMPTS, "r.jsonl")
    assert _prompt(batches[0]["action__yes1"]).endswith("Turn 3. TUTOR: yes1")


def test_no_stage_two_batch_when_nothing_is_routed(monkeypatch):
    sent = []
    monkeypatch.setattr(
        C,
        "run_entries",
        lambda entries, cfg, **k: (
            sent.append(entries) or {"action__no1": _action("no", "no")}
        ),
    )
    records, _ = CR.classify([_replay("no1")], CFG, PROMPTS, "r.jsonl")
    assert len(sent) == 1
    assert records[0]["usage"]["total_tokens"] == 5


def test_a_replay_missing_stage_two_inputs_fails_before_any_batch():
    replay = _replay("x")
    del replay[C.DESCRIPTION_KEY]
    with pytest.raises(KeyError, match="student-state"):
        CR.check_fillable([replay], PROMPTS)


def test_output_sits_beside_the_replay_file():
    assert (
        CR.classification_path(
            "out/a/plain_v1/n1/student-gemini-4-argon_sp1.jsonl", "vendor/m", "9"
        )
        == "out/a/plain_v1/n1/student-gemini-4-argon_sp1__scorer-vendor_m_v9_ctx20.jsonl"
    )
    assert CR.classification_path("s.jsonl", "m", "9", "full") == (
        "s__scorer-m_v9_ctxfull.jsonl"
    )


def test_both_stages_read_the_human_rounds_default_width_unless_asked(batches):
    records, _ = CR.classify(REPLAYS, CFG, PROMPTS, "r.jsonl")
    assert _prompt(batches[0]["action__yes1"]).count("window\n") == 1
    assert "window\n" in _prompt(batches[1]["overscaffold__yes1"])
    assert records[0]["context_turns"] == "20" == C.EXCERPT_WIDTH

    CR.classify(REPLAYS, CFG, PROMPTS, "r.jsonl", width="full")
    assert "whole\n" in _prompt(batches[2]["action__yes1"])


def test_scorer_defaults_to_the_replay_block_and_overrides_like_classify_excerpts():
    default = CR.scorer_spec()
    assert default.model == "gemini-4-argon"
    assert default.thinking == {"include_thoughts": True, "thinking_level": "high"}
    # An override resolves exactly as classify_excerpts --model does.
    override = CR.scorer_spec("claude-opus-5")
    assert override.thinking == C.phase_config(model="claude-opus-5").thinking


def _write(tmp_path, records):
    path = tmp_path / "student-s_sp1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(path)


def _cli(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(
        C,
        "load_student_state",
        lambda *a, **k: {r["moment_id"]: r[C.DESCRIPTION_KEY] for r in REPLAYS},
    )
    plain = [{k: v for k, v in r.items() if k != C.DESCRIPTION_KEY} for r in REPLAYS]
    path = _write(tmp_path, plain)
    return path, CR.main([path, *extra])


def test_dry_run_makes_no_call(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(C, "run_entries", lambda *a, **k: pytest.fail("no calls"))
    _, code = _cli(monkeypatch, tmp_path, "--dry-run")
    assert code == 0
    assert "stage 1 action_direction: 4 request(s)" in capsys.readouterr().out


def test_cli_writes_once_and_refuses_to_overwrite(monkeypatch, tmp_path, batches):
    path, code = _cli(monkeypatch, tmp_path)
    assert code == 0
    out = CR.classification_path(path, "gemini-4-argon", PROMPTS.version, "20")
    lines = open(out).read().splitlines()
    assert len(lines) == 4
    assert C.DESCRIPTION_KEY not in json.loads(lines[0])

    _, code = _cli(monkeypatch, tmp_path)
    assert code == 1


def _serve(monkeypatch, *responses):
    """Serve each batch from `responses` in turn, recording the keys sent."""
    sent = []

    def run_entries(entries, cfg, batch_id=None, display_name=""):
        sent.append((display_name, sorted(e["key"] for e in entries)))
        return responses[len(sent) - 1]

    monkeypatch.setattr(C, "run_entries", run_entries)
    return sent


FAILED = {"text": "", "error": "overloaded", "usage": {"input_tokens": 2}}


def test_a_failed_request_is_resubmitted_alone_and_its_usage_kept(monkeypatch):
    replays = [_replay("ok"), _replay("flaky")]
    sent = _serve(
        monkeypatch,
        {"action__ok": _action("no", "yes")},  # flaky: no result at all
        {"action__flaky": FAILED},
        {"action__flaky": _action("no", "yes")},
    )
    records, counts = CR.classify(replays, CFG, PROMPTS, "r.jsonl")

    assert sent == [
        ("v2_classify_replays_action", ["action__flaky", "action__ok"]),
        ("v2_classify_replays_action_retry1", ["action__flaky"]),
        ("v2_classify_replays_action_retry2", ["action__flaky"]),
    ]
    by_id = {r["moment_id"]: r for r in records}
    assert by_id["flaky"]["action_direction"]["rigor"] is True
    assert by_id["flaky"]["attempts"] == {"action_direction": 3, "over_scaffolding": 0}
    assert by_id["ok"]["attempts"]["action_direction"] == 1
    # The failed attempt's 2 input tokens are summed with the success's 4.
    assert by_id["flaky"]["action_direction"]["usage"]["input_tokens"] == 6
    assert not counts["action_direction_failed"]


def test_retries_stop_after_three_and_the_failure_is_recorded(monkeypatch):
    sent = _serve(monkeypatch, *[{"action__x": FAILED}] * 4)
    records, counts = CR.classify([_replay("x")], CFG, PROMPTS, "r.jsonl")

    assert len(sent) == 1 + CR.MAX_RETRIES == 4
    record = records[0]
    assert record["attempts"]["action_direction"] == 4
    assert record["over_scaffolding_routing"]["reason"] == CR.NOT_ROUTED_FAILED
    assert record["action_direction"]["error"] == "overloaded"
    assert counts["action_direction_failed"] == 1
    assert "still failed after 3 retries: 1" in CR.report(records, counts)


def test_an_unparseable_response_is_not_resubmitted(batches):
    records, _ = CR.classify(REPLAYS, CFG, PROMPTS, "r.jsonl")
    assert len(batches) == 2  # stage 1 and stage 2, no retry batch
    bad = next(r for r in records if r["moment_id"] == "bad1")
    assert bad["attempts"]["action_direction"] == 1


def test_stage_two_failures_are_retried_too(monkeypatch):
    sent = _serve(
        monkeypatch,
        {"action__s": _action("yes", "no")},
        {"overscaffold__s": FAILED},
        {"overscaffold__s": {"text": '{"over-scaffolding": "no"}'}},
    )
    records, _ = CR.classify([_replay("s")], CFG, PROMPTS, "r.jsonl")
    assert [name for name, _ in sent][-1] == (
        "v2_classify_replays_over_scaffolding_retry1"
    )
    assert records[0]["over_scaffolding"]["over_scaffolding"] is False
    assert records[0]["attempts"] == {"action_direction": 1, "over_scaffolding": 2}
