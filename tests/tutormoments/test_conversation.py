"""Tests for tutormoments.conversation: multi-turn orchestration.

TDD: tests written first, then implementation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tutormoments.moments import Moment as Scenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(text, usage=None, latency=None):
    """Build a fake LLM response."""
    return SimpleNamespace(text=text, usage=usage or {}, latency_seconds=latency)


_FROZEN_TRAIT = {
    "persona": "fake-persona",
    "trait_mode": "joined-3",
    "generator_model": "claude-opus-4-6",
    "generated_at": "2026-06-18T00:00:00",
}


def _make_scenario(cut_turn=5, persona="fake-persona"):
    """Build a minimal Scenario for testing (carries a frozen student trait)."""
    context = [
        {
            "turn_number": i + 1,
            "role": "tutor" if i % 2 == 0 else "student",
            "text": f"text {i + 1}",
        }
        for i in range(cut_turn)
    ]
    return Scenario(
        id="test:conv__hum_3_5",
        context=context,
        dimension="scaffolding",
        student={
            "mode": "oracle",
            "reference": "Turn 6. TUTOR: Hi",
            "context": "Grade 5",
            "trait": {**_FROZEN_TRAIT, "persona": persona},
        },
        rubric={"gold": "scaffolding", "hint": "hint"},
        provenance={
            "conv_id": "conv-test",
            "cut_turn": cut_turn,
            "turn_start": 3,
            "turn_end": 5,
            "moment_id": None,
            "annotator_id": None,
            "chosen_cut_turn": cut_turn,
            "cut_votes": {},
            "cluster_size": 1,
            "representative": None,
        },
    )


def _patch_all(monkeypatch, tutor_responses, student_responses):
    """Patch resolve_tutor, resolve_student, build_*_system_prompt.

    The student persona comes from the scenario's frozen student.trait —
    the runtime no longer generates traits.
    """
    tutor_client = MagicMock()
    tutor_client.model = "fake-tutor"
    tutor_client.generate = MagicMock(side_effect=tutor_responses)

    student_client = MagicMock()
    student_client.model = "fake-student"
    student_client.generate = MagicMock(side_effect=student_responses)

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {"kind": "hosted", "client": tutor_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {"kind": "hosted", "client": student_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt",
        lambda mode, **kw: "TUTOR_SYS",
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt",
        lambda **kw: "STUDENT_SYS",
    )
    return tutor_client, student_client


# ---------------------------------------------------------------------------
# Tests: module imports and Transcript dataclass
# ---------------------------------------------------------------------------


def test_imports():
    """Module and Transcript are importable."""
    from tutormoments.conversation import Transcript, run_conversation

    assert Transcript is not None
    assert run_conversation is not None


def test_transcript_defaults():
    """Transcript has sensible defaults."""
    from tutormoments.conversation import Transcript

    t = Transcript(scenario_id="abc", tutor_model="m")
    assert t.scenario_id == "abc"
    assert t.tutor_model == "m"
    assert t.generated_turns == []
    assert t.tutor_usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert t.student_usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert t.tutor_latencies == []
    assert t.student_latencies == []
    assert t.completed is False
    assert t.ended_via == ""


def test_transcript_to_dict():
    """to_dict() returns all fields."""
    from tutormoments.conversation import Transcript

    t = Transcript(scenario_id="x", tutor_model="y")
    d = t.to_dict()
    assert d["scenario_id"] == "x"
    assert "generated_turns" in d
    assert "completed" in d


# ---------------------------------------------------------------------------
# Tests: private helpers
# ---------------------------------------------------------------------------


def test_parse_tutor_tokens_no_tokens():
    from tutormoments.conversation import _parse_tutor_tokens

    text, ended, changed = _parse_tutor_tokens("Hello there!")
    assert text == "Hello there!"
    assert ended is False
    assert changed is False


def test_parse_tutor_tokens_end():
    from tutormoments.conversation import _parse_tutor_tokens

    text, ended, changed = _parse_tutor_tokens("Great work! [END]")
    assert "[END]" not in text
    assert ended is True
    assert changed is False


def test_parse_tutor_tokens_problem_change():
    from tutormoments.conversation import _parse_tutor_tokens

    text, ended, changed = _parse_tutor_tokens("OK, let's move on. [PROBLEM_CHANGE]")
    assert "[PROBLEM_CHANGE]" not in text
    assert ended is False
    assert changed is True


def test_parse_tutor_tokens_next_problem_legacy():
    """[NEXT_PROBLEM] is the legacy alias for [PROBLEM_CHANGE]."""
    from tutormoments.conversation import _parse_tutor_tokens

    text, ended, changed = _parse_tutor_tokens("Done! [NEXT_PROBLEM]")
    assert "[NEXT_PROBLEM]" not in text
    assert ended is False
    assert changed is True


def test_parse_tutor_tokens_end_takes_precedence():
    """When both END and PROBLEM_CHANGE appear, ended wins."""
    from tutormoments.conversation import _parse_tutor_tokens

    text, ended, changed = _parse_tutor_tokens("[END] [PROBLEM_CHANGE]")
    assert ended is True
    assert changed is False


def test_split_messages_single():
    from tutormoments.conversation import _split_messages

    assert _split_messages("Hello") == ["Hello"]


def test_split_messages_next_delimiter():
    from tutormoments.conversation import _split_messages

    result = _split_messages("Msg1 [NEXT] Msg2")
    assert result == ["Msg1", "Msg2"]


def test_split_messages_new_message_delimiter():
    from tutormoments.conversation import _split_messages

    result = _split_messages("Msg1 [NEW_MESSAGE] Msg2")
    assert result == ["Msg1", "Msg2"]


def test_split_messages_empty_string():
    from tutormoments.conversation import _split_messages

    assert _split_messages("") == []


def test_split_messages_whitespace_only():
    from tutormoments.conversation import _split_messages

    assert _split_messages("   ") == []


def test_add_usage():
    from tutormoments.conversation import _add_usage

    total = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    _add_usage(total, {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
    assert total == {"input_tokens": 13, "output_tokens": 7, "total_tokens": 20}


def test_add_usage_missing_keys():
    """Missing keys in new dict are treated as 0."""
    from tutormoments.conversation import _add_usage

    total = {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}
    _add_usage(total, {})
    assert total["input_tokens"] == 10


def test_format_transcript_prefix():
    """_format_transcript_prefix uses real turn_number and UPPERCASE role."""
    from tutormoments.conversation import _format_transcript_prefix

    context = [
        {"turn_number": 26, "role": "tutor", "text": "Hello!"},
        {"turn_number": 27, "role": "student", "text": "Hi there."},
    ]
    result = _format_transcript_prefix(context)
    assert result == "Turn 26. TUTOR: Hello!\nTurn 27. STUDENT: Hi there."


def test_format_transcript_prefix_real_turn_numbers():
    """Non-sequential turn numbers (e.g. from enrichments) are preserved exactly."""
    from tutormoments.conversation import _format_transcript_prefix

    context = [
        {"turn_number": 5, "role": "tutor", "text": "A"},
        {"turn_number": 7, "role": "student", "text": "B"},
        {"turn_number": 10, "role": "tutor", "text": "C"},
    ]
    result = _format_transcript_prefix(context)
    assert result == "Turn 5. TUTOR: A\nTurn 7. STUDENT: B\nTurn 10. TUTOR: C"


def test_format_transcript_prefix_empty():
    from tutormoments.conversation import _format_transcript_prefix

    assert _format_transcript_prefix([]) == ""


# ---------------------------------------------------------------------------
# Tests: run_conversation happy path
# ---------------------------------------------------------------------------


def test_run_conversation_single_round_end(monkeypatch):
    """Tutor says [END] after first turn -> conversation stops after 1 tutor turn."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [
        _resp(
            "Good job! [END]",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
    ]
    student_resp = []  # never called

    tutor_client, student_client = _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=6)

    assert result.completed is True
    assert result.ended_via == "END"
    assert len(result.generated_turns) == 1
    assert result.generated_turns[0]["role"] == "TUTOR"
    assert "[END]" not in result.generated_turns[0]["text"]
    # Student never called
    student_client.generate.assert_not_called()


def test_run_conversation_problem_change(monkeypatch):
    """[PROBLEM_CHANGE] token stops loop correctly."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [_resp("Let's switch. [PROBLEM_CHANGE]")]
    student_resp = []

    _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=6)

    assert result.ended_via == "PROBLEM_CHANGE"
    assert result.completed is True
    assert len(result.generated_turns) == 1


def test_run_conversation_max_turns(monkeypatch):
    """Loop stops at max_turns with ended_via=MAX_TURNS."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    # 2 speaking turns: 1 tutor + 1 student
    tutor_resp = [_resp("Keep going.")]
    student_resp = [_resp("OK.")]

    _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=2)

    assert result.ended_via == "MAX_TURNS"
    assert result.completed is True
    assert len(result.generated_turns) == 2  # 1 tutor + 1 student


def test_run_conversation_turn_numbering(monkeypatch):
    """Generated turns start at cut_turn + 1."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [_resp("A. [END]")]
    _patch_all(monkeypatch, tutor_resp, [])

    result = run_conversation(scenario, "fake-tutor", max_turns=4)

    assert result.generated_turns[0]["turn_number"] == 6  # cut_turn + 1


def test_run_conversation_multi_round(monkeypatch):
    """Multiple T-S rounds accumulate turns correctly."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resps = [_resp("T1"), _resp("T2 [END]")]
    student_resps = [_resp("S1")]

    _patch_all(monkeypatch, tutor_resps, student_resps)

    result = run_conversation(scenario, "fake-tutor", max_turns=6)

    assert result.ended_via == "END"
    assert len(result.generated_turns) == 3
    assert result.generated_turns[0]["role"] == "TUTOR"
    assert result.generated_turns[1]["role"] == "STUDENT"
    assert result.generated_turns[2]["role"] == "TUTOR"


def test_run_conversation_empty_tutor_response_becomes_ellipsis(monkeypatch):
    """Empty tutor text (no END/CHANGE) becomes '...' placeholder."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [_resp(""), _resp("[END]")]
    student_resp = [_resp("OK")]

    _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=4)

    assert result.generated_turns[0]["text"] == "..."


def test_run_conversation_empty_student_response_becomes_ellipsis(monkeypatch):
    """Empty student text becomes '...'."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [_resp("Go ahead."), _resp("[END]")]
    student_resp = [_resp("")]

    _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=4)

    student_turn = next(t for t in result.generated_turns if t["role"] == "STUDENT")
    assert student_turn["text"] == "..."


def test_run_conversation_multi_message_split(monkeypatch):
    """[NEXT] delimiter splits one LLM response into multiple generated turns."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [_resp("Part1 [NEXT] Part2"), _resp("[END]")]
    student_resp = [_resp("OK")]

    _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=6)

    tutor_turns = [t for t in result.generated_turns if t["role"] == "TUTOR"]
    # First speaking turn split into 2 messages
    assert tutor_turns[0]["text"] == "Part1"
    assert tutor_turns[1]["text"] == "Part2"


# ---------------------------------------------------------------------------
# Tests: usage accumulation
# ---------------------------------------------------------------------------


def test_run_conversation_usage_accumulated(monkeypatch):
    """Token usage is summed across all tutor calls."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    usage1 = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    usage2 = {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}

    tutor_resp = [_resp("T1", usage=usage1), _resp("[END]", usage=usage2)]
    student_resp = [
        _resp("S1", usage={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})
    ]

    _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=6)

    assert result.tutor_usage["input_tokens"] == 18
    assert result.tutor_usage["output_tokens"] == 8
    assert result.student_usage["input_tokens"] == 4


def test_run_conversation_latencies_collected(monkeypatch):
    """Latency_seconds values are appended to the correct list."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [_resp("T1", latency=1.2), _resp("[END]", latency=0.8)]
    student_resp = [_resp("S1", latency=0.5)]

    _patch_all(monkeypatch, tutor_resp, student_resp)

    result = run_conversation(scenario, "fake-tutor", max_turns=6)

    assert result.tutor_latencies == [1.2, 0.8]
    assert result.student_latencies == [0.5]


def test_run_conversation_none_latency_not_appended(monkeypatch):
    """None latency is not appended to the list."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_resp = [_resp("[END]", latency=None)]
    _patch_all(monkeypatch, tutor_resp, [])

    result = run_conversation(scenario, "fake-tutor", max_turns=4)

    assert result.tutor_latencies == []


# ---------------------------------------------------------------------------
# Tests: scenario_id and tutor_model on Transcript
# ---------------------------------------------------------------------------


def test_run_conversation_transcript_ids(monkeypatch):
    """Transcript carries correct scenario_id and tutor_model."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    _patch_all(monkeypatch, [_resp("[END]")], [])

    result = run_conversation(scenario, "fake-tutor", max_turns=4)

    assert result.scenario_id == scenario.id
    assert result.tutor_model == "fake-tutor"


# ---------------------------------------------------------------------------
# Tests: registered (callable) tutor/student
# ---------------------------------------------------------------------------


def test_registered_tutor(monkeypatch):
    """Registered tutor callable is invoked with generated_turns."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    call_log = []

    def fake_tutor(turns):
        call_log.append(turns)
        return "[END]"

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {"kind": "registered", "fn": fake_tutor},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {"kind": "registered", "fn": lambda turns: "student reply"},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt", lambda mode, **kw: "SYS"
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt", lambda **kw: "SYS"
    )

    result = run_conversation(scenario, "custom-tutor", max_turns=4)

    assert result.ended_via == "END"
    assert len(call_log) == 1  # called once before [END]


def test_registered_student(monkeypatch):
    """Registered student callable runs and its output is recorded."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)

    tutor_call = [0]

    def fake_tutor(turns):
        tutor_call[0] += 1
        if tutor_call[0] == 2:
            return "[END]"
        return "Hello student"

    student_replies = []

    def fake_student(turns):
        student_replies.append(len(turns))
        return "I understand!"

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {"kind": "registered", "fn": fake_tutor},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {"kind": "registered", "fn": fake_student},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt", lambda mode, **kw: "SYS"
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt", lambda **kw: "SYS"
    )

    result = run_conversation(scenario, "custom-tutor", max_turns=4)

    student_turn = next(t for t in result.generated_turns if t["role"] == "STUDENT")
    assert student_turn["text"] == "I understand!"


# ---------------------------------------------------------------------------
# Tests: cacheable prefix passed to generate
# ---------------------------------------------------------------------------


def test_cacheable_prefix_passed_to_client(monkeypatch):
    """Client.generate is called with cacheable_prefix (the static head)."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    tutor_client = MagicMock()
    tutor_client.model = "fake-tutor"
    tutor_client.generate = MagicMock(return_value=_resp("[END]"))

    student_client = MagicMock()
    student_client.model = "fake-student"
    student_client.generate = MagicMock(return_value=_resp("OK"))

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {"kind": "hosted", "client": tutor_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {"kind": "hosted", "client": student_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt",
        lambda mode, **kw: "TUTOR_SYS",
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt",
        lambda **kw: "STUDENT_SYS",
    )

    run_conversation(scenario, "fake-tutor", max_turns=4)

    call_kwargs = tutor_client.generate.call_args
    assert "cacheable_prefix" in call_kwargs.kwargs
    head = call_kwargs.kwargs["cacheable_prefix"]
    # head must contain the system prompt and transcript_prefix
    assert "TUTOR_SYS" in head
    assert "Turn 1." in head


# ---------------------------------------------------------------------------
# Tests: frozen student trait (schema v2 — the runtime consumes, never generates)
# ---------------------------------------------------------------------------


def test_frozen_persona_flows_into_student_system_prompt(monkeypatch):
    """The persona handed to build_student_system_prompt is the scenario's
    frozen student.trait.persona — no trait generation happens."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5, persona="oracle-frozen-persona")
    seen = {}

    tutor_client = MagicMock()
    tutor_client.model = "fake-tutor"
    tutor_client.generate = MagicMock(
        side_effect=[_resp("Hello there"), _resp("[END]")]
    )
    student_client = MagicMock()
    student_client.model = "fake-student"
    student_client.generate = MagicMock(return_value=_resp("OK"))

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {"kind": "hosted", "client": tutor_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {"kind": "hosted", "client": student_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt",
        lambda mode, **kw: "TUTOR_SYS",
    )

    def capture_student_prompt(**kw):
        seen.update(kw)
        return "STUDENT_SYS"

    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt", capture_student_prompt
    )

    run_conversation(scenario, "fake-tutor", max_turns=4)
    assert seen.get("persona") == "oracle-frozen-persona"


def test_hosted_student_without_trait_is_hard_error(monkeypatch):
    """A hosted student on a moment without student.trait must raise —
    regenerating personas locally would be a different student population."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    del scenario.student["trait"]

    tutor_client = MagicMock()
    tutor_client.model = "fake-tutor"
    tutor_client.generate = MagicMock(return_value=_resp("[END]"))
    student_client = MagicMock()
    student_client.model = "fake-student"
    student_client.generate = MagicMock(return_value=_resp("OK"))

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {"kind": "hosted", "client": tutor_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {"kind": "hosted", "client": student_client, "kwargs": {}},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt",
        lambda mode, **kw: "TUTOR_SYS",
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt",
        lambda **kw: "STUDENT_SYS",
    )

    with pytest.raises(RuntimeError, match="frozen student trait"):
        run_conversation(scenario, "fake-tutor", max_turns=4)


def test_registered_student_needs_no_trait(monkeypatch):
    """Callable students own their behavior — a trait-less moment is fine."""
    from tutormoments.conversation import run_conversation

    scenario = _make_scenario(cut_turn=5)
    del scenario.student["trait"]

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {"kind": "registered", "fn": lambda turns: "[END]"},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {"kind": "registered", "fn": lambda turns: "student reply"},
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt", lambda mode, **kw: "SYS"
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt", lambda **kw: "SYS"
    )

    result = run_conversation(scenario, "custom-tutor", max_turns=4)
    assert result.ended_via == "END"


# ---------------------------------------------------------------------------
# Helpers for batch tests
# ---------------------------------------------------------------------------


def _make_scenario_id(sid: str, cut_turn: int = 5) -> "Scenario":
    """Like _make_scenario but with a custom scenario id."""
    from tutormoments.moments import Moment as Scenario

    context = [
        {
            "turn_number": i + 1,
            "role": "tutor" if i % 2 == 0 else "student",
            "text": f"text {i + 1}",
        }
        for i in range(cut_turn)
    ]
    return Scenario(
        id=sid,
        context=context,
        dimension="scaffolding",
        student={
            "mode": "oracle",
            "reference": "Turn 6. TUTOR: Hi",
            "context": "Grade 5",
            "trait": dict(_FROZEN_TRAIT),
        },
        rubric={"gold": "scaffolding", "hint": "hint"},
        provenance={
            "conv_id": f"conv-{sid}",
            "cut_turn": cut_turn,
            "turn_start": 3,
            "turn_end": 5,
            "moment_id": None,
            "annotator_id": None,
            "chosen_cut_turn": cut_turn,
            "cut_votes": {},
            "cluster_size": 1,
            "representative": None,
        },
    )


def _patch_batch(
    monkeypatch,
    tutor_batch_results: list[dict],
    student_batch_results: list[dict],
    tutor_kwargs: dict | None = None,
    student_kwargs: dict | None = None,
):
    """Patch resolve_tutor, resolve_student, system prompts, trait, and tutormoments.client.run_batch.

    tutor_batch_results: list of {sid: {"text": ...}} dicts, one per round.
    student_batch_results: list of {sid: {"text": ...}} dicts, one per round.
    tutor_kwargs/student_kwargs: roster generation kwargs returned by the
    patched resolve_* (defaults to {}).

    run_batch is called alternately: tutor round 1, student round 1, tutor round 2, ...
    We interleave them in the call sequence.
    """
    tutor_client = MagicMock()
    tutor_client.model = "fake-tutor-batch"
    student_client = MagicMock()
    student_client.model = "fake-student-batch"

    monkeypatch.setattr(
        "tutormoments.conversation.resolve_tutor",
        lambda id: {
            "kind": "hosted",
            "client": tutor_client,
            "kwargs": tutor_kwargs or {},
        },
    )
    monkeypatch.setattr(
        "tutormoments.conversation.resolve_student",
        lambda id=None: {
            "kind": "hosted",
            "client": student_client,
            "kwargs": student_kwargs or {},
        },
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_tutor_system_prompt",
        lambda mode, **kw: "TUTOR_SYS",
    )
    monkeypatch.setattr(
        "tutormoments.conversation.build_student_system_prompt",
        lambda **kw: "STUDENT_SYS",
    )

    # Interleave: tutor round 1, student round 1, tutor round 2, ...
    interleaved = []
    for t_res, s_res in zip(tutor_batch_results, student_batch_results):
        interleaved.append(t_res)
        interleaved.append(s_res)
    # If tutor_batch_results has one more (e.g. last round student skipped),
    # append the trailing tutor result.
    if len(tutor_batch_results) > len(student_batch_results):
        interleaved.append(tutor_batch_results[-1])

    run_batch_mock = MagicMock(side_effect=interleaved)
    # Patch at tutormoments.client since run_conversations_batch imports lazily from there.
    monkeypatch.setattr("tutormoments.client.run_batch", run_batch_mock)
    monkeypatch.setattr(
        "tutormoments.client.build_batch_entry",
        lambda key, tail, **kw: {"key": key, "_tail": tail},
    )
    return run_batch_mock


# ---------------------------------------------------------------------------
# Tests: run_conversations_batch
# ---------------------------------------------------------------------------


def test_batch_import():
    """run_conversations_batch is importable."""
    from tutormoments.conversation import run_conversations_batch

    assert run_conversations_batch is not None


def test_batch_two_scenarios_returns_two_transcripts(monkeypatch):
    """A 2-scenario batch returns exactly 2 Transcripts."""
    from tutormoments.conversation import run_conversations_batch

    s1 = _make_scenario_id("sid-1", cut_turn=3)
    s2 = _make_scenario_id("sid-2", cut_turn=3)

    # Round 1: both tutors reply normally, both students reply normally.
    # max_turns=2 => 1 round (1 tutor + 1 student = 2 speaking turns).
    tutor_round1 = {
        "sid-1": {"text": "Hello from tutor"},
        "sid-2": {"text": "Hi from tutor"},
    }
    student_round1 = {
        "sid-1": {"text": "Student reply 1"},
        "sid-2": {"text": "Student reply 2"},
    }

    _patch_batch(monkeypatch, [tutor_round1], [student_round1])

    results = run_conversations_batch(
        [s1, s2], tutor_id="fake-tutor", max_turns=2, poll_interval=0
    )

    assert len(results) == 2
    assert results[0].scenario_id == "sid-1"
    assert results[1].scenario_id == "sid-2"
    assert all(r.completed for r in results)


def test_batch_forwards_roster_thinking_and_effort_to_run_batch(monkeypatch):
    """Benchmark fidelity: batch mode must forward the roster's generation
    kwargs (thinking/effort/...) to run_batch, exactly as run_conversation
    merges them into client.generate(). Regression test for the bug where
    batch tutors silently ran without thinking/effort."""
    from tutormoments.conversation import run_conversations_batch

    s1 = _make_scenario_id("sid-1", cut_turn=3)
    tutor_round1 = {"sid-1": {"text": "Hello from tutor"}}
    student_round1 = {"sid-1": {"text": "Student reply"}}

    run_batch_mock = _patch_batch(
        monkeypatch,
        [tutor_round1],
        [student_round1],
        tutor_kwargs={"thinking": True, "effort": "xhigh"},
        student_kwargs={"thinking": False},
    )

    run_conversations_batch([s1], tutor_id="fake-tutor", max_turns=2, poll_interval=0)

    tutor_call, student_call = run_batch_mock.call_args_list
    assert tutor_call.kwargs["thinking"] is True
    assert tutor_call.kwargs["effort"] == "xhigh"
    assert student_call.kwargs["thinking"] is False
    assert student_call.kwargs["effort"] == ""


def test_batch_ended_via_tracked_per_scenario(monkeypatch):
    """ended_via is tracked per-scenario correctly."""
    from tutormoments.conversation import run_conversations_batch

    s1 = _make_scenario_id("sid-1", cut_turn=3)
    s2 = _make_scenario_id("sid-2", cut_turn=3)

    # s1 tutor emits [END], s2 tutor replies normally, then student replies.
    tutor_round1 = {
        "sid-1": {"text": "All done! [END]"},
        "sid-2": {"text": "Keep going."},
    }
    student_round1 = {"sid-2": {"text": "OK, continuing."}}

    # With max_turns=4, s1 ends in round 1 via END; s2 ends via MAX_TURNS.
    _patch_batch(monkeypatch, [tutor_round1], [student_round1])

    results = run_conversations_batch(
        [s1, s2], tutor_id="fake-tutor", max_turns=2, poll_interval=0
    )

    by_id = {r.scenario_id: r for r in results}
    assert by_id["sid-1"].ended_via == "END"
    assert by_id["sid-2"].ended_via == "MAX_TURNS"


def test_batch_ended_scenario_pruned_from_later_rounds(monkeypatch):
    """A scenario that emits [END] is not included in subsequent rounds."""
    from tutormoments.conversation import run_conversations_batch

    s1 = _make_scenario_id("sid-1", cut_turn=3)
    s2 = _make_scenario_id("sid-2", cut_turn=3)

    # Round 1 tutor: s1 ends, s2 continues.
    tutor_round1 = {
        "sid-1": {"text": "Bye! [END]"},
        "sid-2": {"text": "More to go."},
    }
    # Round 1 student: only s2 (s1 pruned).
    student_round1 = {"sid-2": {"text": "Got it."}}
    # Round 2 tutor: only s2 active.
    tutor_round2 = {"sid-2": {"text": "Final turn. [END]"}}

    run_batch_mock = _patch_batch(
        monkeypatch, [tutor_round1, tutor_round2], [student_round1]
    )

    results = run_conversations_batch(
        [s1, s2], tutor_id="fake-tutor", max_turns=6, poll_interval=0
    )

    # run_batch calls: tutor_r1, student_r1, tutor_r2 (student_r2 skipped since all ended).
    # Total calls: 3
    assert run_batch_mock.call_count == 3

    by_id = {r.scenario_id: r for r in results}
    assert by_id["sid-1"].ended_via == "END"
    assert by_id["sid-2"].ended_via == "END"


def test_batch_latencies_omitted(monkeypatch):
    """Batch mode does not populate tutor_latencies or student_latencies."""
    from tutormoments.conversation import run_conversations_batch

    s1 = _make_scenario_id("sid-1", cut_turn=3)

    tutor_round1 = {
        "sid-1": {
            "text": "Hello",
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        }
    }
    student_round1 = {"sid-1": {"text": "Hi back"}}

    _patch_batch(monkeypatch, [tutor_round1], [student_round1])

    results = run_conversations_batch(
        [s1], tutor_id="fake-tutor", max_turns=2, poll_interval=0
    )

    r = results[0]
    assert r.tutor_latencies == []
    assert r.student_latencies == []


def test_batch_round_based_ordering(monkeypatch):
    """Generated turns follow T-S-T-S order across rounds."""
    from tutormoments.conversation import run_conversations_batch

    s1 = _make_scenario_id("sid-1", cut_turn=2)

    tutor_round1 = {"sid-1": {"text": "Tutor1"}}
    student_round1 = {"sid-1": {"text": "Student1"}}
    tutor_round2 = {"sid-1": {"text": "Tutor2 [END]"}}

    _patch_batch(monkeypatch, [tutor_round1, tutor_round2], [student_round1])

    results = run_conversations_batch(
        [s1], tutor_id="fake-tutor", max_turns=6, poll_interval=0
    )

    turns = results[0].generated_turns
    roles = [t["role"] for t in turns]
    assert roles == ["TUTOR", "STUDENT", "TUTOR"]


# ---------------------------------------------------------------------------
# End-to-end integration test: tutor.py + student.py + conversation.py
# ---------------------------------------------------------------------------


def test_run_conversation_e2e_mocked_models(monkeypatch):
    """End-to-end mocked conversation: verifies tutor + student + conversation compose.

    This is the seam-level guard that all three modules work together:
    - Tutor returns 2 messages (split by [NEW_MESSAGE]) then [END]
    - Oracle student returns a canned reply
    - Transcript has correct alternating turns, numbering from cut_turn+1,
      ended_via="END", and accumulated usage.
    """
    from tutormoments.conversation import run_conversation

    # Create a real Scenario with 5 context turns
    scenario = _make_scenario(cut_turn=5)

    # Tutor responses: first turn splits into 2 messages via [NEW_MESSAGE],
    # then second turn ends.
    tutor_resps = [
        _resp(
            "Great start! [NEW_MESSAGE] Keep thinking about that.",
            usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
        ),
        _resp(
            "[END]", usage={"input_tokens": 95, "output_tokens": 5, "total_tokens": 100}
        ),
    ]

    # Student returns a canned reply
    student_resps = [
        _resp(
            "I see. Let me try again.",
            usage={"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
        )
    ]

    tutor_client, student_client = _patch_all(monkeypatch, tutor_resps, student_resps)

    # Run the conversation
    result = run_conversation(scenario, "fake-tutor", max_turns=8)

    # Assert completion and termination
    assert result.completed is True
    assert result.ended_via == "END"

    # Assert turn structure: T (2 msgs split by [NEW_MESSAGE]) + S (1 msg) + [END] stops loop
    assert len(result.generated_turns) == 3
    assert result.generated_turns[0]["role"] == "TUTOR"
    assert result.generated_turns[0]["text"] == "Great start!"
    assert result.generated_turns[1]["role"] == "TUTOR"
    assert result.generated_turns[1]["text"] == "Keep thinking about that."
    assert result.generated_turns[2]["role"] == "STUDENT"
    assert result.generated_turns[2]["text"] == "I see. Let me try again."

    # Assert turn numbering starts at cut_turn + 1 = 6
    assert result.generated_turns[0]["turn_number"] == 6
    assert result.generated_turns[1]["turn_number"] == 7
    assert result.generated_turns[2]["turn_number"] == 8

    # Assert usage accumulation
    assert result.tutor_usage["input_tokens"] == 195  # 100 + 95
    assert result.tutor_usage["output_tokens"] == 35  # 30 + 5
    assert result.tutor_usage["total_tokens"] == 230  # 130 + 100
    assert result.student_usage["input_tokens"] == 80
    assert result.student_usage["output_tokens"] == 20
    assert result.student_usage["total_tokens"] == 100

    # Assert no [END] token leaked into any message text
    for turn in result.generated_turns:
        assert "[END]" not in turn["text"]

    # Assert clients were called
    tutor_client.generate.assert_called()
    student_client.generate.assert_called()


def test_transcript_from_dict_roundtrip():
    """Transcript.from_dict reconstructs a written transcript (extra keys ignored)."""
    from tutormoments.conversation import Transcript

    t = Transcript(scenario_id="s1", tutor_model="m1")
    t.generated_turns = [{"turn_number": 2, "role": "TUTOR", "text": "Hi"}]
    t.completed = True
    t.ended_via = "MAX_TURNS"
    d = t.to_dict()
    d["some_future_field"] = "ignored"

    t2 = Transcript.from_dict(d)
    assert t2.scenario_id == "s1"
    assert t2.tutor_model == "m1"
    assert t2.generated_turns == t.generated_turns
    assert t2.completed is True
    assert t2.ended_via == "MAX_TURNS"
