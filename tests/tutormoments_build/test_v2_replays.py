"""v2 replays: what each side sees, the loop, records and paths."""

import threading
from types import SimpleNamespace

import pytest

from tutormoments.conversation import _parse_tutor_tokens, _split_messages
from tutormoments_build.v2 import classify_excerpts as C
from tutormoments_build.v2 import excerpts
from tutormoments_build.v2 import replay_data as RD
from tutormoments_build.v2 import replay_prompts as RP
from tutormoments_build.v2 import replays as R


def _row(index, turn, role, text):
    return {"turn_index": index, "turn_number": turn, "role": role, "text": text}


ROWS = [
    _row(0, None, "", "[SCREEN UPDATE] board opens"),
    _row(1, 1, "TUTOR", "[TUTOR] hi there"),
    _row(2, None, "", "[PAUSE] 12 seconds"),
    _row(3, 2, "STUDENT", "hello"),  # cut row
    _row(4, None, "", "[SCREEN INTERACTION] tutor draws"),
    _row(5, 3, "TUTOR", "real post-cut tutor"),
    _row(6, 4, "STUDENT", "real post-cut student"),  # end row
    _row(7, 5, "TUTOR", "after the moment"),
]

PROBLEMS = [
    {"objective": "Adding money using the column method.", "year": 4},
    {"objective": "Adding money using the column method.", "year": 4},
    {"objective": "Subtracting money", "year": 5},
]


def _moment(moment_id="m1", problems=(), rows=ROWS, cut_index=3, end_index=6):
    return RD.ReplayMoment(
        moment={
            "moment_id": moment_id,
            "source": "tutoring_provider_c" if problems else "tutoring_provider_a",
            "transcript_id": "t1",
            "start_turn": 1,
            "cut_turn": 2,
            "end_turn": 4,
            "labels": {"scaffolding_present": True},
            "situation_type": "scaffold_only",
        },
        conversation_id="c1",
        rows=rows,
        cut_index=cut_index,
        end_index=end_index,
        problems=list(problems),
    )


class ScriptedClient:
    """Returns the scripted texts in order; records every call."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        with self._lock:
            self.calls.append({"prompt": prompt, **kwargs})
            text = self.texts.pop(0)
        return SimpleNamespace(
            text=text, usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        )


TUTOR_PROMPTS = RP.load_tutor_prompts("1")
STUDENT_PROMPTS = RP.load_student_prompts("1")


def _replay(moment, turns, tutor, student=None, persona="Quiet."):
    return R.replay(
        moment,
        turns=turns,
        tutor_client=tutor,
        tutor_thinking={"thinking_level": "high"},
        tutor_mode="plain",
        tutor_prompts=TUTOR_PROMPTS,
        student_client=student,
        student_thinking={"thinking_level": "low"},
        student_prompts=STUDENT_PROMPTS,
        persona=persona,
    )


# ---------------------------------------------------------------- what is seen


def test_transcript_so_far_stops_at_the_cut_row_with_enrichments():
    assert R.transcript_so_far(_moment()) == (
        "[SCREEN UPDATE] board opens\n"
        "Turn 1. TUTOR: hi there\n"
        "[PAUSE] 12 seconds\n"
        "Turn 2. STUDENT: hello"
    )


def test_reference_runs_from_after_the_cut_through_end_index():
    assert R.reference_transcript(_moment()) == (
        "[SCREEN INTERACTION] tutor draws\n"
        "Turn 3. TUTOR: real post-cut tutor\n"
        "Turn 4. STUDENT: real post-cut student"
    )
    assert R.reference_transcript(_moment(end_index=3)) == ""


def test_provider_c_context_lists_years_and_objectives_once():
    assert R.student_context(_moment(problems=PROBLEMS)) == (
        "Years 4-5. Learning objectives covered in this session:\n"
        "- Adding money using the column method.\n"
        "- Subtracting money"
    )
    one_year = [{"objective": "Fractions", "year": 3}]
    assert R.student_context(_moment(problems=one_year)).startswith("Year 3. ")


def test_non_consecutive_years_are_listed_not_ranged():
    problems = [{"objective": "a", "year": 3}, {"objective": "b", "year": 5}]
    assert R.student_context(_moment(problems=problems)).startswith("Years 3, 5. ")


def test_provider_a_gets_v1s_fallback_context():
    assert R.student_context(_moment()) == "K-12 tutoring session"


def test_tutor_prompt_holds_context_and_transcript_but_no_cut_marker():
    tutor = ScriptedClient("ok")
    _replay(_moment(problems=PROBLEMS), 1, tutor)
    head = tutor.calls[0]["cacheable_prefix"]

    assert "Learning objectives covered in this session" in head
    assert head.endswith(R.transcript_so_far(_moment()))
    assert excerpts.CUT_POINT not in head + tutor.calls[0]["prompt"]
    assert "real post-cut" not in head  # the tutor never sees the reference


# ---------------------------------------------------------------- the loop


def test_v1_turn_parsing_is_reproduced():
    for text in ["a [NEW_MESSAGE] b", "a [NEXT] b [END]", "x [PROBLEM_CHANGE]", " "]:
        assert R.parse_tutor_tokens(text) == _parse_tutor_tokens(text)
        assert R.split_messages(text) == _split_messages(text)


def test_n1_is_one_tutor_turn_and_no_student_call():
    tutor = ScriptedClient("first [NEW_MESSAGE] second")
    result = _replay(_moment(), 1, tutor, student=None, persona=None)

    assert [
        (r["turn_number"], r["role"], r["text"]) for r in result["generated_rows"]
    ] == [
        (3, "TUTOR", "first"),
        (4, "TUTOR", "second"),
    ]
    assert result["ended_via"] == "TURNS" and not result["ended_early"]
    assert [c["role"] for c in result["calls"]] == ["tutor"]
    assert tutor.calls[0]["max_tokens"] == 0


def test_n_controls_tutor_turns_and_ends_on_a_tutor_turn():
    tutor = ScriptedClient("t1", "t2", "t3")
    student = ScriptedClient("s1 [NEW_MESSAGE] s1b", "s2")
    result = _replay(_moment(), 3, tutor, student)

    assert [c["role"] for c in result["calls"]] == [
        "tutor",
        "student",
        "tutor",
        "student",
        "tutor",
    ]
    rows = result["generated_rows"]
    assert [r["text"] for r in rows] == ["t1", "s1", "s1b", "t2", "s2", "t3"]
    assert [r["turn_number"] for r in rows] == [3, 4, 5, 6, 7, 8]
    assert rows[-1]["role"] == "TUTOR"
    assert result["tutor_turns"] == 3
    # Generated turns reach later calls through the tail, not the cached head.
    assert "Turn 5. STUDENT: s1b" in tutor.calls[2]["prompt"]
    assert all(
        c["cacheable_prefix"] == tutor.calls[0]["cacheable_prefix"] for c in tutor.calls
    )
    assert result["usage"]["tutor"]["total_tokens"] == 21
    assert result["usage"]["student"]["total_tokens"] == 14


def test_student_prompt_carries_persona_and_reference():
    tutor = ScriptedClient("t1", "t2")
    student = ScriptedClient("s1")
    _replay(_moment(), 2, tutor, student, persona="A very particular persona.")
    head = student.calls[0]["cacheable_prefix"]

    assert "A very particular persona." in head
    assert "Turn 4. STUDENT: real post-cut student" in head
    assert "after the moment" not in head  # past end_index


def test_empty_reference_uses_the_fallback():
    student = ScriptedClient("s1")
    _replay(_moment(end_index=3), 2, ScriptedClient("t1", "t2"), student)
    assert (
        STUDENT_PROMPTS.texts["empty_reference"] in student.calls[0]["cacheable_prefix"]
    )


def test_empty_responses_are_recorded_as_ellipses():
    result = _replay(_moment(), 2, ScriptedClient("", "t2"), ScriptedClient("  "))
    assert [r["text"] for r in result["generated_rows"]][:2] == ["...", "..."]


def test_end_token_stops_the_replay_early():
    student = ScriptedClient()
    result = _replay(_moment(), 3, ScriptedClient("bye [END]"), student)

    assert [r["text"] for r in result["generated_rows"]] == ["bye"]
    assert result["ended_via"] == "END" and result["ended_early"]
    assert student.calls == []


def test_a_student_turn_without_a_persona_raises():
    with pytest.raises(ValueError, match="no persona"):
        _replay(_moment(), 2, ScriptedClient("t1"), ScriptedClient("s1"), persona=None)


# ---------------------------------------------------------------- records


def _record(turns=2):
    result = _replay(_moment(), turns, ScriptedClient("t1", "t2"), ScriptedClient("s1"))
    settings = {"tutor": {"arm": "a"}, "student": {}, "turns": turns}
    return R.build_record(_moment(), result, settings)


def test_record_excerpt_is_a_human_excerpt_with_generated_rows():
    record = _record()
    excerpt = C.excerpt_at(record, excerpts.FULL_KEY)

    assert excerpt == (
        R.transcript_so_far(_moment())
        + f"\n{excerpts.CUT_POINT}\n"
        + "Turn 3. TUTOR: t1\nTurn 4. STUDENT: s1\nTurn 5. TUTOR: t2"
    )
    assert "real post-cut" not in excerpt
    assert C.has_post_cut_content(record)
    assert record["labels"] == {"scaffolding_present": True}
    assert record["situation_type"] == "scaffold_only"
    assert "split" not in record


def test_replay_lead_up_matches_the_human_excerpt_at_every_width():
    moment = _moment()
    bounds = {"start_index": 1, "cut_index": 3, "end_index": 6}
    record = _record()
    assert set(record["excerpts"]) == {
        excerpts.width_key(w) for w in excerpts.DEFAULT_CONTEXT_WIDTHS
    }
    for width in (*excerpts.DEFAULT_CONTEXT_WIDTHS, 1):
        human, human_first = excerpts.render_excerpt(ROWS, bounds, context_turns=width)
        replay, first = R.replay_excerpt(moment, record["generated_rows"], width)
        cut = f"{excerpts.CUT_POINT}\n"
        assert replay.split(cut)[0] == human.split(cut)[0]
        assert first == human_first
    # A window narrower than the lead-up opens on a dialogue turn, as humans do.
    assert R.replay_excerpt(moment, [], 1)[0].startswith("Turn 1. TUTOR: hi there")


def test_output_path_names_every_setting():
    settings = {
        "tutor": {"arm": "gemini-4-argon", "prompt": "plain", "prompt_version": "1"},
        "student": {"model": "gemini-4-argon", "prompt_version": "1"},
        "turns": 1,
    }
    assert R.replay_path("out", settings) == (
        "out/gemini-4-argon/plain_v1/n1/student-gemini-4-argon_sp1.jsonl"
    )


def test_dry_run_makes_no_call(monkeypatch, capsys):
    import tutormoments.client

    monkeypatch.setattr(RD, "load_moments", lambda *a, **k: ([_moment()], {}))
    monkeypatch.setattr(
        tutormoments.client,
        "get_client",
        lambda *a, **k: pytest.fail("dry run must not build a client"),
    )
    assert R.main(["--tutor", "gemini-4-argon", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "to replay: 1" in out
    assert "gemini-4-argon/plain_v1/n1/" in out


def test_missing_personas_stop_a_multi_turn_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(RD, "load_moments", lambda *a, **k: ([_moment()], {}))
    code = R.main(
        [
            "--tutor",
            "gemini-4-argon",
            "--turns",
            "2",
            "--dry-run",
            "--traits",
            str(tmp_path / "none.jsonl"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "no persona" in capsys.readouterr().err
