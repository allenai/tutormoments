"""v2 replay prompts: version 1 renders byte-identically to the v1 runtime."""

import pytest

from tutormoments.conversation import _build_role_prompt
from tutormoments.student import build_student_system_prompt
from tutormoments.tutor import build_tutor_system_prompt
from tutormoments_build.v2 import replay_prompts as RP

CONTEXT = "Year 4. Learning objectives covered in this session:\n- Adding money."
PREFIX = "Turn 1. Tutor: hi\n[SCREEN INTERACTION] clicked\nTurn 2. Student: hello"
EXTRA = "Turn 3. TUTOR: what is 2+2?\nTurn 4. STUDENT: 4"
PERSONA = "Quiet; answers briefly."
REFERENCE = "Turn 3. Tutor: ok\nTurn 4. Student: yes"


@pytest.fixture(scope="module")
def tutor():
    return RP.load_tutor_prompts("1")


@pytest.fixture(scope="module")
def student():
    return RP.load_student_prompts("1")


@pytest.mark.parametrize("mode", RP.TUTOR_MODES)
def test_tutor_prompt_matches_v1(tutor, mode):
    v1_head, v1_tail = _build_role_prompt(
        "TUTOR", PREFIX, EXTRA, CONTEXT, tutor_mode=mode
    )
    system = RP.tutor_system_prompt(tutor, mode, CONTEXT)

    assert system == build_tutor_system_prompt(mode, student_context=CONTEXT)
    assert RP.role_prompt(tutor, system, PREFIX, EXTRA) == (v1_head, v1_tail)


@pytest.mark.parametrize("reference", [REFERENCE, "", None])
def test_student_prompt_matches_v1(student, reference):
    """Includes the empty-reference fallback, now held in a prompt file."""
    v1_head, v1_tail = _build_role_prompt(
        "STUDENT",
        PREFIX,
        EXTRA,
        CONTEXT,
        reference_transcript=reference,
        persona=PERSONA,
    )
    system = RP.student_system_prompt(
        student,
        student_context=CONTEXT,
        reference_transcript=reference,
        persona=PERSONA,
    )

    assert system == build_student_system_prompt(
        student_context=CONTEXT, reference_transcript=reference, persona=PERSONA
    )
    assert RP.role_prompt(student, system, PREFIX, EXTRA) == (v1_head, v1_tail)


def test_no_placeholder_survives_rendering(tutor, student):
    rendered = [
        RP.tutor_system_prompt(tutor, mode, CONTEXT) for mode in RP.TUTOR_MODES
    ] + [
        RP.student_system_prompt(
            student, student_context=CONTEXT, reference_transcript="", persona=PERSONA
        )
    ]
    for text in rendered:
        assert "[[" not in text
        assert "{student_context}" not in text
        assert "{shared_generation_instructions}" not in text


def test_the_newest_version_is_the_default():
    assert RP.load_tutor_prompts().version == RP.load_tutor_prompts("1").version


def test_v1s_oracle_tutor_is_not_offered(tutor):
    with pytest.raises(ValueError, match="unknown tutor mode"):
        RP.tutor_system_prompt(tutor, "oracle", CONTEXT)


def test_a_missing_version_names_the_available_ones():
    with pytest.raises(FileNotFoundError, match="versions available: 1"):
        RP.load_student_prompts("999")
