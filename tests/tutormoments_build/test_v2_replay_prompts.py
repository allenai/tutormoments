"""Tests for loading the replay prompt versions and filling their placeholders."""

import pytest

from tutormoments_build.v2 import replay_prompts as RP

CONTEXT = (
    "Year 4.\n\nLearning objective: Adding money.\n\nSession Content 1:\nslide text"
)
PERSONA = "Quiet; answers briefly."


@pytest.fixture(scope="module")
def tutor():
    return RP.load_tutor_prompts("1")


@pytest.fixture(scope="module")
def student():
    return RP.load_student_prompts("1")


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


def test_an_empty_context_leaves_no_placeholder_or_gap(tutor, student):
    rendered = [RP.tutor_system_prompt(tutor, mode, "") for mode in RP.TUTOR_MODES] + [
        RP.student_system_prompt(
            student, student_context="", reference_transcript="", persona=PERSONA
        )
    ]
    for text in rendered:
        assert "{student_context}" not in text
        assert "[[NEXT_CONVERSATION_INFORMATION_HERE]]" not in text
        assert "\n\n\n" not in text


def test_fill_context_drops_a_lone_placeholder_line_with_one_blank_after_it():
    text = "Goal.\n\n{c}\n\n## Next"
    assert RP.fill_context(text, "{c}", "") == "Goal.\n\n## Next"
    assert RP.fill_context(text, "{c}", "ctx") == "Goal.\n\nctx\n\n## Next"
    assert RP.fill_context("Inline: {c}.", "{c}", "") == "Inline: ."
