"""Tests for tutorsim.tutor module."""

import os
from unittest.mock import patch

import pytest

from tutorsim import register_tutor
from tutorsim.config import _TUTOR_REGISTRY, _reset_config_cache
from tutorsim.tutor import build_tutor_system_prompt, resolve_tutor


def test_plain_mode_loads_and_substitutes():
    """Test that 'plain' mode loads plain.txt and substitutes {student_context}."""
    context = "Grade 5, fractions"
    prompt = build_tutor_system_prompt("plain", student_context=context)

    assert context in prompt
    assert "You are an online tutor" in prompt
    # Ensure the placeholder is replaced, not present in output
    assert "{student_context}" not in prompt


def test_scaffolding_rigor_mode_loads_and_substitutes():
    """Test that 'scaffolding_rigor' mode loads its file and substitutes {student_context}."""
    context = "Grade 3, addition"
    prompt = build_tutor_system_prompt("scaffolding_rigor", student_context=context)

    assert context in prompt
    assert "Expert K-12 math tutor" in prompt
    assert "Over-scaffolding" in prompt
    # Ensure the placeholder is replaced
    assert "{student_context}" not in prompt


def test_oracle_mode_substitutes_reference_transcript():
    """Test that 'oracle' mode substitutes {reference_transcript}."""
    context = "Grade 4, geometry"
    reference = "Tutor: Great job! Now let's try the next one."
    prompt = build_tutor_system_prompt(
        "oracle", student_context=context, reference_transcript=reference
    )

    assert context in prompt
    assert reference in prompt
    assert "The real conversation continued past this point" in prompt
    # Ensure placeholders are replaced
    assert "{student_context}" not in prompt
    assert "{reference_transcript}" not in prompt


def test_oracle_mode_requires_reference_transcript():
    """Test that oracle mode raises ValueError if reference_transcript is not provided."""
    with pytest.raises(ValueError, match="reference_transcript"):
        build_tutor_system_prompt("oracle", student_context="Grade 5")


def test_default_mode_with_none():
    """Test that mode=None loads the default/plain prompt."""
    context = "Grade 2, counting"
    prompt = build_tutor_system_prompt(None, student_context=context)

    assert context in prompt
    assert "You are an online tutor" in prompt


def test_default_mode_with_empty_string():
    """Test that mode='' loads the default/plain prompt."""
    context = "Grade 6, fractions"
    prompt = build_tutor_system_prompt("", student_context=context)

    assert context in prompt
    assert "You are an online tutor" in prompt


def test_default_mode_explicit():
    """Test that mode='default' loads the default/plain prompt."""
    context = "Grade 1, basics"
    prompt = build_tutor_system_prompt("default", student_context=context)

    assert context in prompt
    assert "You are an online tutor" in prompt


def test_student_context_can_be_empty():
    """Test that student_context can be an empty string."""
    prompt = build_tutor_system_prompt("plain", student_context="")

    assert "You are an online tutor" in prompt
    assert "{student_context}" not in prompt


def test_resolve_tutor_hosted():
    """Test resolve_tutor for a hosted model."""
    # Patch the Anthropic SDK at the real import path
    with patch("anthropic.Anthropic"):
        # Set up the environment
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            result = resolve_tutor("claude-opus-4-8")

            # Verify the result structure
            assert result["kind"] == "hosted"
            assert "client" in result
            assert "kwargs" in result

            # Verify the client was instantiated correctly
            assert result["client"].model == "claude-opus-4-8"

            # Verify kwargs from config (expected: thinking=True, effort=xhigh for opus-4-8)
            assert isinstance(result["kwargs"], dict)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]


def test_resolve_tutor_registered():
    """Test resolve_tutor for a registered callable."""
    # Clean up registry first
    _TUTOR_REGISTRY.clear()

    # Register a dummy tutor
    @register_tutor("dummy-tutor")
    def dummy_fn(conversation):
        return "dummy response"

    try:
        result = resolve_tutor("dummy-tutor")

        # Verify the result structure
        assert result["kind"] == "registered"
        assert "fn" in result
        assert callable(result["fn"])
        assert result["fn"] is dummy_fn
    finally:
        # Clean up
        _TUTOR_REGISTRY.clear()


def test_resolve_tutor_unknown_raises():
    """Test that resolve_tutor raises ValueError for unknown tutors."""
    # Clean up registry
    _TUTOR_REGISTRY.clear()
    _reset_config_cache()

    # Attempt to resolve an unknown tutor
    with pytest.raises(ValueError, match="Model 'not-a-model' not in roster"):
        resolve_tutor("not-a-model")
