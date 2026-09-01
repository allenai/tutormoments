"""Tests for tutormoments.student module - oracle prompt render + student resolution.

Trait generation moved to tutormoments_build (frozen personas ship in the
release); its tests live in tests/tutormoments_build/test_traits.py.
"""

# ---------------------------------------------------------------------------
# Tests: build_student_system_prompt (oracle mode) -- Task 4
# ---------------------------------------------------------------------------


class TestBuildStudentSystemPrompt:
    """Tests for build_student_system_prompt() -- oracle prompt render + substitutions."""

    def test_substitutes_reference_transcript(self):
        """[[REFERENCE_TRANSCRIPT_HERE]] must be replaced with the supplied reference."""
        from tutormoments.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5 math class",
            reference_transcript="Turn 8: TUTOR: What is 3+3? STUDENT: 6",
            persona="A curious, energetic student.",
        )
        assert "Turn 8: TUTOR: What is 3+3? STUDENT: 6" in result
        assert "[[REFERENCE_TRANSCRIPT_HERE]]" not in result

    def test_substitutes_persona(self):
        """[[PERSONA_DESCRIPTION_HERE]] must be replaced with the supplied persona."""
        from tutormoments.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5 math class",
            reference_transcript="Turn 8: TUTOR: Hi STUDENT: Hello",
            persona="A curious, energetic student.",
        )
        assert "A curious, energetic student." in result
        assert "[[PERSONA_DESCRIPTION_HERE]]" not in result

    def test_substitutes_student_context(self):
        """[[NEXT_CONVERSATION_INFORMATION_HERE]] must be replaced with student_context."""
        from tutormoments.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5 math class",
            reference_transcript="Turn 8: TUTOR: Hi STUDENT: Hello",
            persona="A curious student.",
        )
        assert "Grade 5 math class" in result
        assert "[[NEXT_CONVERSATION_INFORMATION_HERE]]" not in result

    def test_substitutes_shared_generation_instructions(self):
        """{shared_generation_instructions} must be replaced (no literal brace placeholder)."""
        from tutormoments.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5",
            reference_transcript="Turn 8...",
            persona="A curious student.",
        )
        assert "{shared_generation_instructions}" not in result
        # The actual shared instructions contain this sentinel text.
        assert "Do *not* generate any turns as the tutor" in result

    def test_no_unreplaced_brackets(self):
        """No double-bracket placeholders should remain in the output."""
        from tutormoments.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="context text",
            reference_transcript="ref text",
            persona="persona text",
        )
        assert "[[" not in result
        assert "]]" not in result

    def test_empty_reference_uses_fallback_message(self):
        """When reference_transcript is empty/None, the fallback message is substituted."""
        from tutormoments.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5",
            reference_transcript="",
            persona="A student.",
        )
        assert "The real conversation ends at the cut point" in result
        assert "[[REFERENCE_TRANSCRIPT_HERE]]" not in result

    def test_none_reference_uses_fallback_message(self):
        """When reference_transcript is None, the fallback message is substituted."""
        from tutormoments.student import build_student_system_prompt

        result = build_student_system_prompt(
            student_context="Grade 5",
            reference_transcript=None,
            persona="A student.",
        )
        assert "The real conversation ends at the cut point" in result


# ---------------------------------------------------------------------------
# Tests: resolve_student -- Task 4
# ---------------------------------------------------------------------------


class TestResolveStudent:
    """Tests for resolve_student() -- hosted model resolution + registry."""

    def test_resolve_student_no_id_returns_hosted_claude_opus(self, monkeypatch):
        """resolve_student() with no args returns hosted claude-opus-4-6 with thinking off."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Patch anthropic.Anthropic so no real HTTP call is made.
        import unittest.mock as mock

        from tutormoments.config import _reset_config_cache

        _reset_config_cache()
        with mock.patch("anthropic.Anthropic"):
            from tutormoments.student import resolve_student

            result = resolve_student()

        assert result["kind"] == "hosted"
        assert result["client"] is not None
        # The default config states the student's thinking-off condition
        # explicitly ({type: disabled}) so it survives model swaps.
        assert result["kwargs"]["thinking"] == {"thinking": {"type": "disabled"}}

    def test_resolve_student_registered(self, monkeypatch):
        """A @register_student-decorated callable is returned by resolve_student(name)."""
        import unittest.mock as mock

        from tutormoments.config import _STUDENT_REGISTRY, register_student

        # Register a dummy student.
        @register_student("dummy")
        def _dummy_student(conversation):
            return "I am a dummy student"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with mock.patch("anthropic.Anthropic"):
            from tutormoments.student import resolve_student

            result = resolve_student("dummy")

        assert result["kind"] == "registered"
        assert result["fn"] is _dummy_student

        # Cleanup registry.
        _STUDENT_REGISTRY.pop("dummy", None)

    def test_resolve_student_registered_from_config_model(self, tmp_path, monkeypatch):
        """student.model can name a registered callable."""
        from tutormoments.config import (
            _STUDENT_REGISTRY,
            _reset_config_cache,
            register_student,
        )
        from tutormoments.student import resolve_student

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
providers:
  anthropic: { env: ANTHROPIC_API_KEY }
  openai:    { env: OPENAI_API_KEY }
  gemini:    { env: GEMINI_API_KEY }
  together:  { env: TOGETHER_API_KEY }
benchmark_models:
  claude-opus-4-8: { thinking: { type: adaptive }, effort: xhigh, condition: xhigh }
# A registered (scripted) student model is a code callable: no wire form,
# no thinking keys.
student: { model: dummy-from-config, mode: oracle }
scorer:  { model: claude-opus-4-6, thinking: { type: adaptive } }
defaults: { trials: 1, max_turns: 5 }
retry:    { max_retries: 5, base_delay: 5 }
batch:    { timeout: 86400 }
""",
            encoding="utf-8",
        )
        monkeypatch.setenv("TUTORMOMENTS_CONFIG", str(config_path))
        _reset_config_cache()

        @register_student("dummy-from-config")
        def _dummy_student(conversation):
            return "student turn"

        result = resolve_student()
        assert result["kind"] == "registered"
        assert result["fn"] is _dummy_student

        _STUDENT_REGISTRY.pop("dummy-from-config", None)
        _reset_config_cache()
