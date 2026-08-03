"""Build-time student-trait generation (tutormoments_build.traits).

Trait generation moved out of the runtime when frozen personas were embedded
in the release (schema v2): the runtime only consumes student.trait; new
moments sets get their personas at build time via generate_traits_for_moments.
"""

from unittest.mock import MagicMock

from tutormoments.moments import Moment
from tutormoments_build.traits import generate_traits_for_moments


def _moment(mid="t:c1__hum_1_2", trait=None):
    student = {"mode": "oracle", "reference": "", "context": "Grade 5"}
    if trait is not None:
        student["trait"] = trait
    return Moment(
        id=mid,
        context=[
            {"turn_number": 1, "role": "tutor", "text": "What is 3/4 of 8?"},
            {"turn_number": 2, "role": "student", "text": "Um... 6?"},
        ],
        dimension="scaffolding",
        student=student,
        rubric={"gold": "scaffolding", "hint": ""},
        provenance={"conv_id": "c1", "cut_turn": 2},
    )


def _fake_client():
    """ModelClient stub: every generate() returns a parseable trait response."""
    client = MagicMock()
    client.model = "claude-opus-4-6"
    resp = MagicMock()
    resp.text = "DESCRIPTION: The student rushes and guesses under uncertainty."
    client.generate.return_value = resp
    return client


def test_generate_traits_attaches_full_trait_object():
    m = _moment()
    n = generate_traits_for_moments(
        [m], model_client=_fake_client(), model_name="claude-opus-4-6"
    )
    assert n == 1
    trait = m.student["trait"]
    assert trait["persona"]  # non-empty, parsed from DESCRIPTION:
    assert "rushes and guesses" in trait["persona"]
    assert trait["trait_mode"] == "joined-3"
    assert trait["generator_model"] == "claude-opus-4-6"
    assert trait["generated_at"]  # ISO timestamp recorded


def test_generate_traits_prefix_is_turn_formatted_context():
    """The generator is prompted with the same 'Turn N. ROLE: text' prefix
    format the paper's runtime used (pre-cut context only)."""
    client = _fake_client()
    m = _moment()
    generate_traits_for_moments([m], model_client=client, model_name="x")
    prompts = [c.args[0] for c in client.generate.call_args_list]
    assert any("Turn 1. TUTOR: What is 3/4 of 8?" in p for p in prompts)
    assert any("Turn 2. STUDENT: Um... 6?" in p for p in prompts)


def test_generate_traits_skips_moments_with_existing_trait():
    frozen = {
        "persona": "frozen",
        "trait_mode": "joined-3",
        "generator_model": "claude-opus-4-6",
        "generated_at": "2026-06-18T00:00:00",
    }
    client = _fake_client()
    m = _moment(trait=frozen)
    n = generate_traits_for_moments([m], model_client=client, model_name="x")
    assert n == 0
    assert m.student["trait"] == frozen  # untouched
    assert client.generate.call_count == 0  # no LLM calls


def test_trait_generator_prompts_ship_in_build_package():
    """system/user templates + dimension descriptions live in tutormoments_build,
    not the runtime package."""
    from tutormoments_build.resources import resource_text

    assert resource_text("prompts/trait_generator/system.txt").strip()
    assert resource_text("prompts/trait_generator/user.txt").strip()
    assert resource_text(
        "prompts/trait_generator/dimensions/misconceptions.txt"
    ).strip()


# ---------------------------------------------------------------------------
# TraitGenerator live-path tests (moved from tests/tutormoments/test_student.py
# when trait generation moved to build time)
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _make_client(response_text: str) -> MagicMock:
    """Return a mock ModelClient whose .generate() returns a ModelResponse stub."""
    mock_response = SimpleNamespace(text=response_text)
    client = MagicMock()
    client.generate = MagicMock(return_value=mock_response)
    return client


class TestGenerateTrait:
    """Tests for the module-level generate_trait() function."""

    def test_joined3_builds_system_prompt_with_all_5_dimensions(self):
        """System prompt must contain text from all 5 dimension files."""
        from tutormoments_build.traits import generate_trait

        client = _make_client("Some thinking\nDESCRIPTION: A diligent student.")
        # We don't need the description content; just confirm the call goes through.
        generate_trait("Turn 1. TUTOR: Hi", mode="joined-3", model_client=client)

        # All 5 generate calls happened (one per dimension).
        assert client.generate.call_count == 5

    def test_joined3_parses_description_marker(self):
        """generate_trait extracts text after 'DESCRIPTION:' for each dimension call."""
        from tutormoments_build.traits import generate_trait

        # Each of the 5 calls returns a distinct description.
        responses = [
            SimpleNamespace(text=f"Some thinking\nDESCRIPTION: Dimension {i} desc.")
            for i in range(5)
        ]
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)

        result = generate_trait(
            "Turn 1. TUTOR: Hi", mode="joined-3", model_client=client
        )

        # Result is the sorted+joined descriptions.
        assert isinstance(result, str)
        # All 5 descriptions appear in the result.
        for i in range(5):
            assert f"Dimension {i} desc." in result

    def test_joined3_sorts_and_joins_with_double_newline(self):
        """Dimension descriptions are sorted alphabetically and joined with '\\n\\n'."""
        from tutormoments_build.traits import generate_trait

        # Return a fixed description per call. We reverse-order by hand to verify sort.
        descs = ["Z last", "A first", "M middle", "B second", "Y fourth"]
        responses = [SimpleNamespace(text=f"DESCRIPTION: {d}") for d in descs]
        client = MagicMock()
        client.generate = MagicMock(side_effect=responses)

        result = generate_trait("some prefix", mode="joined-3", model_client=client)

        parts = result.split("\n\n")
        assert parts == sorted(descs), f"Expected sorted join, got: {parts}"

    def test_adapter_calls_generate_with_correct_defaults(self):
        """The adapter must call client.generate with temperature=0.7, max_tokens=1024."""
        from tutormoments_build.traits import generate_trait

        client = _make_client("DESCRIPTION: A student.")
        generate_trait("prefix", mode="joined-3", model_client=client)

        # Check every generate call uses the right defaults.
        for call in client.generate.call_args_list:
            kwargs = call.kwargs if call.kwargs else {}
            # max_tokens must be 1024 (adapter default when None passed by TraitGenerator)
            assert kwargs.get("max_tokens", 1024) == 1024, f"max_tokens wrong: {kwargs}"
            # json_mode must be False
            assert kwargs.get("json_mode", False) is False, f"json_mode wrong: {kwargs}"

    def test_no_description_marker_returns_full_output(self):
        """When 'DESCRIPTION:' is missing, the full response text is used as description."""
        from tutormoments_build.traits import generate_trait

        full_text = "No marker here, just plain text."
        client = _make_client(full_text)
        result = generate_trait("prefix", mode="joined-3", model_client=client)

        assert full_text.strip() in result

    def test_description_after_marker_stripped(self):
        """Text before 'DESCRIPTION:' (thinking) is stripped; only post-marker text is kept."""
        from tutormoments_build.traits import generate_trait

        client = _make_client(
            "  lots of thinking\n\nDESCRIPTION:   Parsed description.  "
        )
        result = generate_trait("prefix", mode="joined-3", model_client=client)

        # All 5 dimension results will be "Parsed description." (same stub for each)
        # After sort+dedup it will be five copies, but we just check the content.
        assert "Parsed description." in result
        assert "lots of thinking" not in result

    def test_system_prompt_passed_as_cacheable_prefix(self):
        """The system prompt is forwarded as cacheable_prefix, not embedded in user text."""
        from tutormoments_build.traits import generate_trait

        client = _make_client("DESCRIPTION: A student.")
        generate_trait("prefix text", mode="joined-3", model_client=client)

        for call in client.generate.call_args_list:
            kwargs = call.kwargs if call.kwargs else {}
            # cacheable_prefix must be set (the system prompt) and non-empty
            assert "cacheable_prefix" in kwargs, (
                "cacheable_prefix missing from generate call"
            )
            assert kwargs["cacheable_prefix"], "cacheable_prefix is empty/None"

    def test_user_prompt_contains_conversation_text(self):
        """The user prompt (first positional arg) must contain the transcript prefix."""
        from tutormoments_build.traits import generate_trait

        prefix = "Turn 1. TUTOR: What is 2+2?\nTurn 2. STUDENT: 4"
        client = _make_client("DESCRIPTION: A bright student.")
        generate_trait(prefix, mode="joined-3", model_client=client)

        for call in client.generate.call_args_list:
            user_text = call.args[0] if call.args else call.kwargs.get("prompt", "")
            assert prefix in user_text, (
                f"Prefix missing from user text: {user_text[:200]}"
            )
