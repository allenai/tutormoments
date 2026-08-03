"""Oracle student simulator prompt + student resolution for tutormoments.

Trait (persona) generation moved to tutormoments_build.traits when frozen
personas were embedded in the moments release (schema v2, 2026-07-03):
the runtime reads student.trait.persona from each released moment and
never generates or caches traits.

Public API:
    build_student_system_prompt(student_context, reference_transcript, persona) -> str
    resolve_student(student_id) -> dict
"""

import logging

from tutormoments.resources import resource_text

logger = logging.getLogger(__name__)

_PROMPTS_DIR = "prompts/student"

# ---------------------------------------------------------------------------
# Oracle student system prompt builder (Task 4)
# ---------------------------------------------------------------------------

_STUDENT_PROMPTS_DIR = _PROMPTS_DIR

_EMPTY_REFERENCE_FALLBACK = (
    "(The real conversation ends at the cut point — the student made no "
    "further turns. Acknowledge any final tutor message briefly, then end "
    "the session.)"
)


def _get_shared_generation_instructions() -> str:
    """Return the shared generation instructions text."""
    return (
        "You may generate multiple student turns in a row as needed. "
        "Do *not* generate any turns as the tutor. "
        "You should only generate turns that involve student utterances or actions. "
        "Remember to wait for the tutor to respond before generating your next turns."
    )


def _render_oracle_template() -> str:
    """Load prompts/student/oracle.txt and substitute {shared_generation_instructions}."""
    template = resource_text(f"{_STUDENT_PROMPTS_DIR}/oracle.txt").rstrip("\n")
    shared = _get_shared_generation_instructions()
    return template.replace("{shared_generation_instructions}", shared)


def build_student_system_prompt(
    *,
    student_context: str,
    reference_transcript: str | None,
    persona: str,
) -> str:
    """Assemble the STUDENT system prompt for oracle mode.

    Loads prompts/student/oracle.txt, substitutes {shared_generation_instructions},
    then fills the three bracket placeholders:
      [[REFERENCE_TRANSCRIPT_HERE]]           <- reference_transcript (fallback if empty)
      [[PERSONA_DESCRIPTION_HERE]]            <- persona
      [[NEXT_CONVERSATION_INFORMATION_HERE]]  <- student_context

    Args:
        student_context: Text for the student's current context / conversation info.
        reference_transcript: Full post-cut real transcript the oracle student imitates.
            If empty or None, the empty-reference fallback message is used instead.
        persona: The moment's frozen student.trait.persona text.

    Returns:
        Fully-substituted system prompt string (ASCII-safe, no unreplaced placeholders).
    """
    out = _render_oracle_template()

    # Substitute shared + bracket placeholders.
    out = out.replace("[[NEXT_CONVERSATION_INFORMATION_HERE]]", student_context or "")
    out = out.replace("[[PERSONA_DESCRIPTION_HERE]]", persona or "")

    # Empty reference fallback: same logic as students.py lines 155-160.
    ref = reference_transcript or _EMPTY_REFERENCE_FALLBACK
    out = out.replace("[[REFERENCE_TRANSCRIPT_HERE]]", ref)

    return out


# ---------------------------------------------------------------------------
# resolve_student (Task 4)
# ---------------------------------------------------------------------------


def resolve_student(student_id: str | None = None) -> dict:
    """Decide what the student is: registered callable or hosted model.

    Mirrors resolve_tutor() in tutor.py. The default hosted student is the
    model in the active config's student.model block.

    Args:
        student_id: Student name registered via @register_student, or None
            to use the default hosted student from config.

    Returns:
        Dictionary with:
        - kind == "registered": {"kind": "registered", "fn": <callable>}
        - kind == "hosted": {"kind": "hosted", "client": ModelClient, "kwargs": dict}
    """
    from tutormoments.client import ModelClient
    from tutormoments.config import get_registered_student, student_spec

    spec = student_spec()
    model = student_id or spec["model"]

    # FIRST: check registry for either an explicit student_id or config student.model.
    if model is not None:
        registered_fn = get_registered_student(model)
        if registered_fn is not None:
            return {"kind": "registered", "fn": registered_fn}

    # ELSE: use the hosted student model from config or the explicit id.
    client = ModelClient(model)
    kwargs = {k: v for k, v in spec.items() if k not in {"model", "mode"}}
    kwargs.setdefault("thinking", False)
    return {
        "kind": "hosted",
        "client": client,
        "kwargs": kwargs,
    }
