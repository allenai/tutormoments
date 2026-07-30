"""Build the tutor system prompt and resolve tutor callables/clients.

Single source of truth for synthesizing the TUTOR side of an exchange.
Loads the appropriate packaged prompt template and substitutes
the documented placeholders.

Modes (and their template paths):
- None / "" / "default" -> prompts/tutor/plain.txt (the default tutor)
- "plain"               -> prompts/tutor/plain.txt
- "scaffolding_rigor"   -> prompts/tutor/scaffolding_rigor.txt
- "oracle"              -> prompts/tutor/oracle.txt (requires `reference_transcript`)

Placeholders:
- {student_context}        <- student_context (always available)
- {reference_transcript}   <- reference_transcript (oracle only)
"""

from tutorsim.resources import resource_text

SUPPORTED_MODES = {"plain", "scaffolding_rigor", "oracle"}


def _load_template(filename: str) -> str:
    """Load a prompt template from the tutor prompts directory."""
    return resource_text(f"prompts/tutor/{filename}").strip()


def build_tutor_system_prompt(
    mode: str | None,
    *,
    student_context: str,
    reference_transcript: str = "",
) -> str:
    """Assemble the TUTOR system prompt.

    Args:
        mode: None (or "" or "default") for the plain tutor; "plain" for
            the basic tutor; "scaffolding_rigor" for the expert tutor;
            "oracle" for the reference-aware tutor that mimics the real
            human tutor's post-cut continuation.
        student_context: substituted for `{student_context}`.
        reference_transcript: required when mode == "oracle"; used for
            `{reference_transcript}` substitution.

    Returns:
        The fully-substituted system prompt string.

    Raises:
        ValueError: if mode == "oracle" but reference_transcript is not provided.
    """
    # Default mode: load plain.txt
    if mode in (None, "", "default"):
        template = _load_template("plain.txt")
    else:
        template = _load_template(f"{mode}.txt")

    # Substitute student_context
    out = template.replace("{student_context}", student_context or "")

    # Oracle-specific: substitute reference_transcript
    if mode == "oracle":
        if not reference_transcript:
            raise ValueError("mode='oracle' requires a non-empty reference_transcript")
        out = out.replace("{reference_transcript}", reference_transcript)

    return out


def resolve_tutor(tutor_id: str) -> dict:
    """Decide what the tutor is: registered callable or hosted model.

    Args:
        tutor_id: Tutor identifier (registered name or model roster id).

    Returns:
        Dictionary with:
        - kind == "registered": {"kind": "registered", "fn": <callable>}
        - kind == "hosted": {"kind": "hosted", "client": ModelClient, "kwargs": dict}

    Raises:
        ValueError: If tutor_id is not registered and not in model roster.
    """
    from tutorsim.client import ModelClient
    from tutorsim.config import get_registered_tutor, resolve_model

    # FIRST: check registry
    registered_fn = get_registered_tutor(tutor_id)
    if registered_fn is not None:
        return {"kind": "registered", "fn": registered_fn}

    # ELSE: resolve as hosted model (raises ValueError if unknown)
    model_spec = resolve_model(tutor_id)
    client = ModelClient(tutor_id)
    return {
        "kind": "hosted",
        "client": client,
        "kwargs": model_spec["kwargs"],
    }
