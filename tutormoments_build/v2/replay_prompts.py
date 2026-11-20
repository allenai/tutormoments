"""Tutor and student prompts for v2 replays.

The prompts are versioned copies of the v1 runtime's, under
``prompts/v2/tutor/<version>/`` and ``prompts/v2/student/<version>/``, so they
can be revised without mixing results. Version 1 renders byte-identically to
v1 (``tutormoments.tutor`` / ``tutormoments.student`` /
``tutormoments.conversation._build_role_prompt``): the text v1 keeps in Python
-- the student's shared generation instructions, the empty-reference fallback,
the "Here is the conversation so far:" joiner and the role instructions -- lives
in the version folder instead.

Every file a version needs is read when it is loaded, so a version missing one
fails before any moment is sent.
"""

from dataclasses import dataclass

from tutormoments_build.resources import resource_text
from tutormoments_build.v2.classify_excerpts import (
    available_prompt_versions,
    latest_prompt_version,
)

TUTOR_PROMPT_ROOT = "prompts/v2/tutor"
STUDENT_PROMPT_ROOT = "prompts/v2/student"

# v1's plain and scaffolding_rigor tutors; v1's oracle tutor is not copied.
TUTOR_MODES = ("plain", "scaffolding_rigor")

_SHARED_FILES = {"joiner": "conversation_so_far.md", "respond": "respond.md"}
TUTOR_FILES = {**{mode: f"{mode}.md" for mode in TUTOR_MODES}, **_SHARED_FILES}
STUDENT_FILES = {
    "oracle": "oracle.md",
    "empty_reference": "empty_reference.md",
    **_SHARED_FILES,
}


@dataclass(frozen=True)
class RolePrompts:
    """One role's prompt files at one version, read and trimmed."""

    role: str
    version: str
    root: str
    texts: dict[str, str]


def _load(role: str, root: str, files: dict[str, str], version: str | None):
    version = version or latest_prompt_version(root)
    texts = {}
    for name, filename in files.items():
        path = f"{root}/{version}/{filename}"
        try:
            raw = resource_text(path)
        except (FileNotFoundError, NotADirectoryError) as exc:
            available = available_prompt_versions(root)
            raise FileNotFoundError(
                f"no {role} prompt at {path}; versions available: "
                f"{', '.join(available) or '(none)'}"
            ) from exc
        # v1 strips the tutor template and only right-strips newlines off the
        # student template; matching each keeps the rendering byte-identical.
        texts[name] = raw.rstrip("\n") if name == "oracle" else raw.strip()
    return RolePrompts(role=role, version=version, root=root, texts=texts)


def load_tutor_prompts(version: str | None = None) -> RolePrompts:
    """The tutor prompt files at ``version`` (default: the newest)."""
    return _load("TUTOR", TUTOR_PROMPT_ROOT, TUTOR_FILES, version)


def load_student_prompts(version: str | None = None) -> RolePrompts:
    """The student prompt files at ``version`` (default: the newest)."""
    return _load("STUDENT", STUDENT_PROMPT_ROOT, STUDENT_FILES, version)


def tutor_system_prompt(prompts: RolePrompts, mode: str, student_context: str) -> str:
    """The tutor system prompt for ``mode``, as v1's build_tutor_system_prompt."""
    if mode not in TUTOR_MODES:
        raise ValueError(f"unknown tutor mode {mode!r}; choose from {TUTOR_MODES}")
    return prompts.texts[mode].replace("{student_context}", student_context or "")


def student_system_prompt(
    prompts: RolePrompts,
    *,
    student_context: str,
    reference_transcript: str | None,
    persona: str,
) -> str:
    """The oracle student system prompt, as v1's build_student_system_prompt.

    An empty reference (the real conversation ends at the cut) is replaced by
    the version's fallback text.
    """
    out = prompts.texts["oracle"]
    out = out.replace("[[NEXT_CONVERSATION_INFORMATION_HERE]]", student_context or "")
    out = out.replace("[[PERSONA_DESCRIPTION_HERE]]", persona or "")
    ref = reference_transcript or prompts.texts["empty_reference"]
    return out.replace("[[REFERENCE_TRANSCRIPT_HERE]]", ref)


def role_prompt(
    prompts: RolePrompts, system_prompt: str, transcript_prefix: str, extra: str
) -> tuple[str, str]:
    """``(cacheable head, tail)`` for one call, as v1's _build_role_prompt.

    The head (system prompt + the pre-cut transcript) is byte-stable across a
    replay's calls, so it is passed as the cacheable prefix; generated turns
    flow through ``extra`` in the tail.
    """
    head = f"{system_prompt}\n\n{prompts.texts['joiner']}\n\n{transcript_prefix}"
    tail = f"{extra}\n\n{prompts.texts['respond']}"
    return head, tail
