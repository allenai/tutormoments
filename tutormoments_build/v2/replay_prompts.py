"""Loads and fills the tutor and student prompts used in replays.

Prompts live in `prompts/v2/tutor/<version>/` and `prompts/v2/student/<version>/`.
The newest version is the default. All of a version's files are read when it
loads, so a missing file fails before any API call. See README.md, "Replays".
"""

import re
from dataclasses import dataclass

from tutormoments_build.resources import resource_text
from tutormoments_build.v2.classify_excerpts import (
    available_prompt_versions,
    latest_prompt_version,
)

TUTOR_PROMPT_ROOT = "prompts/v2/tutor"
STUDENT_PROMPT_ROOT = "prompts/v2/student"

# v1 also had an oracle tutor; it isn't offered here.
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
    """One role's prompt files at one version."""

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
        # Trimmed as v1 did: tutor files fully, the student prompt only of
        # trailing newlines.
        texts[name] = raw.rstrip("\n") if name == "oracle" else raw.strip()
    return RolePrompts(role=role, version=version, root=root, texts=texts)


def load_tutor_prompts(version: str | None = None) -> RolePrompts:
    """The tutor prompt files at `version` (default: the newest)."""
    return _load("TUTOR", TUTOR_PROMPT_ROOT, TUTOR_FILES, version)


def load_student_prompts(version: str | None = None) -> RolePrompts:
    """The student prompt files at `version` (default: the newest)."""
    return _load("STUDENT", STUDENT_PROMPT_ROOT, STUDENT_FILES, version)


def fill_context(text: str, placeholder: str, context: str) -> str:
    """Put the student context where `placeholder` is.

    If the context is empty and the placeholder has a line to itself, that line and
    one blank line after it are removed, so no gap is left.
    """
    if context:
        return text.replace(placeholder, context)
    alone = re.compile(rf"^[ \t]*{re.escape(placeholder)}[ \t]*\n(\n)?", re.M)
    return alone.sub("", text).replace(placeholder, "")


def tutor_system_prompt(prompts: RolePrompts, mode: str, student_context: str) -> str:
    """The tutor's system prompt for `mode`, with the student context filled in."""
    if mode not in TUTOR_MODES:
        raise ValueError(f"unknown tutor mode {mode!r}; choose from {TUTOR_MODES}")
    return fill_context(prompts.texts[mode], "{student_context}", student_context)


def student_system_prompt(
    prompts: RolePrompts,
    *,
    student_context: str,
    reference_transcript: str | None,
    persona: str,
) -> str:
    """The student's system prompt, with context, persona and reference filled in.

    With no reference transcript (the real conversation ends at the cut), the
    version's `empty_reference.md` text is used.
    """
    out = prompts.texts["oracle"]
    out = fill_context(
        out, "[[NEXT_CONVERSATION_INFORMATION_HERE]]", student_context or ""
    )
    out = out.replace("[[PERSONA_DESCRIPTION_HERE]]", persona or "")
    ref = reference_transcript or prompts.texts["empty_reference"]
    return out.replace("[[REFERENCE_TRANSCRIPT_HERE]]", ref)


def role_prompt(
    prompts: RolePrompts, system_prompt: str, transcript_prefix: str, extra: str
) -> tuple[str, str]:
    """Split one call's prompt into (head, tail).

    The head is the system prompt plus the transcript up to the cut. It is the same
    on every call in a replay, so it is sent as the cacheable prefix. The tail is
    the turns generated so far plus the "respond as" instruction.
    """
    head = f"{system_prompt}\n\n{prompts.texts['joiner']}\n\n{transcript_prefix}"
    tail = f"{extra}\n\n{prompts.texts['respond']}"
    return head, tail
