"""Multi-turn conversation orchestration: tutor and student alternate.

Provides a synchronous per-scenario loop.
"""

import logging
from dataclasses import asdict, dataclass, field, fields
from types import SimpleNamespace

from tutormoments.moments import Moment, _build_reference_transcript
from tutormoments.student import build_student_system_prompt, resolve_student
from tutormoments.tutor import build_tutor_system_prompt, resolve_tutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Delimiters / control tokens
# ---------------------------------------------------------------------------

NEXT_DELIMITER = "[NEXT]"
NEW_MESSAGE_DELIMITER = "[NEW_MESSAGE]"
END_TOKEN = "[END]"
PROBLEM_CHANGE_TOKEN = "[PROBLEM_CHANGE]"
NEXT_PROBLEM_TOKEN = "[NEXT_PROBLEM]"  # legacy alias (v5 and earlier prompts)


# ---------------------------------------------------------------------------
# Transcript dataclass (renamed from Exchange)
# ---------------------------------------------------------------------------


@dataclass
class Transcript:
    scenario_id: str
    tutor_model: str
    generated_turns: list[dict] = field(default_factory=list)
    tutor_usage: dict = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    student_usage: dict = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    # Per-call end-to-end wall-clock seconds. Meaning is unchanged by the
    # streaming work, so the paper's Figure 7 pipeline and the website
    # refresh script -- which read these directly -- stay comparable.
    tutor_latencies: list[float] = field(default_factory=list)
    student_latencies: list[float] = field(default_factory=list)
    # Per-call streaming timing, one entry per LLM call in order. Shape:
    #   {ttfc_seconds, ttft_seconds, ttlt_seconds, output_tokens,
    #    cache_read_input_tokens, output_tps, turn_index, cache_state}
    tutor_timings: list[dict] = field(default_factory=list)
    student_timings: list[dict] = field(default_factory=list)
    completed: bool = False
    # "END" | "PROBLEM_CHANGE" | "MAX_TURNS" | "" (in-progress)
    ended_via: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        """Reconstruct a Transcript from a written transcript dict.

        Unknown keys are ignored so older/newer on-disk transcripts load.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_tutor_tokens(text: str) -> tuple[str, bool, bool]:
    """Strip tutor control tokens and report which were present.

    Returns (cleaned_text, ended, problem_change).
    END takes precedence: if both tokens appear, ended=True, problem_change=False.
    [NEXT_PROBLEM] is the legacy alias for [PROBLEM_CHANGE].
    """
    has_end = END_TOKEN in text
    has_change = (PROBLEM_CHANGE_TOKEN in text) or (NEXT_PROBLEM_TOKEN in text)
    cleaned = (
        text.replace(END_TOKEN, "")
        .replace(PROBLEM_CHANGE_TOKEN, "")
        .replace(NEXT_PROBLEM_TOKEN, "")
        .rstrip()
    )
    if has_end:
        return cleaned, True, False
    return cleaned, False, has_change


def _split_messages(text: str) -> list[str]:
    """Split LLM output into multiple messages on either delimiter.

    Recognizes [NEXT] (v1-v4) and [NEW_MESSAGE] (v5+). Either token splits
    the text into separate chat messages.
    """
    normalized = text.replace(NEW_MESSAGE_DELIMITER, NEXT_DELIMITER)
    parts = normalized.split(NEXT_DELIMITER)
    messages = [p.strip() for p in parts]
    return [m for m in messages if m]


def _add_usage(total: dict, new: dict) -> None:
    """Accumulate token usage."""
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + new.get(key, 0)


def _record_timing(timings: list[dict], response, turn_index: int) -> None:
    """Append this call's streaming timing, tagged with its cache state.

    `cache_state` is read off `cache_read_input_tokens` rather than inferred
    from `turn_index`. Turn position is a bad proxy for two reasons: the
    minimum cacheable prefix is model-dependent and not monotonic across
    generations (a short pre-cut transcript can cache on one roster model and
    silently fail to on another), and only the Anthropic path sends a real
    cache breakpoint at all -- Gemini, Together and OpenAI concatenate the
    prefix into the prompt and depend on the provider's automatic caching, so
    a hit there need not mean this conversation was served from cache.
    Providers that report nothing get "unknown", never a guess.
    """
    timing = getattr(response, "timing", None)
    if not timing:
        return
    cache_read = timing.get("cache_read_input_tokens")
    if cache_read is None:
        cache_state = "unknown"
    else:
        cache_state = "hit" if cache_read > 0 else "miss"
    timings.append({**timing, "turn_index": turn_index, "cache_state": cache_state})


def _append_turns_to_extra(
    transcript: Transcript,
    messages: list[str],
    role: str,
    extra: str,
    next_turn_num: int,
) -> tuple[str, int]:
    """Append messages as turns and grow the `extra` suffix.

    transcript_prefix stays fixed; this only mutates the per-round growing portion.
    """
    for msg in messages:
        turn = {"turn_number": next_turn_num, "role": role, "text": msg}
        transcript.generated_turns.append(turn)
        extra += f"\nTurn {next_turn_num}. {role}: {msg}"
        next_turn_num += 1
    return extra, next_turn_num


# ---------------------------------------------------------------------------
# Transcript prefix formatter
# ---------------------------------------------------------------------------


def _format_transcript_prefix(context: list[dict]) -> str:
    """Format scenario.context turns into 'Turn N. ROLE: text' string.

    Roles are uppercased (tutor -> TUTOR, student -> STUDENT).
    Turn numbers come from the real turn_number stored in each context entry,
    preserving the original (non-sequential) numbering from the source transcript.
    """
    lines = []
    for turn in context:
        n = turn["turn_number"]
        role = turn["role"].upper()
        text = turn["text"]
        lines.append(f"Turn {n}. {role}: {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Role prompt builder
# ---------------------------------------------------------------------------


def _build_role_prompt(
    role: str,
    transcript_prefix: str,
    extra: str,
    student_context: str,
    *,
    tutor_mode: str | None = None,
    reference_transcript: str | None = None,
    persona: str = "",
) -> tuple[str, str]:
    """Build (cacheable_head, tail) for either tutor or student.

    head = system_prompt + "Here is the conversation so far:\\n\\n" + transcript_prefix
    tail = extra + "\\n\\n" + role_instruction

    The head is byte-stable across rounds (system + static cut prefix), so the
    prompt cache hits on round 2+. Generated turns flow through `extra` (tail).

    Prompts are built through the tutormoments adapter signatures
    (build_tutor_system_prompt / build_student_system_prompt).
    """
    if role == "TUTOR":
        system_prompt = build_tutor_system_prompt(
            tutor_mode,
            student_context=student_context,
            reference_transcript=reference_transcript or "",
        )
        role_instruction = (
            "Respond as the TUTOR. Give only your response, no labels or prefixes."
        )
    else:
        system_prompt = build_student_system_prompt(
            student_context=student_context,
            reference_transcript=reference_transcript,
            persona=persona,
        )
        role_instruction = (
            "Respond as the STUDENT. Give only your response, no labels or prefixes."
        )

    head = f"{system_prompt}\n\nHere is the conversation so far:\n\n{transcript_prefix}"
    tail = f"{extra}\n\n{role_instruction}"
    return head, tail


# ---------------------------------------------------------------------------
# Frozen student trait (schema v2: the release embeds the paper's personas;
# the runtime consumes them and never generates its own)
# ---------------------------------------------------------------------------


def _frozen_persona(scenario: Moment) -> str:
    """Return the moment's frozen student persona, or raise.

    Regenerating personas locally would evaluate tutors against a different
    student population, silently breaking comparability with the paper — so
    a hosted student without a frozen trait is a hard error.
    """
    persona = (scenario.student.get("trait") or {}).get("persona", "")
    if not persona:
        raise RuntimeError(
            f"Moment {scenario.id} carries no frozen student trait "
            "(student.trait.persona). This dataset predates schema v2 — "
            "re-download the released dataset, or rebuild it with "
            "`tutormoments-build dataset build-from-run` / `dataset build`."
        )
    return persona


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_conversation(
    scenario: Moment,
    tutor_id: str,
    student_id: str | None = None,
    *,
    max_turns: int,
    tutor_mode: str | None = None,
    student_mode: str = "oracle",
    tutor_max_tokens: int = 0,
    student_max_tokens: int = 0,
    images: list[str] | None = None,
    tutor_kwargs: dict | None = None,
    student_kwargs: dict | None = None,
) -> Transcript:
    """Sync multi-turn conversation: tutor and student alternate.

    Both [END] and [PROBLEM_CHANGE] (or legacy [NEXT_PROBLEM]) terminate the
    loop; the termination reason is recorded on Transcript.ended_via.

    Each tutor/student call passes transcript_prefix's head as cacheable_prefix
    so the static head hits the prompt cache on round 2+.

    The student persona comes from the moment's frozen student.trait
    (schema v2) — the runtime never generates traits.

    Args:
        scenario: Fully-hydrated Moment object.
        tutor_id: Registered tutor name or model roster id.
        student_id: Registered student name, or None for default hosted student.
        max_turns: Maximum speaking turns (each LLM call = 1 speaking turn).
        tutor_mode: Prompt mode for the tutor (None/"plain"/oracle/etc.).
        student_mode: Student simulator mode label (recorded; oracle only).
        tutor_max_tokens: Max tokens for tutor responses; 0 (the default)
            means the model's maximum. The benchmark deliberately imposes no
            output cap -- a cap a thinking model can exhaust before emitting
            any visible text yields an empty response, which is recorded as
            "..." below and scored as if the tutor said that. Measured on
            claude-opus-4-8 at effort=xhigh, a 1500-token cap did this to
            17.5% of tutor turns.
        student_max_tokens: Max tokens for student responses; 0 = model max.
        images: Optional list of image paths/URLs forwarded to both clients.
        tutor_kwargs: Extra kwargs merged into tutor client.generate() calls.
        student_kwargs: Extra kwargs merged into student client.generate() calls.

    Returns:
        Transcript with all generated turns, usage, latencies, and termination info.
    """
    tutor_res = resolve_tutor(tutor_id)
    student_res = resolve_student(student_id)

    # Compute static transcript prefix from scenario.context (does not change).
    transcript_prefix = _format_transcript_prefix(scenario.context)

    # Determine tutor_model name for Transcript.
    if tutor_res["kind"] == "hosted":
        tutor_model = tutor_res["client"].model
    else:
        tutor_model = tutor_id

    transcript = Transcript(scenario_id=scenario.id, tutor_model=tutor_model)

    # Loop state
    extra = ""
    next_turn_num = scenario.provenance["cut_turn"] + 1
    ended_via = ""
    speaking_turns = 0

    # Reference transcript and student context are pre-baked in scenario.student.
    reference_transcript = scenario.student.get("reference", "")
    student_context = scenario.student.get("context", "")

    # The oracle student's persona is the moment's frozen trait (schema v2).
    persona = ""
    if student_res["kind"] == "hosted":
        persona = _frozen_persona(scenario)

    while speaking_turns < max_turns:
        # ----------------------------------------------------------------
        # Tutor turn
        # ----------------------------------------------------------------
        head, tail = _build_role_prompt(
            "TUTOR",
            transcript_prefix,
            extra,
            student_context,
            tutor_mode=tutor_mode,
            reference_transcript=reference_transcript,
        )

        if tutor_res["kind"] == "hosted":
            client = tutor_res["client"]
            kwargs = tutor_res["kwargs"]
            response = client.generate(
                tail,
                json_mode=False,
                max_tokens=tutor_max_tokens,
                images=images,
                cacheable_prefix=head,
                stream=True,
                **{**kwargs, **(tutor_kwargs or {})},
            )
        else:
            raw_text = tutor_res["fn"](transcript.generated_turns)
            response = SimpleNamespace(
                text=raw_text, usage={}, latency_seconds=None, timing=None
            )

        _add_usage(transcript.tutor_usage, response.usage)
        if response.latency_seconds is not None:
            transcript.tutor_latencies.append(response.latency_seconds)
        _record_timing(
            transcript.tutor_timings, response, len(transcript.tutor_timings)
        )
        speaking_turns += 1

        text, ended, problem_change = _parse_tutor_tokens(response.text)
        messages = _split_messages(text)
        if not messages and not (ended or problem_change):
            messages = ["..."]
        if messages:
            extra, next_turn_num = _append_turns_to_extra(
                transcript, messages, "TUTOR", extra, next_turn_num
            )

        if ended:
            ended_via = "END"
            break
        if problem_change:
            ended_via = "PROBLEM_CHANGE"
            break
        if speaking_turns >= max_turns:
            ended_via = "MAX_TURNS"
            break

        # ----------------------------------------------------------------
        # Student turn
        # ----------------------------------------------------------------
        head, tail = _build_role_prompt(
            "STUDENT",
            transcript_prefix,
            extra,
            student_context,
            reference_transcript=reference_transcript,
            persona=persona,
        )

        if student_res["kind"] == "hosted":
            client = student_res["client"]
            kwargs = student_res["kwargs"]
            response = client.generate(
                tail,
                json_mode=False,
                max_tokens=student_max_tokens,
                images=images,
                cacheable_prefix=head,
                stream=True,
                **{**kwargs, **(student_kwargs or {})},
            )
        else:
            raw_text = student_res["fn"](transcript.generated_turns)
            response = SimpleNamespace(
                text=raw_text, usage={}, latency_seconds=None, timing=None
            )

        _add_usage(transcript.student_usage, response.usage)
        if response.latency_seconds is not None:
            transcript.student_latencies.append(response.latency_seconds)
        _record_timing(
            transcript.student_timings, response, len(transcript.student_timings)
        )
        speaking_turns += 1

        messages = _split_messages(response.text) or ["..."]
        extra, next_turn_num = _append_turns_to_extra(
            transcript, messages, "STUDENT", extra, next_turn_num
        )

    if not ended_via:
        ended_via = "MAX_TURNS"

    transcript.completed = True
    transcript.ended_via = ended_via
    return transcript
