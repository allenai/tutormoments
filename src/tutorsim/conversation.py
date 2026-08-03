"""Multi-turn conversation orchestration: tutor and student alternate.

Provides a synchronous per-scenario loop and a round-based batch loop.
"""

import logging
import math
from dataclasses import asdict, dataclass, field, fields
from types import SimpleNamespace

from tutorsim.moments import Moment, _build_reference_transcript
from tutorsim.student import build_student_system_prompt, resolve_student
from tutorsim.tutor import build_tutor_system_prompt, resolve_tutor

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
    tutor_latencies: list[float] = field(default_factory=list)
    student_latencies: list[float] = field(default_factory=list)
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

    Prompts are built through the tutorsim adapter signatures
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
            "`tutorsim-build dataset build-from-run` / `dataset build`."
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
    tutor_max_tokens: int = 1500,
    student_max_tokens: int = 1000,
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
        tutor_max_tokens: Max tokens for tutor responses.
        student_max_tokens: Max tokens for student responses.
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
                **{**kwargs, **(tutor_kwargs or {})},
            )
        else:
            raw_text = tutor_res["fn"](transcript.generated_turns)
            response = SimpleNamespace(text=raw_text, usage={}, latency_seconds=None)

        _add_usage(transcript.tutor_usage, response.usage)
        if response.latency_seconds is not None:
            transcript.tutor_latencies.append(response.latency_seconds)
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
                **{**kwargs, **(student_kwargs or {})},
            )
        else:
            raw_text = student_res["fn"](transcript.generated_turns)
            response = SimpleNamespace(text=raw_text, usage={}, latency_seconds=None)

        _add_usage(transcript.student_usage, response.usage)
        if response.latency_seconds is not None:
            transcript.student_latencies.append(response.latency_seconds)
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


def _batch_gen_kwargs(roster_kwargs: dict) -> dict:
    """Map roster generation kwargs to the subset run_batch understands.

    run_conversation forwards the full roster kwargs to client.generate();
    run_batch exposes the same generation knobs as explicit parameters, so
    pick them out here. Keeping the mapping explicit (rather than **splat)
    means an unknown roster key fails loudly in sync mode first.
    """
    return {
        "thinking": roster_kwargs.get("thinking", False),
        "thinking_budget": roster_kwargs.get("thinking_budget", 0),
        "reasoning_effort": roster_kwargs.get("reasoning_effort", ""),
        "effort": roster_kwargs.get("effort", ""),
    }


def run_conversations_batch(
    scenarios: list[Moment],
    *,
    tutor_id: str,
    tutor_mode: str | None = None,
    student_id: str | None = None,
    max_turns: int,
    tutor_max_tokens: int = 1500,
    student_max_tokens: int = 1000,
    poll_interval: int = 60,
    save_callback: callable = None,
    images_by_scenario: dict[str, list[str]] | None = None,
    transcripts: dict[str, dict] | None = None,
) -> list[Transcript]:
    """Batch mode multi-turn conversations across all scenarios.

    Per-scenario state tracks `extra` (growing suffix) separate from the
    static scenario.transcript_prefix; the head is passed as cacheable_prefix
    on every per-scenario batch entry.

    Round-based batching: all active scenarios' tutor calls go in one batch
    via tutorsim.client.run_batch, then all active student calls go in the
    next batch. A scenario that emits [END] or [PROBLEM_CHANGE] is pruned
    from later rounds. Latencies are omitted in batch mode (not available
    from the batch API).

    Args:
        scenarios: List of fully-hydrated Moment objects.
        tutor_id: Registered tutor name or model roster id.
        tutor_mode: Prompt mode for the tutor (None/"plain"/oracle/etc.).
        student_id: Registered student name, or None for default.
        max_turns: Maximum speaking turns (each LLM call = 1 speaking turn).
        tutor_max_tokens: Max tokens for tutor responses.
        student_max_tokens: Max tokens for student responses.
        poll_interval: Seconds between batch status polls (pass 0 in tests).
        save_callback: Optional callable(scenario_id, transcript) called after
            each round for each scenario.
        images_by_scenario: Optional {scenario_id: [image_paths]} forwarded to
            batch entries.
        transcripts: Full conversations keyed by conv_id, required when
            scenario.student["reference"] is absent and needs to be derived.

    Returns:
        List of Transcripts in the same order as `scenarios`.
    """
    # Lazy import to avoid module-level SDK import.
    from tutorsim.client import build_batch_entry, run_batch

    tutor_res = resolve_tutor(tutor_id)
    student_res = resolve_student(student_id)

    if tutor_res["kind"] == "hosted":
        tutor_client = tutor_res["client"]
        tutor_model = tutor_client.model
    else:
        raise ValueError(
            "run_conversations_batch: registered (callable) tutors are not "
            "supported in batch mode. Use run_conversation for sync execution."
        )

    if student_res["kind"] != "hosted":
        raise ValueError(
            "run_conversations_batch: registered (callable) students are not "
            "supported in batch mode. Use run_conversation for sync execution."
        )
    student_client = student_res["client"]

    # Generation kwargs from the model roster (thinking/effort/etc.) must be
    # forwarded to run_batch, mirroring run_conversation's **kwargs merge --
    # otherwise batch runs silently diverge from sync benchmark runs (e.g.
    # tutors losing thinking/effort).
    tutor_gen_kwargs = _batch_gen_kwargs(tutor_res["kwargs"])
    student_gen_kwargs = _batch_gen_kwargs(student_res["kwargs"])

    # Per-scenario state
    transcript_map: dict[str, Transcript] = {}
    transcript_prefix_map: dict[str, str] = {}
    extras: dict[str, str] = {}
    next_turns: dict[str, int] = {}
    ended_via: dict[str, str] = {}
    refs: dict[str, str] = {}
    personas: dict[str, str] = {}

    for scenario in scenarios:
        sid = scenario.id
        transcript_map[sid] = Transcript(scenario_id=sid, tutor_model=tutor_model)
        prefix = _format_transcript_prefix(scenario.context)
        transcript_prefix_map[sid] = prefix
        extras[sid] = ""
        next_turns[sid] = scenario.provenance["cut_turn"] + 1

        # Reference transcript: use frozen scenario.student["reference"] first
        # (same as sync path); fall back to deriving from transcripts dict.
        ref = scenario.student.get("reference", "")
        if not ref and transcripts:
            conv = transcripts.get(scenario.provenance["conv_id"])
            if conv:
                ref = _build_reference_transcript(conv, scenario.provenance["cut_turn"])
        refs[sid] = ref

        # The oracle student's persona is the moment's frozen trait (schema v2).
        personas[sid] = _frozen_persona(scenario)

    scenario_map = {s.id: s for s in scenarios}
    active_ids = list(scenario_map.keys())

    # max_turns counts SPEAKING TURNS (LLM calls), alternating T-S-T-S-...
    # Each round = 1 tutor LLM call + 1 student LLM call = 2 speaking turns.
    # We may skip the student batch in the last round if max_turns is odd.
    for round_num in range(math.ceil(max_turns / 2)):
        if not active_ids:
            break

        # --- Tutor batch ---
        logger.info(
            "Round %d - tutor batch (%d scenarios)", round_num + 1, len(active_ids)
        )
        tutor_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            head, tail = _build_role_prompt(
                "TUTOR",
                transcript_prefix_map[sid],
                extras[sid],
                scenario.student.get("context", ""),
                tutor_mode=tutor_mode,
                reference_transcript=refs.get(sid),
            )
            scenario_images = (images_by_scenario or {}).get(sid)
            tutor_entries.append(
                build_batch_entry(
                    sid,
                    tail,
                    json_mode=False,
                    max_tokens=tutor_max_tokens,
                    images=scenario_images,
                    cacheable_prefix=head,
                )
            )

        tutor_raw = run_batch(
            tutor_client,
            tutor_entries,
            json_mode=False,
            display_name=f"tutor_round_{round_num + 1}",
            poll_interval=poll_interval,
            **tutor_gen_kwargs,
        )

        failed = []
        ended_this_round = []
        for sid in active_ids:
            result = tutor_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                logger.warning("tutor failed for %s", sid[:50])
                failed.append(sid)
                continue

            transcript = transcript_map[sid]
            if result.get("usage"):
                _add_usage(transcript.tutor_usage, result["usage"])
            # Latencies omitted in batch mode (not available from batch API).

            text, ended, problem_change = _parse_tutor_tokens(result["text"])
            messages = _split_messages(text)
            if not messages and not (ended or problem_change):
                messages = ["..."]
            if messages:
                extras[sid], next_turns[sid] = _append_turns_to_extra(
                    transcript,
                    messages,
                    "TUTOR",
                    extras[sid],
                    next_turns[sid],
                )

            if ended:
                ended_via[sid] = "END"
                ended_this_round.append(sid)
                continue
            if problem_change:
                ended_via[sid] = "PROBLEM_CHANGE"
                ended_this_round.append(sid)
                continue
            # Speaking-turns budget check: after the tutor batch in round_num,
            # each active scenario has done (2*round_num + 1) speaking turns.
            if (2 * round_num + 1) >= max_turns:
                ended_via[sid] = "MAX_TURNS"
                ended_this_round.append(sid)

        for sid in failed:
            if sid in active_ids:
                active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        # --- Student batch ---
        if not active_ids:
            if save_callback:
                for sid in scenario_map:
                    save_callback(sid, transcript_map[sid])
            continue

        logger.info(
            "Round %d - student batch (%d scenarios)", round_num + 1, len(active_ids)
        )
        student_entries = []
        for sid in active_ids:
            scenario = scenario_map[sid]
            head, tail = _build_role_prompt(
                "STUDENT",
                transcript_prefix_map[sid],
                extras[sid],
                scenario.student.get("context", ""),
                reference_transcript=refs.get(sid),
                persona=personas[sid],
            )
            scenario_images = (images_by_scenario or {}).get(sid)
            student_entries.append(
                build_batch_entry(
                    sid,
                    tail,
                    json_mode=False,
                    max_tokens=student_max_tokens,
                    images=scenario_images,
                    cacheable_prefix=head,
                )
            )

        student_raw = run_batch(
            student_client,
            student_entries,
            json_mode=False,
            display_name=f"student_round_{round_num + 1}",
            poll_interval=poll_interval,
            **student_gen_kwargs,
        )

        failed = []
        ended_this_round = []
        for sid in active_ids:
            result = student_raw.get(sid, {})
            if "error" in result or not result.get("text"):
                logger.warning("student failed for %s", sid[:50])
                failed.append(sid)
                continue

            transcript = transcript_map[sid]
            if result.get("usage"):
                _add_usage(transcript.student_usage, result["usage"])
            # Latencies omitted in batch mode (not available from batch API).

            messages = _split_messages(result["text"]) or ["..."]
            extras[sid], next_turns[sid] = _append_turns_to_extra(
                transcript,
                messages,
                "STUDENT",
                extras[sid],
                next_turns[sid],
            )

            # After the student batch in round_num, each active scenario has
            # done (2*round_num + 2) speaking turns.
            if (2 * round_num + 2) >= max_turns:
                ended_via[sid] = "MAX_TURNS"
                ended_this_round.append(sid)

        for sid in failed:
            if sid in active_ids:
                active_ids.remove(sid)
        for sid in ended_this_round:
            if sid in active_ids:
                active_ids.remove(sid)

        if save_callback:
            for sid in scenario_map:
                save_callback(sid, transcript_map[sid])

    for sid in scenario_map:
        transcript_map[sid].completed = True
        transcript_map[sid].ended_via = ended_via.get(sid, "MAX_TURNS")

    logger.info("Conversations complete: %d scenarios", len(scenario_map))
    return [transcript_map[s.id] for s in scenarios]
