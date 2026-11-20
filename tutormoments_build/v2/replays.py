"""Replays v2 moments: an LM tutor and a simulated student continue a real
transcript from its cut point.

Usage, what each side sees and the output paths are in README.md, "Replays".
The loop follows v1's `tutormoments.conversation.run_conversation`.
"""

import argparse
import datetime
import logging
import os
import random
import sys

from tutormoments.conversation import (
    END_TOKEN,
    NEW_MESSAGE_DELIMITER,
    NEXT_DELIMITER,
    NEXT_PROBLEM_TOKEN,
    PROBLEM_CHANGE_TOKEN,
)
from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments.usage import add_usage
from tutormoments_build.v2 import (
    excerpts,
    jsonl_runs,
    replay_config,
    replay_data,
    replay_prompts,
    traits,
)

logger = logging.getLogger("tutormoments_build.v2.replays")

DEFAULT_OUT_DIR = "outputs/v2_replays"
# Provider A has no session information, so its context is empty.
NO_CONTEXT = ""
EMPTY_TURN = "..."


# ===========================================================================
# What the tutor and student see
# ===========================================================================


def transcript_so_far(moment: replay_data.ReplayMoment) -> str:
    """The transcript through the cut row, screen-activity and pause rows included."""
    return "\n".join(
        excerpts.format_row(row) for row in moment.rows[: moment.cut_index + 1]
    )


def reference_transcript(moment: replay_data.ReplayMoment) -> str:
    """The real rows after the cut, through the moment's end ('' if none)."""
    return "\n".join(
        excerpts.format_row(row)
        for row in moment.rows[moment.cut_index + 1 : moment.end_index + 1]
    )


def _years(years: list[int]) -> str:
    if len(years) == 1:
        return f"Year {years[0]}."
    if years == list(range(years[0], years[-1] + 1)):
        return f"Years {years[0]}-{years[-1]}."
    # Not consecutive, so list them; a range would imply a missing year.
    return f"Years {', '.join(map(str, years))}."


def student_context(moment: replay_data.ReplayMoment) -> str:
    """The student context: provider C's year(s), then each objective with its slides.

    Provider A has none, so it gets an empty string. Objectives and slides keep the
    session's order. The transcripts don't say which slide was up when, so the
    whole session's slides are given.
    """
    if not moment.problems:
        return NO_CONTEXT
    years = sorted({p["year"] for p in moment.problems if p["year"] is not None})
    decks: dict[str, list[str]] = {}
    for problem in moment.problems:
        if problem["objective"]:
            decks.setdefault(problem["objective"], []).append(problem["slide_text"])
    if not decks:
        return NO_CONTEXT
    lines = [_years(years)] if years else []
    for title, slides in decks.items():
        lines += (
            ["", f"Learning objective: {title}"]
            if lines
            else [f"Learning objective: {title}"]
        )
        for number, text in enumerate(slides, 1):
            lines += ["", f"Session Content {number}:", text]
    return "\n".join(lines)


# ===========================================================================
# Turn handling (as in v1)
# ===========================================================================


def parse_tutor_tokens(text: str) -> tuple[str, bool, bool]:
    """Remove `[END]` / `[PROBLEM_CHANGE]` from a tutor reply, as v1 does.

    Returns (text, ended, problem_change).
    """
    has_end = END_TOKEN in text
    has_change = PROBLEM_CHANGE_TOKEN in text or NEXT_PROBLEM_TOKEN in text
    cleaned = (
        text.replace(END_TOKEN, "")
        .replace(PROBLEM_CHANGE_TOKEN, "")
        .replace(NEXT_PROBLEM_TOKEN, "")
        .rstrip()
    )
    if has_end:
        return cleaned, True, False
    return cleaned, False, has_change


def split_messages(text: str) -> list[str]:
    """Split a reply into messages on `[NEW_MESSAGE]` or `[NEXT]`, as v1 does."""
    parts = text.replace(NEW_MESSAGE_DELIMITER, NEXT_DELIMITER).split(NEXT_DELIMITER)
    return [p.strip() for p in parts if p.strip()]


def next_turn_number(moment: replay_data.ReplayMoment) -> int:
    """The first generated turn's number: one past the last real turn up to the cut."""
    numbers = [
        row["turn_number"]
        for row in moment.rows[: moment.cut_index + 1]
        if row["turn_number"] is not None
    ]
    return max(numbers) + 1


# ===========================================================================
# The loop
# ===========================================================================


def replay(
    moment: replay_data.ReplayMoment,
    *,
    turns: int,
    tutor_client,
    tutor_thinking: dict,
    tutor_mode: str,
    tutor_prompts: replay_prompts.RolePrompts,
    student_client,
    student_thinking: dict,
    student_prompts: replay_prompts.RolePrompts,
    persona: str | None,
) -> dict:
    """Run one replay and return its generated rows, calls, usage and how it ended.

    The tutor speaks first, then student and tutor alternate until there are
    `turns` tutor turns. It stops early if the tutor sends `[END]` or
    `[PROBLEM_CHANGE]`. An empty reply is recorded as "...", as in v1.
    """
    if turns < 1:
        raise ValueError("a replay needs at least one tutor turn")
    prefix = transcript_so_far(moment)
    context = student_context(moment)
    reference = reference_transcript(moment) or None

    tutor_system = replay_prompts.tutor_system_prompt(
        tutor_prompts, tutor_mode, context
    )
    student_system = None
    if turns > 1:
        if not persona:
            raise ValueError(f"moment {moment.moment_id} has no persona")
        student_system = replay_prompts.student_system_prompt(
            student_prompts,
            student_context=context,
            reference_transcript=reference,
            persona=persona,
        )

    extra = ""
    number = next_turn_number(moment)
    rows: list[dict] = []
    calls: list[dict] = []
    usage = {"tutor": {}, "student": {}}

    def call(role, client, thinking, prompts, system):
        head, tail = replay_prompts.role_prompt(prompts, system, prefix, extra)
        response = client.generate(
            tail,
            json_mode=False,
            max_tokens=0,
            cacheable_prefix=head,
            stream=True,
            thinking=thinking,
        )
        call_usage = dict(response.usage or {})
        add_usage(usage[role], call_usage)
        calls.append({"role": role, "text": response.text or "", "usage": call_usage})
        return response.text or ""

    def append(role, messages):
        nonlocal extra, number
        for message in messages:
            row = {"turn_number": number, "role": role.upper(), "text": message}
            rows.append(row)
            extra += f"\n{excerpts.format_row(row)}"
            number += 1

    ended_via = "TURNS"
    for tutor_turn in range(1, turns + 1):
        raw = call("tutor", tutor_client, tutor_thinking, tutor_prompts, tutor_system)
        text, ended, problem_change = parse_tutor_tokens(raw)
        messages = split_messages(text)
        if not messages and not (ended or problem_change):
            messages = [EMPTY_TURN]
        append("tutor", messages)
        if ended or problem_change:
            ended_via = "END" if ended else "PROBLEM_CHANGE"
            break
        if tutor_turn == turns:
            break
        raw = call(
            "student", student_client, student_thinking, student_prompts, student_system
        )
        append("student", split_messages(raw) or [EMPTY_TURN])

    tutor_turns = sum(c["role"] == "tutor" for c in calls)
    return {
        "generated_rows": rows,
        "tutor_turns": tutor_turns,
        # How the replay stopped, and whether it stopped before `turns`.
        "ended_via": ended_via,
        "ended_early": tutor_turns < turns,
        "calls": calls,
        "usage": usage,
        "student_context": context,
    }


def replay_excerpt(
    moment: replay_data.ReplayMoment, rows: list[dict], width=excerpts.FULL
) -> tuple[str, int]:
    """The replay as an excerpt: lead-up, the CUT POINT marker, then the generated rows.

    The lead-up at `width` starts where the human excerpt at that width starts, so
    both are scored on the same context. Returns (text, index of the first row).
    """
    first = excerpts.context_start(moment.rows, moment.cut_index, width)
    lines = [
        excerpts.format_row(row) for row in moment.rows[first : moment.cut_index + 1]
    ]
    lines.append(excerpts.CUT_POINT)
    lines += [excerpts.format_row(row) for row in rows]
    return "\n".join(lines), first


def replay_excerpts(moment: replay_data.ReplayMoment, rows: list[dict]) -> dict:
    """The replay excerpt at each width the human excerpts have."""
    out = {}
    for width in excerpts.DEFAULT_CONTEXT_WIDTHS:
        text, first = replay_excerpt(moment, rows, width)
        out[excerpts.width_key(width)] = {
            "excerpt": text,
            "context_turns": width,
            "context_start_index": first,
            "context_rows": moment.cut_index - first,
        }
    return out


def build_record(
    moment: replay_data.ReplayMoment, result: dict, settings: dict
) -> dict:
    m = moment.moment
    rows = result["generated_rows"]
    return {
        "moment_id": moment.moment_id,
        "source": m.get("source"),
        "transcript_id": m["transcript_id"],
        "conversation_id": moment.conversation_id,
        "start_turn": m["start_turn"],
        "cut_turn": m["cut_turn"],
        "end_turn": m["end_turn"],
        "cut_index": moment.cut_index,
        "end_index": moment.end_index,
        # Gold labels for the moment at the cut. Classification reads
        # situation_type for the over-scaffolding prompt.
        "labels": m["labels"],
        "situation_type": m.get("situation_type"),
        "settings": settings,
        # The student context the tutor and student were given.
        "student_context": result["student_context"],
        "generated_rows": rows,
        "post_cut_rows": len(rows),
        "post_cut_dialogue_rows": len(rows),
        "tutor_turns": result["tutor_turns"],
        "ended_via": result["ended_via"],
        "ended_early": result["ended_early"],
        "usage": result["usage"],
        "calls": result["calls"],
        "excerpts": replay_excerpts(moment, rows),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }


# ===========================================================================
# Settings and paths
# ===========================================================================


def settings_for(
    arm, tutor_mode, tutor_prompts, student, student_prompts, turns
) -> dict:
    """The settings that define a replay run, stored on every record."""
    return {
        "tutor": {
            "arm": arm.name,
            "model": arm.model,
            "thinking": arm.thinking,
            "prompt": tutor_mode,
            "prompt_version": tutor_prompts.version,
        },
        "student": {
            "model": student.model,
            "thinking": student.thinking,
            "prompt": "oracle",
            "prompt_version": student_prompts.version,
        },
        "turns": turns,
    }


def replay_path(out_dir: str, settings: dict) -> str:
    """Where a run's replays are written (see README.md, "Replays")."""
    tutor, student = settings["tutor"], settings["student"]
    return os.path.join(
        out_dir,
        tutor["arm"],
        f"{tutor['prompt']}_v{tutor['prompt_version']}",
        f"n{settings['turns']}",
        f"student-{student['model']}_sp{student['prompt_version']}.jsonl",
    )


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.replays",
        description="Replay v2 moments with an LM tutor and a simulated student",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument(
        "--tutor",
        nargs="+",
        required=True,
        metavar="ARM",
        help="benchmark_models arm(s) to replay as the tutor",
    )
    parser.add_argument(
        "--tutor-prompt", default="plain", choices=replay_prompts.TUTOR_MODES
    )
    parser.add_argument(
        "--tutor-prompt-version",
        default=None,
        metavar="N",
        help="Default: the newest under prompts/v2/tutor/",
    )
    parser.add_argument(
        "--student-prompt-version",
        default=None,
        metavar="N",
        help="Default: the newest under prompts/v2/student/",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        metavar="N",
        help="Generated tutor turns (a student reply between each)",
    )
    parser.add_argument(
        "--ground-truth", default=excerpts.DEFAULT_GROUND_TRUTH_DIR, metavar="DIR"
    )
    parser.add_argument(
        "--transcripts",
        nargs="+",
        default=list(excerpts.DEFAULT_TRANSCRIPTS),
        metavar="FILE",
    )
    parser.add_argument(
        "--slides",
        default=replay_data.DEFAULT_SLIDES,
        metavar="FILE",
        help="Provider C lesson-slides file (slide text for the student context)",
    )
    parser.add_argument("--traits", default=traits.DEFAULT_OUT, metavar="FILE")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="A seeded random N of the pending moments",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4, metavar="N")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace replays already written"
    )
    parser.add_argument(
        "--print",
        dest="print_moment",
        metavar="MOMENT_ID",
        help="Print a written replay's excerpt (id prefix accepted) and exit",
    )
    parser.add_argument(
        "--show-prompt",
        metavar="MOMENT_ID",
        help="Print the first tutor call's prompt for a moment and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without any API call"
    )
    return parser


def _find(moments, prefix):
    for moment in moments:
        if moment.moment_id.startswith(prefix):
            return moment
    return None


def main(argv: list[str] | None = None) -> int:
    from tutormoments.config import resolve_arm

    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)
    if args.turns < 1:
        print("--turns must be at least 1", file=sys.stderr)
        return 2

    # Load config and prompts first, so a mistake fails before any API call.
    arms = [resolve_arm(name) for name in args.tutor]
    student = replay_config.role_spec("student")
    tutor_prompts = replay_prompts.load_tutor_prompts(args.tutor_prompt_version)
    student_prompts = replay_prompts.load_student_prompts(args.student_prompt_version)
    moments, dropped = replay_data.load_moments(
        args.ground_truth, args.transcripts, args.slides
    )
    personas = traits.load_traits(args.traits) if args.turns > 1 else {}

    if args.show_prompt:
        moment = _find(moments, args.show_prompt)
        if moment is None:
            print(f"no moment matching {args.show_prompt!r}", file=sys.stderr)
            return 1
        system = replay_prompts.tutor_system_prompt(
            tutor_prompts, args.tutor_prompt, student_context(moment)
        )
        head, tail = replay_prompts.role_prompt(
            tutor_prompts, system, transcript_so_far(moment), ""
        )
        print(head + tail)
        return 0

    for arm in arms:
        settings = settings_for(
            arm, args.tutor_prompt, tutor_prompts, student, student_prompts, args.turns
        )
        path = replay_path(args.out_dir, settings)

        if args.print_moment:
            record = next(
                (
                    r
                    for k, r in jsonl_runs.load_by_id(path).items()
                    if k.startswith(args.print_moment)
                ),
                None,
            )
            if record is None:
                print(
                    f"no replay matching {args.print_moment!r} in {path}",
                    file=sys.stderr,
                )
                return 1
            print(record["excerpts"][excerpts.FULL_KEY]["excerpt"])
            continue

        todo = jsonl_runs.pending(moments, path, args.overwrite)
        if args.sample is not None:
            todo = random.Random(args.seed).sample(todo, min(args.sample, len(todo)))
        if args.limit is not None:
            todo = todo[: args.limit]
        if args.turns > 1:
            missing = [m.moment_id for m in todo if m.moment_id not in personas]
            if missing:
                print(
                    f"{len(missing)} moment(s) have no persona in {args.traits} "
                    f"(e.g. {missing[0]}); run `python -m tutormoments_build.v2.traits`",
                    file=sys.stderr,
                )
                return 1

        print(
            f"replays: tutor {arm.name} ({arm.model} {arm.thinking}), "
            f"prompt {args.tutor_prompt} v{tutor_prompts.version}, N={args.turns}\n"
            f"  student: {student.model} {student.thinking}, oracle "
            f"v{student_prompts.version}"
            + ("" if args.turns > 1 else " (not called at N=1)")
            + f"\n  moments: {len(moments)} located"
            + (f", dropped {dict(dropped)}" if dropped else "")
            + f"\n  output: {path}\n  to replay: {len(todo)} "
            f"({len(todo) * args.turns} tutor + {len(todo) * (args.turns - 1)} "
            "student call(s) at most)"
        )
        if args.dry_run or not todo:
            continue

        from tutormoments.client import get_client

        tutor_client = get_client(arm.model)
        student_client = get_client(student.model) if args.turns > 1 else None

        def work(
            moment,
            settings=settings,
            tutor_client=tutor_client,
            student_client=student_client,
        ):
            result = replay(
                moment,
                turns=args.turns,
                tutor_client=tutor_client,
                tutor_thinking=arm.thinking,
                tutor_mode=args.tutor_prompt,
                tutor_prompts=tutor_prompts,
                student_client=student_client,
                student_thinking=student.thinking,
                student_prompts=student_prompts,
                persona=(
                    traits.persona_for(personas, moment.moment_id)
                    if args.turns > 1
                    else None
                ),
            )
            return build_record(moment, result, settings)

        written = jsonl_runs.run(
            todo, path, work, overwrite=args.overwrite, concurrency=args.concurrency
        )
        totals = {"tutor": {}, "student": {}}
        for record in written:
            for role in totals:
                add_usage(totals[role], record["usage"][role])
        print(f"  wrote {len(written)} replay(s)")
        for role, usage in totals.items():
            if usage:
                print(
                    f"  {role} tokens: {usage.get('input_tokens', 0)} in / "
                    f"{usage.get('output_tokens', 0)} out / "
                    f"{usage.get('total_tokens', 0)} total"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
