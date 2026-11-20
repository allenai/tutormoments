"""Classifies replays with the same v2 scorer prompts used on human moments.

Two batches: `action_direction` on every replay, then `over-scaffolding` only
where the first one predicted scaffolding. Usage, routing and output paths are
in README.md, "Replays".
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments.models import infer_provider
from tutormoments.usage import add_usage
from tutormoments_build.v2 import classify_excerpts as C
from tutormoments_build.v2 import jsonl_runs, replay_config

logger = logging.getLogger("tutormoments_build.v2.classify_replays")

# Score on the same excerpt width as the human round by default.
WIDTH = C.EXCERPT_WIDTH

NOT_ROUTED_UNPARSEABLE = "not routed: action_direction unparseable"
NOT_ROUTED_NO_SCAFFOLDING = "not routed: predicted no scaffolding"
NOT_ROUTED_FAILED = "not routed: action_direction request failed"

# Failed requests are resent at most this many times. A reply that arrived
# but doesn't parse is kept; resending it would re-roll the answer.
MAX_RETRIES = 3


# ===========================================================================
# Config and paths
# ===========================================================================


def scorer_spec(model: str | None = None, config_path=None) -> C.V2Spec:
    """The scorer's settings: `v2_replay.scorer`, or an override `model`.

    An override is looked up as `classify_excerpts --model` looks it up, and uses
    only its own config entry; nothing is taken from the default scorer.
    """
    from tutormoments.config import load_config

    scorer = replay_config.role_spec("scorer", config_path)
    poll = replay_config.poll_interval(config_path)
    if not model or model == scorer.model:
        return C.V2Spec(scorer.model, scorer.provider, scorer.thinking, poll)
    scoped = (load_config(config_path).get("v2") or {}).get("models") or {}
    block, model_id, entry = C.model_knobs(model, scoped, config_path)
    return C.V2Spec(
        model=model_id,
        provider=infer_provider(model_id),
        thinking=C._native_thinking(block, model_id, entry),
        poll_interval=poll,
    )


def classification_path(
    replay_file: str, model: str, prompt_version: str, width: str = WIDTH
) -> str:
    """The replay file path plus `__scorer-<model>_v<version>_ctx<width>`."""
    stem, ext = os.path.splitext(replay_file)
    model = model.replace("/", "_")
    return f"{stem}__scorer-{model}_v{prompt_version}_ctx{width}{ext}"


# ===========================================================================
# Routing
# ===========================================================================


def action_result(record: dict, raw: dict) -> dict | None:
    """The parsed stage-1 result for a replay; None if nothing follows the cut."""
    if not C.has_post_cut_content(record):
        return None
    return C._pass_result(
        raw.get(f"{C.ACTION_PREFIX}__{record['moment_id']}"),
        C.parse_action_direction,
    )


def route(action: dict | None) -> tuple[bool, str | None]:
    """Whether a replay goes to stage 2, and if not, why. Returns (routed, reason)."""
    if action is None:
        return False, "not classified: no post-cut content"
    if action["scaffolding"] is None:
        failed_request = action["error"] and not action["raw"]
        return False, NOT_ROUTED_FAILED if failed_request else NOT_ROUTED_UNPARSEABLE
    if action["scaffolding"] is False:
        return False, NOT_ROUTED_NO_SCAFFOLDING
    return True, None


def predicted_presence(actions: dict[str, dict | None]):
    """Fills `{action}` from the stage-1 prediction instead of the gold labels."""

    def presence(record: dict) -> tuple[bool, bool]:
        action = actions[record["moment_id"]]
        return bool(action["scaffolding"]), bool(action["rigor"])

    return presence


def check_fillable(records: list[dict], prompts: C.PromptSet) -> None:
    """Fail before stage 1 if any replay couldn't be given the stage-2 prompt.

    We only know which replays go to stage 2 after stage 1 is paid for, so every
    replay is checked as if it would.
    """
    template = prompts.templates["over_scaffolding"]
    for record in records:
        if C.has_post_cut_content(record):
            C.over_scaffolding_slots(template, record, lambda r: (True, False))


# ===========================================================================
# Records
# ===========================================================================


def build_record(
    replay: dict,
    action: dict | None,
    over_raw: dict,
    actions: dict,
    cfg: C.V2Spec,
    prompts: C.PromptSet,
    replay_file: str,
    width: str = WIDTH,
) -> dict:
    moment_id = replay["moment_id"]
    routed, reason = route(action)
    over = over_slots = description_from = None
    if routed:
        over = C._pass_result(
            over_raw.get(f"{C.OVER_SCAFFOLDING_PREFIX}__{moment_id}"),
            C.parse_over_scaffolding,
        )
        over_slots = C.over_scaffolding_slots(
            prompts.templates["over_scaffolding"], replay, predicted_presence(actions)
        )
        if over_slots and C.DESCRIPTIONS_SLOT in over_slots:
            state = replay[C.DESCRIPTION_KEY]
            description_from = {
                k: state[k]
                for k in ("adjudicators", "situation_method", "model", "prompt")
            }
    classified = action is not None
    return {
        "moment_id": moment_id,
        "source": replay.get("source"),
        "transcript_id": replay["transcript_id"],
        "conversation_id": replay.get("conversation_id"),
        "replay": {
            "file": replay_file,
            "settings": replay["settings"],
            "generated_at": replay.get("generated_at"),
            "tutor_turns": replay.get("tutor_turns"),
            "ended_via": replay.get("ended_via"),
        },
        "model": cfg.model,
        "provider": cfg.provider,
        "thinking": dict(cfg.thinking),
        "prompt_version": prompts.version,
        "prompts": dict(prompts.paths),
        "context_turns": width,
        "classified": classified,
        "skipped": None if classified else "no_post_cut_content",
        "action_direction": action,
        # Stage 2 is decided by this replay's own stage-1 prediction, not gold.
        "over_scaffolding_routing": {
            "gated_on": "predicted",
            "routed": routed,
            "reason": reason,
            "from": None
            if action is None
            else {"scaffolding": action["scaffolding"], "rigor": action["rigor"]},
        },
        "over_scaffolding": over,
        "over_scaffolding_asked": over is not None,
        # {action} comes from the prediction; the other slots are gold.
        "prompt_slots": {"over_scaffolding": over_slots},
        "description_from": description_from,
        "situation_type": replay.get("situation_type"),
        "labels": replay.get("labels"),
        "usage": C._sum_usage((action or {}).get("usage"), (over or {}).get("usage")),
    }


def write_records(path: str, records: list[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


# ===========================================================================
# Running
# ===========================================================================


def failed(result: dict | None) -> bool:
    """Whether a batch result is a failed request (an error, or no result)."""
    return result is None or bool(result.get("error"))


def run_with_retries(
    entries: list[dict],
    cfg: C.V2Spec,
    *,
    batch_id: str | None,
    display_name: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[dict, dict[str, int]]:
    """Run one batch, resending failed requests up to `max_retries` times.

    Each retry is a new batch of just the failures. Token usage is added up over
    every attempt. Returns (results by key, attempts by key).
    """
    results = dict(
        C.run_entries(entries, cfg, batch_id=batch_id, display_name=display_name)
    )
    attempts = {entry["key"]: 1 for entry in entries}
    for retry in range(1, max_retries + 1):
        todo = [e for e in entries if failed(results.get(e["key"]))]
        if not todo:
            break
        logger.warning(
            "%s: resubmitting %d failed request(s) (retry %d of %d)",
            display_name,
            len(todo),
            retry,
            max_retries,
        )
        again = C.run_entries(todo, cfg, display_name=f"{display_name}_retry{retry}")
        for entry in todo:
            key = entry["key"]
            attempts[key] += 1
            new = again.get(key)
            if new is None:
                continue
            usage: dict = {}
            for result in (results.get(key), new):
                add_usage(usage, (result or {}).get("usage") or {})
            results[key] = {**new, "usage": usage}
    still = sum(failed(results.get(e["key"])) for e in entries)
    if still:
        logger.warning(
            "%s: %d request(s) still failed after %d retries",
            display_name,
            still,
            max_retries,
        )
    return results, attempts


def classify(
    replays: list[dict],
    cfg: C.V2Spec,
    prompts: C.PromptSet,
    replay_file: str,
    *,
    action_batch_id: str | None = None,
    over_batch_id: str | None = None,
    width: str = WIDTH,
) -> tuple[list[dict], Counter]:
    """Run both stages on one replay file. Returns (records, counts)."""
    entries, counts = C.build_entries(replays, prompts, ("action_direction",), width)
    raw1, attempts1 = (
        run_with_retries(
            entries,
            cfg,
            batch_id=action_batch_id,
            display_name="v2_classify_replays_action",
        )
        if entries
        else ({}, {})
    )
    actions = {r["moment_id"]: action_result(r, raw1) for r in replays}

    routed = {mid for mid, action in actions.items() if route(action)[0]}
    over_entries, over_counts = C.build_entries(
        replays,
        prompts,
        ("over_scaffolding",),
        width,
        gate=lambda r: r["moment_id"] in routed,
        presence=predicted_presence(actions),
    )
    counts["over_scaffolding"] = over_counts["over_scaffolding"]
    raw2, attempts2 = (
        run_with_retries(
            over_entries,
            cfg,
            batch_id=over_batch_id,
            display_name="v2_classify_replays_over_scaffolding",
        )
        if over_entries
        else ({}, {})
    )

    records = [
        build_record(
            r, actions[r["moment_id"]], raw2, actions, cfg, prompts, replay_file, width
        )
        for r in replays
    ]
    for record in records:
        moment_id = record["moment_id"]
        record["attempts"] = {
            "action_direction": attempts1.get(f"{C.ACTION_PREFIX}__{moment_id}", 0),
            "over_scaffolding": attempts2.get(
                f"{C.OVER_SCAFFOLDING_PREFIX}__{moment_id}", 0
            ),
        }
        for pass_name in ("action_direction", "over_scaffolding"):
            result = record[pass_name]
            if result and result["error"]:
                counts[f"{pass_name}_failed"] += 1
        reason = record["over_scaffolding_routing"]["reason"]
        if reason:
            counts[reason] += 1
        if record["action_direction"] and record["action_direction"]["parse_error"]:
            counts["action_direction_unparsed"] += 1
        if record["over_scaffolding"] and record["over_scaffolding"]["parse_error"]:
            counts["over_scaffolding_unparsed"] += 1
    return records, counts


def _yes_no(records, pass_name, field) -> str:
    values = [r[pass_name][field] for r in records if r.get(pass_name)]
    return (
        f"{sum(v is True for v in values)} / {sum(v is False for v in values)} / "
        f"{sum(v is None for v in values)}"
    )


def report(records: list[dict], counts: Counter) -> str:
    usage = C._sum_usage(*(r["usage"] for r in records))
    lines = [
        f"  replays: {len(records)}",
        f"  scaffolding      {_yes_no(records, 'action_direction', 'scaffolding')}",
        f"  rigor            {_yes_no(records, 'action_direction', 'rigor')}",
        "  over-scaffolding "
        f"{_yes_no(records, 'over_scaffolding', 'over_scaffolding')}"
        "  (asked only where scaffolding was predicted)",
        "  (counts are yes / no / unparsed)",
    ]
    for reason in (
        NOT_ROUTED_NO_SCAFFOLDING,
        NOT_ROUTED_UNPARSEABLE,
        NOT_ROUTED_FAILED,
    ):
        if counts[reason]:
            lines.append(f"  {reason}: {counts[reason]}")
    for pass_name in ("action_direction", "over_scaffolding"):
        if counts[f"{pass_name}_failed"]:
            lines.append(
                f"  {pass_name} requests still failed after {MAX_RETRIES} retries: "
                f"{counts[f'{pass_name}_failed']}"
            )
    lines.append(
        f"  tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out "
        f"/ {usage['total_tokens']:,} total"
    )
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.classify_replays",
        description="Classify v2 replays with the v2 scorer prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("replays", metavar="REPLAYS.jsonl", help="A replays.py file")
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Scorer override, looked up under `v2.models` then the arms. "
        "Default: `v2_replay.scorer`.",
    )
    parser.add_argument(
        "--prompt-version",
        default=C.latest_prompt_version(),
        metavar="VERSION",
        help=f"Prompt set under tutormoments_build/{C.PROMPT_ROOT}/",
    )
    parser.add_argument(
        "--context-turns",
        type=C.context_width,
        default=WIDTH,
        metavar="WIDTH",
        help="Which excerpt rendering the prompts read, as classify_excerpts: "
        "`full` or N dialogue turns before the cut",
    )
    parser.add_argument(
        "--student-state", default=C.DEFAULT_STUDENT_STATE, metavar="FILE"
    )
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--action-batch-id",
        default=None,
        metavar="ID",
        help="Resume the stage-1 batch instead of submitting one",
    )
    parser.add_argument(
        "--over-batch-id",
        default=None,
        metavar="ID",
        help="Resume the stage-2 batch instead of submitting one",
    )
    parser.add_argument(
        "--print",
        dest="print_moment",
        metavar="MOMENT_ID",
        help="Print one replay's action_direction prompt (id prefix) and exit",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    replays = list(jsonl_runs.load_by_id(args.replays).values())
    if not replays:
        print(f"no replays in {args.replays}", file=sys.stderr)
        return 1
    if args.limit is not None:
        replays = replays[: args.limit]

    prompts = C.load_prompt_set(args.prompt_version)
    if "{" + C.DESCRIPTIONS_SLOT + "}" in prompts.templates["over_scaffolding"]:
        C.attach_descriptions(
            {"replays": replays}, C.load_student_state(args.student_state)
        )

    if args.print_moment:
        record = next(
            (r for r in replays if r["moment_id"].startswith(args.print_moment)), None
        )
        if record is None:
            print(f"no replay matching {args.print_moment!r}", file=sys.stderr)
            return 1
        print(
            C.fill(
                prompts.templates["action_direction"],
                C.excerpt_at(record, args.context_turns),
            )
        )
        return 0

    # Check everything before either batch is paid for.
    cfg = scorer_spec(args.model)
    check_fillable(replays, prompts)
    width = args.context_turns
    out_path = classification_path(args.replays, cfg.model, prompts.version, width)
    entries, counts = C.build_entries(replays, prompts, ("action_direction",), width)
    knobs = ", ".join(f"{k}: {v}" for k, v in sorted(cfg.thinking.items()))
    print(
        f"classify replays: {args.replays}\n"
        f"  scorer: {cfg.model} ({knobs}), prompts v{prompts.version}, "
        f"excerpt {width}\n"
        f"  output: {out_path}\n"
        f"  stage 1 action_direction: {counts['action_direction']} request(s)\n"
        f"  stage 2 over-scaffolding: at most {counts['action_direction']} "
        "(only where stage 1 predicts scaffolding)"
        + (
            f"\n  not classified (nothing after the cut): {counts['no_post_cut_content']}"
            if counts["no_post_cut_content"]
            else ""
        )
    )
    if args.dry_run:
        return 0
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1

    records, counts = classify(
        replays,
        cfg,
        prompts,
        args.replays,
        action_batch_id=args.action_batch_id,
        over_batch_id=args.over_batch_id,
        width=width,
    )
    write_records(out_path, records)
    print(report(records, counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
