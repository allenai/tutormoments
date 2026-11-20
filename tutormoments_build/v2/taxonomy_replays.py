"""Labels the fine-grained tutor actions in classified replays.

Runs the same `classify_action` prompt as `taxonomy.py`, but `{action}` and
whether a replay is asked come from that replay's own `action_direction`
prediction (from `classify_replays.py`), not from the human tutor's gold labels.

Reads a `classify_replays.py` output file, plus the replay file it names
(`replay.file` on each record; `--replays` overrides). Writes
`<classification stem>__taxonomy-<model>_v<version>.jsonl` beside it. The
excerpt width defaults to the one the classification was made at.

A replay is not asked when nothing follows its cut, when its action_direction
request failed or was unparseable, or when it was predicted to neither scaffold
nor push for rigor. Each case is written with its reason.

Usage::

    python -m tutormoments_build.v2.taxonomy_replays CLASSIFIED.jsonl --dry-run
    python -m tutormoments_build.v2.taxonomy_replays CLASSIFIED.jsonl --print 02e89625
    python -m tutormoments_build.v2.taxonomy_replays CLASSIFIED.jsonl --limit 20
    python -m tutormoments_build.v2.taxonomy_replays CLASSIFIED.jsonl
"""

import argparse
import os
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.v2 import classify_excerpts as C
from tutormoments_build.v2 import classify_replays as R
from tutormoments_build.v2 import jsonl_runs
from tutormoments_build.v2 import taxonomy as T

PREDICTED_NEITHER = "predicted_neither"
ACTION_UNPARSEABLE = "action_direction unparseable"
ACTION_FAILED = "action_direction request failed"


# ===========================================================================
# Joining replays to their classification
# ===========================================================================


def replay_file(classified: list[dict]) -> str:
    """The one replay file every classification record names."""
    files = {r["replay"]["file"] for r in classified}
    if len(files) != 1:
        raise ValueError(f"classification records name {len(files)} replay files")
    return files.pop()


def join(classified: list[dict], replays: dict[str, dict]) -> list[dict]:
    """Each replay record with its classification attached under `classification`.

    Raises if a classified moment is missing from the replay file, or the replay
    was regenerated (different settings) after it was classified.
    """
    out = []
    for record in classified:
        moment_id = record["moment_id"]
        replay = replays.get(moment_id)
        if replay is None:
            raise KeyError(f"{moment_id} is classified but not in the replay file")
        if replay["settings"] != record["replay"]["settings"]:
            raise ValueError(
                f"{moment_id}: replay settings differ from the classified run"
            )
        out.append({**replay, "classification": record})
    return out


def predicted_presence(record: dict) -> tuple[bool, bool]:
    action = record["classification"]["action_direction"]
    return bool(action["scaffolding"]), bool(action["rigor"])


def unusable_reason(record: dict) -> str | None:
    """Why a replay's prediction can't fill `{action}`, or None if it can."""
    action = record["classification"].get("action_direction")
    if action is None:
        return T.NO_POST_CUT
    if action["scaffolding"] is None or action["rigor"] is None:
        failed_request = action.get("error") and not action.get("raw")
        return ACTION_FAILED if failed_request else ACTION_UNPARSEABLE
    return None


def skip_reason(record: dict) -> str | None:
    return unusable_reason(record) or T.skip_reason(
        record, predicted_presence, PREDICTED_NEITHER
    )


# ===========================================================================
# Paths, records, running
# ===========================================================================


def output_path(classification_file: str, model: str, prompt_version: str) -> str:
    stem, ext = os.path.splitext(classification_file)
    return f"{stem}__taxonomy-{model.replace('/', '_')}_v{prompt_version}{ext}"


def build_record(
    record: dict,
    raw_entries: dict,
    attempts: dict,
    cfg: C.V2Spec,
    prompt: T.ActionPrompt,
    categories: dict,
    classification_file: str,
    width: str,
) -> dict:
    moment_id = record["moment_id"]
    classification = record["classification"]
    reason = skip_reason(record)
    taxonomy = None if reason else T.taxonomy_result(raw_entries, moment_id, categories)
    action = classification.get("action_direction")
    return {
        "moment_id": moment_id,
        "source": record.get("source"),
        "transcript_id": record["transcript_id"],
        "conversation_id": record.get("conversation_id"),
        "replay": classification["replay"],
        "classification": {
            "file": classification_file,
            "model": classification["model"],
            "thinking": classification["thinking"],
            "prompt_version": classification["prompt_version"],
            "context_turns": classification["context_turns"],
        },
        "model": cfg.model,
        "provider": cfg.provider,
        "thinking": dict(cfg.thinking),
        "prompt": prompt.path,
        "prompt_version": prompt.version,
        "prompt_sha256": T.prompt_sha256(prompt.template),
        "context_turns": width,
        # {action} is the replay's predicted action, not gold.
        "action": None if reason else T.action_slot(record, predicted_presence),
        "action_from": {
            "gated_on": "predicted",
            "scaffolding": None if action is None else action["scaffolding"],
            "rigor": None if action is None else action["rigor"],
        },
        "asked": reason is None,
        "skipped": reason,
        "attempts": attempts.get(f"{T.ENTRY_PREFIX}__{moment_id}", 0),
        "taxonomy": taxonomy,
        "situation_type": record.get("situation_type"),
        "labels": record.get("labels"),
        "post_cut_rows": record.get("post_cut_rows"),
        "post_cut_dialogue_rows": record.get("post_cut_dialogue_rows"),
        "usage": (taxonomy or {}).get("usage", dict(C.ZERO_USAGE)),
    }


def entries_for(records, template, width):
    """Batch entries for the replays whose prediction can fill `{action}`."""
    usable = [r for r in records if unusable_reason(r) is None]
    entries, counts = T.build_entries(
        usable, template, width, presence=predicted_presence, neither=PREDICTED_NEITHER
    )
    counts.update(r for r in map(unusable_reason, records) if r)
    return entries, counts


def classify(
    records: list[dict],
    cfg: C.V2Spec,
    prompt: T.ActionPrompt,
    classification_file: str,
    *,
    width: str,
    batch_id: str | None = None,
) -> tuple[list[dict], Counter]:
    categories = T.parse_categories(prompt.template)
    entries, counts = entries_for(records, prompt.template, width)
    raw, attempts = (
        R.run_with_retries(
            entries, cfg, batch_id=batch_id, display_name="v2_taxonomy_replays"
        )
        if entries
        else ({}, {})
    )
    out = [
        build_record(
            r, raw, attempts, cfg, prompt, categories, classification_file, width
        )
        for r in records
    ]
    for record in out:
        taxonomy = record["taxonomy"]
        if taxonomy and taxonomy["parse_error"]:
            counts["unparsed"] += 1
        if taxonomy and R.failed(taxonomy):
            counts["failed"] += 1
    return out, counts


def report(counts: Counter, records: list[dict] | None = None) -> str:
    lines = [f"  asked: {counts['asked']}"]
    for wording in T.ACTION_WORDING.values():
        lines.append(f"    {wording:<30}{counts[f'asked_{wording}']:>5}")
    for reason in (PREDICTED_NEITHER, ACTION_UNPARSEABLE, ACTION_FAILED, T.NO_POST_CUT):
        if counts[reason]:
            lines.append(f"  not asked, {reason}: {counts[reason]}")
    if records is None:
        return "\n".join(lines)
    tally = Counter(
        a["category"]
        for r in records
        if r["taxonomy"] and r["taxonomy"]["actions"]
        for a in r["taxonomy"]["actions"]
    )
    lines.append("  labels returned:")
    lines += [f"    {c:<30}{n:>5}" for c, n in tally.most_common()]
    for key in ("unparsed", "failed"):
        if counts[key]:
            lines.append(f"  {key}: {counts[key]}")
    usage = C._sum_usage(*(r["usage"] for r in records))
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
        prog="python -m tutormoments_build.v2.taxonomy_replays",
        description="Label fine-grained tutor actions in classified replays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument(
        "classified", metavar="CLASSIFIED.jsonl", help="A classify_replays.py file"
    )
    parser.add_argument(
        "--replays",
        default=None,
        metavar="REPLAYS.jsonl",
        help="Replay file (default: the one the classification names)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model override, configured as for taxonomy.py. "
        "Default: the config's v2 model.",
    )
    parser.add_argument(
        "--prompt-version", default=T.default_prompt_version(), metavar="VERSION"
    )
    parser.add_argument(
        "--context-turns",
        type=C.context_width,
        default=None,
        metavar="WIDTH",
        help="Excerpt width (default: the width the classification used)",
    )
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-id", default=None, metavar="ID")
    parser.add_argument(
        "--print",
        dest="print_moment",
        metavar="MOMENT_ID",
        help="Print one replay's filled prompt (id prefix) and exit",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    classified = list(jsonl_runs.load_by_id(args.classified).values())
    if not classified:
        print(f"no records in {args.classified}", file=sys.stderr)
        return 1
    if args.limit is not None:
        classified = classified[: args.limit]
    replays_path = args.replays or replay_file(classified)
    records = join(classified, jsonl_runs.load_by_id(replays_path))

    widths = {r["context_turns"] for r in classified}
    if args.context_turns is None and len(widths) != 1:
        print(
            "classification mixes excerpt widths; pass --context-turns", file=sys.stderr
        )
        return 1
    width = args.context_turns or widths.pop()

    prompt = T.load_prompt(args.prompt_version)
    T.parse_categories(prompt.template)

    if args.print_moment:
        record = next(
            (r for r in records if r["moment_id"].startswith(args.print_moment)), None
        )
        if record is None:
            print(f"no replay matching {args.print_moment!r}", file=sys.stderr)
            return 1
        reason = skip_reason(record)
        print(
            f"not asked: {reason}"
            if reason
            else T.fill_prompt(prompt.template, record, width, predicted_presence)
        )
        return 0

    cfg = C.phase_config(model=args.model)
    out_path = output_path(args.classified, cfg.model, prompt.version)
    knobs = ", ".join(f"{k}: {v}" for k, v in sorted(cfg.thinking.items()))
    print(
        f"taxonomy on replays: {args.classified}\n"
        f"  replays: {replays_path}\n"
        f"  model: {cfg.model} ({knobs}), prompt {prompt.path}, excerpt {width}\n"
        f"  output: {out_path}"
    )
    if args.dry_run:
        _, counts = entries_for(records, prompt.template, width)
        print(report(counts))
        return 0
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to replace it", file=sys.stderr)
        return 1

    out, counts = classify(
        records, cfg, prompt, args.classified, width=width, batch_id=args.batch_id
    )
    R.write_records(out_path, out)
    print(report(counts, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
