"""Label the fine-grained tutor actions after the cut in human-human moment excerpts.

Runs ``prompts/v2_helper/classify_action/<version>/classify_action.md`` over the
excerpts ``excerpts.py`` builds. ``--prompt-version`` defaults to the newest
numbered version directory present, and each record names the version, path and
hash of the prompt it was made with. The prompt lists scaffolding and rigor action types and asks which ones
the tutor uses after the cut, as a JSON array of
``{"evidence": <turn numbers>, "category": <action type>}``.

The prompt is told what the tutor did, through its ``{action}`` slot, which is
filled from the gold labels:

  scaffolding_present and rigor_present    "scaffold and push for rigor"
  scaffolding_present only                 "scaffold"
  rigor_present only                       "push for rigor"

Every moment with either gold label is asked. A moment whose gold marks neither
has no action to state and is not asked; nor is a moment with nothing after its
cut point. Both are still written, so the output accounts for every excerpt.

The action types are read from the prompt's ``### Scaffolding`` and ``### Rigor``
sections, so each returned label is tagged with its family, and a label the
prompt does not list is kept but flagged.

The excerpt and model setup are shared with ``classify_excerpts``: by default
20 dialogue turns before the cut through the moment's last row
(``--context-turns full`` for the whole transcript), and ``--model`` resolved
the same way (``v2`` block, ``v2.models``, or a ``benchmark_models`` arm).

Predictions go to ``<out-dir>/<model>/<prompt-version>/<width>/<split>.jsonl``:
a prompt edit changes what the labels mean, so rounds made with different
versions never share a file. Existing files are
only replaced with ``--overwrite``. Only the ``iteration`` split runs by default.

Usage::

    python -m tutormoments_build.v2.taxonomy --dry-run
    python -m tutormoments_build.v2.taxonomy --print
    python -m tutormoments_build.v2.taxonomy --print 02e89625
    python -m tutormoments_build.v2.taxonomy --limit 20
    python -m tutormoments_build.v2.taxonomy
    python -m tutormoments_build.v2.taxonomy --splits test
    python -m tutormoments_build.v2.taxonomy --model claude-opus-5
    python -m tutormoments_build.v2.taxonomy --context-turns full
    python -m tutormoments_build.v2.taxonomy --prompt-version 1
    python -m tutormoments_build.v2.taxonomy --batch-id msgbatch_01ABC...
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.resources import resource_text
from tutormoments_build.v2.classify_excerpts import (
    DEFAULT_EXCERPT_DIR,
    DEFAULT_SPLITS,
    EXCERPT_WIDTH,
    RANDOM,
    SPLIT_STEMS,
    ZERO_USAGE,
    V2Spec,
    _sum_usage,
    available_prompt_versions,
    check_writable,
    context_width,
    excerpt_at,
    fill,
    has_post_cut_content,
    latest_prompt_version,
    load_excerpts,
    phase_config,
    pick_moment,
    run_entries,
    write_split,
)

logger = logging.getLogger("tutormoments_build.v2.taxonomy")

# Numbered version directories, each holding PROMPT_FILE. A bare run uses the
# newest one present (latest_prompt_version).
PROMPT_ROOT = "prompts/v2_helper/classify_action"
PROMPT_FILE = "classify_action.md"

DEFAULT_OUT_DIR = "outputs/v2_taxonomy"

# (scaffolding_present, rigor_present) -> {action}. (False, False) has no
# wording: such moments are not asked.
ACTION_WORDING = {
    (True, True): "scaffold and push for rigor",
    (True, False): "scaffold",
    (False, True): "push for rigor",
}

# Prompt section heading -> family tag on each returned label.
FAMILY_HEADINGS = {"scaffolding": "scaffolding", "rigor": "rigor"}

# Batch-entry key prefix. A moment_id is a UUID, so a key splits back apart on
# the first "__".
ENTRY_PREFIX = "taxonomy"

NO_POST_CUT = "no_post_cut_content"
NO_GOLD_ACTION = "gold_marks_neither"

_HEADING = re.compile(r"^###\s+(.+?)\s*$")
_CATEGORY_LINE = re.compile(r"^`([a-z_]+)`\s*:")


# ===========================================================================
# Prompt
# ===========================================================================


@dataclass(frozen=True)
class ActionPrompt:
    """One numbered version of the classify_action prompt, loaded from disk."""

    version: str
    path: str
    template: str


def default_prompt_version() -> str:
    """The newest numbered version under PROMPT_ROOT."""
    return latest_prompt_version(PROMPT_ROOT)


def prompt_path(version: str) -> str:
    """Package-relative path to one version's template."""
    return f"{PROMPT_ROOT}/{version}/{PROMPT_FILE}"


def load_prompt(version: str | None = None) -> ActionPrompt:
    """Load one prompt version (default: the newest), naming the others if missing."""
    version = version or default_prompt_version()
    path = prompt_path(version)
    try:
        template = resource_text(path)
    except (FileNotFoundError, NotADirectoryError) as exc:
        available = available_prompt_versions(PROMPT_ROOT)
        raise FileNotFoundError(
            f"no classify_action prompt at {path}; prompt versions available: "
            f"{', '.join(available) or '(none)'}"
        ) from exc
    return ActionPrompt(version=version, path=path, template=template)


def load_template(version: str | None = None) -> str:
    """The template text of one prompt version (default: the newest)."""
    return load_prompt(version).template


def prompt_sha256(template: str) -> str:
    """Hash of the template, recorded on every prediction.

    Recorded beside the version, so an edit made inside a version directory
    after a round ran still shows up as a different prompt.
    """
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def parse_categories(template: str) -> dict[str, str]:
    """Return {action type: family} from the prompt's action-type sections.

    Reads the backticked names opening each line under a ``### Scaffolding`` or
    ``### Rigor`` heading. Raises if either family comes up empty, so a prompt
    restructured in a way this reader misses fails before a batch is paid for
    rather than flagging every label as unknown.
    """
    out: dict[str, str] = {}
    family = None
    for line in template.splitlines():
        heading = _HEADING.match(line)
        if heading:
            family = FAMILY_HEADINGS.get(heading.group(1).strip().lower())
            continue
        if line.startswith("## "):
            family = None
            continue
        match = _CATEGORY_LINE.match(line.strip())
        if family and match:
            out[match.group(1)] = family
    missing = [f for f in FAMILY_HEADINGS.values() if f not in out.values()]
    if missing:
        raise ValueError(
            f"prompt: no action types found under {', '.join(missing)} heading(s)"
        )
    return out


def action_slot(record: dict) -> str | None:
    """The ``{action}`` value from the gold labels, or None if gold marks neither."""
    labels = record.get("labels") or {}
    key = (bool(labels.get("scaffolding_present")), bool(labels.get("rigor_present")))
    return ACTION_WORDING.get(key)


def skip_reason(record: dict) -> str | None:
    """Why a moment is not sent, or None if it is."""
    if not has_post_cut_content(record):
        return NO_POST_CUT
    if action_slot(record) is None:
        return NO_GOLD_ACTION
    return None


def fill_prompt(template: str, record: dict, width: str = EXCERPT_WIDTH) -> str:
    return fill(template, excerpt_at(record, width), {"action": action_slot(record)})


def build_entries(
    records: list[dict], template: str, width: str = EXCERPT_WIDTH
) -> tuple[list[dict], Counter]:
    """One batch entry per askable moment. Returns (entries, counts)."""
    from tutormoments.client import build_batch_entry

    entries: list[dict] = []
    counts: Counter = Counter()
    for record in records:
        reason = skip_reason(record)
        if reason:
            counts[reason] += 1
            continue
        # Not JSON mode: the prompt asks for a top-level array, and OpenAI's
        # JSON mode only allows an object. parse_actions reads the array out of
        # surrounding prose or a code fence.
        entries.append(
            build_batch_entry(
                f"{ENTRY_PREFIX}__{record['moment_id']}",
                fill_prompt(template, record, width),
                json_mode=False,
            )
        )
        counts["asked"] += 1
        counts[f"asked_{action_slot(record)}"] += 1
    return entries, counts


# ===========================================================================
# Parsing
# ===========================================================================


def _loads_list(text: str) -> list | None:
    """Parse model output into a list of label objects, or None.

    Accepts a bare array, an array inside a code fence or prose, an object
    wrapping one array value, or a single label object.
    """
    if not text:
        return None
    candidates = [text]
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            if "category" in parsed:
                return [parsed]
            lists = [v for v in parsed.values() if isinstance(v, list)]
            if len(lists) == 1:
                return lists[0]
    return None


def parse_actions(text: str, categories: dict[str, str]) -> tuple[dict, bool]:
    """Parse a classify_action response. Returns (fields, had_error).

    ``actions`` is None when the response could not be parsed, distinct from
    ``[]`` (parsed, no labels). Each action carries its ``family`` from the
    prompt, or None for a label the prompt does not list; those labels are also
    collected in ``unknown_categories``.
    """
    items = _loads_list(text)
    if items is None:
        return {"actions": None, "unknown_categories": []}, True

    actions = []
    unknown = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "") or "").strip().lower()
        if not category:
            continue
        family = categories.get(category)
        if family is None:
            unknown.append(category)
        actions.append(
            {
                "category": category,
                "family": family,
                "evidence": str(item.get("evidence", "") or "").strip(),
            }
        )
    return {"actions": actions, "unknown_categories": unknown}, False


# ===========================================================================
# Assembly
# ===========================================================================


def build_record(
    excerpt_record: dict,
    raw_entries: dict,
    cfg: V2Spec,
    prompt: ActionPrompt,
    categories: dict[str, str],
    width: str = EXCERPT_WIDTH,
) -> dict:
    """One prediction record. Unasked moments get ``taxonomy`` None and a reason."""
    moment_id = excerpt_record["moment_id"]
    reason = skip_reason(excerpt_record)

    taxonomy = None
    if reason is None:
        raw = raw_entries.get(f"{ENTRY_PREFIX}__{moment_id}") or {}
        text = raw.get("text", "")
        fields, had_error = parse_actions(text, categories)
        taxonomy = {
            **fields,
            "parse_error": bool(had_error),
            "error": raw.get("error"),
            "raw": text,
            "usage": raw.get("usage", dict(ZERO_USAGE)),
        }

    return {
        "moment_id": moment_id,
        "transcript_id": excerpt_record["transcript_id"],
        "conversation_id": excerpt_record.get("conversation_id"),
        "split": excerpt_record["split"],
        "model": cfg.model,
        "provider": cfg.provider,
        "thinking": dict(cfg.thinking),
        "prompt": prompt.path,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt_sha256(prompt.template),
        "context_turns": width,
        # The gold-filled {action} the model was told; None if not asked.
        "action": action_slot(excerpt_record) if reason is None else None,
        "situation_type": excerpt_record.get("situation_type"),
        "post_cut_rows": excerpt_record.get("post_cut_rows"),
        # Dialogue turns only; evaluate_taxonomy's short-moment block reads it.
        "post_cut_dialogue_rows": excerpt_record.get("post_cut_dialogue_rows"),
        "asked": reason is None,
        "skipped": reason,
        "taxonomy": taxonomy,
        "labels": excerpt_record.get("labels"),
        "usage": (taxonomy or {}).get("usage", dict(ZERO_USAGE)),
    }


def classify(
    excerpts: dict[str, list[dict]],
    cfg: V2Spec,
    prompt: ActionPrompt,
    *,
    batch_id: str | None = None,
    width: str = EXCERPT_WIDTH,
) -> tuple[dict[str, list[dict]], Counter]:
    """Classify every split in one pooled batch. Returns (records by split, counts)."""
    categories = parse_categories(prompt.template)
    flat = [record for records in excerpts.values() for record in records]
    entries, counts = build_entries(flat, prompt.template, width)

    raw_entries = (
        run_entries(entries, cfg, batch_id=batch_id, display_name="v2_taxonomy")
        if entries
        else {}
    )

    out: dict[str, list[dict]] = {}
    for stem, records in excerpts.items():
        built = [
            build_record(record, raw_entries, cfg, prompt, categories, width)
            for record in records
        ]
        for record in built:
            if record["taxonomy"] and record["taxonomy"]["parse_error"]:
                counts["unparsed"] += 1
        out[stem] = built
    return out, counts


def prediction_dir(
    out_dir: str, model: str, prompt_version: str, width: str = EXCERPT_WIDTH
) -> str:
    """``<out_dir>/<model>/<prompt_version>/<width>``, "/" in the model id flattened."""
    return os.path.join(out_dir, model.replace("/", "_"), prompt_version, width)


# ===========================================================================
# Reporting
# ===========================================================================


def report(
    out: dict[str, list[dict]],
    counts: Counter,
    dry_run: bool,
    cfg: V2Spec | None = None,
    out_dir: str | None = None,
    width: str = EXCERPT_WIDTH,
    prompt: ActionPrompt | None = None,
) -> str:
    lines = [
        "",
        "DRY RUN -- no API calls made, nothing written"
        if dry_run
        else "Predictions written",
        "",
    ]
    if cfg:
        knobs = ", ".join(f"{k}: {v}" for k, v in sorted(cfg.thinking.items()))
        lines += [f"  model:   {cfg.model} ({knobs or 'no reasoning params sent'})"]
    if prompt:
        lines.append(f"  prompt:  {prompt.path} (version {prompt.version})")
    lines.append(f"  excerpt: {width}")
    if out_dir:
        lines.append(f"  output:  {out_dir}/")
    lines.append("")

    lines.append(f"  {'requests' if dry_run else 'asked'}: {counts['asked']}")
    for wording in ACTION_WORDING.values():
        lines.append(f"    {wording:<30}{counts[f'asked_{wording}']:>5}")
    lines.append(f"  not asked, nothing after the cut:  {counts[NO_POST_CUT]}")
    lines.append(f"  not asked, gold marks neither:     {counts[NO_GOLD_ACTION]}")

    if dry_run:
        return "\n".join(lines + [""])

    results = [
        r["taxonomy"]
        for records in out.values()
        for r in records
        if r["taxonomy"] and r["taxonomy"]["actions"] is not None
    ]
    tally = Counter(a["category"] for t in results for a in t["actions"])
    unknown = Counter(c for t in results for c in t["unknown_categories"])

    lines += ["", "  labels returned:"]
    for category, n in tally.most_common():
        flag = "  (not in prompt)" if category in unknown else ""
        lines.append(f"    {category:<30}{n:>5}{flag}")
    if counts["unparsed"]:
        lines += ["", f"  unparsed responses: {counts['unparsed']}"]

    usage = _sum_usage(*(r["usage"] for records in out.values() for r in records))
    lines += [
        "",
        f"  tokens: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out "
        f"/ {usage['total_tokens']:,} total",
        "",
    ]
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.taxonomy",
        description="Label fine-grained tutor actions after the cut in moment excerpts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--excerpt-dir", default=DEFAULT_EXCERPT_DIR, metavar="DIR")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model to classify with, configured as for classify_excerpts "
        "(`v2.models` or a `benchmark_models` arm). Default: the config's v2 model.",
    )
    parser.add_argument(
        "--prompt-version",
        default=default_prompt_version(),
        metavar="VERSION",
        help=f"Which numbered prompt under tutormoments_build/{PROMPT_ROOT}/ to "
        "classify with. Defaults to the newest one present. Predictions land in "
        "<out-dir>/<model>/<version>/<width>/.",
    )
    parser.add_argument(
        "--context-turns",
        type=context_width,
        default=EXCERPT_WIDTH,
        metavar="WIDTH",
        help="Excerpt rendering to read: `full`, or N dialogue turns before the cut.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_STEMS, default=list(DEFAULT_SPLITS)
    )
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument(
        "--batch-id",
        default=None,
        metavar="ID",
        help="Resume polling an in-flight batch submitted with the same flags.",
    )
    parser.add_argument(
        "--print",
        dest="print_moment",
        nargs="?",
        const=RANDOM,
        metavar="MOMENT_ID",
        help="Print one moment's filled prompt and exit (bare for a random one).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _print_prompt(excerpts, prefix, prompt, width) -> int:
    record = pick_moment(excerpts, prefix)
    if record is None:
        print(
            "no moments loaded to pick from"
            if prefix is RANDOM
            else f"no moment matching {prefix!r}",
            file=sys.stderr,
        )
        return 1
    print(f"===== moment {record['moment_id']} ({record['split']}) =====\n")
    reason = skip_reason(record)
    if reason:
        print(f"not asked: {reason}")
        return 0
    print(f"===== {prompt.path} (@{width} excerpt) =====\n")
    print(fill_prompt(prompt.template, record, width))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    excerpts = load_excerpts(args.excerpt_dir, stems=tuple(args.splits))
    if args.limit is not None:
        excerpts = {stem: records[: args.limit] for stem, records in excerpts.items()}

    prompt = load_prompt(args.prompt_version)
    parse_categories(prompt.template)  # fail on a prompt this can't read, on every path
    width = args.context_turns

    if args.print_moment:
        return _print_prompt(excerpts, args.print_moment, prompt, width)

    cfg = phase_config(model=args.model)
    out_dir = prediction_dir(args.out_dir, cfg.model, prompt.version, width)

    if args.dry_run:
        flat = [record for records in excerpts.values() for record in records]
        _, counts = build_entries(flat, prompt.template, width)
        print(
            report(
                {},
                counts,
                dry_run=True,
                cfg=cfg,
                out_dir=out_dir,
                width=width,
                prompt=prompt,
            )
        )
        return 0

    check_writable(out_dir, args.splits, args.overwrite)

    logger.info(
        "labelling %d moment(s) with model=%s thinking=%s excerpt=%s",
        sum(len(records) for records in excerpts.values()),
        cfg.model,
        cfg.thinking,
        width,
    )
    out, counts = classify(excerpts, cfg, prompt, batch_id=args.batch_id, width=width)

    for stem, records in out.items():
        logger.info(
            "wrote %d record(s) to %s",
            len(records),
            write_split(out_dir, stem, records),
        )
    print(
        report(
            out,
            counts,
            dry_run=False,
            cfg=cfg,
            out_dir=out_dir,
            width=width,
            prompt=prompt,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
