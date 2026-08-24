"""Run the v2 classification prompts over human-human moment excerpts.

``excerpts.py`` creates excerpts. 

Using
prompts in ``tutormoments_build/prompts/v2/``: 

  action_direction.md    scaffolding yes/no, rigor yes/no, plus a description
  over-scaffolding.md    over-scaffolding yes/no, plus a description

Together they predict ``scaffolding_present``, ``rigor_present``, ``over_scaffolding_present``,
in a manner that lines up for direct comparison with gold labels. 

**The over-scaffolding pass is gated on the gold labels**. 
It is sent only where the annotators marked
``scaffolding_appropriate`` *and* ``scaffolding_present``. 

What is left is the question the prompt was written for: scaffolding was called
for, the tutor scaffolded, did they scaffold too much? 

Moments outside the gate get no over-scaffolding call at all: the field is
``null``. 

Both passes ride in one batch.

Model and thinking default to the ``v2`` block in the runtime config. 
``--model`` sets the model, with hyperparameters 
(thinking, effort, reasoning_effort, thinking_budget) from
that model's entry under ``v2.models`` in the config. Falling back to the
top-level ``models`` tutor roster for an id already configured there.

Predictions are written per model: ``<out-dir>/<model>/<split>.jsonl``. 

Only the ``iteration`` split runs by default. The ``test`` split is held out
while the prompts are still changing, and is classified with an explicit
``--splits test`` once they are frozen.

Examples of models it could run with:
- claude-opus-5
- gemini-3.5-flash
- gpt-5.6-sol

Usage::

    python -m tutormoments_build.v2.classify_excerpts --dry-run
    python -m tutormoments_build.v2.classify_excerpts --limit 20
    python -m tutormoments_build.v2.classify_excerpts
    python -m tutormoments_build.v2.classify_excerpts --splits test
    python -m tutormoments_build.v2.classify_excerpts --model claude-opus-5
    python -m tutormoments_build.v2.classify_excerpts --print
    python -m tutormoments_build.v2.classify_excerpts --print 02e89625
    python -m tutormoments_build.v2.classify_excerpts --batch-id msgbatch_01ABC...
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.resources import resource_text

logger = logging.getLogger("tutormoments_build.v2.classify_excerpts")

DEFAULT_EXCERPT_DIR = "data/excerpts"
DEFAULT_OUT_DIR = "outputs/v2_predictions"

# Excerpt file stems, which are also the prediction output stems.
SPLIT_STEMS = ("iteration", "test")

# Only the iteration split is classified unless the test split is asked for by
# name. The test split is the held-out half: every prediction run over it while
# the prompts are still being revised turns it into a second iteration set. It
# is classified once the prompts are frozen, with `--splits test`.
DEFAULT_SPLITS = ("iteration",)

ACTION_DIRECTION_PROMPT = "prompts/v2/action_direction.md"
OVER_SCAFFOLDING_PROMPT = "prompts/v2/over-scaffolding.md"

# Lead-up each prompt reads, in dialogue turns before the cut. They differ
# because the questions differ: action direction asks only what the tutor's
# *first* move after the cut is, which the immediate exchange settles, while
# over-scaffolding asks whether the student had already shown they could do the
# work -- a judgment that needs to see enough of the session to answer.
# The excerpt file must have been built at both widths (it is, by default).
ACTION_CONTEXT_TURNS = 5
OVER_SCAFFOLDING_CONTEXT_TURNS = 20

# Batch-entry key prefixes. Neither contains "__", and a moment_id is a UUID, so
# a key splits back apart on the first "__".
ACTION_PREFIX = "action"
OVER_SCAFFOLDING_PREFIX = "overscaffold"

# Used when the config carries no `v2` block, so the script still runs against a
# stripped-down custom config. The config file is the source of truth.
FALLBACK_SPEC = {
    "model": "claude-opus-5",
    "thinking": "adaptive",
    "effort": "xhigh",
    "poll_interval": 60,
}

ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# Sentinel for `--print` passed with no id: pick a moment at random.
RANDOM = object()


# ===========================================================================
# Config
# ===========================================================================


# Per-model knobs read off a `v2.models` (or tutor roster) entry when --model
# names one. poll_interval is not among them: it is a property of the round, not
# the model, so it stays whatever the v2 block says.
ROSTER_KEYS = ("thinking", "thinking_budget", "reasoning_effort", "effort")


def model_knobs(model: str, scoped: dict, config_path=None) -> dict:
    """Generation knobs for a --model id: ``v2.models`` first, tutor roster second.

    Scoring models for this round belong under the v2 block: they classify
    moments and are never replayed as tutors, while the top-level ``models``
    roster is the candidate list `tutormoments run` draws from. Ids already on
    that roster still resolve, so a model configured there needs no second
    entry.

    Either lookup missing raises rather than defaulting: an unconfigured model
    has no declared thinking config, and guessing one would misreport how a
    label was made.
    """
    from tutormoments.client import infer_provider
    from tutormoments.config import resolve_model

    if model in scoped:
        # The tutor roster path gets this check from resolve_model. An id no
        # client can route has to fail here rather than at batch submission.
        infer_provider(model)
        return scoped[model] or {}

    try:
        return resolve_model(model, config_path)["kwargs"]
    except ValueError as exc:
        # Name both places a comparison model can be configured, so the fix is
        # obvious from the error.
        raise ValueError(
            f"{exc} Models under the config `v2.models`: "
            f"{', '.join(scoped) or '(none)'}"
        ) from exc


def phase_config(config_path=None, model: str | None = None) -> dict:
    """Return the v2 classification phase config (model/thinking/poll_interval).

    ``model`` overrides the v2 block's model. Its generation knobs then come
    from ``model_knobs``, so a comparison run is configured in the config file
    rather than on the command line; anything that entry does not set keeps the
    v2 block's value. The v2 block's own model needs no entry -- it is already
    fully specified there.
    """
    from tutormoments.config import load_config

    cfg = dict(FALLBACK_SPEC)
    block = load_config(config_path).get("v2")
    if not block:
        logger.warning(
            "no `v2` block in the config; falling back to %s", FALLBACK_SPEC["model"]
        )
    else:
        cfg.update(block)

    # `v2.models` is a lookup table for --model, not a setting of the round, so
    # it must not ride along in the returned spec (which is recorded per
    # prediction and passed to the batch).
    scoped = cfg.pop("models", None) or {}

    if model and model != cfg["model"]:
        kwargs = model_knobs(model, scoped, config_path)
        cfg["model"] = model
        # The entry is the override's *complete* generation config, so every
        # per-model knob is cleared before it is applied. Inheriting them would
        # cross vendors -- an OpenAI model carrying the baseline's Anthropic
        # `effort` -- and misreport the depth a label was made at. `thinking`
        # then defaults off rather than to the v2 block's adaptive: a configured
        # model that sets no thinking key means thinking off.
        for key in ROSTER_KEYS:
            cfg.pop(key, None)
        cfg["thinking"] = False
        cfg.update({k: v for k, v in kwargs.items() if k in ROSTER_KEYS})

    return cfg


def use_thinking(cfg: dict) -> bool:
    """Normalise the config ``thinking`` value to the bool run_batch expects.

    Mirrors ``tutormoments_build.groundtruth._use_thinking``: a string like
    "adaptive"/"enabled" (or True) enables thinking, anything else disables it.
    Passing the raw config string through would make any non-empty string --
    "disabled" included -- truthy.
    """
    return cfg.get("thinking", False) in ("adaptive", "enabled", True)


# ===========================================================================
# Reading
# ===========================================================================


def load_excerpts(excerpt_dir: str, stems=DEFAULT_SPLITS) -> dict[str, list[dict]]:
    """Return {split stem: [excerpt record, ...]} in file order.

    A missing split file is not an error -- a round may have produced only one.
    """
    out: dict[str, list[dict]] = {}
    for stem in stems:
        path = os.path.join(excerpt_dir, f"{stem}.jsonl")
        if not os.path.exists(path):
            logger.warning("no excerpt file at %s; skipping split", path)
            continue
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        out[stem] = records
    if not out:
        raise FileNotFoundError(
            f"no excerpt splits found in {excerpt_dir}; run "
            "`python -m tutormoments_build.v2.excerpts` first"
        )
    return out


# ===========================================================================
# Prompt building
# ===========================================================================


def fill(template: str, excerpt: str) -> str:
    """Substitute ``{excerpt}`` into a v2 prompt template.

    ``str.replace``, not ``str.format``: both templates print a literal JSON
    object in their output-format section, and ``format`` would read those
    braces as fields and raise.
    """
    return template.replace("{excerpt}", excerpt)


def excerpt_at(record: dict, context_turns: int) -> str:
    """The moment's excerpt rendered at ``context_turns`` of lead-up.

    Excerpt files carry one rendering per width built (see
    ``excerpts.DEFAULT_CONTEXT_WIDTHS``). A missing width is a build/consumer
    mismatch rather than a data problem, so it raises with the command that
    fixes it instead of silently falling back to a window the prompt was not
    written for.
    """
    available = record.get("excerpts") or {}
    rendered = available.get(str(context_turns))
    if rendered is None:
        raise KeyError(
            f"moment {record.get('moment_id')} has no {context_turns}-turn excerpt "
            f"(built: {', '.join(sorted(available)) or 'none'}); rebuild with "
            f"`python -m tutormoments_build.v2.excerpts --context-turns "
            f"{OVER_SCAFFOLDING_CONTEXT_TURNS} {ACTION_CONTEXT_TURNS}`"
        )
    return rendered["excerpt"]


def wants_over_scaffolding(record: dict) -> bool:
    """Whether the over-scaffolding prompt applies to this moment.

    Gold-gated on both halves of the premise -- see the module docstring. The
    prompt asks whether a tutor who *should* have scaffolded scaffolded too
    much, so it is asked only where the annotators marked both
    ``scaffolding_appropriate`` and ``scaffolding_present``.
    """
    labels = record.get("labels") or {}
    return bool(labels.get("scaffolding_appropriate")) and bool(
        labels.get("scaffolding_present")
    )


def skip_reason(record: dict) -> str:
    """Which half of the over-scaffolding premise this moment fails.

    Reported separately because the two are different situations, not one
    exclusion: "no scaffolding" is a moment the prompt has nothing to weigh in,
    while "not appropriate" is one the ground truth already labels
    over-scaffolding by rule.
    """
    labels = record.get("labels") or {}
    if not labels.get("scaffolding_present"):
        return "no_scaffolding"
    return "not_appropriate"


def has_post_cut_content(record: dict) -> bool:
    """Whether there is anything after the cut point for a prompt to classify.

    Both prompts ask what the tutor does *after* the cut, so a moment ending at
    its own cut point gives them nothing to read and is not sent at all. Screen
    activity counts as content: an enrichment row like "[SCREEN INTERACTION]
    Tutor writes 3x7 on the board" is a pedagogical move, so a moment whose
    post-cut span is all enrichment is still classified.

    This has excluded nothing so far -- every annotated moment has had at least
    one post-cut row. It is a guard for later rounds, where a redrawn boundary
    could put the cut on a moment's last row. The run report counts what it
    actually excludes.

    Excerpt files built before ``post_cut_rows`` existed have no field to read;
    those moments are classified, and the caller warns.
    """
    return record.get("post_cut_rows", 1) > 0


def build_entries(records: list[dict]) -> tuple[list[dict], Counter]:
    """Build the batch entries for one round. Returns (entries, per-pass counts)."""
    from tutormoments.client import build_batch_entry

    action_template = resource_text(ACTION_DIRECTION_PROMPT)
    over_template = resource_text(OVER_SCAFFOLDING_PROMPT)

    entries: list[dict] = []
    counts: Counter = Counter()

    for record in records:
        moment_id = record["moment_id"]

        if "post_cut_rows" not in record:
            counts["post_cut_field_missing"] += 1
        if not has_post_cut_content(record):
            counts["no_post_cut_content"] += 1
            continue

        entries.append(
            build_batch_entry(
                f"{ACTION_PREFIX}__{moment_id}",
                fill(action_template, excerpt_at(record, ACTION_CONTEXT_TURNS)),
                json_mode=True,
            )
        )
        counts["action_direction"] += 1

        if wants_over_scaffolding(record):
            entries.append(
                build_batch_entry(
                    f"{OVER_SCAFFOLDING_PREFIX}__{moment_id}",
                    fill(
                        over_template,
                        excerpt_at(record, OVER_SCAFFOLDING_CONTEXT_TURNS),
                    ),
                    json_mode=True,
                )
            )
            counts["over_scaffolding"] += 1
        else:
            counts["over_scaffolding_not_asked"] += 1
            counts[f"over_scaffolding_skip_{skip_reason(record)}"] += 1

    if counts["post_cut_field_missing"]:
        logger.warning(
            "%d excerpt record(s) predate the post_cut_rows field and were not "
            "checked for post-cut content; rebuild with "
            "`python -m tutormoments_build.v2.excerpts` to apply that check",
            counts["post_cut_field_missing"],
        )

    return entries, counts


# ===========================================================================
# Parsing
# ===========================================================================


def _loads_object(text: str) -> dict | None:
    """Parse model output into a JSON object, or None if it isn't one.

    Unwraps a single-element list (models occasionally wrap the object), and
    falls back to the first braced span when the response carries prose or a
    code fence around the JSON.
    """
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    return parsed if isinstance(parsed, dict) else None


def _description(obj: dict | None) -> str:
    return str((obj or {}).get("description", "") or "").strip()


# Inverse of tutormoments.scoring._YES_NO_TO_ACTION_LABEL, which collapses the
# prompt's two yes/no fields into one label. "unclear" (either field missing or
# unparseable) has no entry: it yields (None, None).
_LABEL_TO_PRESENCE = {
    "both": (True, True),
    "scaffolding": (True, False),
    "rigor": (False, True),
    "neither": (False, False),
}


def parse_action_direction(text: str) -> tuple[dict, bool]:
    """Parse an action_direction response. Returns (fields, had_error).

    The prompt's ``{"scaffolding": ..., "rigor": ...}`` shape is the one the v1
    structure pass already reads, so the yes/no extraction reuses
    ``tutormoments.scoring.parse_action_label`` -- including its regex fallback
    for responses with extra text around the JSON -- and expands the label it
    returns back into the two booleans. ``description`` is read separately,
    since that parser drops everything but the label.

    ``scaffolding``/``rigor`` are None when the response could not be parsed.
    """
    from tutormoments.scoring import parse_action_label

    label, had_error = parse_action_label(text)
    scaffolding, rigor = _LABEL_TO_PRESENCE.get(label, (None, None))
    return {
        "scaffolding": scaffolding,
        "rigor": rigor,
        "description": _description(_loads_object(text)),
    }, had_error


# The prompt asks for the hyphenated key; the others are spellings models drift to.
_OVER_SCAFFOLDING_KEYS = ("over-scaffolding", "over_scaffolding", "overscaffolding")
_YES_NO = {"yes": True, "no": False}


def parse_over_scaffolding(text: str) -> tuple[dict, bool]:
    """Parse an over-scaffolding response. Returns (fields, had_error).

    ``over_scaffolding`` is None when the response could not be parsed.
    """
    obj = _loads_object(text)
    if obj is not None:
        for key in _OVER_SCAFFOLDING_KEYS:
            value = _YES_NO.get(str(obj.get(key, "")).strip().lower())
            if value is not None:
                return {
                    "over_scaffolding": value,
                    "description": _description(obj),
                }, False

    match = re.search(
        r'["\']?over[-_ ]?scaffolding["\']?\s*:\s*["\']?(yes|no)', text or "", re.I
    )
    if match:
        return {
            "over_scaffolding": _YES_NO[match.group(1).lower()],
            "description": _description(obj),
        }, False

    return {"over_scaffolding": None, "description": _description(obj)}, True


# ===========================================================================
# Running
# ===========================================================================


def run_entries(entries: list[dict], cfg: dict, batch_id: str | None = None) -> dict:
    """Submit one batch and return {key: {"text", "usage"} | {"error", ...}}.

    ``batch_id`` resumes polling an in-flight batch instead of submitting a new
    one; the entries passed must be the ones that batch was submitted with,
    since they drive result parsing.
    """
    from tutormoments.client import ModelClient, run_batch

    client = ModelClient(cfg["model"])

    def _created(created_id):
        logger.info(
            "batch %s submitted; resume with --batch-id %s", created_id, created_id
        )

    return run_batch(
        client,
        entries,
        json_mode=True,
        display_name="v2_classify_excerpts",
        poll_interval=cfg.get("poll_interval", 60),
        thinking=use_thinking(cfg),
        thinking_budget=cfg.get("thinking_budget", 0),
        reasoning_effort=cfg.get("reasoning_effort", ""),
        effort=cfg.get("effort", ""),
        existing_batch_id=batch_id,
        on_batch_created=_created,
    )


# ===========================================================================
# Assembly
# ===========================================================================


def _sum_usage(*usages: dict) -> dict:
    """Sum input/output/total tokens across usage dicts."""
    out = dict(ZERO_USAGE)
    for usage in usages:
        if isinstance(usage, dict):
            for field in out:
                out[field] += int(usage.get(field, 0) or 0)
    return out


def _pass_result(raw: dict | None, parse) -> dict:
    """Shape one pass's output: parsed fields, plus the raw text and any error.

    The raw response is kept so a parse failure or a surprising label can be
    read back without re-running the batch, and the token usage so the round's
    cost is recoverable per moment.
    """
    raw = raw or {}
    text = raw.get("text", "")
    error = raw.get("error")
    fields, had_error = parse(text)
    return {
        **fields,
        "parse_error": bool(had_error),
        "error": error,
        "raw": text,
        "usage": raw.get("usage", dict(ZERO_USAGE)),
    }


def build_record(excerpt_record: dict, raw_entries: dict, cfg: dict) -> dict:
    """Assemble one prediction record from an excerpt record and the batch results.

    ``over_scaffolding`` is None when the pass was not asked about this moment
    (the gold gate), which is distinct from a pass that ran and answered "no".

    A moment with nothing after its cut point is not classified at all: both
    passes are None and ``skipped`` says why. The record is still written, so a
    prediction file accounts for every moment in the excerpt file and a join
    against the ground truth never comes up short.
    """
    moment_id = excerpt_record["moment_id"]
    classified = has_post_cut_content(excerpt_record)

    action = (
        _pass_result(
            raw_entries.get(f"{ACTION_PREFIX}__{moment_id}"), parse_action_direction
        )
        if classified
        else None
    )

    over = None
    if classified and wants_over_scaffolding(excerpt_record):
        over = _pass_result(
            raw_entries.get(f"{OVER_SCAFFOLDING_PREFIX}__{moment_id}"),
            parse_over_scaffolding,
        )

    return {
        "moment_id": moment_id,
        "transcript_id": excerpt_record["transcript_id"],
        "conversation_id": excerpt_record.get("conversation_id"),
        "split": excerpt_record["split"],
        "model": cfg["model"],
        "thinking": cfg.get("thinking"),
        # The reasoning-depth knobs the round actually ran with, under the names
        # the two APIs use (`effort` is Anthropic's, `reasoning_effort` OpenAI's;
        # a model uses one or neither). Recorded because they are part of how a
        # label was produced: the same model at a pinned effort and at the API
        # default can answer a borderline moment differently, and without these
        # two rounds write identical-looking metadata.
        "effort": cfg.get("effort") or None,
        "reasoning_effort": cfg.get("reasoning_effort") or None,
        "thinking_budget": cfg.get("thinking_budget") or None,
        "prompts": {
            "action_direction": ACTION_DIRECTION_PROMPT,
            "over_scaffolding": OVER_SCAFFOLDING_PROMPT,
        },
        "context_turns": {
            "action_direction": ACTION_CONTEXT_TURNS,
            "over_scaffolding": OVER_SCAFFOLDING_CONTEXT_TURNS,
        },
        "post_cut_rows": excerpt_record.get("post_cut_rows"),
        "post_cut_dialogue_rows": excerpt_record.get("post_cut_dialogue_rows"),
        "classified": classified,
        "skipped": None if classified else "no_post_cut_content",
        "action_direction": action,
        "over_scaffolding": over,
        "over_scaffolding_asked": over is not None,
        "labels": excerpt_record.get("labels"),
        "usage": _sum_usage((action or {}).get("usage"), (over or {}).get("usage")),
    }


def classify(
    excerpts: dict[str, list[dict]],
    cfg: dict,
    *,
    batch_id: str | None = None,
) -> tuple[dict[str, list[dict]], Counter]:
    """Classify every split in one pooled batch.

    Returns ({split stem: [prediction record, ...]}, counts). All splits share
    the batch -- moment ids are unique across them, and the records are split
    apart again on the way out.
    """
    flat = [record for records in excerpts.values() for record in records]
    entries, counts = build_entries(flat)

    raw_entries = run_entries(entries, cfg, batch_id=batch_id) if entries else {}

    out: dict[str, list[dict]] = {}
    for stem, records in excerpts.items():
        built = [build_record(record, raw_entries, cfg) for record in records]
        for record in built:
            if not record["classified"]:
                continue
            if record["action_direction"]["parse_error"]:
                counts["action_direction_unparsed"] += 1
            if record["over_scaffolding"] and record["over_scaffolding"]["parse_error"]:
                counts["over_scaffolding_unparsed"] += 1
        out[stem] = built

    return out, counts


def model_dir(out_dir: str, model: str) -> str:
    """The output directory for one model's predictions.

    Predictions are filed under the model that made them so a comparison run
    does not overwrite the round before it. Model ids can carry a vendor prefix
    ("deepseek-ai/DeepSeek-V4-Pro"), which would otherwise nest a directory;
    the separator is flattened so every model gets exactly one level.
    """
    return os.path.join(out_dir, model.replace("/", "_"))


def write_split(out_dir: str, stem: str, records: list[dict]) -> str:
    """Write one split's predictions JSONL atomically. Returns the path written."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.jsonl")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
    return path


# ===========================================================================
# Reporting
# ===========================================================================


# Why a moment fell outside the over-scaffolding gate, in report wording.
_SKIP_LABELS = {
    "no_scaffolding": "gold says no scaffolding here",
    "not_appropriate": "scaffolding not called for (gold labels it over-scaffolding by rule)",
}


def _skip_breakdown(counts: Counter) -> list[str]:
    """Per-reason lines for the moments the over-scaffolding gate excluded."""
    return [
        f"      {counts[f'over_scaffolding_skip_{reason}']} {label}"
        for reason, label in _SKIP_LABELS.items()
        if counts[f"over_scaffolding_skip_{reason}"]
    ]


def _yes_no_counts(records: list[dict], pass_name: str, field: str) -> str:
    """``yes / no / unparsed`` tally for one predicted field."""
    values = [
        r[pass_name][field]
        for r in records
        if r.get("classified") and r.get(pass_name) is not None
    ]
    yes = sum(1 for v in values if v is True)
    no = sum(1 for v in values if v is False)
    bad = sum(1 for v in values if v is None)
    return f"{yes} / {no} / {bad}"


def _model_line(cfg: dict | None) -> list[str]:
    """The model the round ran (or would run) against, and its reasoning settings.

    The effort knobs are shown only when set, so a dry run reports the depth the
    round will actually be submitted at rather than leaving it to be inferred
    from the config file.
    """
    if not cfg:
        return []
    knobs = [f"thinking: {cfg.get('thinking')}"]
    knobs += [
        f"{name}: {cfg[name]}"
        for name in ("effort", "reasoning_effort", "thinking_budget")
        if cfg.get(name)
    ]
    return [f"  model: {cfg['model']} ({', '.join(knobs)})", ""]


def report(
    out: dict[str, list[dict]],
    counts: Counter,
    dry_run: bool,
    cfg: dict | None = None,
) -> str:
    lines = [
        "",
        "DRY RUN -- no API calls made, nothing written"
        if dry_run
        else "Predictions written",
        "",
    ]
    lines += _model_line(cfg)

    if dry_run:
        if counts["no_post_cut_content"]:
            lines.append(
                f"  {counts['no_post_cut_content']} moment(s) not classified: "
                "nothing after the cut point"
            )
            lines.append("")
        lines.append("  requests that would be sent:")
        for name in ("action_direction", "over_scaffolding"):
            lines.append(f"    {name:<28}{counts[name]:>5}")
        lines.append(
            f"    {'(over-scaffolding not asked)':<28}"
            f"{counts['over_scaffolding_not_asked']:>5}"
        )
        lines += _skip_breakdown(counts)
        lines.append("")
        return "\n".join(lines)

    header = (
        f"  {'split':<12}{'moments':>9}{'scaffolding':>16}{'rigor':>16}"
        f"{'over-scaffolding':>20}"
    )
    lines += [header, f"  {'-' * (len(header) - 2)}"]

    for stem, records in out.items():
        if not records:
            lines.append(f"  {stem:<12}{0:>9}")
            continue
        lines.append(
            f"  {stem:<12}{len(records):>9}"
            f"{_yes_no_counts(records, 'action_direction', 'scaffolding'):>16}"
            f"{_yes_no_counts(records, 'action_direction', 'rigor'):>16}"
            f"{_yes_no_counts(records, 'over_scaffolding', 'over_scaffolding'):>20}"
        )

    lines += ["", "  (counts are yes / no / unparsed)"]

    if counts["no_post_cut_content"]:
        lines.append(
            f"  {counts['no_post_cut_content']} moment(s) not classified: "
            "nothing after the cut point"
        )

    asked = counts["over_scaffolding"]
    not_asked = counts["over_scaffolding_not_asked"]
    lines.append(
        f"  over-scaffolding asked on {asked} moment(s); {not_asked} not asked"
    )
    lines += _skip_breakdown(counts)

    unparsed = counts["action_direction_unparsed"] + counts["over_scaffolding_unparsed"]
    if unparsed:
        lines += ["", f"  unparsed responses: {unparsed}"]
        for name in ("action_direction_unparsed", "over_scaffolding_unparsed"):
            if counts[name]:
                lines.append(f"    {name:<32}{counts[name]:>5}")

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
        prog="python -m tutormoments_build.v2.classify_excerpts",
        description="Run the v2 classification prompts over rendered moment excerpts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument("--excerpt-dir", default=DEFAULT_EXCERPT_DIR, metavar="DIR")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Scoring model to classify with, for comparing models on the same "
        "excerpts. Must be configured under `v2.models` in the config (or on "
        "the `models` tutor roster), which is where its thinking/effort "
        "settings come from. Default: the config's v2 model. Predictions land "
        "in <out-dir>/<model>/.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_STEMS,
        default=list(DEFAULT_SPLITS),
        help="Which excerpt splits to classify. Defaults to the iteration "
        "split alone; pass `test` explicitly to spend the held-out split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Classify only the first N moments of each split, for a smoke test",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        metavar="ID",
        help="Resume polling an in-flight batch instead of submitting a new one. "
        "Must be a batch submitted with the same excerpts and flags.",
    )
    parser.add_argument(
        "--print",
        dest="print_moment",
        nargs="?",
        const=RANDOM,
        metavar="MOMENT_ID",
        help="Print one moment's filled prompts to stdout and exit (id prefix "
        "accepted). Pass it bare to print a random moment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the prompts and report; make no API calls and write nothing",
    )
    return parser


def pick_moment(excerpts: dict[str, list[dict]], prefix: str) -> dict | None:
    """Find one moment to print: the first id matching ``prefix``, or a random one.

    ``prefix`` is RANDOM when ``--print`` was passed bare. Random picks are drawn
    across every loaded split, so ``--splits`` narrows the pool.
    """
    flat = [record for records in excerpts.values() for record in records]
    if prefix is RANDOM:
        return random.choice(flat) if flat else None
    return next((r for r in flat if r["moment_id"].startswith(prefix)), None)


def _print_prompts(excerpts: dict[str, list[dict]], prefix: str) -> int:
    record = pick_moment(excerpts, prefix)
    if record is None:
        if prefix is RANDOM:
            print("no moments loaded to pick from", file=sys.stderr)
        else:
            print(f"no moment matching {prefix!r}", file=sys.stderr)
        return 1

    action_template = resource_text(ACTION_DIRECTION_PROMPT)
    over_template = resource_text(OVER_SCAFFOLDING_PROMPT)

    # Naming the moment makes a random pick reproducible: the id printed here is
    # what `--print <id>` takes to render this exact moment again.
    print(f"===== moment {record['moment_id']} ({record['split']}) =====\n")
    print(
        f"===== {ACTION_DIRECTION_PROMPT} ({ACTION_CONTEXT_TURNS}-turn window) =====\n"
    )
    print(fill(action_template, excerpt_at(record, ACTION_CONTEXT_TURNS)))
    if wants_over_scaffolding(record):
        print(
            f"\n\n===== {OVER_SCAFFOLDING_PROMPT} "
            f"({OVER_SCAFFOLDING_CONTEXT_TURNS}-turn window) =====\n"
        )
        print(fill(over_template, excerpt_at(record, OVER_SCAFFOLDING_CONTEXT_TURNS)))
    else:
        print(
            f"\n\n===== {OVER_SCAFFOLDING_PROMPT} =====\n\n"
            f"not asked: {_SKIP_LABELS[skip_reason(record)]}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    excerpts = load_excerpts(args.excerpt_dir, stems=tuple(args.splits))
    if args.limit is not None:
        excerpts = {stem: records[: args.limit] for stem, records in excerpts.items()}

    if args.print_moment:
        return _print_prompts(excerpts, args.print_moment)

    # Resolved before the dry run too, so an unknown --model fails there rather
    # than only once a real round is submitted.
    cfg = phase_config(model=args.model)

    if args.dry_run:
        _, counts = build_entries(
            [record for records in excerpts.values() for record in records]
        )
        print(report({}, counts, dry_run=True, cfg=cfg))
        return 0

    logger.info(
        "classifying %d moment(s) with model=%s thinking=%s",
        sum(len(records) for records in excerpts.values()),
        cfg["model"],
        cfg.get("thinking"),
    )

    out, counts = classify(excerpts, cfg, batch_id=args.batch_id)

    out_dir = model_dir(args.out_dir, cfg["model"])
    for stem, records in out.items():
        logger.info(
            "wrote %d prediction(s) to %s",
            len(records),
            write_split(out_dir, stem, records),
        )

    print(report(out, counts, dry_run=False, cfg=cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
