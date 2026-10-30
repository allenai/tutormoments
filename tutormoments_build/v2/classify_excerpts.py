"""Run the v2 classification prompts over human-human moment excerpts.

``excerpts.py`` creates excerpts.

Using prompts in ``tutormoments_build/prompts/v2/<version>/``:

  action_direction.md    scaffolding yes/no, rigor yes/no, plus a description
  over-scaffolding.md    over-scaffolding yes/no, plus a description

The prompts are versioned by directory and ``--prompt-version`` defaults to the newest version present,
and outputs are saved based on prompt version.

Together they predict ``scaffolding_present``, ``rigor_present``, ``over_scaffolding_present``,
in a manner that allows for comparison w/ gold labels.

**The over-scaffolding prompt is gated on the gold labels**.
It is sent only where the gold marks ``scaffolding_present``, and its answer is
scored against ``over_scaffolding_present``.

From prompt v8 the over-scaffolding prompt is also told what the adjudicators
wrote about the student before the cut (``{descriptions}``): the atomic facets
``decompose_situation.py`` extracted into ``data/ground_truth/student_state.jsonl``,
as a bullet list -- an oracle input like the gold-filled slots.

Both prompts are run in one batch, and both read the same excerpt: the whole
transcript from its first row through the moment's last
(the ``full`` rendering created by ``excerpts.py``). ``--passes`` runs only the
one(s) named, e.g. the over-scaffolding pass alone when only its prompt changed.

``--model`` sets the model; looking up settings in default_config.yaml.
Examples of models this script could run with:
- claude-opus-5
- gemini-3.5-flash
- gpt-5.6-sol

Predictions are written per model and prompt version:
``<out-dir>/<model>/<prompt version>/<split>.jsonl``. An existing file is never
replaced unless ``--overwrite`` says to.

Only the ``iteration`` split runs by default. The ``test`` split is held out
unless specified w/ ``--splits test``.

Usage::

    python -m tutormoments_build.v2.classify_excerpts --dry-run
    python -m tutormoments_build.v2.classify_excerpts --limit 20
    python -m tutormoments_build.v2.classify_excerpts
    python -m tutormoments_build.v2.classify_excerpts --splits test
    python -m tutormoments_build.v2.classify_excerpts --model claude-opus-5
    python -m tutormoments_build.v2.classify_excerpts --prompt-version 2
    python -m tutormoments_build.v2.classify_excerpts --passes over_scaffolding
    python -m tutormoments_build.v2.classify_excerpts --overwrite
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
from dataclasses import dataclass
from types import SimpleNamespace

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments.models import (
    NATIVE_THINKING_KEYS,
    ThinkingConfigError,
    infer_provider,
    resolve_thinking,
)
from tutormoments_build.resources import resource_path, resource_text
from tutormoments_build.v2 import excerpts

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

# Prompts are versioned: each revision is a numbered directory under
# PROMPT_ROOT holding a full set, and predictions are filed under the version
# that produced them. A prompt edit changes what the labels mean, so a run made
# before it and one made after it are not comparable and must not land in the
# same file.
PROMPT_ROOT = "prompts/v2"

# Version a bare run falls back to when the prompt root cannot be read at all.
# The real default is the newest version present -- see latest_prompt_version.
FALLBACK_PROMPT_VERSION = "1"

# Pass name -> its template's filename inside a version directory.
PROMPT_FILES = {
    "action_direction": "action_direction.md",
    "over_scaffolding": "over-scaffolding.md",
}

# Every pass, in the order a round runs them. ``--passes`` narrows a round to a
# subset, e.g. re-running only the over-scaffolding pass when only its prompt
# changed; a pass left out is recorded as not run rather than as failed.
PASSES = tuple(PROMPT_FILES)

# What both prompts read: the excerpt rendered with no lead-up window at all --
# the transcript from its first row through the moment's last. Neither question
# is settled by the immediate exchange alone (over-scaffolding asks whether the
# student had already shown they could do the work, and even action direction
# reads differently against what the tutor has already tried), so rather than
# guess a width that holds for both, they get the whole session. The excerpt
# file must have been built at this width (it is, by default).
EXCERPT_WIDTH = excerpts.FULL_KEY

# Gold-filled template slots. From prompt v5 the over-scaffolding prompt is told
# the teachers' situation judgement ({is_isnt}) and what the tutor did after the
# cut ({action}), both read from the gold labels -- an oracle condition, so its
# scores are an upper bound rather than comparable with versions that lack them.
# From v7 the situation judgement is {is_appropriate_for}, which names what the
# moment calls for rather than only whether it calls for rigor. From v8
# {descriptions} carries the student-state facets decompose_situation.py
# extracted from the adjudicators' accounts of the situation before the cut.
DESCRIPTIONS_SLOT = "descriptions"
GOLD_SLOTS = ("is_isnt", "is_appropriate_for", "action", DESCRIPTIONS_SLOT)

# Where an excerpt record carries its student-state facets once attached (see
# attach_descriptions). Set in memory only; excerpt files never hold it.
DESCRIPTION_KEY = "student_state"

# decompose_situation.py's output: one record per ground-truth moment.
DEFAULT_STUDENT_STATE = "data/ground_truth/student_state.jsonl"

# {descriptions} for a moment whose adjudicators wrote only about the tutor, so
# decomposition found no facet about the student.
NO_FACETS_LINE = "- (no description of the student)"

# Situation types in which teachers tend to judge the moment calls for rigor.
# Every other type, fifty_fifty included, fills {is_isnt} with "isn't".
RIGOR_SITUATIONS = ("rigor_only", "rigor_maj")

# situation_type -> {is_appropriate_for}. Unlike {is_isnt}, an even split gets
# its own wording instead of falling in with the scaffolding side.
APPROPRIATE_FOR_WORDING = {
    "rigor_only": "is appropriate for a push for rigor",
    "rigor_maj": "is appropriate for a push for rigor",
    "fifty_fifty": "is appropriate for both a push for rigor and scaffolding",
    "scaffold_maj": "is appropriate for scaffolding",
    "scaffold_only": "is appropriate for scaffolding",
}

# Any {name} placeholder in a template. The JSON output-format blocks never
# match: their braces open onto a newline or a quote, not a bare name.
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

# (scaffolding_present, rigor_present) -> {action}. Neither has no wording: the
# gold gate only asks about moments where scaffolding is present.
ACTION_WORDING = {
    (True, True): "scaffolding and pushing for rigor",
    (True, False): "scaffolding",
    (False, True): "pushing for rigor",
}

# Batch-entry key prefixes. Neither contains "__", and a moment_id is a UUID, so
# a key splits back apart on the first "__".
ACTION_PREFIX = "action"
OVER_SCAFFOLDING_PREFIX = "overscaffold"

ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# Sentinel for `--print` passed with no id: pick a moment at random.
RANDOM = object()


# ===========================================================================
# Prompt versions
# ===========================================================================


@dataclass(frozen=True)
class PromptSet:
    """One numbered revision of the v2 prompts, loaded from disk.

    ``paths`` is what a prediction record reports it was made with, so the file
    names the exact templates rather than only their version number.
    """

    version: str
    paths: dict[str, str]
    templates: dict[str, str]


def prompt_paths(version: str) -> dict[str, str]:
    """Package-relative paths to one version's templates, by pass name."""
    return {
        name: f"{PROMPT_ROOT}/{version}/{filename}"
        for name, filename in PROMPT_FILES.items()
    }


def available_prompt_versions() -> list[str]:
    """Version directories present under PROMPT_ROOT, numerically where possible.

    Used to pick the default version and to make a bad --prompt-version say what
    the alternatives are, so a missing prompt root is an empty list rather than
    an error -- the caller either has a fallback or is already reporting a
    failure and should not be derailed by a second one.
    """
    try:
        root = resource_path(PROMPT_ROOT)
        names = [entry.name for entry in root.iterdir() if entry.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        return []
    return sorted(names, key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else n))


def latest_prompt_version() -> str:
    """The newest numbered prompt version present under PROMPT_ROOT.

    This is what a bare run classifies with. A prompt revision is written to be
    run, and a default pinned to a number would leave a bare run quietly
    classifying with an older set and filing its predictions under that older
    version -- the edit never exercised, and the comparison it was made to
    settle never made.

    Numbered versions sort numerically, so this is v10 rather than v9 once there
    are ten of them. Unnumbered directories are ignored: a scratch copy left
    beside the numbered sets must not become what a bare run uses. Falls back to
    FALLBACK_PROMPT_VERSION when there is nothing to read, leaving the failure to
    ``load_prompt_set``, which reports it with the versions that do exist.
    """
    numbered = [name for name in available_prompt_versions() if name.isdigit()]
    return numbered[-1] if numbered else FALLBACK_PROMPT_VERSION


def load_prompt_set(version: str) -> PromptSet:
    """Load every template for one prompt version.

    Both templates are read up front so a version missing one fails here, before
    any moment is sent, rather than half way through a round that has already
    been paid for.
    """
    paths = prompt_paths(version)
    templates = {}
    for name, path in paths.items():
        try:
            templates[name] = resource_text(path)
        except (FileNotFoundError, NotADirectoryError) as exc:
            available = available_prompt_versions()
            raise FileNotFoundError(
                f"no {name} prompt at {path}; prompt versions available: "
                f"{', '.join(available) or '(none)'}"
            ) from exc
    return PromptSet(version=version, paths=paths, templates=templates)


# ===========================================================================
# Config
# ===========================================================================


@dataclass(frozen=True)
class V2Spec:
    """One classification round's model and the request parameters it runs at.

    ``thinking`` holds the provider-native parameters exactly as the config
    stated them -- ``{"thinking": {"type": "adaptive"}, "effort": "xhigh"}``
    for Anthropic, ``{"reasoning": "xhigh"}`` for OpenAI, ``{"thinking_budget":
    -1, "include_thoughts": True}`` for Gemini -- and is handed to ``run_batch``
    unchanged. Recorded on every prediction, so what a label was made at is
    readable off the record without consulting a mapping.
    """

    model: str
    provider: str
    thinking: dict
    poll_interval: int


def _native_thinking(block: str, model: str, entry: dict) -> dict:
    """Pull the provider-native thinking parameters out of a config entry.

    The legal keys depend on the model's provider, so an entry that names
    another vendor's knob (an OpenAI model carrying Anthropic's ``effort``) is
    a config error rather than a silently dropped setting. Validation is the
    runtime's own ``resolve_thinking``, and it runs at config-read time: a
    round that cannot be submitted as configured must fail before a batch is
    built, not after it is paid for.
    """
    allowed = NATIVE_THINKING_KEYS[infer_provider(model)]
    unknown = set(entry) - allowed
    if unknown:
        raise ThinkingConfigError(
            f"config {block}: unknown key(s) {sorted(unknown)} for model "
            f"{model}. Allowed: {sorted(allowed)}."
        )
    try:
        resolve_thinking(model, entry)
    except ThinkingConfigError as exc:
        raise ThinkingConfigError(f"config {block}: {exc}") from None
    return dict(entry)


def model_knobs(model: str, scoped: dict, config_path=None) -> tuple[str, str, dict]:
    """Request parameters for a ``--model`` id: ``v2.models``, then the arms.

    Returns ``(block name, model id, the entry's raw provider-native keys)``.
    The model id is resolved rather than echoed back: an arm may be named for
    the filesystem while carrying a vendor-prefixed id, and it is the id the
    client has to be pointed at.

    Scoring models for this round belong under ``v2.models``: they classify
    moments and are never replayed as tutors, while ``benchmark_models`` is the
    arm roster ``tutormoments run`` draws its candidates from. A model already
    on that roster needs no second entry -- it resolves to its arm, so it
    classifies at exactly the parameters the benchmark runs it with.

    A model in neither place raises rather than defaulting: an unconfigured
    model has no declared reasoning condition, and guessing one would misreport
    how a label was made. A model id carried by several arms is equally
    ambiguous -- the arms differ precisely in the parameters being asked for --
    so it raises too, naming them so the arm can be picked by name.
    """
    from tutormoments.config import load_config, resolve_arm

    if model in scoped:
        return f"v2.models.{model}", model, dict(scoped[model] or {})

    # An arm is named by its key but carries a model id, and the two differ
    # where an arm name had to be spelled for the filesystem
    # (deepseek-v4-pro -> deepseek-ai/DeepSeek-V4-Pro). Match either, with an
    # exact key match winning outright.
    raw_arms = load_config(config_path).get("benchmark_models") or {}
    if model in raw_arms:
        matches = [model]
    else:
        matches = [
            name
            for name, entry in raw_arms.items()
            if (entry or {}).get("model", name) == model
        ]
    if len(matches) > 1:
        raise ValueError(
            f"Model '{model}' is carried by several arms "
            f"({', '.join(matches)}), which differ in exactly the parameters "
            "being asked for. Name the arm instead."
        )
    if matches:
        arm = resolve_arm(matches[0], config_path)
        return f"benchmark_models.{arm.name}", arm.model, dict(arm.thinking)

    # Name both places a comparison model can be configured, so the fix is
    # obvious from the error.
    raise ValueError(
        f"Model '{model}' is configured nowhere. Models under the config "
        f"`v2.models`: {', '.join(scoped) or '(none)'}. Arms under "
        f"`benchmark_models`: {', '.join(raw_arms) or '(none)'}"
    )


def phase_config(config_path=None, model: str | None = None) -> V2Spec:
    """Return the v2 classification phase spec (model/thinking/poll_interval).

    ``model`` overrides the v2 block's model. Its request parameters then come
    wholesale from ``model_knobs``, never merged with the v2 block's: the
    entry is the override's *complete* reasoning condition, and inheriting
    anything would cross vendors -- an OpenAI model carrying the baseline's
    Anthropic ``effort`` -- or misreport the depth a label was made at. Every
    entry therefore states its condition in full; a provider with a reasoning
    knob and an entry that sets none of it is a config error, not a default.

    ``poll_interval`` is a property of the round, not of the model, so it stays
    whatever the v2 block says under an override.
    """
    from tutormoments.config import load_config

    # The config is the only place the scoring model and its parameters are
    # stated. No defaults are filled in from code: a fallback here would run a
    # round on a model, or at an effort, that no config names.
    block = load_config(config_path).get("v2")
    if not block or not block.get("model"):
        raise ValueError(
            "config has no `v2` block with a `model`. State the classification "
            "model and its provider-native reasoning parameters there "
            "(see docs/thinking.md)."
        )
    cfg = dict(block)

    # `v2.models` is a lookup table for --model, not a setting of the round.
    scoped = cfg.pop("models", None) or {}
    poll_interval = int(cfg.pop("poll_interval", 60))

    if model and model != cfg["model"]:
        block_name, model, entry = model_knobs(model, scoped, config_path)
    else:
        model = cfg.pop("model")
        block_name, entry = "v2", dict(cfg)

    return V2Spec(
        model=model,
        provider=infer_provider(model),
        thinking=_native_thinking(block_name, model, entry),
        poll_interval=poll_interval,
    )


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


def load_student_state(path: str = DEFAULT_STUDENT_STATE) -> dict[str, dict]:
    """Return {moment_id: student-state record} from decompose_situation.py.

    Records whose facets are None (no description, or an unparsed response)
    are left out, so filling a prompt for that moment raises (see
    ``description_slot``) rather than telling the model nothing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no student-state file at {path}; run "
            "`python -m tutormoments_build.v2.decompose_situation` first"
        )
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                if record.get("facets") is not None:
                    out[record["moment_id"]] = record
    return out


def attach_descriptions(
    excerpts: dict[str, list[dict]], states: dict[str, dict]
) -> None:
    """Set each excerpt record's student-state facets, in place.

    A moment with no facets is left without them; filling a prompt that asks
    for them then raises (see ``description_slot``).
    """
    for records in excerpts.values():
        for record in records:
            state = states.get(record["moment_id"])
            if state:
                record[DESCRIPTION_KEY] = state


# ===========================================================================
# Prompt building
# ===========================================================================


def fill(template: str, excerpt: str, slots: dict[str, str] | None = None) -> str:
    """Substitute ``{excerpt}`` and any gold ``slots`` into a v2 prompt template.

    ``str.replace``, not ``str.format``: both templates print a literal JSON
    object in their output-format section, and ``format`` would read those
    braces as fields and raise.

    Any placeholder left unfilled raises rather than reaching the model as a
    literal ``{is_isnt}`` -- including one this module has never heard of, so
    renaming a slot in a new prompt version fails loudly instead of silently.
    Every placeholder is substituted in one pass over the template, so neither
    transcript text nor a teacher's written account is itself searched for slot
    names.
    """
    values = {**(slots or {}), "excerpt": excerpt}
    unfilled = [
        name
        for name in dict.fromkeys(PLACEHOLDER.findall(template))
        if name not in values
    ]
    if unfilled:
        raise ValueError(f"template slot(s) left unfilled: {', '.join(unfilled)}")
    return PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def uses_gold_slots(template: str) -> bool:
    """Whether a template asks for any gold-filled slot."""
    return any("{" + name + "}" in template for name in GOLD_SLOTS)


def gold_slots(record: dict) -> dict[str, str]:
    """The gold-filled slot values for one moment.

    Raises rather than guessing when the moment lacks what a slot needs: a
    prompt telling the model the wrong situation would misreport the condition
    its label was made under.
    """
    situation = record.get("situation_type")
    if situation is None:
        raise KeyError(
            f"moment {record.get('moment_id')} has no situation_type; rebuild "
            "the excerpts with `python -m tutormoments_build.v2.excerpts`"
        )
    labels = record.get("labels") or {}
    key = (bool(labels.get("scaffolding_present")), bool(labels.get("rigor_present")))
    if key not in ACTION_WORDING:
        raise ValueError(
            f"moment {record.get('moment_id')}: gold marks neither scaffolding "
            "nor rigor, so there is no {action} to state"
        )
    if situation not in APPROPRIATE_FOR_WORDING:
        raise ValueError(
            f"moment {record.get('moment_id')}: unknown situation_type {situation!r}"
        )
    return {
        "is_isnt": "is" if situation in RIGOR_SITUATIONS else "isn't",
        "is_appropriate_for": APPROPRIATE_FOR_WORDING[situation],
        "action": ACTION_WORDING[key],
    }


def description_slot(record: dict) -> str:
    """The ``{descriptions}`` value: the student-state facets as a bullet list.

    Starts on a new line, since the templates put the slot after a colon. An
    empty facet list is NO_FACETS_LINE. A moment with no facets at all raises
    rather than leaving the slot blank: a prompt saying "teachers describe the
    situation as:" followed by nothing would misreport the condition its label
    was made under.
    """
    state = record.get(DESCRIPTION_KEY)
    if not state:
        raise KeyError(
            f"moment {record.get('moment_id')} has no student-state facets in "
            f"{DEFAULT_STUDENT_STATE}"
        )
    bullets = [f"- {facet}" for facet in state["facets"]] or [NO_FACETS_LINE]
    return "\n" + "\n".join(bullets)


def over_scaffolding_slots(template: str, record: dict) -> dict[str, str] | None:
    """Gold slots for the over-scaffolding prompt, or None if it takes none.

    Only the slots the template asks for, so the record states exactly what the
    model was told.
    """
    if not uses_gold_slots(template):
        return None
    values = gold_slots(record)
    if "{" + DESCRIPTIONS_SLOT + "}" in template:
        values[DESCRIPTIONS_SLOT] = description_slot(record)
    return {
        name: value for name, value in values.items() if "{" + name + "}" in template
    }


def excerpt_at(record: dict, width: str = EXCERPT_WIDTH) -> str:
    """The moment's excerpt rendered at ``width`` of lead-up.

    Excerpt files carry one rendering per width built, keyed by
    ``excerpts.width_key`` (see ``excerpts.DEFAULT_CONTEXT_WIDTHS``). A missing
    width is a build/consumer mismatch rather than a data problem, so it raises
    with the command that fixes it instead of silently falling back to a window
    the prompt was not written for -- an excerpt file built before ``full``
    existed would otherwise hand the prompts a truncated session.
    """
    available = record.get("excerpts") or {}
    rendered = available.get(width)
    if rendered is None:
        raise KeyError(
            f"moment {record.get('moment_id')} has no {width!r} excerpt "
            f"(built: {', '.join(sorted(available)) or 'none'}); rebuild with "
            f"`python -m tutormoments_build.v2.excerpts --context-turns {width}`"
        )
    return rendered["excerpt"]


def wants_over_scaffolding(record: dict) -> bool:
    """Whether the over-scaffolding prompt applies to this moment.

    Gold-gated -- see the module docstring. The prompt asks whether the tutor
    scaffolded too much, which presumes they scaffolded, so it is asked only
    where the gold marks ``scaffolding_present``. The gold never marks
    over-scaffolding without it, so the gate drops no positive.
    """
    labels = record.get("labels") or {}
    return bool(labels.get("scaffolding_present"))


def has_post_cut_content(record: dict) -> bool:
    """Whether there is anything after the cut point for a prompt to classify.

    Both prompts ask what the tutor does *after* the cut, so a moment ending at
    its own cut point gives them nothing to read and is not sent at all. Screen
    activity counts as content: an enrichment row like "[SCREEN INTERACTION]
    Tutor writes 3x7 on the board" is a pedagogical move, so a moment whose
    post-cut span is all enrichment is still classified.

    It does exclude moments: an annotator can draw the cut on the moment's last
    turn, often in a moment only one turn long. The run report counts what it
    actually excludes.

    Excerpt files built before ``post_cut_rows`` existed have no field to read;
    those moments are classified, and the caller warns.
    """
    return record.get("post_cut_rows", 1) > 0


def build_entries(
    records: list[dict], prompts: PromptSet, passes=PASSES
) -> tuple[list[dict], Counter]:
    """Build the batch entries for one round. Returns (entries, per-pass counts).

    Only the ``passes`` named get entries.
    """
    from tutormoments.client import build_batch_entry

    action_template = prompts.templates["action_direction"]
    over_template = prompts.templates["over_scaffolding"]

    entries: list[dict] = []
    counts: Counter = Counter()

    for record in records:
        moment_id = record["moment_id"]

        if "post_cut_rows" not in record:
            counts["post_cut_field_missing"] += 1
        if not has_post_cut_content(record):
            counts["no_post_cut_content"] += 1
            continue

        if "action_direction" in passes:
            entries.append(
                build_batch_entry(
                    f"{ACTION_PREFIX}__{moment_id}",
                    fill(action_template, excerpt_at(record)),
                    json_mode=True,
                )
            )
            counts["action_direction"] += 1

        if "over_scaffolding" not in passes:
            continue
        if wants_over_scaffolding(record):
            entries.append(
                build_batch_entry(
                    f"{OVER_SCAFFOLDING_PREFIX}__{moment_id}",
                    fill(
                        over_template,
                        excerpt_at(record),
                        over_scaffolding_slots(over_template, record),
                    ),
                    json_mode=True,
                )
            )
            counts["over_scaffolding"] += 1
        else:
            counts["over_scaffolding_not_asked"] += 1

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


def with_max_tokens(entry: dict, max_tokens: int) -> dict:
    """A copy of one batch entry asking for ``max_tokens`` of output.

    Copied, not mutated: the caller keeps the entries to parse results against.
    """
    request = entry["request"]
    config = {**request.get("generation_config", {}), "max_output_tokens": max_tokens}
    return {**entry, "request": {**request, "generation_config": config}}


def run_entries(
    entries: list[dict],
    cfg: V2Spec,
    batch_id: str | None = None,
    display_name: str = "v2_classify_excerpts",
) -> dict:
    """Submit one batch and return {key: {"text", "usage"} | {"error", ...}}.

    ``batch_id`` resumes polling an in-flight batch instead of submitting a new
    one; the entries passed must be the ones that batch was submitted with,
    since they drive result parsing.
    """
    from tutormoments.client import ModelClient, resolve_max_tokens, run_batch

    client = ModelClient(cfg.model)

    # No output cap of our own: every entry asks for the model's maximum.
    # build_batch_entry's default (65536) is half what Anthropic and OpenAI
    # allow, and a thinking model that exhausts it returns no text to parse.
    # Resolved here rather than passed as 0 because the Gemini batch path sends
    # the entry's value through as is.
    max_tokens = (
        resolve_max_tokens(SimpleNamespace(provider=cfg.provider, model=cfg.model), 0)
        or 0
    )
    entries = [with_max_tokens(entry, max_tokens) for entry in entries]

    def _created(created_id):
        logger.info(
            "batch %s submitted; resume with --batch-id %s", created_id, created_id
        )

    return run_batch(
        client,
        entries,
        json_mode=True,
        display_name=display_name,
        poll_interval=cfg.poll_interval,
        thinking=cfg.thinking,
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


def build_record(
    excerpt_record: dict,
    raw_entries: dict,
    cfg: V2Spec,
    prompts: PromptSet,
    passes=PASSES,
) -> dict:
    """Assemble one prediction record from an excerpt record and the batch results.

    ``over_scaffolding`` is None when the pass was not asked about this moment
    (the gold gate), which is distinct from a pass that ran and answered "no".
    A pass left out of ``passes`` is None too; ``passes`` on the record says
    which ran.

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
        if classified and "action_direction" in passes
        else None
    )

    over = None
    over_slots = None
    description_from = None
    if (
        classified
        and "over_scaffolding" in passes
        and wants_over_scaffolding(excerpt_record)
    ):
        over = _pass_result(
            raw_entries.get(f"{OVER_SCAFFOLDING_PREFIX}__{moment_id}"),
            parse_over_scaffolding,
        )
        over_slots = over_scaffolding_slots(
            prompts.templates["over_scaffolding"], excerpt_record
        )
        if over_slots and DESCRIPTIONS_SLOT in over_slots:
            state = excerpt_record[DESCRIPTION_KEY]
            description_from = {
                "adjudicators": state["adjudicators"],
                "situation_method": state["situation_method"],
                "model": state["model"],
                "prompt": state["prompt"],
            }

    return {
        "moment_id": moment_id,
        "transcript_id": excerpt_record["transcript_id"],
        "conversation_id": excerpt_record.get("conversation_id"),
        "split": excerpt_record["split"],
        "model": cfg.model,
        "provider": cfg.provider,
        # The request parameters the round actually ran with, in the provider's
        # own parlance and exactly as they went on the wire. Recorded because
        # they are part of how a label was produced: the same model at a pinned
        # effort and at the API default can answer a borderline moment
        # differently, and without them two rounds write identical-looking
        # metadata.
        "thinking": dict(cfg.thinking),
        # Which prompts wrote this label. The version is what the output path
        # is keyed on; the paths are kept beside it so a record still names the
        # templates if the directories are ever reorganised.
        "prompt_version": prompts.version,
        "prompts": dict(prompts.paths),
        "passes": list(passes),
        # Which rendering of the excerpt the prompts read. Both passes get the
        # same one; it is recorded per pass so a later round that splits them
        # apart again stays readable against this one.
        "context_turns": {
            "action_direction": EXCERPT_WIDTH,
            "over_scaffolding": EXCERPT_WIDTH,
        },
        # What the gold-filled slots told the model, per pass; None for a pass
        # whose template has none (or that was not asked).
        "prompt_slots": {"over_scaffolding": over_slots},
        # Whose accounts the {descriptions} facets came from, how they were
        # merged, and the model and prompt that extracted them; None when the
        # prompt has no such slot.
        "description_from": description_from,
        "situation_type": excerpt_record.get("situation_type"),
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
    cfg: V2Spec,
    prompts: PromptSet,
    *,
    batch_id: str | None = None,
    passes=PASSES,
) -> tuple[dict[str, list[dict]], Counter]:
    """Classify every split in one pooled batch.

    Returns ({split stem: [prediction record, ...]}, counts). All splits share
    the batch -- moment ids are unique across them, and the records are split
    apart again on the way out.
    """
    flat = [record for records in excerpts.values() for record in records]
    entries, counts = build_entries(flat, prompts, passes)

    raw_entries = run_entries(entries, cfg, batch_id=batch_id) if entries else {}

    out: dict[str, list[dict]] = {}
    for stem, records in excerpts.items():
        built = [
            build_record(record, raw_entries, cfg, prompts, passes)
            for record in records
        ]
        for record in built:
            if not record["classified"]:
                continue
            if record["action_direction"] and record["action_direction"]["parse_error"]:
                counts["action_direction_unparsed"] += 1
            if record["over_scaffolding"] and record["over_scaffolding"]["parse_error"]:
                counts["over_scaffolding_unparsed"] += 1
        out[stem] = built

    return out, counts


def prediction_dir(out_dir: str, model: str, prompt_version: str) -> str:
    """The output directory for one model's predictions at one prompt version.

    Predictions are filed under the model that made them so a comparison run
    does not overwrite the round before it, and under the prompt version below
    that so a prompt revision does not either -- an edited prompt asks a
    different question, and its answers are a different result set rather than a
    fresher copy of the old one. Model ids can carry a vendor prefix
    ("deepseek-ai/DeepSeek-V4-Pro"), which would otherwise nest a directory;
    the separator is flattened so every model gets exactly one level.
    """
    return os.path.join(out_dir, model.replace("/", "_"), prompt_version)


def existing_outputs(out_dir: str, stems) -> list[str]:
    """Prediction files in ``out_dir`` that a round over ``stems`` would replace."""
    paths = [os.path.join(out_dir, f"{stem}.jsonl") for stem in stems]
    return [path for path in paths if os.path.exists(path)]


def check_writable(out_dir: str, stems, overwrite: bool) -> None:
    """Refuse to clobber existing predictions unless --overwrite says to.

    Called before the batch is submitted, not before the write: a round costs
    real money, and discovering at the end that the results have nowhere to go
    would either throw them away or force the overwrite this check exists to
    prevent.
    """
    if overwrite:
        return
    existing = existing_outputs(out_dir, stems)
    if existing:
        raise FileExistsError(
            f"{len(existing)} prediction file(s) already here: "
            f"{', '.join(existing)}. Pass --overwrite to replace them, or "
            "--out-dir to write elsewhere."
        )


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
NOT_ASKED_LABEL = "gold says no scaffolding here"


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


def _model_line(cfg: "V2Spec | None") -> list[str]:
    """The model the round ran (or would run) against, and its reasoning settings.

    The parameters are printed in the provider's own parlance, so a dry run
    reports the depth the round will actually be submitted at rather than
    leaving it to be inferred from the config file.
    """
    if not cfg:
        return []
    knobs = ", ".join(f"{k}: {v}" for k, v in sorted(cfg.thinking.items()))
    return [f"  model: {cfg.model} ({knobs or 'no reasoning params sent'})", ""]


def _prompt_line(
    prompts: "PromptSet | None", out_dir: str | None, passes=PASSES
) -> list[str]:
    """The prompt version the round ran, and where its predictions land.

    Printed on dry runs too: the version decides both which templates are filled
    and which file is written, so it is the one setting worth confirming before
    paying for a round. A round narrowed by --passes says so.
    """
    if prompts is None:
        return []
    lines = [f"  prompts: v{prompts.version} ({PROMPT_ROOT}/{prompts.version}/)"]
    if tuple(passes) != PASSES:
        lines.append(f"  passes:  {', '.join(passes)} only")
    if out_dir:
        lines.append(f"  output:  {out_dir}/")
    return lines + [""]


def report(
    out: dict[str, list[dict]],
    counts: Counter,
    dry_run: bool,
    cfg: "V2Spec | None" = None,
    prompts: "PromptSet | None" = None,
    out_dir: str | None = None,
    passes=PASSES,
) -> str:
    lines = [
        "",
        "DRY RUN -- no API calls made, nothing written"
        if dry_run
        else "Predictions written",
        "",
    ]
    lines += _model_line(cfg)
    lines += _prompt_line(prompts, out_dir, passes)

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
            f"{counts['over_scaffolding_not_asked']:>5}  {NOT_ASKED_LABEL}"
        )
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
        f"  over-scaffolding asked on {asked} moment(s); {not_asked} not asked "
        f"({NOT_ASKED_LABEL})"
    )

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
        "excerpts. Must be configured under `v2.models` in the config, or be a "
        "`benchmark_models` arm (by arm name or model id); that entry is where "
        "its provider-native reasoning parameters come from. Default: the "
        "config's v2 model. Predictions land in <out-dir>/<model>/.",
    )
    parser.add_argument(
        "--prompt-version",
        default=latest_prompt_version(),
        metavar="VERSION",
        help="Which numbered prompt set under "
        f"tutormoments_build/{PROMPT_ROOT}/ to classify with. Defaults to the "
        "newest one present. Predictions land in <out-dir>/<model>/<version>/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace prediction files that are already there. Without it a "
        "round that would overwrite one refuses before making any API call.",
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
        "--passes",
        nargs="+",
        choices=PASSES,
        default=list(PASSES),
        help="Which passes to run. Name one to re-run it alone, e.g. "
        "`over_scaffolding` when only that prompt changed; the other pass is "
        "recorded as not run.",
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


def _print_prompts(
    excerpts: dict[str, list[dict]], prefix: str, prompts: PromptSet, passes=PASSES
) -> int:
    record = pick_moment(excerpts, prefix)
    if record is None:
        if prefix is RANDOM:
            print("no moments loaded to pick from", file=sys.stderr)
        else:
            print(f"no moment matching {prefix!r}", file=sys.stderr)
        return 1

    action_template = prompts.templates["action_direction"]
    over_template = prompts.templates["over_scaffolding"]
    action_path = prompts.paths["action_direction"]
    over_path = prompts.paths["over_scaffolding"]

    # Naming the moment makes a random pick reproducible: the id printed here is
    # what `--print <id>` takes to render this exact moment again.
    print(f"===== moment {record['moment_id']} ({record['split']}) =====\n")
    if "action_direction" in passes:
        print(f"===== {action_path} (@{EXCERPT_WIDTH} excerpt) =====\n")
        print(fill(action_template, excerpt_at(record)))
    if "over_scaffolding" not in passes:
        return 0
    if wants_over_scaffolding(record):
        print(f"\n\n===== {over_path} (@{EXCERPT_WIDTH} excerpt) =====\n")
        print(
            fill(
                over_template,
                excerpt_at(record),
                over_scaffolding_slots(over_template, record),
            )
        )
    else:
        print(f"\n\n===== {over_path} =====\n\nnot asked: {NOT_ASKED_LABEL}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    excerpts = load_excerpts(args.excerpt_dir, stems=tuple(args.splits))
    if args.limit is not None:
        excerpts = {stem: records[: args.limit] for stem, records in excerpts.items()}

    # Loaded before anything else uses it so an unknown --prompt-version fails
    # on every path, --print and --dry-run included.
    prompts = load_prompt_set(args.prompt_version)
    # Ordered as PASSES whatever order they were named in, and deduplicated.
    passes = tuple(name for name in PASSES if name in args.passes)

    # Before --print and --dry-run, which fill the prompts too. Only a round
    # filling {descriptions} reads the student-state file.
    over_template = prompts.templates["over_scaffolding"]
    if "over_scaffolding" in passes and "{" + DESCRIPTIONS_SLOT + "}" in over_template:
        attach_descriptions(excerpts, load_student_state())

    if args.print_moment:
        return _print_prompts(excerpts, args.print_moment, prompts, passes)

    # Resolved before the dry run too, so an unknown --model fails there rather
    # than only once a real round is submitted.
    cfg = phase_config(model=args.model)
    out_dir = prediction_dir(args.out_dir, cfg.model, prompts.version)

    if args.dry_run:
        _, counts = build_entries(
            [record for records in excerpts.values() for record in records],
            prompts,
            passes,
        )
        print(
            report(
                {},
                counts,
                dry_run=True,
                cfg=cfg,
                prompts=prompts,
                out_dir=out_dir,
                passes=passes,
            )
        )
        return 0

    # Before the batch, not after it: see check_writable.
    check_writable(out_dir, args.splits, args.overwrite)

    logger.info(
        "classifying %d moment(s) with model=%s thinking=%s prompts=v%s",
        sum(len(records) for records in excerpts.values()),
        cfg.model,
        cfg.thinking,
        prompts.version,
    )

    out, counts = classify(
        excerpts, cfg, prompts, batch_id=args.batch_id, passes=passes
    )

    for stem, records in out.items():
        logger.info(
            "wrote %d prediction(s) to %s",
            len(records),
            write_split(out_dir, stem, records),
        )

    print(
        report(
            out,
            counts,
            dry_run=False,
            cfg=cfg,
            prompts=prompts,
            out_dir=out_dir,
            passes=passes,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
