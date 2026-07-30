"""Pure situation/effectiveness classification helpers for the ground-truth build.

Only the pure, LLM-free helpers live here -- disk/storage/CLI machinery is
intentionally out of scope. The orchestration that calls these lives in
``tutorsim.groundtruth``.

The historical "label" terminology is dropped in favour of
"classify" to match the paper; the strategy-template loaders keep the
"labeller" name only because that is what the on-disk prompt directory and the
``groundtruth.labeller`` config key are called.
"""

import json
import logging
import re

from tutorsim_build.resources import resource_text

logger = logging.getLogger(__name__)

# Placeholder/test annotation text that should be skipped rather than sent to
# the model. Shared by the decompose/structure passes (tutorsim.scoring.JUNK_TEXTS
# carries the same set) and by the situation/effectiveness passes here.
JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}


# ===========================================================================
# Situation classification
# ===========================================================================

_SITUATION_PROMPT_RESOURCE = (
    "prompts/classify/situation_labeller/classify_scaffolding.md"
)

VALID_SITUATION_LABELS = {"yes", "no", "unclear", "no_mention"}


def _parse_situation_label(text: str) -> tuple[dict, bool]:
    """Parse a situation label from model output text.

    Returns (situation_label dict, had_error).
    Tries json.loads first; falls back to regex extraction field-by-field.
    A list-wrapped response (e.g. [{...}]) is unwrapped automatically.
    """

    def _coerce(val: str) -> str:
        v = val.strip().lower()
        return v if v in VALID_SITUATION_LABELS else "unclear"

    # --- attempt 1: standard JSON parse ---
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        return {
            "scaffolding": _coerce(str(parsed.get("scaffolding", "unclear"))),
            "rigor": _coerce(str(parsed.get("rigor", "unclear"))),
        }, False
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        pass

    # --- attempt 2: regex field extraction ---
    # Handles unquoted keys, unquoted values, and extra surrounding text.
    result = {}
    for field in ("scaffolding", "rigor"):
        m = re.search(rf'["\']?{field}["\']?\s*:\s*["\']?([a-z_]+)["\']?', text)
        result[field] = _coerce(m.group(1)) if m else "unclear"

    had_error = result["scaffolding"] == "unclear" and result["rigor"] == "unclear"
    return result, had_error


def load_situation_prompt() -> str:
    """Load the situation labeller prompt template."""
    logger.info("Loading situation labeller prompt: %s", _SITUATION_PROMPT_RESOURCE)
    return resource_text(_SITUATION_PROMPT_RESOURCE)


# ===========================================================================
# Effectiveness (strategy) classification
# ===========================================================================

_LABELLER_PROMPTS_DIR = "prompts/classify/labeller"

VALID_LABELS = {"effective", "partial", "ineffective"}
VALID_LABELS_BINARY = {"effective", "ineffective"}


def load_labeller_prompt(name: str) -> str:
    """Load a labeller prompt from the prompts/classify/labeller/ directory."""
    resource = f"{_LABELLER_PROMPTS_DIR}/{name}.txt"
    logger.info("Loading labeller prompt: %s", resource)
    return resource_text(resource)


def load_labeller_templates(labeller_cfg: "str | dict") -> "dict[str | None, str]":
    """Resolve the labeller config value to {annotation_type: template}.

    If `labeller_cfg` is a string (e.g. "classify_v2"), returns {None: <template>}
    -- the None key is the fallback used for every type. If it's a dict (e.g.
    {"scaffolding": "classify_scaffolding", ...}), returns one entry per type.
    """
    if isinstance(labeller_cfg, dict):
        return {
            ann_type: load_labeller_prompt(name)
            for ann_type, name in labeller_cfg.items()
        }
    return {None: load_labeller_prompt(labeller_cfg)}


def pick_template(templates: "dict[str | None, str]", annotation_type: str) -> str:
    """Pick the per-type template; fall back to None key if type is unmapped."""
    if annotation_type in templates:
        return templates[annotation_type]
    if None in templates:
        return templates[None]
    raise KeyError(
        f"No labeller template for annotation_type={annotation_type!r}. "
        f"Available keys: {list(templates.keys())}"
    )
