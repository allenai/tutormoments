"""The model registry: provider routing and thinking-ladder wire translation.

This module is the single source of truth for per-model facts. The data lives
in the packaged ``models.yaml`` (families = capability shapes, models =
per-model entries with pricing); this module loads it, validates it, and
translates the canonical thinking ladder into each provider's wire knob.

The translation is fail-closed: an explicit thinking level on a model that is
not registered, or a rung the model's family does not support, raises
ThinkingConfigError at config-load time rather than letting a run proceed
under a condition the provider cannot honor (or would silently ignore).

Deliberately SDK-free: config loading must not drag provider SDKs in.
"""

import threading
from dataclasses import dataclass
from enum import Enum

import yaml

from tutormoments.resources import resource_text

_REGISTRY_RESOURCE = "models.yaml"

# Provider routing fallback for model ids that are not registered. Routing is
# deliberately permissive (a dev can point a ModelClient at any claude-* id);
# thinking translation below is strict. Order matters (first match wins).
_PROVIDER_PREFIXES = [
    ("gemini", "gemini"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    ("deepseek-ai/", "together"),
    ("moonshotai/", "together"),
    ("minimaxai/", "together"),
    ("google/gemma", "together"),
    ("meta-llama/", "together"),
    ("qwen/", "together"),
]

_KNOWN_PROVIDERS = {"anthropic", "gemini", "openai", "together"}
_KNOWN_MECHANISMS = {
    "anthropic-adaptive",
    "anthropic-budget",
    "gemini",
    "openai-effort",
    "noop",
}
_ANTHROPIC_ADAPTIVE_SPECIALS = {"omit", "disabled", "adaptive"}
_ANTHROPIC_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
_LADDER_DOC = "none/minimal/low/medium/high/xhigh/max/dynamic"


class ThinkingConfigError(ValueError):
    """A thinking condition that cannot be honored or expressed."""


class ThinkingLevel(str, Enum):
    """The canonical thinking ladder.

    `none` means thinking verifiably off; `minimal`..`max` are explicit depth
    rungs; `dynamic` means the model decides (provider auto mechanisms).
    Each family maps the rungs it supports to its own wire knob in models.yaml;
    a missing rung is unsatisfiable there and rejected.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    DYNAMIC = "dynamic"

    @classmethod
    def coerce(cls, value) -> "ThinkingLevel | None":
        """Coerce a config value to a ladder level (None passes through).

        Rejects the retired raw-knob spellings with a migration hint. Booleans
        are called out specially because YAML 1.1 parses no/on/off as bools --
        a user who wrote `thinking: off` meant the string and needs the quoted
        ladder spelling instead.
        """
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ThinkingConfigError(
                f"thinking: {str(value).lower()} is not a valid thinking level. "
                f"Raw on/off knobs were replaced by the ladder ({_LADDER_DOC}); "
                "note YAML parses unquoted no/on/off as booleans, so spell the "
                "level as a plain word (e.g. thinking: none, thinking: high)."
            )
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                raise ThinkingConfigError(
                    f"thinking: {value!r} is not a valid thinking level. "
                    f"Valid levels: {_LADDER_DOC}."
                ) from None
        raise ThinkingConfigError(
            f"thinking: {value!r} is not a valid thinking level (got "
            f"{type(value).__name__}). Numeric budgets and provider knobs "
            f"moved into the model registry (models.yaml); config states one "
            f"of: {_LADDER_DOC}."
        )


@dataclass(frozen=True)
class WireThinking:
    """The provider wire fragment one resolved thinking level produces.

    Exactly one provider-specific field is populated (or none, for the
    omit/noop cases). Consumers copy fields into their request kwargs; the
    dicts are freshly built per resolve call, never shared.
    """

    provider: str
    level: "ThinkingLevel | None"
    # {"type": "adaptive"} | {"type": "disabled"} | {"type": "enabled",
    # "budget_tokens": N} | None (omit the param entirely).
    anthropic_thinking: dict | None = None
    # output_config.effort value; only ever set alongside adaptive thinking.
    anthropic_effort: str | None = None
    # thinking_config dict for generation_config | None (omit).
    gemini_thinking_config: dict | None = None
    # reasoning_effort value | None (omit).
    openai_reasoning_effort: str | None = None

    def describe(self) -> str:
        """Human-readable wire form, for logs and the smoke report."""
        if self.anthropic_thinking is not None:
            parts = [f"thinking={self.anthropic_thinking}"]
            if self.anthropic_effort:
                parts.append(f"effort={self.anthropic_effort}")
            return " ".join(parts)
        if self.gemini_thinking_config is not None:
            return f"thinking_config={self.gemini_thinking_config}"
        if self.openai_reasoning_effort is not None:
            return f"reasoning_effort={self.openai_reasoning_effort}"
        return "(none sent)"


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: dict | None = None


def _load_registry() -> dict:
    """Load and validate models.yaml once (module-level cache)."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = _validate_registry(
                    yaml.safe_load(resource_text(_REGISTRY_RESOURCE))
                )
    return _REGISTRY


def _reset_registry_cache() -> None:
    """Clear the registry cache (for testing)."""
    global _REGISTRY
    _REGISTRY = None


def _validate_registry(raw: dict) -> dict:
    """Validate the registry shape; raise ValueError naming the defect."""
    families = raw.get("families") or {}
    models = raw.get("models") or {}
    if not families or not models:
        raise ValueError("models.yaml must define non-empty families: and models:")

    for fam_name, fam in families.items():
        provider = fam.get("provider")
        if provider not in _KNOWN_PROVIDERS:
            raise ValueError(
                f"models.yaml family '{fam_name}': unknown provider {provider!r}"
            )
        thinking = fam.get("thinking") or {}
        mechanism = thinking.get("mechanism")
        if mechanism not in _KNOWN_MECHANISMS:
            raise ValueError(
                f"models.yaml family '{fam_name}': unknown mechanism {mechanism!r}"
            )
        rungs = thinking.get("rungs") or {}
        if not rungs:
            raise ValueError(f"models.yaml family '{fam_name}': empty rungs map")
        for rung_name, wire_value in rungs.items():
            try:
                ThinkingLevel(rung_name)
            except ValueError:
                raise ValueError(
                    f"models.yaml family '{fam_name}': '{rung_name}' is not a "
                    f"ladder level ({_LADDER_DOC})"
                ) from None
            _check_rung_value(fam_name, mechanism, rung_name, wire_value)
        for rung_name in thinking.get("unverified") or []:
            if rung_name not in rungs:
                raise ValueError(
                    f"models.yaml family '{fam_name}': unverified rung "
                    f"'{rung_name}' is not in the rungs map"
                )

    for model_key, entry in models.items():
        fam_name = (entry or {}).get("family")
        if fam_name not in families:
            raise ValueError(
                f"models.yaml model '{model_key}': unknown family {fam_name!r}"
            )
        cap = entry.get("max_output_cap")
        if cap is not None and (not isinstance(cap, int) or cap <= 0):
            raise ValueError(
                f"models.yaml model '{model_key}': max_output_cap must be a "
                f"positive integer, got {cap!r}"
            )

    return raw


def _check_rung_value(fam_name, mechanism, rung_name, wire_value) -> None:
    ctx = f"models.yaml family '{fam_name}' rung '{rung_name}'"
    if mechanism == "anthropic-adaptive":
        if (
            wire_value not in _ANTHROPIC_ADAPTIVE_SPECIALS
            and wire_value not in _ANTHROPIC_EFFORT_LEVELS
        ):
            raise ValueError(
                f"{ctx}: anthropic-adaptive values must be omit/disabled/"
                f"adaptive or an effort level, got {wire_value!r}"
            )
    elif mechanism == "anthropic-budget":
        if wire_value != "omit" and not (
            isinstance(wire_value, int)
            and not isinstance(wire_value, bool)
            and wire_value > 0
        ):
            raise ValueError(
                f"{ctx}: anthropic-budget values must be 'omit' or a positive "
                f"integer budget, got {wire_value!r}"
            )
    elif mechanism == "gemini":
        is_budget = isinstance(wire_value, int) and not isinstance(wire_value, bool)
        if not (is_budget or isinstance(wire_value, str)):
            raise ValueError(
                f"{ctx}: gemini values must be an integer thinking_budget or a "
                f"thinking_level string, got {wire_value!r}"
            )
    elif mechanism == "openai-effort":
        if not isinstance(wire_value, str):
            raise ValueError(
                f"{ctx}: openai-effort values must be reasoning_effort "
                f"strings, got {wire_value!r}"
            )
    elif mechanism == "noop":
        if wire_value is not None:
            raise ValueError(f"{ctx}: noop rungs must map to null")


def _find_model_entry(model: str) -> tuple[str, dict] | None:
    """Find the registry entry for a model id.

    Exact (case-insensitive) key match first, then the LONGEST matching
    prefix -- so a dated point release resolves to its base entry, and
    gemini-2.5-flash-lite would prefer a gemini-2.5-flash-lite entry over
    gemini-2.5-flash if both existed.
    """
    registry = _load_registry()
    models = registry["models"]
    model_lower = model.lower()
    by_lower = {key.lower(): (key, entry) for key, entry in models.items()}
    if model_lower in by_lower:
        return by_lower[model_lower]
    best = None
    for key_lower, (key, entry) in by_lower.items():
        if model_lower.startswith(key_lower):
            if best is None or len(key_lower) > len(best[0].lower()):
                best = (key, entry)
    return best


def _family_for(model: str) -> tuple[str, dict] | None:
    found = _find_model_entry(model)
    if found is None:
        return None
    _, entry = found
    fam_name = entry["family"]
    return fam_name, _load_registry()["families"][fam_name]


def infer_provider(model: str) -> str:
    """Infer provider from a model id.

    Registered models answer from the registry; unregistered ids fall back to
    the generic prefix table so routing (dev/direct client use) stays
    permissive while thinking translation stays strict.
    """
    family = _family_for(model)
    if family is not None:
        return family[1]["provider"]
    model_lower = model.lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Expected prefix: {', '.join(p for p, _ in _PROVIDER_PREFIXES)}"
    )


def max_output_cap(model: str) -> int | None:
    """Per-model max output token cap, or None when the provider default rules."""
    found = _find_model_entry(model)
    if found is None:
        return None
    return found[1].get("max_output_cap")


def get_pricing(model: str) -> dict:
    """Per-model pricing grid (per-MTok rates). Empty = not priced yet."""
    found = _find_model_entry(model)
    if found is None:
        return {}
    return dict(found[1].get("pricing") or {})


def resolve_thinking(model: str, level) -> WireThinking:
    """Translate a thinking level into the wire fragment for `model`.

    THE fail-closed chokepoint: called at config load (so unsatisfiable
    conditions die before any tokens are spent) and again by the client before
    the retry loop (so direct callers get the same contract).

    level=None means no stated condition -- nothing is sent and nothing is
    checked. That path exists for non-benchmark direct calls only; every
    config-driven path passes an explicit level (config requires one).
    """
    level = ThinkingLevel.coerce(level)
    if level is None:
        return WireThinking(provider=infer_provider(model), level=None)

    family = _family_for(model)
    if family is None:
        raise ThinkingConfigError(
            f"Model '{model}' is not in the model registry, so "
            f"thinking: {level.value} cannot be translated to a wire format. "
            "Add an entry for it in src/tutormoments/models.yaml (family + "
            "pricing) before benchmarking it."
        )
    fam_name, fam = family
    thinking = fam["thinking"]
    rungs = thinking["rungs"]

    if level.value not in rungs:
        notes = fam.get("notes")
        reason = f" {notes.strip()}" if notes else ""
        raise ThinkingConfigError(
            f"Model '{model}' (family {fam_name}) does not support "
            f"thinking: {level.value}.{reason} Supported levels: "
            f"{', '.join(rungs)}."
        )
    if level.value in (thinking.get("unverified") or []):
        raise ThinkingConfigError(
            f"thinking: {level.value} on '{model}' (family {fam_name}) has a "
            "documented wire form that has not been verified against the live "
            "API from this codebase. Verify it with `tutormoments smoke`, "
            "then remove the rung from the family's `unverified` list in "
            "src/tutormoments/models.yaml."
        )

    provider = fam["provider"]
    mechanism = thinking["mechanism"]
    wire_value = rungs[level.value]

    if mechanism == "anthropic-adaptive":
        if wire_value == "omit":
            return WireThinking(provider=provider, level=level)
        if wire_value == "disabled":
            return WireThinking(
                provider=provider,
                level=level,
                anthropic_thinking={"type": "disabled"},
            )
        if wire_value == "adaptive":
            return WireThinking(
                provider=provider,
                level=level,
                anthropic_thinking={"type": "adaptive"},
            )
        return WireThinking(
            provider=provider,
            level=level,
            anthropic_thinking={"type": "adaptive"},
            anthropic_effort=wire_value,
        )
    if mechanism == "anthropic-budget":
        if wire_value == "omit":
            return WireThinking(provider=provider, level=level)
        return WireThinking(
            provider=provider,
            level=level,
            anthropic_thinking={"type": "enabled", "budget_tokens": wire_value},
        )
    if mechanism == "gemini":
        if isinstance(wire_value, str):
            config = {"include_thoughts": True, "thinking_level": wire_value}
        elif wire_value == 0:
            config = {"include_thoughts": False, "thinking_budget": 0}
        else:
            config = {"include_thoughts": True, "thinking_budget": wire_value}
        return WireThinking(
            provider=provider, level=level, gemini_thinking_config=config
        )
    if mechanism == "openai-effort":
        return WireThinking(
            provider=provider, level=level, openai_reasoning_effort=wire_value
        )
    # noop
    return WireThinking(provider=provider, level=level)
