"""Provider routing, per-model facts, and thinking-parameter translation.

The packaged ``models.yaml`` stores stable per-model facts: provider, pricing,
and output caps. It says nothing about reasoning conditions -- those are
stated explicitly, in provider parlance, wherever a model is configured
(``benchmark_models:`` arms and the role blocks), so the run config is
auditable without consulting a mapping.

Validation is fail-closed: a malformed provider-native thinking mapping raises
ThinkingConfigError at config-load time rather than letting a run proceed
under parameters the provider cannot honor or would silently ignore. Values
the provider vocabulary may extend (effort tiers, reasoning levels) are
shape-checked here and proven live with `tutormoments smoke`.

Deliberately SDK-free: config loading must not drag provider SDKs in.
"""

import threading
from copy import deepcopy
from dataclasses import dataclass

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

# The provider-native thinking keys config may state, per provider. These
# mirror exactly the knobs the client knows how to send; an unknown key is a
# config error, not a passthrough.
NATIVE_THINKING_KEYS = {
    "anthropic": {"thinking", "effort"},
    "gemini": {"thinking_budget", "thinking_level", "include_thoughts"},
    "openai": {"reasoning"},
    "together": set(),
}


class ThinkingConfigError(ValueError):
    """A thinking condition that cannot be honored or expressed."""


@dataclass(frozen=True)
class WireThinking:
    """The provider wire fragment one resolved thinking config produces.

    Exactly one provider-specific field is populated (or none, for the
    omit/noop cases). Consumers copy fields into their request kwargs; the
    dicts are freshly built per resolve call, never shared.
    """

    provider: str
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
    models = (raw or {}).get("models") or {}
    if not models:
        raise ValueError("models.yaml must define a non-empty models: map")

    for model_key, entry in models.items():
        entry = entry or {}
        provider = entry.get("provider")
        if provider not in _KNOWN_PROVIDERS:
            raise ValueError(
                f"models.yaml model '{model_key}': unknown provider {provider!r}"
            )
        cap = entry.get("max_output_cap")
        if cap is not None and (not isinstance(cap, int) or cap <= 0):
            raise ValueError(
                f"models.yaml model '{model_key}': max_output_cap must be a "
                f"positive integer, got {cap!r}"
            )
        pricing = entry.get("pricing")
        if pricing is not None and not isinstance(pricing, dict):
            raise ValueError(
                f"models.yaml model '{model_key}': pricing must be a mapping, "
                f"got {pricing!r}"
            )

    return raw


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


def infer_provider(model: str) -> str:
    """Infer provider from a model id.

    Registered models answer from the registry; unregistered ids fall back to
    the generic prefix table so routing (dev/direct client use) stays
    permissive while thinking translation stays strict.
    """
    found = _find_model_entry(model)
    if found is not None:
        return found[1]["provider"]
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


def resolve_thinking(model: str, thinking: dict | None) -> WireThinking:
    """Validate and translate provider-native thinking params from config.

    THE fail-closed chokepoint: called at config load (so malformed or
    unsupported settings die before any tokens are spent) and again by the
    client before the retry loop (so direct callers get the same contract).

    The mapping is intentionally small and mirrors the knobs this client knows
    how to send today:

    - Gemini: ``thinking_budget`` or ``thinking_level`` plus optional
      ``include_thoughts``.
    - OpenAI: ``reasoning`` (forwarded by the chat adapter as
      ``reasoning_effort``).
    - Anthropic: ``thinking`` (a provider thinking block, or null to omit the
      param) plus optional ``effort``.
    - Together/open-weight reasoners: no exposed knobs.

    thinking=None means "no stated condition": nothing is sent and nothing is
    checked. That path exists for non-benchmark direct calls only; every
    config-driven path passes an explicit mapping (config requires one).
    """
    if thinking is None:
        return WireThinking(provider=infer_provider(model))
    if not isinstance(thinking, dict):
        raise ThinkingConfigError(
            f"thinking config for '{model}' must be a mapping of provider-"
            f"native keys (or None for direct calls), got "
            f"{type(thinking).__name__}: {thinking!r}. The thinking-ladder "
            "levels were removed; state the provider's own parameters."
        )
    provider = infer_provider(model)
    params = dict(thinking)
    if provider == "gemini":
        return _resolve_gemini(params)
    if provider == "openai":
        return _resolve_openai(params)
    if provider == "anthropic":
        return _resolve_anthropic(params)
    if provider == "together":
        _check_native_keys(provider, params, NATIVE_THINKING_KEYS["together"])
        return WireThinking(provider=provider)
    raise ValueError(f"Unsupported provider: {provider}")


def _check_native_keys(provider: str, params: dict, allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ThinkingConfigError(
            f"{provider} thinking config has unknown key(s) {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}."
        )


def _resolve_gemini(params: dict) -> WireThinking:
    _check_native_keys("gemini", params, NATIVE_THINKING_KEYS["gemini"])
    if "thinking_budget" in params and "thinking_level" in params:
        raise ThinkingConfigError(
            "gemini thinking config cannot set both thinking_budget and thinking_level."
        )
    if "thinking_budget" not in params and "thinking_level" not in params:
        raise ThinkingConfigError(
            "gemini thinking config must set thinking_budget or thinking_level."
        )
    if "include_thoughts" in params and not isinstance(
        params["include_thoughts"], bool
    ):
        raise ThinkingConfigError("gemini include_thoughts must be true or false.")
    config: dict = {}
    if "include_thoughts" in params:
        config["include_thoughts"] = params["include_thoughts"]
    if "thinking_budget" in params:
        budget = params["thinking_budget"]
        if not isinstance(budget, int) or isinstance(budget, bool):
            raise ThinkingConfigError("gemini thinking_budget must be an integer.")
        config["thinking_budget"] = budget
        config.setdefault("include_thoughts", budget != 0)
    else:
        level = params["thinking_level"]
        if not isinstance(level, str) or not level:
            raise ThinkingConfigError(
                "gemini thinking_level must be a non-empty string."
            )
        config["thinking_level"] = level
        config.setdefault("include_thoughts", True)
    return WireThinking(provider="gemini", gemini_thinking_config=config)


def _resolve_openai(params: dict) -> WireThinking:
    _check_native_keys("openai", params, NATIVE_THINKING_KEYS["openai"])
    if "reasoning" not in params:
        raise ThinkingConfigError("openai thinking config must set reasoning.")
    reasoning = params["reasoning"]
    if not isinstance(reasoning, str) or not reasoning:
        raise ThinkingConfigError("openai reasoning must be a non-empty string.")
    return WireThinking(provider="openai", openai_reasoning_effort=reasoning)


def _resolve_anthropic(params: dict) -> WireThinking:
    _check_native_keys("anthropic", params, NATIVE_THINKING_KEYS["anthropic"])
    if "thinking" not in params:
        raise ThinkingConfigError(
            "anthropic thinking config must set thinking (a thinking block, "
            "or null to send no thinking param)."
        )
    thinking = params["thinking"]
    if thinking is None:
        thinking_config = None
    elif isinstance(thinking, dict):
        thinking_config = deepcopy(thinking)
    else:
        raise ThinkingConfigError("anthropic thinking must be a mapping or null.")
    effort = params.get("effort")
    if effort is not None and (not isinstance(effort, str) or not effort):
        raise ThinkingConfigError("anthropic effort must be a non-empty string.")
    if thinking_config is None:
        if effort is not None:
            raise ThinkingConfigError(
                "anthropic effort requires thinking.type adaptive."
            )
    else:
        kind = thinking_config.get("type")
        if kind not in {"adaptive", "enabled", "disabled"}:
            raise ThinkingConfigError(
                "anthropic thinking.type must be adaptive, enabled, or disabled."
            )
        if kind == "enabled":
            budget = thinking_config.get("budget_tokens")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise ThinkingConfigError(
                    "anthropic enabled thinking requires a positive integer "
                    "budget_tokens."
                )
        elif "budget_tokens" in thinking_config:
            raise ThinkingConfigError(
                "anthropic budget_tokens is only valid with thinking.type enabled."
            )
        if effort is not None and kind != "adaptive":
            raise ThinkingConfigError(
                "anthropic effort requires thinking.type adaptive."
            )
    return WireThinking(
        provider="anthropic",
        anthropic_thinking=thinking_config,
        anthropic_effort=effort,
    )
