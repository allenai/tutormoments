"""Contract tests for the model registry and thinking-ladder translation.

The matrix below pins the exact wire fragment every (family, rung) pair
produces. These are benchmark-defining mappings: a change here changes what
gets sent to providers, so the expected values are written out literally
rather than derived.
"""

import pytest

from tutormoments.models import (
    ThinkingConfigError,
    ThinkingLevel,
    _validate_registry,
    get_pricing,
    infer_provider,
    max_output_cap,
    resolve_thinking,
)

# ---------------------------------------------------------------------------
# The ladder -> wire matrix, one row per (model, level) that must SUCCEED.
# Expected is (anthropic_thinking, anthropic_effort, gemini_thinking_config,
# openai_reasoning_effort).
# ---------------------------------------------------------------------------

ADAPTIVE = {"type": "adaptive"}


def _anth(thinking, effort=None):
    return (thinking, effort, None, None)


def _gem(config):
    return (None, None, config, None)


def _oai(effort):
    return (None, None, None, effort)


NOTHING = (None, None, None, None)

WIRE_MATRIX = [
    # --- anthropic 4.6 adaptive tier ---
    ("claude-opus-4-6", "none", NOTHING),
    ("claude-opus-4-6", "low", _anth(ADAPTIVE, "low")),
    ("claude-opus-4-6", "high", _anth(ADAPTIVE, "high")),
    ("claude-opus-4-6", "dynamic", _anth(ADAPTIVE)),
    ("claude-sonnet-4-6", "high", _anth(ADAPTIVE, "high")),
    # --- anthropic 4.7+ adaptive tier ---
    ("claude-opus-4-8", "none", NOTHING),
    ("claude-opus-4-8", "xhigh", _anth(ADAPTIVE, "xhigh")),
    ("claude-opus-4-8", "dynamic", _anth(ADAPTIVE)),
    # --- sonnet 5: off must be explicit ---
    ("claude-sonnet-5", "none", _anth({"type": "disabled"})),
    ("claude-sonnet-5", "xhigh", _anth(ADAPTIVE, "xhigh")),
    ("claude-sonnet-5", "dynamic", _anth(ADAPTIVE)),
    # --- anthropic legacy (enabled + budget) ---
    ("claude-haiku-4-5", "none", NOTHING),
    ("claude-haiku-4-5", "low", _anth({"type": "enabled", "budget_tokens": 4096})),
    ("claude-haiku-4-5", "high", _anth({"type": "enabled", "budget_tokens": 16384})),
    ("claude-3-7-sonnet", "high", _anth({"type": "enabled", "budget_tokens": 16384})),
    # --- gemini 2.5 pro (always-thinking) ---
    (
        "gemini-2.5-pro",
        "high",
        _gem({"include_thoughts": True, "thinking_budget": 16384}),
    ),
    (
        "gemini-2.5-pro",
        "dynamic",
        _gem({"include_thoughts": True, "thinking_budget": -1}),
    ),
    # --- gemini 2.5 flash (off is real) ---
    (
        "gemini-2.5-flash",
        "none",
        _gem({"include_thoughts": False, "thinking_budget": 0}),
    ),
    (
        "gemini-2.5-flash",
        "low",
        _gem({"include_thoughts": True, "thinking_budget": 4096}),
    ),
    (
        "gemini-2.5-flash",
        "dynamic",
        _gem({"include_thoughts": True, "thinking_budget": -1}),
    ),
    # --- gemini 3.x: dynamic keeps the proven budget -1 shape ---
    (
        "gemini-3.5-flash",
        "dynamic",
        _gem({"include_thoughts": True, "thinking_budget": -1}),
    ),
    # --- openai gpt-5 line ---
    ("gpt-5.5", "low", _oai("low")),
    ("gpt-5.5", "high", _oai("high")),
    ("gpt-5.4-mini", "high", _oai("high")),
    # --- openai o-series ---
    ("o3", "low", _oai("low")),
    ("o4", "high", _oai("high")),
    # --- together internal reasoners ---
    ("deepseek-ai/DeepSeek-V4-Pro", "dynamic", NOTHING),
]


@pytest.mark.parametrize("model,level,expected", WIRE_MATRIX)
def test_wire_matrix(model, level, expected):
    wire = resolve_thinking(model, level)
    anth_thinking, anth_effort, gem_config, oai_effort = expected
    assert wire.anthropic_thinking == anth_thinking
    assert wire.anthropic_effort == anth_effort
    assert wire.gemini_thinking_config == gem_config
    assert wire.openai_reasoning_effort == oai_effort
    assert wire.level == ThinkingLevel(level)


# ---------------------------------------------------------------------------
# Unsatisfiable rungs must raise with a reason.
# ---------------------------------------------------------------------------

UNSATISFIABLE = [
    ("gemini-2.5-pro", "none"),  # API rejects budget 0
    ("gemini-3.5-flash", "none"),  # thinking_level floor doesn't guarantee off
    ("o3", "none"),  # no off switch
    ("o3", "xhigh"),
    ("deepseek-ai/DeepSeek-V4-Pro", "none"),  # always-thinking
    ("deepseek-ai/DeepSeek-V4-Pro", "high"),  # no depth knob at all
    ("claude-opus-4-6", "xhigh"),  # 4.6 tier has no xhigh
    ("claude-haiku-4-5", "dynamic"),  # legacy models have no adaptive mode
]


@pytest.mark.parametrize("model,level", UNSATISFIABLE)
def test_unsatisfiable_rungs_raise(model, level):
    with pytest.raises(ThinkingConfigError) as exc_info:
        resolve_thinking(model, level)
    assert "does not support" in str(exc_info.value)


def test_unsatisfiable_error_carries_family_reason():
    with pytest.raises(ThinkingConfigError, match="rejects thinking_budget"):
        resolve_thinking("gemini-2.5-pro", "none")
    with pytest.raises(ThinkingConfigError, match="off switch"):
        resolve_thinking("o3-mini", "none")


# ---------------------------------------------------------------------------
# Unverified rungs are fail-closed until proven live.
# ---------------------------------------------------------------------------

UNVERIFIED = [
    ("gemini-3.5-flash", "low"),
    ("gemini-3.5-flash", "high"),
    ("gpt-5.5", "none"),
    ("gpt-5.5", "xhigh"),
]


@pytest.mark.parametrize("model,level", UNVERIFIED)
def test_unverified_rungs_raise(model, level):
    with pytest.raises(ThinkingConfigError, match="not been verified"):
        resolve_thinking(model, level)


# ---------------------------------------------------------------------------
# Registration and matching rules.
# ---------------------------------------------------------------------------


def test_unregistered_model_with_explicit_level_raises():
    with pytest.raises(ThinkingConfigError, match="not in the model registry"):
        resolve_thinking("brand-new-model-9000", "high")


def test_unregistered_model_with_none_level_is_a_noop():
    wire = resolve_thinking("claude-experimental-dev-model", None)
    assert wire.level is None
    assert wire.anthropic_thinking is None
    assert wire.anthropic_effort is None


def test_point_release_resolves_via_prefix():
    wire = resolve_thinking("claude-haiku-4-5-20251001", "high")
    assert wire.anthropic_thinking == {"type": "enabled", "budget_tokens": 16384}
    wire = resolve_thinking("gpt-5.4-mini-2026-03-17", "high")
    assert wire.openai_reasoning_effort == "high"
    wire = resolve_thinking("gpt-5.5-2026-04-23", "high")
    assert wire.openai_reasoning_effort == "high"


def test_longest_prefix_wins():
    # claude-opus-4-20250514 (legacy) must not be swallowed by any shorter
    # entry; and its dated id resolves to itself, not claude-opus-4-...
    wire = resolve_thinking("claude-opus-4-20250514", "high")
    assert wire.anthropic_thinking == {"type": "enabled", "budget_tokens": 16384}


def test_matching_is_case_insensitive():
    wire = resolve_thinking("Claude-Opus-4-8", "xhigh")
    assert wire.anthropic_effort == "xhigh"
    wire = resolve_thinking("GPT-5.5-2026-04-23", "high")
    assert wire.openai_reasoning_effort == "high"
    wire = resolve_thinking("deepseek-ai/deepseek-v4-pro", "dynamic")
    assert wire.provider == "together"


# ---------------------------------------------------------------------------
# ThinkingLevel.coerce: the config-boundary gate.
# ---------------------------------------------------------------------------


def test_coerce_accepts_ladder_strings_case_insensitively():
    assert ThinkingLevel.coerce("HIGH") is ThinkingLevel.HIGH
    assert ThinkingLevel.coerce("none") is ThinkingLevel.NONE
    assert ThinkingLevel.coerce(None) is None
    assert ThinkingLevel.coerce(ThinkingLevel.DYNAMIC) is ThinkingLevel.DYNAMIC


@pytest.mark.parametrize("bad", [True, False])
def test_coerce_rejects_booleans_with_migration_hint(bad):
    with pytest.raises(ThinkingConfigError, match="ladder"):
        ThinkingLevel.coerce(bad)


def test_coerce_rejects_unknown_strings():
    with pytest.raises(ThinkingConfigError, match="not a valid thinking level"):
        ThinkingLevel.coerce("adaptive")  # retired spelling; dynamic replaced it
    with pytest.raises(ThinkingConfigError, match="not a valid thinking level"):
        ThinkingLevel.coerce("enabled")


@pytest.mark.parametrize("dropped", ["minimal", "medium", "max"])
def test_coerce_rejects_dropped_rungs(dropped):
    # The ladder is deliberately small: these rungs were removed as unused.
    # Re-adding one is a reviewed registry line + a smoke verification.
    with pytest.raises(ThinkingConfigError, match="not a valid thinking level"):
        ThinkingLevel.coerce(dropped)


def test_coerce_rejects_numbers():
    with pytest.raises(ThinkingConfigError, match="registry"):
        ThinkingLevel.coerce(16384)


# ---------------------------------------------------------------------------
# Provider inference and per-model facts.
# ---------------------------------------------------------------------------


def test_infer_provider_registered_models():
    assert infer_provider("claude-sonnet-5") == "anthropic"
    assert infer_provider("gemini-3.5-flash") == "gemini"
    assert infer_provider("gpt-5.5-2026-04-23") == "openai"
    assert infer_provider("deepseek-ai/DeepSeek-V4-Pro") == "together"


def test_infer_provider_unregistered_falls_back_to_prefixes():
    assert infer_provider("claude-hypothetical-6") == "anthropic"
    assert infer_provider("gpt-4o") == "openai"
    assert infer_provider("o3-mini") == "openai"
    assert infer_provider("qwen/Qwen3-235B") == "together"


def test_infer_provider_is_case_insensitive():
    assert infer_provider("GEMINI-2.5-PRO") == "gemini"


def test_infer_provider_unknown_raises():
    with pytest.raises(ValueError, match="Cannot infer provider"):
        infer_provider("mystery-model")


def test_max_output_cap():
    assert max_output_cap("claude-haiku-4-5") == 64000
    assert max_output_cap("claude-haiku-4-5-20251001") == 64000
    assert max_output_cap("claude-opus-4-8") is None
    assert max_output_cap("not-registered") is None


def test_get_pricing_shape():
    # Pricing is schema-room for the cost-tracking workstream: present, dict,
    # empty until populated. Empty means "not priced yet", never "free".
    assert get_pricing("claude-opus-4-8") == {}
    assert get_pricing("unregistered-model") == {}


def test_describe_renders_wire_form():
    assert "adaptive" in resolve_thinking("claude-opus-4-8", "xhigh").describe()
    assert "effort=xhigh" in resolve_thinking("claude-opus-4-8", "xhigh").describe()
    assert "thinking_budget" in resolve_thinking("gemini-2.5-pro", "dynamic").describe()
    assert (
        resolve_thinking("deepseek-ai/DeepSeek-V4-Pro", "dynamic").describe()
        == "(none sent)"
    )


# ---------------------------------------------------------------------------
# Registry schema validation (bad registries must fail loudly at load).
# ---------------------------------------------------------------------------


def _minimal_registry():
    return {
        "families": {
            "fam": {
                "provider": "anthropic",
                "thinking": {
                    "mechanism": "anthropic-adaptive",
                    "rungs": {"none": "omit", "high": "high"},
                },
            }
        },
        "models": {"claude-x": {"family": "fam", "pricing": {}}},
    }


def test_validate_registry_accepts_minimal():
    assert _validate_registry(_minimal_registry())


def test_validate_registry_rejects_unknown_provider():
    bad = _minimal_registry()
    bad["families"]["fam"]["provider"] = "closedai"
    with pytest.raises(ValueError, match="unknown provider"):
        _validate_registry(bad)


def test_validate_registry_rejects_unknown_mechanism():
    bad = _minimal_registry()
    bad["families"]["fam"]["thinking"]["mechanism"] = "vibes"
    with pytest.raises(ValueError, match="unknown mechanism"):
        _validate_registry(bad)


def test_validate_registry_rejects_non_ladder_rung():
    bad = _minimal_registry()
    bad["families"]["fam"]["thinking"]["rungs"]["turbo"] = "high"
    with pytest.raises(ValueError, match="not a ladder level"):
        _validate_registry(bad)


def test_validate_registry_rejects_bad_rung_value_types():
    bad = _minimal_registry()
    bad["families"]["fam"]["thinking"]["rungs"]["high"] = 16384
    with pytest.raises(ValueError, match="anthropic-adaptive values"):
        _validate_registry(bad)


def test_validate_registry_rejects_unknown_family_reference():
    bad = _minimal_registry()
    bad["models"]["claude-x"]["family"] = "ghost"
    with pytest.raises(ValueError, match="unknown family"):
        _validate_registry(bad)


def test_validate_registry_rejects_unverified_not_in_rungs():
    bad = _minimal_registry()
    bad["families"]["fam"]["thinking"]["unverified"] = ["max"]
    with pytest.raises(ValueError, match="unverified rung"):
        _validate_registry(bad)


def test_packaged_registry_is_valid():
    # The shipped models.yaml must always pass its own validation.
    from tutormoments.models import _load_registry

    registry = _load_registry()
    assert "families" in registry and "models" in registry
