"""Contract tests for the model registry and thinking-parameter translation.

The matrix below pins the exact wire fragment every provider-native thinking
config produces. These are benchmark-defining mappings: a change here changes
what gets sent to providers, so the expected values are written out literally
rather than derived.
"""

import pytest

from tutormoments.models import (
    ThinkingConfigError,
    _validate_registry,
    get_pricing,
    infer_provider,
    max_output_cap,
    resolve_thinking,
)

# ---------------------------------------------------------------------------
# The native-config -> wire matrix, one row per (model, thinking) that must
# SUCCEED. Expected is (anthropic_thinking, anthropic_effort,
# gemini_thinking_config, openai_reasoning_effort).
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
    # --- anthropic: thinking null means "send no thinking param" ---
    ("claude-opus-4-6", {"thinking": None}, NOTHING),
    (
        "claude-opus-4-6",
        {"thinking": ADAPTIVE, "effort": "high"},
        _anth(ADAPTIVE, "high"),
    ),
    (
        "claude-opus-4-8",
        {"thinking": ADAPTIVE, "effort": "xhigh"},
        _anth(ADAPTIVE, "xhigh"),
    ),
    ("claude-opus-4-8", {"thinking": ADAPTIVE}, _anth(ADAPTIVE)),
    (
        "claude-sonnet-5",
        {"thinking": {"type": "disabled"}},
        _anth({"type": "disabled"}),
    ),
    (
        "claude-haiku-4-5",
        {"thinking": {"type": "enabled", "budget_tokens": 16384}},
        _anth({"type": "enabled", "budget_tokens": 16384}),
    ),
    # --- gemini: budget or level; include_thoughts defaults on unless off ---
    (
        "gemini-2.5-pro",
        {"thinking_budget": 16384},
        _gem({"include_thoughts": True, "thinking_budget": 16384}),
    ),
    (
        "gemini-2.5-pro",
        {"include_thoughts": True, "thinking_budget": -1},
        _gem({"include_thoughts": True, "thinking_budget": -1}),
    ),
    (
        "gemini-2.5-flash",
        {"thinking_budget": 0},
        _gem({"include_thoughts": False, "thinking_budget": 0}),
    ),
    (
        "gemini-3.5-flash",
        {"thinking_level": "high"},
        _gem({"include_thoughts": True, "thinking_level": "high"}),
    ),
    # --- openai ---
    ("gpt-5.5", {"reasoning": "low"}, _oai("low")),
    ("gpt-5.5-2026-04-23", {"reasoning": "high"}, _oai("high")),
    ("o3", {"reasoning": "low"}, _oai("low")),
    # --- together internal reasoners: no knobs, nothing sent ---
    ("deepseek-ai/DeepSeek-V4-Pro", {}, NOTHING),
]


@pytest.mark.parametrize("model,thinking,expected", WIRE_MATRIX)
def test_wire_matrix(model, thinking, expected):
    wire = resolve_thinking(model, thinking)
    anth_thinking, anth_effort, gem_config, oai_effort = expected
    assert wire.anthropic_thinking == anth_thinking
    assert wire.anthropic_effort == anth_effort
    assert wire.gemini_thinking_config == gem_config
    assert wire.openai_reasoning_effort == oai_effort


# ---------------------------------------------------------------------------
# Malformed or provider-mismatched configs must raise before any tokens.
# ---------------------------------------------------------------------------


def test_rejects_mismatched_provider_keys():
    with pytest.raises(ThinkingConfigError, match="unknown key"):
        resolve_thinking("gpt-5.5-2026-04-23", {"thinking_budget": 4096})
    with pytest.raises(ThinkingConfigError, match="unknown key"):
        resolve_thinking("gemini-2.5-pro", {"reasoning": "low"})
    with pytest.raises(ThinkingConfigError, match="unknown key"):
        resolve_thinking("deepseek-ai/DeepSeek-V4-Pro", {"thinking_budget": -1})


def test_gemini_requires_exactly_one_depth_knob():
    with pytest.raises(ThinkingConfigError, match="cannot set both"):
        resolve_thinking(
            "gemini-2.5-pro", {"thinking_budget": 4096, "thinking_level": "low"}
        )
    with pytest.raises(ThinkingConfigError, match="must set"):
        resolve_thinking("gemini-2.5-pro", {})
    with pytest.raises(ThinkingConfigError, match="integer"):
        resolve_thinking("gemini-2.5-pro", {"thinking_budget": "lots"})
    with pytest.raises(ThinkingConfigError, match="include_thoughts"):
        resolve_thinking(
            "gemini-2.5-pro", {"thinking_budget": -1, "include_thoughts": "yes"}
        )


def test_openai_requires_reasoning():
    with pytest.raises(ThinkingConfigError, match="must set reasoning"):
        resolve_thinking("gpt-5.5", {})
    with pytest.raises(ThinkingConfigError, match="non-empty string"):
        resolve_thinking("gpt-5.5", {"reasoning": 3})


def test_anthropic_requires_explicit_thinking():
    # The key must be present -- null is the explicit "send nothing".
    with pytest.raises(ThinkingConfigError, match="must set thinking"):
        resolve_thinking("claude-opus-4-8", {})
    with pytest.raises(ThinkingConfigError, match="mapping or null"):
        resolve_thinking("claude-opus-4-8", {"thinking": "adaptive"})


def test_rejects_bad_anthropic_shapes():
    with pytest.raises(ThinkingConfigError, match="thinking.type"):
        resolve_thinking("claude-opus-4-8", {"thinking": {"type": "vibes"}})
    with pytest.raises(ThinkingConfigError, match="positive integer"):
        resolve_thinking("claude-opus-4-8", {"thinking": {"type": "enabled"}})
    with pytest.raises(ThinkingConfigError, match="only valid with"):
        resolve_thinking(
            "claude-opus-4-8",
            {"thinking": {"type": "adaptive", "budget_tokens": 4096}},
        )
    with pytest.raises(ThinkingConfigError, match="effort requires"):
        resolve_thinking(
            "claude-opus-4-8", {"thinking": {"type": "disabled"}, "effort": "high"}
        )
    with pytest.raises(ThinkingConfigError, match="effort requires"):
        resolve_thinking("claude-opus-4-8", {"thinking": None, "effort": "high"})


def test_rejects_retired_ladder_levels():
    # The canonical ladder was removed; a stray level string must not pass
    # silently as if it configured anything.
    with pytest.raises(ThinkingConfigError, match="must be a mapping"):
        resolve_thinking("claude-opus-4-8", "xhigh")
    with pytest.raises(ThinkingConfigError, match="must be a mapping"):
        resolve_thinking("gemini-2.5-pro", "dynamic")
    with pytest.raises(ThinkingConfigError, match="must be a mapping"):
        resolve_thinking("gpt-5.5", True)


def test_none_means_no_stated_condition():
    # Direct (non-benchmark) calls: nothing sent, nothing checked.
    wire = resolve_thinking("claude-experimental-dev-model", None)
    assert wire.anthropic_thinking is None
    assert wire.anthropic_effort is None
    assert wire.gemini_thinking_config is None
    assert wire.openai_reasoning_effort is None


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


def test_point_release_resolves_via_prefix():
    assert max_output_cap("claude-haiku-4-5-20251001") == 64000
    assert infer_provider("gpt-5.4-mini-2026-03-17") == "openai"


def test_matching_is_case_insensitive():
    assert max_output_cap("Claude-Haiku-4-5") == 64000
    assert infer_provider("deepseek-ai/deepseek-v4-pro") == "together"


def test_max_output_cap():
    assert max_output_cap("claude-haiku-4-5") == 64000
    assert max_output_cap("claude-opus-4-8") is None
    assert max_output_cap("not-registered") is None


def test_get_pricing_shape():
    # Pricing is schema-room for the cost-tracking workstream: present, dict,
    # empty until populated. Empty means "not priced yet", never "free".
    assert get_pricing("claude-opus-4-8") == {}
    assert get_pricing("unregistered-model") == {}


def test_describe_renders_wire_form():
    adaptive_xhigh = {"thinking": {"type": "adaptive"}, "effort": "xhigh"}
    assert "adaptive" in resolve_thinking("claude-opus-4-8", adaptive_xhigh).describe()
    assert (
        "effort=xhigh" in resolve_thinking("claude-opus-4-8", adaptive_xhigh).describe()
    )
    assert (
        "thinking_budget"
        in resolve_thinking("gemini-2.5-pro", {"thinking_budget": -1}).describe()
    )
    assert (
        resolve_thinking("deepseek-ai/DeepSeek-V4-Pro", {}).describe() == "(none sent)"
    )


# ---------------------------------------------------------------------------
# Registry schema validation (bad registries must fail loudly at load).
# ---------------------------------------------------------------------------


def _minimal_registry():
    return {
        "models": {
            "claude-x": {"provider": "anthropic", "pricing": {}},
        },
    }


def test_validate_registry_accepts_minimal():
    assert _validate_registry(_minimal_registry())


def test_validate_registry_rejects_unknown_provider():
    bad = _minimal_registry()
    bad["models"]["claude-x"]["provider"] = "closedai"
    with pytest.raises(ValueError, match="unknown provider"):
        _validate_registry(bad)


def test_validate_registry_rejects_missing_provider():
    bad = _minimal_registry()
    del bad["models"]["claude-x"]["provider"]
    with pytest.raises(ValueError, match="unknown provider"):
        _validate_registry(bad)


def test_validate_registry_rejects_bad_output_cap():
    bad = _minimal_registry()
    bad["models"]["claude-x"]["max_output_cap"] = -1
    with pytest.raises(ValueError, match="max_output_cap"):
        _validate_registry(bad)


def test_validate_registry_rejects_bad_pricing():
    bad = _minimal_registry()
    bad["models"]["claude-x"]["pricing"] = "cheap"
    with pytest.raises(ValueError, match="pricing"):
        _validate_registry(bad)


def test_validate_registry_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _validate_registry({"models": {}})


def test_packaged_registry_is_valid():
    # The shipped models.yaml must always pass its own validation.
    from tutormoments.models import _load_registry

    registry = _load_registry()
    assert "models" in registry
    assert "families" not in registry  # the thinking ladder is gone
