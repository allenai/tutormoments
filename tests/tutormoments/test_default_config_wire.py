"""Pin the shipped config's effective wire behavior.

This is the machine-checkable statement of exactly what the packaged
default_config.yaml sends. If a config edit changes any row here, that is a
benchmark-condition change and must be deliberate (and called out in the PR).

The provider-native config migration changed zero wire bytes for the tutor
arms and the scorer/groundtruth roles. One deliberate change was made for the
thinking-off roles (student, taxonomy): they now send an explicit
`thinking: {"type": "disabled"}` instead of omitting the param. On the
Anthropic 4.x models these roles use, omission already meant thinking off, so
the *condition* is unchanged -- but omission means adaptive on Sonnet 5+, so
the explicit block keeps the condition stable under model swaps and lets
`tutormoments smoke` assert it (an omitted param judges as "na").
"""

import pytest

from tutormoments.config import (
    get_groundtruth_phase_config,
    load_config,
    resolve_arm,
    scorer_spec,
    student_spec,
    taxonomy_spec,
)
from tutormoments.models import resolve_thinking

ADAPTIVE = {"type": "adaptive"}
DISABLED = {"type": "disabled"}

# arm/role -> (model, anthropic_thinking, anthropic_effort,
#              gemini_thinking_config, openai_reasoning_effort)
EXPECTED_ARM_WIRE = {
    "claude-opus-4-8": ("claude-opus-4-8", ADAPTIVE, "xhigh", None, None),
    "claude-sonnet-4-6": ("claude-sonnet-4-6", ADAPTIVE, "high", None, None),
    "claude-sonnet-5": ("claude-sonnet-5", ADAPTIVE, "xhigh", None, None),
    "gemini-2.5-pro": (
        "gemini-2.5-pro",
        None,
        None,
        {"include_thoughts": True, "thinking_budget": -1},
        None,
    ),
    "gemini-3.5-flash": (
        "gemini-3.5-flash",
        None,
        None,
        {"include_thoughts": True, "thinking_budget": -1},
        None,
    ),
    "gpt-5.4-mini-2026-03-17": ("gpt-5.4-mini-2026-03-17", None, None, None, "high"),
    "gpt-5.5-2026-04-23": ("gpt-5.5-2026-04-23", None, None, None, "high"),
    "deepseek-v4-pro": (
        "deepseek-ai/DeepSeek-V4-Pro",
        None,
        None,
        None,
        None,
    ),
}


def _assert_wire(model, level, expected):
    exp_model, anth, effort, gem, oai = expected
    assert model == exp_model
    wire = resolve_thinking(model, level)
    assert wire.anthropic_thinking == anth
    assert wire.anthropic_effort == effort
    assert wire.gemini_thinking_config == gem
    assert wire.openai_reasoning_effort == oai


def test_default_roster_covers_expected_arms():
    load_config()
    arms = load_config()["benchmark_models"]
    assert set(arms) == set(EXPECTED_ARM_WIRE)


@pytest.mark.parametrize("arm_name", sorted(EXPECTED_ARM_WIRE))
def test_default_arm_wire(arm_name):
    arm = resolve_arm(arm_name)
    _assert_wire(arm.model, arm.thinking, EXPECTED_ARM_WIRE[arm_name])


def test_default_student_wire():
    # Deliberate change from the pre-migration config (see module docstring):
    # explicit disabled instead of omitting the param. Same condition on 4.x.
    spec = student_spec()
    _assert_wire(
        spec.model, spec.thinking, ("claude-opus-4-6", DISABLED, None, None, None)
    )


def test_default_scorer_wire():
    # Pre-migration: scorer thinking="adaptive" sent {"type": "adaptive"}.
    spec = scorer_spec()
    _assert_wire(
        spec.model, spec.thinking, ("claude-opus-4-6", ADAPTIVE, None, None, None)
    )


def test_default_taxonomy_wire():
    # Deliberate change from the pre-migration config (see module docstring):
    # explicit disabled instead of omitting the param. Same condition on 4.x.
    spec = taxonomy_spec()
    _assert_wire(
        spec.model, spec.thinking, ("claude-opus-4-8", DISABLED, None, None, None)
    )


def test_default_groundtruth_wire():
    # Pre-migration: groundtruth thinking="adaptive" sent {"type": "adaptive"}
    # (thinking_budget: 0 and reasoning_effort: "" were no-ops).
    spec = get_groundtruth_phase_config()
    _assert_wire(
        spec.model, spec.thinking, ("claude-opus-4-8", ADAPTIVE, None, None, None)
    )
