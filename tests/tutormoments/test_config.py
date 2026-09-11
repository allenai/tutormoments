import dataclasses

import pytest

from tutormoments import config as cfgmod
from tutormoments.config import ArmSpec
from tutormoments.models import ThinkingConfigError

# A complete, valid config skeleton for override tests. Individual tests
# append or replace blocks on top of it.
_BASE_ARM = (
    "benchmark_models:\n"
    "  claude-opus-4-8: { model: claude-opus-4-8, thinking: { type: adaptive }, "
    "effort: xhigh, condition: xhigh }"
)
_BASE_YAML = f"""
providers:
  anthropic: {{ env: ANTHROPIC_API_KEY }}
  openai:    {{ env: OPENAI_API_KEY }}
  gemini:    {{ env: GEMINI_API_KEY }}
  together:  {{ env: TOGETHER_API_KEY }}
{_BASE_ARM}
student: {{ model: claude-opus-4-6, mode: oracle, thinking: null }}
scorer:  {{ model: claude-opus-4-6, thinking: {{ type: adaptive }} }}
defaults: {{ trials: 1, max_turns: 2 }}
retry:    {{ max_retries: 1, base_delay: 1 }}
batch:    {{ timeout: 123 }}
"""


def _write_config(tmp_path, text):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    cfgmod._reset_config_cache()
    return config_path


def _with_roster(tmp_path, roster_yaml: str):
    return _write_config(tmp_path, _BASE_YAML.replace(_BASE_ARM, roster_yaml))


def test_packaged_default_config_parses_and_has_expected_roster():
    cfgmod._reset_config_cache()
    cfg = cfgmod.load_config()
    assert set(cfg["providers"]) == {"anthropic", "openai", "gemini", "together"}
    assert set(cfg["benchmark_models"]) == {
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "gemini-2.5-pro",
        "gemini-3.5-flash",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.5-2026-04-23",
        "deepseek-v4-pro",
    }
    assert cfg["benchmark_models"]["claude-opus-4-8"] == {
        "model": "claude-opus-4-8",
        "thinking": {"type": "adaptive"},
        "effort": "xhigh",
        "condition": "xhigh",
    }
    assert cfg["benchmark_models"]["gpt-5.5-2026-04-23"] == {
        "model": "gpt-5.5-2026-04-23",
        "reasoning": "high",
        "condition": "high",
    }
    assert cfg["benchmark_models"]["deepseek-v4-pro"] == {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "condition": "dynamic",
    }
    assert cfg["student"] == {
        "model": "claude-opus-4-6",
        "mode": "oracle",
        "thinking": {"type": "disabled"},
    }
    assert cfg["scorer"] == {
        "model": "claude-opus-4-6",
        "thinking": {"type": "adaptive"},
    }
    assert cfg["taxonomy"] == {
        "model": "claude-opus-4-8",
        "thinking": {"type": "disabled"},
        "batch_size": 50,
    }
    assert cfg["defaults"] == {"trials": 1, "max_turns": 5}
    assert cfg["retry"] == {"max_retries": 5, "base_delay": 5}
    assert cfg["batch"] == {"timeout": 86400}


def test_get_retry_config_reads_yaml():
    cfgmod._reset_config_cache()
    assert cfgmod.get_retry_config() == {"max_retries": 5, "base_delay": 5}


def test_get_batch_timeout_reads_yaml():
    cfgmod._reset_config_cache()
    assert cfgmod.get_batch_timeout() == 86400


def test_load_config_returns_parsed_dict():
    cfgmod._reset_config_cache()
    c = cfgmod.load_config()
    assert c["scorer"]["model"] == "claude-opus-4-6"
    assert "benchmark_models" in c and "providers" in c
    assert cfgmod.describe_config_source() == "tutormoments:default_config.yaml"


def test_load_config_accepts_explicit_override(tmp_path):
    config_path = _write_config(tmp_path, _BASE_YAML)
    cfg = cfgmod.load_config(config_path)
    assert cfg["defaults"]["max_turns"] == 2
    rc = cfgmod.build_run_config(
        tutors=["claude-opus-4-8"],
        config_path=config_path,
    )
    assert rc.max_turns == 2
    assert rc.config_source == str(config_path)


def test_resolve_arm_known():
    cfgmod._reset_config_cache()
    arm = cfgmod.resolve_arm("claude-opus-4-8")
    assert isinstance(arm, ArmSpec)
    assert arm.name == "claude-opus-4-8"
    assert arm.model == "claude-opus-4-8"
    assert arm.provider == "anthropic"
    assert arm.thinking == {"thinking": {"type": "adaptive"}, "effort": "xhigh"}
    assert arm.condition == "xhigh"


def test_resolve_arm_together():
    arm = cfgmod.resolve_arm("deepseek-v4-pro")
    assert arm.provider == "together"
    assert arm.model == "deepseek-ai/DeepSeek-V4-Pro"
    assert arm.thinking == {}
    assert arm.condition == "dynamic"


def test_resolve_arm_gemini():
    arm = cfgmod.resolve_arm("gemini-2.5-pro")
    assert arm.provider == "gemini"
    assert arm.thinking == {"include_thoughts": True, "thinking_budget": -1}
    assert arm.condition == "dynamic"


def test_resolve_arm_unknown_raises():
    with pytest.raises(ValueError, match="not in roster"):
        cfgmod.resolve_arm("gpt-9-imaginary")


def test_scorer_and_student_specs():
    cfgmod._reset_config_cache()
    assert cfgmod.scorer_spec().model == "claude-opus-4-6"
    assert cfgmod.scorer_spec().thinking == {"thinking": {"type": "adaptive"}}
    assert cfgmod.student_spec().thinking == {"thinking": {"type": "disabled"}}


def test_build_run_config_defaults():
    cfgmod._reset_config_cache()
    rc = cfgmod.build_run_config(tutors=["claude-opus-4-8"])
    assert rc.modes == ["plain", "scaffolding_rigor"]
    # The default config pins the released HF dataset; the runnable set defaults
    # to the "moments" config within it.
    assert rc.dataset == "allenai/tutormoments-preview"
    assert rc.data_path is None
    assert rc.dataset_config == "moments"
    assert rc.max_turns == 5 and rc.trials == 1
    # --seed was removed: it never seeded anything (artifact of an older
    # row-sampling implementation) and only misled about reproducibility.
    assert not hasattr(rc, "seed")
    assert rc.sample is None
    arm = rc.resolved_tutors["claude-opus-4-8"]
    assert isinstance(arm, ArmSpec)
    assert arm.model == "claude-opus-4-8"
    assert arm.thinking == {"thinking": {"type": "adaptive"}, "effort": "xhigh"}
    assert arm.condition == "xhigh"


def test_build_run_config_overrides():
    rc = cfgmod.build_run_config(
        tutors=["gpt-5.5-2026-04-23"], modes=["plain"], sample=10, trials=3
    )
    assert rc.modes == ["plain"] and rc.sample == 10 and rc.trials == 3
    arm = rc.resolved_tutors["gpt-5.5-2026-04-23"]
    assert isinstance(arm, ArmSpec)
    assert arm.model == "gpt-5.5-2026-04-23"
    assert arm.provider == "openai"
    assert arm.thinking == {"reasoning": "high"}
    assert arm.condition == "high"


def test_register_and_lookup_tutor():
    from tutormoments import register_tutor

    @register_tutor("my-model")
    def my_tutor(conversation):
        return "next turn"

    assert cfgmod.get_registered_tutor("my-model") is my_tutor
    rc = cfgmod.build_run_config(tutors=["my-model"])
    assert "my-model" in rc.tutors
    # A registered tutor has no model spec: it maps to None in resolved_tutors.
    assert rc.resolved_tutors["my-model"] is None


def test_register_and_lookup_student():
    from tutormoments import register_student

    @register_student("my-student")
    def my_student(conversation):
        return "student turn"

    assert cfgmod.get_registered_student("my-student") is my_student


def test_registered_tutor_lookup_returns_none_for_unknown():
    assert cfgmod.get_registered_tutor("nonexistent") is None


def test_registered_student_lookup_returns_none_for_unknown():
    assert cfgmod.get_registered_student("nonexistent") is None


def test_groundtruth_phase_config_shape():
    cfgmod._reset_config_cache()
    gt = cfgmod.get_groundtruth_phase_config()
    assert gt.model == "claude-opus-4-8"
    assert gt.thinking == {"thinking": {"type": "adaptive"}}
    assert gt.poll_interval == 60
    assert gt.labeller is not None


def test_get_labeller_config_routes_by_type():
    cfgmod._reset_config_cache()
    labeller = cfgmod.get_labeller_config()
    assert labeller == {
        "scaffolding": "classify_scaffolding",
        "rapport": "classify_rapport",
    }


# ---------------------------------------------------------------------------
# Arm roster contract tests.
# ---------------------------------------------------------------------------


def test_arm_model_key_distinct_and_shared_model(tmp_path):
    """An arm's `model:` may differ from its key; two arms may share a model."""
    config_path = _with_roster(
        tmp_path,
        "benchmark_models:\n"
        "  opus-deep:    { model: claude-opus-4-8, thinking: { type: adaptive }, "
        "effort: xhigh, condition: xhigh }\n"
        "  opus-shallow: { model: claude-opus-4-8, thinking: { type: adaptive }, "
        "effort: low, condition: low }",
    )
    deep = cfgmod.resolve_arm("opus-deep", config_path=config_path)
    shallow = cfgmod.resolve_arm("opus-shallow", config_path=config_path)
    assert deep.name == "opus-deep"
    assert deep.model == "claude-opus-4-8"
    # Provider is inferred from the model id, not the (arbitrary) arm key.
    assert deep.provider == "anthropic"
    assert deep.thinking == {"thinking": {"type": "adaptive"}, "effort": "xhigh"}
    # The two arms coexist as distinct conditions on the same model.
    assert shallow.model == deep.model
    assert shallow.thinking == {"thinking": {"type": "adaptive"}, "effort": "low"}


def test_arm_model_defaults_to_key(tmp_path):
    config_path = _with_roster(
        tmp_path,
        "benchmark_models:\n"
        "  claude-opus-4-8: { thinking: { type: adaptive }, effort: xhigh, "
        "condition: xhigh }",
    )
    arm = cfgmod.resolve_arm("claude-opus-4-8", config_path=config_path)
    assert arm.model == "claude-opus-4-8"
    assert arm.provider == "anthropic"


def test_benchmark_models_accept_provider_native_params(tmp_path):
    """benchmark_models states the exact provider-native knobs in config."""
    config_path = _with_roster(
        tmp_path,
        "benchmark_models:\n"
        "  gemini-low: { model: gemini-2.5-pro, thinking_budget: 4096, condition: low }\n"
        "  gpt-low:    { model: gpt-5.5-2026-04-23, reasoning: low, condition: low }\n"
        "  sonnet-hi:  { model: claude-sonnet-4-6, thinking: { type: adaptive }, "
        "effort: high, condition: high }",
    )
    gemini = cfgmod.resolve_arm("gemini-low", config_path=config_path)
    gpt = cfgmod.resolve_arm("gpt-low", config_path=config_path)
    sonnet = cfgmod.resolve_arm("sonnet-hi", config_path=config_path)
    assert gemini.thinking == {"thinking_budget": 4096}
    assert gemini.condition == "low"
    assert gpt.thinking == {"reasoning": "low"}
    assert gpt.condition == "low"
    assert sonnet.thinking == {"thinking": {"type": "adaptive"}, "effort": "high"}
    assert sonnet.condition == "high"


def test_benchmark_models_reject_provider_mismatched_keys(tmp_path):
    config_path = _with_roster(
        tmp_path,
        "benchmark_models:\n"
        "  gpt-bad: { model: gpt-5.5-2026-04-23, thinking_budget: 4096, condition: low }",
    )
    with pytest.raises(ThinkingConfigError, match="unknown key"):
        cfgmod.load_config(config_path)
    # ThinkingConfigError is a ValueError, so broad callers still catch it.
    assert issubclass(ThinkingConfigError, ValueError)


def test_benchmark_models_require_condition(tmp_path):
    config_path = _with_roster(
        tmp_path,
        "benchmark_models:\n  gpt-low: { model: gpt-5.5-2026-04-23, reasoning: low }",
    )
    with pytest.raises(ThinkingConfigError, match="`condition`"):
        cfgmod.load_config(config_path)


def test_benchmark_models_require_explicit_thinking(tmp_path):
    # An anthropic arm must state its thinking block (null = send nothing);
    # leaving it out entirely is not a benchmarkable condition.
    config_path = _with_roster(
        tmp_path,
        "benchmark_models:\n"
        "  opus-implicit: { model: claude-opus-4-8, condition: default }",
    )
    with pytest.raises(ThinkingConfigError, match="must set thinking"):
        cfgmod.load_config(config_path)


def test_retired_models_roster_rejected_with_hint(tmp_path):
    # The pre-release `models:` ladder roster must not load silently.
    config_path = _with_roster(
        tmp_path,
        "models:\n  claude-opus-4-8: { thinking: xhigh }",
    )
    with pytest.raises(ThinkingConfigError, match="benchmark_models"):
        cfgmod.load_config(config_path)


def test_retired_ladder_level_in_role_block_rejected(tmp_path):
    config_path = _write_config(
        tmp_path,
        _BASE_YAML.replace(
            "scorer:  { model: claude-opus-4-6, thinking: { type: adaptive } }",
            "scorer:  { model: claude-opus-4-6, thinking: dynamic }",
        ),
    )
    with pytest.raises(ThinkingConfigError, match="config scorer"):
        cfgmod.load_config(config_path)


def test_malformed_scorer_thinking_fails_at_load(tmp_path):
    config_path = _write_config(
        tmp_path,
        _BASE_YAML.replace(
            "scorer:  { model: claude-opus-4-6, thinking: { type: adaptive } }",
            "scorer:  { model: gemini-2.5-pro, thinking_budget: many }",
        ),
    )
    with pytest.raises(ThinkingConfigError, match="config scorer"):
        cfgmod.load_config(config_path)


def test_provider_mismatched_taxonomy_keys_fail_at_load(tmp_path):
    config_path = _write_config(
        tmp_path,
        _BASE_YAML + "taxonomy: { model: o3, thinking_budget: 4096, batch_size: 10 }\n",
    )
    with pytest.raises(ThinkingConfigError, match="config taxonomy"):
        cfgmod.load_config(config_path)


def test_groundtruth_requires_provider_native_keys(tmp_path):
    config_path = _write_config(
        tmp_path,
        _BASE_YAML
        + "groundtruth: { model: deepseek-ai/DeepSeek-V4-Pro, reasoning: high }\n",
    )
    with pytest.raises(ThinkingConfigError, match="config groundtruth"):
        cfgmod.load_config(config_path)


def test_registered_student_takes_no_thinking_keys(tmp_path):
    from tutormoments import register_student

    @register_student("scripted-student")
    def scripted(conversation):
        return "student turn"

    try:
        config_path = _write_config(
            tmp_path,
            _BASE_YAML.replace(
                "student: { model: claude-opus-4-6, mode: oracle, thinking: null }",
                "student: { model: scripted-student, mode: oracle }",
            ),
        )
        spec = cfgmod.student_spec(config_path)
        assert spec.model == "scripted-student"
        assert spec.thinking is None
    finally:
        cfgmod._STUDENT_REGISTRY.pop("scripted-student", None)


def test_specs_are_frozen():
    """Mutating a shared spec must raise, not silently poison the cache."""
    cfgmod._reset_config_cache()
    arm = cfgmod.resolve_arm("claude-opus-4-8")
    with pytest.raises(dataclasses.FrozenInstanceError):
        arm.thinking = {}
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfgmod.student_spec().model = "other-model"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfgmod.scorer_spec().thinking = {}
    # The cached spec is unchanged.
    assert cfgmod.resolve_arm("claude-opus-4-8").thinking == {
        "thinking": {"type": "adaptive"},
        "effort": "xhigh",
    }


def test_build_run_config_mixes_registered_and_roster_tutors(tmp_path):
    from tutormoments import register_tutor

    config_path = _write_config(tmp_path, _BASE_YAML)

    @register_tutor("scripted-tutor")
    def scripted(conversation):
        return "next turn"

    try:
        rc = cfgmod.build_run_config(
            tutors=["scripted-tutor", "claude-opus-4-8"],
            config_path=config_path,
        )
        assert rc.resolved_tutors["scripted-tutor"] is None
        arm = rc.resolved_tutors["claude-opus-4-8"]
        assert isinstance(arm, ArmSpec)
        assert arm.thinking == {"thinking": {"type": "adaptive"}, "effort": "xhigh"}
    finally:
        cfgmod._TUTOR_REGISTRY.pop("scripted-tutor", None)
