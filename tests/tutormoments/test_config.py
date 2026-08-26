import dataclasses

import pytest

from tutormoments import config as cfgmod
from tutormoments.config import ArmSpec
from tutormoments.models import ThinkingConfigError, ThinkingLevel

# A complete, valid config skeleton for override tests. Individual tests
# append or replace blocks on top of it.
_BASE_YAML = """
providers:
  anthropic: { env: ANTHROPIC_API_KEY }
  openai:    { env: OPENAI_API_KEY }
  gemini:    { env: GEMINI_API_KEY }
  together:  { env: TOGETHER_API_KEY }
models:
  claude-opus-4-8: { thinking: xhigh }
student: { model: claude-opus-4-6, mode: oracle, thinking: none }
scorer:  { model: claude-opus-4-6, thinking: dynamic }
defaults: { trials: 1, max_turns: 2 }
retry:    { max_retries: 1, base_delay: 1 }
batch:    { timeout: 123 }
"""


def _write_config(tmp_path, text):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    cfgmod._reset_config_cache()
    return config_path


def test_packaged_default_config_parses_and_has_expected_roster():
    cfgmod._reset_config_cache()
    cfg = cfgmod.load_config()
    assert set(cfg["providers"]) == {"anthropic", "openai", "gemini", "together"}
    assert set(cfg["models"]) == {
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        "gemini-2.5-pro",
        "gemini-3.5-flash",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.5-2026-04-23",
        "deepseek-ai/DeepSeek-V4-Pro",
    }
    assert cfg["models"]["claude-opus-4-8"] == {"thinking": "xhigh"}
    assert cfg["models"]["claude-sonnet-5"] == {"thinking": "xhigh"}
    assert cfg["models"]["deepseek-ai/DeepSeek-V4-Pro"] == {"thinking": "dynamic"}
    assert cfg["student"] == {
        "model": "claude-opus-4-6",
        "mode": "oracle",
        "thinking": "none",
    }
    assert cfg["scorer"] == {"model": "claude-opus-4-6", "thinking": "dynamic"}
    assert cfg["taxonomy"] == {
        "model": "claude-opus-4-8",
        "thinking": "none",
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
    assert "models" in c and "providers" in c
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
    assert arm.thinking == ThinkingLevel.XHIGH
    assert arm.thinking == "xhigh"  # str-Enum: string comparison works


def test_resolve_arm_together():
    arm = cfgmod.resolve_arm("deepseek-ai/DeepSeek-V4-Pro")
    assert arm.provider == "together"
    assert arm.thinking == "dynamic"


def test_resolve_arm_gemini():
    arm = cfgmod.resolve_arm("gemini-2.5-pro")
    assert arm.provider == "gemini"
    assert arm.thinking == "dynamic"


def test_resolve_arm_unknown_raises():
    with pytest.raises(ValueError, match="not in roster"):
        cfgmod.resolve_arm("gpt-9-imaginary")


def test_scorer_and_student_specs():
    cfgmod._reset_config_cache()
    assert cfgmod.scorer_spec().model == "claude-opus-4-6"
    assert cfgmod.student_spec().thinking == ThinkingLevel.NONE


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
    assert arm.thinking == "xhigh"


def test_build_run_config_overrides():
    rc = cfgmod.build_run_config(
        tutors=["gpt-5.5-2026-04-23"], modes=["plain"], sample=10, trials=3
    )
    assert rc.modes == ["plain"] and rc.sample == 10 and rc.trials == 3
    arm = rc.resolved_tutors["gpt-5.5-2026-04-23"]
    assert isinstance(arm, ArmSpec)
    assert arm.model == "gpt-5.5-2026-04-23"
    assert arm.provider == "openai"
    assert arm.thinking == "high"


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
    assert gt.thinking == ThinkingLevel.DYNAMIC
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
# New-contract tests: the arm roster (thinking ladder registry).
# ---------------------------------------------------------------------------


def test_arm_model_key_distinct_and_shared_model(tmp_path):
    """An arm's `model:` may differ from its key; two arms may share a model."""
    config_path = _write_config(
        tmp_path,
        _BASE_YAML.replace(
            "models:\n  claude-opus-4-8: { thinking: xhigh }",
            "models:\n"
            "  opus-deep:    { model: claude-opus-4-8, thinking: xhigh }\n"
            "  opus-shallow: { model: claude-opus-4-8, thinking: low }",
        ),
    )
    deep = cfgmod.resolve_arm("opus-deep", config_path=config_path)
    shallow = cfgmod.resolve_arm("opus-shallow", config_path=config_path)
    assert deep.name == "opus-deep"
    assert deep.model == "claude-opus-4-8"
    # Provider is inferred from the model id, not the (arbitrary) arm key.
    assert deep.provider == "anthropic"
    assert deep.thinking == "xhigh"
    # The two arms coexist as distinct conditions on the same model.
    assert shallow.model == deep.model
    assert shallow.thinking == "low"


def test_arm_model_defaults_to_key(tmp_path):
    config_path = _write_config(tmp_path, _BASE_YAML)
    arm = cfgmod.resolve_arm("claude-opus-4-8", config_path=config_path)
    assert arm.model == "claude-opus-4-8"
    assert arm.provider == "anthropic"


def test_arm_missing_thinking_fails_at_load(tmp_path):
    config_path = _write_config(
        tmp_path,
        _BASE_YAML.replace(
            "claude-opus-4-8: { thinking: xhigh }", "claude-opus-4-8: {}"
        ),
    )
    with pytest.raises(ThinkingConfigError, match="`thinking` is required"):
        cfgmod.load_config(config_path)
    # ThinkingConfigError is a ValueError, so broad callers still catch it.
    assert issubclass(ThinkingConfigError, ValueError)


@pytest.mark.parametrize(
    "entry",
    [
        "{ thinking: xhigh, thinking_budget: 8192 }",
        "{ reasoning_effort: high }",
        "{ thinking: high, effort: high }",
    ],
)
def test_raw_knob_keys_rejected_with_migration_hint(tmp_path, entry):
    config_path = _write_config(
        tmp_path,
        _BASE_YAML.replace(
            "claude-opus-4-8: { thinking: xhigh }", f"claude-opus-4-8: {entry}"
        ),
    )
    with pytest.raises(ThinkingConfigError, match="thinking ladder"):
        cfgmod.load_config(config_path)


@pytest.mark.parametrize("value", ["true", "false"])
def test_boolean_thinking_rejected_with_hint(tmp_path, value):
    config_path = _write_config(
        tmp_path,
        _BASE_YAML.replace(
            "claude-opus-4-8: { thinking: xhigh }",
            f"claude-opus-4-8: {{ thinking: {value} }}",
        ),
    )
    with pytest.raises(ThinkingConfigError, match="replaced by the ladder"):
        cfgmod.load_config(config_path)


def test_unsatisfiable_scorer_level_fails_at_load(tmp_path):
    # gemini-2.5-pro is always-thinking: `none` has no wire form.
    config_path = _write_config(
        tmp_path,
        _BASE_YAML.replace(
            "scorer:  { model: claude-opus-4-6, thinking: dynamic }",
            "scorer:  { model: gemini-2.5-pro, thinking: none }",
        ),
    )
    with pytest.raises(ThinkingConfigError, match="config scorer"):
        cfgmod.load_config(config_path)


def test_unsatisfiable_taxonomy_level_fails_at_load(tmp_path):
    # The o-series has no off switch: `none` is unsatisfiable.
    config_path = _write_config(
        tmp_path,
        _BASE_YAML + "taxonomy: { model: o3, thinking: none, batch_size: 10 }\n",
    )
    with pytest.raises(ThinkingConfigError, match="config taxonomy"):
        cfgmod.load_config(config_path)


def test_unsatisfiable_groundtruth_level_fails_at_load(tmp_path):
    # DeepSeek reasons internally with no knob: only `dynamic` is expressible.
    config_path = _write_config(
        tmp_path,
        _BASE_YAML
        + "groundtruth: { model: deepseek-ai/DeepSeek-V4-Pro, thinking: high }\n",
    )
    with pytest.raises(ThinkingConfigError, match="config groundtruth"):
        cfgmod.load_config(config_path)


def test_specs_are_frozen():
    """Mutating a shared spec must raise, not silently poison the cache."""
    cfgmod._reset_config_cache()
    arm = cfgmod.resolve_arm("claude-opus-4-8")
    with pytest.raises(dataclasses.FrozenInstanceError):
        arm.thinking = ThinkingLevel.NONE
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfgmod.student_spec().model = "other-model"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfgmod.scorer_spec().thinking = ThinkingLevel.NONE
    # The cached spec is unchanged.
    assert cfgmod.resolve_arm("claude-opus-4-8").thinking == "xhigh"


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
        assert arm.thinking == "xhigh"
    finally:
        cfgmod._TUTOR_REGISTRY.pop("scripted-tutor", None)
