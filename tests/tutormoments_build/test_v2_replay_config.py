"""The v2_replay config block: shipped roles and load-time validation."""

import pytest

from tutormoments.models import ThinkingConfigError, resolve_thinking
from tutormoments_build.v2 import replay_config as RC

# The exact request each shipped role sends. A config edit that changes a row
# here changes how replays are run or scored and must be deliberate.
EXPECTED = {
    "student": {"include_thoughts": True, "thinking_level": "low"},
    "scorer": {"include_thoughts": True, "thinking_level": "high"},
    "traits": {"include_thoughts": True, "thinking_level": "high"},
}


@pytest.mark.parametrize("role", RC.ROLES)
def test_shipped_roles_run_argon_at_their_stated_parameters(role):
    spec = RC.role_spec(role)

    assert spec.model == "gemini-4-argon"
    assert spec.provider == "gemini"
    assert spec.thinking == EXPECTED[role]
    # And the client can put it on the wire.
    assert resolve_thinking(spec.model, spec.thinking).provider == "gemini"


def test_poll_interval_is_read_from_the_block():
    assert RC.poll_interval() == 60


def _patch(monkeypatch, config):
    import tutormoments.config

    monkeypatch.setattr(tutormoments.config, "load_config", lambda path=None: config)


def test_a_cross_vendor_knob_is_refused(monkeypatch):
    _patch(
        monkeypatch,
        {"v2_replay": {"student": {"model": "gemini-4-argon", "effort": "high"}}},
    )
    with pytest.raises(ThinkingConfigError, match="v2_replay.student.*unknown key"):
        RC.role_spec("student")


def test_a_malformed_condition_is_refused(monkeypatch):
    """Shape is checked offline (level vocabularies differ per model and are
    proven live by smoke, see docs/thinking.md)."""
    entry = {"model": "gemini-4-argon", "thinking_level": "low", "thinking_budget": 0}
    _patch(monkeypatch, {"v2_replay": {"scorer": entry}})
    with pytest.raises(ThinkingConfigError, match="v2_replay.scorer"):
        RC.role_spec("scorer")


@pytest.mark.parametrize(
    "config", [{}, {"v2_replay": {}}, {"v2_replay": {"traits": {}}}]
)
def test_no_model_is_picked_in_code(monkeypatch, config):
    _patch(monkeypatch, config)
    with pytest.raises(ValueError, match="v2_replay"):
        RC.role_spec("traits")


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="unknown v2_replay role"):
        RC.role_spec("tutor")
