"""The ``v2_replay`` config block: the student, scorer and traits roles.

The tutor is a ``benchmark_models`` arm at that arm's own parameters; this
block states the other roles. ``tutormoments.config`` validates only the blocks
it knows, so this one is validated here, the way ``classify_excerpts`` validates
``v2``: every role names a model and its provider-native parameters, checked by
the runtime's own ``resolve_thinking`` when the config is read, so a bad
setting fails before any spend. Nothing is filled in from code.
"""

from dataclasses import dataclass

from tutormoments.models import infer_provider
from tutormoments_build.v2.classify_excerpts import _native_thinking

BLOCK = "v2_replay"
ROLES = ("student", "scorer", "traits")


@dataclass(frozen=True)
class RoleSpec:
    """One role's model and the provider-native parameters it runs at."""

    role: str
    model: str
    provider: str
    thinking: dict


def _block(config_path=None) -> dict:
    from tutormoments.config import load_config

    block = load_config(config_path).get(BLOCK)
    if not block:
        raise ValueError(
            f"config has no `{BLOCK}` block. State the student, scorer and "
            "traits models and their provider-native parameters there "
            "(see docs/thinking.md)."
        )
    return block


def role_spec(role: str, config_path=None) -> RoleSpec:
    """The spec for one ``v2_replay`` role, validated."""
    if role not in ROLES:
        raise ValueError(f"unknown {BLOCK} role {role!r}; choose from {ROLES}")
    entry = dict(_block(config_path).get(role) or {})
    model = entry.pop("model", None)
    if not model:
        raise ValueError(f"config {BLOCK}.{role} has no `model`.")
    return RoleSpec(
        role=role,
        model=model,
        provider=infer_provider(model),
        thinking=_native_thinking(f"{BLOCK}.{role}", model, entry),
    )


def poll_interval(config_path=None) -> int:
    """Seconds between batch polls for replay classification."""
    return int(_block(config_path).get("poll_interval", 60))
