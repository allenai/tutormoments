"""Reads and checks the `v2_replay` config block: the student, scorer and traits models.

`tutormoments.config` doesn't check this block, so this module does when the
config is read. A bad setting fails before any API call. See README.md,
"Replays".
"""

from dataclasses import dataclass

from tutormoments.models import infer_provider
from tutormoments_build.v2.classify_excerpts import _native_thinking

BLOCK = "v2_replay"
ROLES = ("student", "scorer", "traits")


@dataclass(frozen=True)
class RoleSpec:
    """One role's model and its provider-native settings."""

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
    """The checked settings for one `v2_replay` role."""
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
    """Seconds to wait between checks on a running batch."""
    return int(_block(config_path).get("poll_interval", 60))
