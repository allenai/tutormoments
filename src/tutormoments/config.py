"""Config loading and tutor/student registries for tutormoments.

The config boundary normalizes every model-bearing block exactly once, at
load time, into frozen spec objects:

- the `models:` roster is a map of ARMS -- benchmark conditions. The key is
  the arm name; `model:` (defaulting to the key) names the provider model id;
  `thinking:` (required) states the arm's reasoning condition as a canonical
  ladder level. The same model may appear under several arms.
- the student/scorer/taxonomy/groundtruth blocks each name a model and its
  required `thinking:` level.

Every stated level is translated through the model registry
(tutormoments.models.resolve_thinking) HERE, so an unsatisfiable or
unregistered condition fails at config load -- before any tokens are spent --
for every block, not just the ones a particular run touches. The retired raw
knobs (boolean `thinking`, `thinking_budget`, `reasoning_effort`, `effort`)
are rejected with a migration hint.

Downstream code consumes the frozen specs; nothing re-reads or re-interprets
the raw YAML values, so validation-time and request-time semantics cannot
diverge.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from tutormoments.models import (
    ThinkingConfigError,
    ThinkingLevel,
    infer_provider,
    resolve_thinking,
)
from tutormoments.resources import resource_text

_CONFIG_CACHE = {}
_NORMALIZED_CACHE = {}
_TUTOR_REGISTRY: dict[str, callable] = {}
_STUDENT_REGISTRY: dict[str, callable] = {}
_CONFIG_ENV_VAR = "TUTORMOMENTS_CONFIG"
_DEFAULT_CONFIG_RESOURCE = "default_config.yaml"

# The retired per-provider knobs. Their meaning moved into the model registry
# (src/tutormoments/models.yaml); config states a ladder level instead.
_RAW_KNOB_KEYS = ("thinking_budget", "reasoning_effort", "effort")
_MIGRATION_HINT = (
    "the raw provider knobs were replaced by the thinking ladder "
    "(none/low/high/xhigh/dynamic); see README "
    '"Configuring thinking".'
)


# ---------------------------------------------------------------------------
# Normalized spec objects (frozen: shared safely from the config cache).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One benchmark arm: a model under a stated reasoning condition."""

    name: str
    model: str
    provider: str
    thinking: ThinkingLevel


@dataclass(frozen=True)
class StudentSpec:
    model: str
    mode: str
    thinking: ThinkingLevel


@dataclass(frozen=True)
class ScorerSpec:
    model: str
    thinking: ThinkingLevel


@dataclass(frozen=True)
class TaxonomySpec:
    model: str
    thinking: ThinkingLevel
    batch_size: int


@dataclass(frozen=True)
class GroundtruthSpec:
    model: str
    thinking: ThinkingLevel
    poll_interval: int
    # Labeller template routing: {annotation_type: prompt_name} or a single
    # prompt name for all types. Routing data, not LM configuration.
    labeller: "dict | str | None"


def _config_source(path: str | os.PathLike | None = None) -> tuple[str, str]:
    """Return the config source kind and identifier in precedence order."""
    if path is not None:
        return ("path", str(Path(path).expanduser()))

    env_path = os.environ.get(_CONFIG_ENV_VAR)
    if env_path:
        return ("path", str(Path(env_path).expanduser()))

    local_path = Path("config.yaml")
    if local_path.exists():
        return ("path", str(local_path))

    return ("package", _DEFAULT_CONFIG_RESOURCE)


def describe_config_source(path: str | os.PathLike | None = None) -> str:
    """Return a user-facing description of the config source that will be used."""
    kind, ident = _config_source(path)
    if kind == "package":
        return f"tutormoments:{ident}"
    return str(Path(ident))


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load, validate, and parse a Tutormoments config file, with caching.

    Model-bearing blocks are normalized and validated on first load (see
    module docstring): a config with an unsatisfiable thinking condition, a
    retired raw knob, or a missing required `thinking:` raises here.

    Args:
        path: Optional explicit config path. If omitted, precedence is
            TUTORMOMENTS_CONFIG, cwd/config.yaml, then packaged default_config.yaml.

    Returns:
        Parsed dict from yaml.safe_load() (the raw config; model-bearing
        blocks are additionally available as frozen specs via the accessors).
    """
    source = _config_source(path)
    if source not in _CONFIG_CACHE:
        kind, ident = source
        if kind == "package":
            content = resource_text(ident)
        else:
            file_path = Path(ident)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Tutormoments config file not found: {file_path}"
                )
            content = file_path.read_text(encoding="utf-8")
        raw = yaml.safe_load(content)
        # Normalize BEFORE caching: a config that fails validation must not
        # poison the cache as if it had loaded.
        _NORMALIZED_CACHE[source] = _normalize_config(raw)
        _CONFIG_CACHE[source] = raw
    return _CONFIG_CACHE[source]


def _normalized(path: str | os.PathLike | None = None) -> dict:
    """Return the normalized spec bundle for the active config."""
    load_config(path)
    return _NORMALIZED_CACHE[_config_source(path)]


def _reset_config_cache() -> None:
    """Clear the config cache (for testing)."""
    _CONFIG_CACHE.clear()
    _NORMALIZED_CACHE.clear()


# ---------------------------------------------------------------------------
# Normalization: raw YAML -> frozen specs, exactly once.
# ---------------------------------------------------------------------------


def _reject_raw_knobs(block_name: str, entry: dict) -> None:
    for knob in _RAW_KNOB_KEYS:
        if knob in entry:
            suggestion = _suggest_ladder(entry)
            hint = (
                f" (this entry maps to `thinking: {suggestion}`)" if suggestion else ""
            )
            raise ThinkingConfigError(
                f"config {block_name}: `{knob}` is no longer a config key -- "
                f"{_MIGRATION_HINT}{hint}"
            )


def _suggest_ladder(entry: dict) -> str | None:
    """Best-effort migration suggestion for a raw-knob entry."""
    thinking = entry.get("thinking")
    if thinking is False:
        return "none"
    effort = entry.get("effort") or entry.get("reasoning_effort")
    if isinstance(effort, str) and effort:
        return effort
    if entry.get("thinking_budget") == -1:
        return "dynamic"
    if thinking is True or thinking == "adaptive":
        return "dynamic"
    return None


def _require_thinking(block_name: str, entry: dict) -> ThinkingLevel:
    """Coerce the block's required `thinking:` value to a ladder level."""
    if "thinking" not in entry:
        raise ThinkingConfigError(
            f"config {block_name}: `thinking` is required -- every benchmarked "
            f"condition must be stated explicitly "
            f"(none/low/high/xhigh/dynamic)."
        )
    try:
        level = ThinkingLevel.coerce(entry["thinking"])
    except ThinkingConfigError as e:
        raise ThinkingConfigError(f"config {block_name}: {e}") from None
    return level


def _check_keys(block_name: str, entry: dict, allowed: set[str]) -> None:
    _reject_raw_knobs(block_name, entry)
    unknown = set(entry) - allowed
    if unknown:
        raise ThinkingConfigError(
            f"config {block_name}: unknown key(s) {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}."
        )


def _validate_wire(block_name: str, model: str, level: ThinkingLevel) -> None:
    """Fail fast if the stated condition cannot be honored on this model."""
    try:
        resolve_thinking(model, level)
    except ThinkingConfigError as e:
        raise ThinkingConfigError(f"config {block_name}: {e}") from None


def _normalize_arm(name: str, entry) -> ArmSpec:
    block = f"models.{name}"
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise ThinkingConfigError(f"config {block}: expected a mapping")
    _check_keys(block, entry, {"model", "thinking"})
    model = entry.get("model", name)
    level = _require_thinking(block, entry)
    _validate_wire(block, model, level)
    return ArmSpec(
        name=name, model=model, provider=infer_provider(model), thinking=level
    )


def _normalize_config(raw: dict) -> dict:
    """Build the frozen spec bundle from a parsed config dict."""
    bundle: dict = {"arms": {}}

    for name, entry in (raw.get("models") or {}).items():
        bundle["arms"][name] = _normalize_arm(name, entry)

    student = raw.get("student")
    if student is not None:
        _check_keys("student", student, {"model", "mode", "thinking"})
        level = _require_thinking("student", student)
        model = student["model"]
        # A registered (scripted) student is a code callable, not an API
        # model; its thinking level is stated but has no wire form to check.
        if get_registered_student(model) is None:
            _validate_wire("student", model, level)
        bundle["student"] = StudentSpec(
            model=model, mode=student.get("mode", ""), thinking=level
        )

    scorer = raw.get("scorer")
    if scorer is not None:
        _check_keys("scorer", scorer, {"model", "thinking"})
        level = _require_thinking("scorer", scorer)
        _validate_wire("scorer", scorer["model"], level)
        bundle["scorer"] = ScorerSpec(model=scorer["model"], thinking=level)

    taxonomy = raw.get("taxonomy")
    if taxonomy is not None:
        _check_keys("taxonomy", taxonomy, {"model", "thinking", "batch_size"})
        level = _require_thinking("taxonomy", taxonomy)
        _validate_wire("taxonomy", taxonomy["model"], level)
        bundle["taxonomy"] = TaxonomySpec(
            model=taxonomy["model"],
            thinking=level,
            batch_size=taxonomy.get("batch_size", 50),
        )

    groundtruth = raw.get("groundtruth")
    if groundtruth is not None:
        _check_keys(
            "groundtruth",
            groundtruth,
            {"model", "thinking", "poll_interval", "labeller"},
        )
        level = _require_thinking("groundtruth", groundtruth)
        _validate_wire("groundtruth", groundtruth["model"], level)
        bundle["groundtruth"] = GroundtruthSpec(
            model=groundtruth["model"],
            thinking=level,
            poll_interval=groundtruth.get("poll_interval", 60),
            labeller=groundtruth.get("labeller"),
        )

    return bundle


def register_tutor(name: str):
    """Decorator to register a tutor callable in the registry.

    Args:
        name: Unique name for the tutor.

    Returns:
        Decorator that stores the callable in _TUTOR_REGISTRY and returns it unchanged.

    The decorated callable must have signature: (conversation: list[dict]) -> str
    where conversation is the chat history and return value is the next turn text.
    """

    def decorator(func: callable) -> callable:
        _TUTOR_REGISTRY[name] = func
        return func

    return decorator


def register_student(name: str):
    """Decorator to register a student callable in the registry.

    Args:
        name: Unique name for the student.

    Returns:
        Decorator that stores the callable in _STUDENT_REGISTRY and returns it unchanged.

    The decorated callable must have signature: (conversation: list[dict]) -> str
    where conversation is the chat history and return value is the next turn text.
    """

    def decorator(func: callable) -> callable:
        _STUDENT_REGISTRY[name] = func
        return func

    return decorator


def get_registered_tutor(name: str) -> Optional[callable]:
    """Look up a registered tutor by name.

    Args:
        name: Tutor name to look up.

    Returns:
        The registered callable, or None if not found.
    """
    return _TUTOR_REGISTRY.get(name)


def get_registered_student(name: str) -> Optional[callable]:
    """Look up a registered student by name.

    Args:
        name: Student name to look up.

    Returns:
        The registered callable, or None if not found.
    """
    return _STUDENT_REGISTRY.get(name)


def get_retry_config(config_path: str | os.PathLike | None = None) -> dict:
    """Return retry configuration for ModelClient.generate().

    Reads from config: retry -> {max_retries, base_delay}.
    """
    return load_config(config_path)["retry"]


def get_batch_timeout(config_path: str | os.PathLike | None = None) -> int:
    """Return batch polling timeout in seconds.

    Reads from config: batch -> timeout.
    """
    return load_config(config_path)["batch"]["timeout"]


def resolve_arm(arm_name: str, config_path: str | os.PathLike | None = None) -> ArmSpec:
    """Resolve an arm name to its normalized spec.

    Args:
        arm_name: Arm identifier (a key of the config models roster).

    Returns:
        The frozen ArmSpec (name, model, provider, thinking).

    Raises:
        ValueError: If arm_name is not in the roster.
    """
    arms = _normalized(config_path)["arms"]
    if arm_name not in arms:
        raise ValueError(
            f"Arm '{arm_name}' not in roster. Valid arms: {', '.join(arms)}"
        )
    return arms[arm_name]


def student_spec(config_path: str | os.PathLike | None = None) -> StudentSpec:
    """Return the normalized student spec from config."""
    bundle = _normalized(config_path)
    if "student" not in bundle:
        raise ValueError("config has no student block")
    return bundle["student"]


def scorer_spec(config_path: str | os.PathLike | None = None) -> ScorerSpec:
    """Return the normalized scorer spec from config."""
    bundle = _normalized(config_path)
    if "scorer" not in bundle:
        raise ValueError("config has no scorer block")
    return bundle["scorer"]


def taxonomy_spec(config_path: str | os.PathLike | None = None) -> TaxonomySpec:
    """Return the normalized taxonomy classifier spec from config.

    Used by the action classifier that runs on every run and by
    `tutormoments taxonomy`.
    """
    bundle = _normalized(config_path)
    if "taxonomy" not in bundle:
        raise ValueError("config has no taxonomy block")
    return bundle["taxonomy"]


def get_groundtruth_phase_config(
    config_path: str | os.PathLike | None = None,
) -> GroundtruthSpec:
    """Return the normalized ground-truth build phase spec.

    Used by the dev-only `tutormoments dataset build-ground-truth` pipeline.
    """
    bundle = _normalized(config_path)
    if "groundtruth" not in bundle:
        raise ValueError("config has no groundtruth block")
    return bundle["groundtruth"]


def get_labeller_config(config_path: str | os.PathLike | None = None):
    """Return the effectiveness-labeller template routing.

    A dict ({annotation_type: prompt_name}) routes per type; a string loads a
    single template for all types. Reads from config: groundtruth -> labeller.
    """
    return get_groundtruth_phase_config(config_path).labeller


@dataclass
class RunConfig:
    """Configuration for a tutormoments run.

    Attributes:
        tutors: List of tutor arm names (or registered tutor names) to run.
        modes: List of evaluation modes (e.g., ["plain", "scaffolding_rigor"]).
        dataset: Hugging Face dataset id holding the released benchmark
            (None when running from a local release dir).
        data_path: Local release directory containing moments.jsonl
            (developer override; wins over dataset).
        dataset_revision: Pinned dataset revision (HF path only).
        dataset_config: Config name within the dataset holding the runnable
            moments set.
        sample: Number of samples from dataset (None = use all).
        trials: Number of trials per tutor/mode/sample.
        max_turns: Maximum turns per conversation.
        replay_concurrency: Number of per-moment replays to run concurrently
            within a cell. Result-preserving (only overlaps network round-trips).
        student: Normalized StudentSpec.
        scorer: Normalized ScorerSpec.
        resolved_tutors: Dict[arm name -> ArmSpec] for roster arms; a
            registered tutor maps to None (it has no model spec).
        config_source: Where the config was loaded from.
    """

    tutors: list[str]
    modes: list[str]
    dataset: str | None
    data_path: str | None
    dataset_revision: str | None
    dataset_config: str
    sample: int | None
    trials: int
    max_turns: int
    replay_concurrency: int
    student: StudentSpec
    scorer: ScorerSpec
    resolved_tutors: dict[str, ArmSpec | None]
    config_source: str


def build_run_config(
    *,
    tutors: list[str],
    modes: list[str] | None = None,
    dataset: str | None = None,
    data_path: str | None = None,
    dataset_revision: str | None = None,
    dataset_config: str | None = None,
    sample: int | None = None,
    trials: int | None = None,
    max_turns: int | None = None,
    replay_concurrency: int | None = None,
    config_path: str | os.PathLike | None = None,
) -> RunConfig:
    """Build a RunConfig from CLI arguments and config defaults.

    Args:
        tutors: List of tutor arm names (required).
        modes: Evaluation modes. Default: ["plain", "scaffolding_rigor"].
        dataset: Hugging Face dataset id. Default: config `dataset.id`.
        data_path: Local release dir with moments.jsonl (wins over dataset).
        dataset_revision: Pinned dataset revision. Default: config
            `dataset.revision`.
        dataset_config: Dataset config holding the runnable moments set.
            Default: config `dataset.config`, then "moments".
        sample: Number of samples to draw. Default: None (use all).
        trials: Number of trials. Default: read from config defaults.
        max_turns: Max turns per conversation. Default: read from config defaults.
        replay_concurrency: Concurrent per-moment replays within a cell.
            Default: read from config `execution.replay_concurrency`, then 4.
        config_path: Optional explicit config path.

    Returns:
        RunConfig with all fields filled.

    Raises:
        ValueError: If any tutor arm name is not in the roster.
    """
    cfg = load_config(config_path)
    d = cfg["defaults"]
    ds = cfg.get("dataset") or {}

    # Fill in defaults
    if modes is None:
        modes = ["plain", "scaffolding_rigor"]
    if dataset is None:
        dataset = ds.get("id")
    if dataset_revision is None:
        dataset_revision = ds.get("revision")
    if dataset_config is None:
        dataset_config = ds.get("config") or "moments"
    if trials is None:
        trials = d["trials"]
    if max_turns is None:
        max_turns = d["max_turns"]
    if replay_concurrency is None:
        # Fall back to 4 if the execution block is absent (older configs).
        replay_concurrency = (cfg.get("execution") or {}).get("replay_concurrency", 4)

    # Resolve tutors (check registry first, then roster). All thinking
    # validation already happened at load_config time.
    resolved_tutors: dict[str, ArmSpec | None] = {}
    for tutor_id in tutors:
        if get_registered_tutor(tutor_id) is not None:
            resolved_tutors[tutor_id] = None
        else:
            resolved_tutors[tutor_id] = resolve_arm(tutor_id, config_path=config_path)

    return RunConfig(
        tutors=tutors,
        modes=modes,
        dataset=dataset,
        data_path=data_path,
        dataset_revision=dataset_revision,
        dataset_config=dataset_config,
        sample=sample,
        trials=trials,
        max_turns=max_turns,
        replay_concurrency=replay_concurrency,
        student=student_spec(config_path),
        scorer=scorer_spec(config_path),
        resolved_tutors=resolved_tutors,
        config_source=describe_config_source(config_path),
    )
