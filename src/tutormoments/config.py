"""Config loading and tutor/student registries for tutormoments.

The config boundary normalizes model-bearing blocks once, at load time, into
frozen spec objects. ``benchmark_models:`` is the tutor roster: each key is an
auditable benchmark arm, ``model`` names the provider model id, provider-
native thinking knobs state the exact request parameters, and ``condition``
gives the result grouping label. The student/scorer/taxonomy/groundtruth role
blocks state their thinking parameters the same provider-native way -- every
configured condition is readable in provider parlance without consulting a
mapping.

Every stated thinking config is validated through
tutormoments.models.resolve_thinking HERE, so a malformed or provider-
mismatched setting fails at config load -- before any tokens are spent -- for
every block, not just the ones a particular run touches. Downstream code
consumes the frozen specs; nothing re-reads or re-interprets the raw YAML
values, so validation-time and request-time semantics cannot diverge.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from tutormoments.models import (
    NATIVE_THINKING_KEYS,
    ThinkingConfigError,
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

_COMMON_ARM_KEYS = {"model", "condition"}


# ---------------------------------------------------------------------------
# Normalized spec objects (frozen: shared safely from the config cache).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One benchmark arm: a model under a stated reasoning condition."""

    name: str
    model: str
    provider: str
    # Exact provider-native thinking config (the keys the config stated).
    thinking: dict
    condition: str = ""


@dataclass(frozen=True)
class StudentSpec:
    model: str
    mode: str
    # Provider-native thinking config; None for registered (scripted)
    # students, which are code callables with no API wire form.
    thinking: dict | None


@dataclass(frozen=True)
class ScorerSpec:
    model: str
    thinking: dict


@dataclass(frozen=True)
class TaxonomySpec:
    model: str
    thinking: dict
    batch_size: int


@dataclass(frozen=True)
class GroundtruthSpec:
    model: str
    thinking: dict
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
    module docstring): a config with malformed or provider-mismatched
    thinking parameters in any arm or role block raises here.

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


def _block_provider(block_name: str, model: str) -> str:
    try:
        return infer_provider(model)
    except ValueError as e:
        raise ThinkingConfigError(f"config {block_name}: {e}") from None


def _native_thinking(
    block_name: str, model: str, entry: dict, extra_keys: set[str]
) -> dict:
    """Extract and validate the entry's provider-native thinking keys.

    Every model-bearing block states its thinking parameters in the provider's
    own parlance; the set of legal keys therefore depends on the model's
    provider. Unknown keys and malformed or provider-mismatched settings fail
    here, at config load.
    """
    provider = _block_provider(block_name, model)
    allowed = extra_keys | NATIVE_THINKING_KEYS[provider]
    unknown = set(entry) - allowed
    if unknown:
        raise ThinkingConfigError(
            f"config {block_name}: unknown key(s) {sorted(unknown)} for "
            f"provider {provider}. Allowed: {sorted(allowed)}."
        )
    thinking = {k: entry[k] for k in NATIVE_THINKING_KEYS[provider] if k in entry}
    try:
        resolve_thinking(model, thinking)
    except ThinkingConfigError as e:
        raise ThinkingConfigError(f"config {block_name}: {e}") from None
    return thinking


def _require_model(block_name: str, entry: dict, default: str | None = None) -> str:
    model = entry.get("model", default)
    if not isinstance(model, str) or not model:
        raise ThinkingConfigError(
            f"config {block_name}: `model` must be a non-empty string."
        )
    return model


def _condition(block_name: str, entry: dict) -> str:
    condition = entry.get("condition")
    if not isinstance(condition, str) or not condition:
        raise ThinkingConfigError(
            f"config {block_name}: `condition` must be a non-empty string."
        )
    return condition


def _normalize_arm(name: str, entry) -> ArmSpec:
    block = f"benchmark_models.{name}"
    if entry is None:
        entry = {}
    if not isinstance(entry, dict):
        raise ThinkingConfigError(f"config {block}: expected a mapping")
    model = _require_model(block, entry, default=name)
    thinking = _native_thinking(block, model, entry, _COMMON_ARM_KEYS)
    return ArmSpec(
        name=name,
        model=model,
        provider=infer_provider(model),
        condition=_condition(block, entry),
        thinking=thinking,
    )


def _normalize_config(raw: dict) -> dict:
    """Build the frozen spec bundle from a parsed config dict."""
    bundle: dict = {"arms": {}}

    if "models" in raw:
        raise ThinkingConfigError(
            "config: the `models:` roster was replaced by `benchmark_models:` "
            "(each arm states provider-native thinking parameters plus a "
            '`condition` label); see README "Running new tutor models".'
        )

    for name, entry in (raw.get("benchmark_models") or {}).items():
        bundle["arms"][name] = _normalize_arm(name, entry)

    student = raw.get("student")
    if student is not None:
        model = _require_model("student", student)
        # A registered (scripted) student is a code callable, not an API
        # model; it has no provider wire form and takes no thinking keys.
        if get_registered_student(model) is not None:
            unknown = set(student) - {"model", "mode"}
            if unknown:
                raise ThinkingConfigError(
                    f"config student: registered student '{model}' takes no "
                    f"thinking keys; unknown key(s) {sorted(unknown)}."
                )
            thinking = None
        else:
            thinking = _native_thinking("student", model, student, {"model", "mode"})
        bundle["student"] = StudentSpec(
            model=model, mode=student.get("mode", ""), thinking=thinking
        )

    scorer = raw.get("scorer")
    if scorer is not None:
        model = _require_model("scorer", scorer)
        bundle["scorer"] = ScorerSpec(
            model=model,
            thinking=_native_thinking("scorer", model, scorer, {"model"}),
        )

    taxonomy = raw.get("taxonomy")
    if taxonomy is not None:
        model = _require_model("taxonomy", taxonomy)
        bundle["taxonomy"] = TaxonomySpec(
            model=model,
            thinking=_native_thinking(
                "taxonomy", model, taxonomy, {"model", "batch_size"}
            ),
            batch_size=taxonomy.get("batch_size", 50),
        )

    groundtruth = raw.get("groundtruth")
    if groundtruth is not None:
        model = _require_model("groundtruth", groundtruth)
        bundle["groundtruth"] = GroundtruthSpec(
            model=model,
            thinking=_native_thinking(
                "groundtruth",
                model,
                groundtruth,
                {"model", "poll_interval", "labeller"},
            ),
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
        arm_name: Arm identifier (a key of the config benchmark_models roster).

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
