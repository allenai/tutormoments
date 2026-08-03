"""Config loading and tutor/student registries for tutormoments."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from tutormoments.client import infer_provider
from tutormoments.resources import resource_text

_CONFIG_CACHE = {}
_TUTOR_REGISTRY: dict[str, callable] = {}
_STUDENT_REGISTRY: dict[str, callable] = {}
_CONFIG_ENV_VAR = "TUTORMOMENTS_CONFIG"
_DEFAULT_CONFIG_RESOURCE = "default_config.yaml"


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
    """Load and parse a Tutormoments config file, with module-level caching.

    Args:
        path: Optional explicit config path. If omitted, precedence is
            TUTORMOMENTS_CONFIG, cwd/config.yaml, then packaged default_config.yaml.

    Returns:
        Parsed dict from yaml.safe_load().
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
        _CONFIG_CACHE[source] = yaml.safe_load(content)
    return _CONFIG_CACHE[source]


def _reset_config_cache() -> None:
    """Clear the config cache (for testing)."""
    _CONFIG_CACHE.clear()


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


def resolve_model(model_id: str, config_path: str | os.PathLike | None = None) -> dict:
    """Resolve a model ID to provider, env var, and kwargs.

    Args:
        model_id: Model identifier (must be in the config models roster).

    Returns:
        Dict with keys: provider (str), env (str), kwargs (dict).

    Raises:
        ValueError: If model_id is not in the roster.
    """
    cfg = load_config(config_path)
    if model_id not in cfg["models"]:
        valid_ids = list(cfg["models"].keys())
        raise ValueError(
            f"Model '{model_id}' not in roster. Valid models: {', '.join(valid_ids)}"
        )
    provider = infer_provider(model_id)
    env = cfg["providers"][provider]["env"]
    kwargs = cfg["models"][model_id]
    return {"provider": provider, "env": env, "kwargs": kwargs}


def student_spec(config_path: str | os.PathLike | None = None) -> dict:
    """Return the student spec block from config.

    Returns:
        Dict with model, mode, thinking keys.
    """
    return load_config(config_path)["student"]


def scorer_spec(config_path: str | os.PathLike | None = None) -> dict:
    """Return the scorer spec block from config.

    Returns:
        Dict with model and thinking keys.
    """
    return load_config(config_path)["scorer"]


def taxonomy_spec(config_path: str | os.PathLike | None = None) -> dict:
    """Return the taxonomy classifier spec block from config.

    Returns:
        Dict with model, thinking, and batch_size keys. Used by the action
        classifier that runs on every run and by `tutormoments taxonomy`.
    """
    return load_config(config_path)["taxonomy"]


def get_groundtruth_phase_config(config_path: str | os.PathLike | None = None) -> dict:
    """Return the ground-truth build phase config block.

    Reads from config: groundtruth -> {model, poll_interval, thinking,
    thinking_budget, reasoning_effort, labeller}. Used by the dev-only
    `tutormoments dataset build-ground-truth` pipeline.
    """
    return load_config(config_path)["groundtruth"]


def get_labeller_config(config_path: str | os.PathLike | None = None):
    """Return the effectiveness-labeller template routing.

    A dict ({annotation_type: prompt_name}) routes per type; a string loads a
    single template for all types. Reads from config: groundtruth -> labeller.
    """
    return load_config(config_path)["groundtruth"]["labeller"]


@dataclass
class RunConfig:
    """Configuration for a tutormoments run.

    Attributes:
        tutors: List of tutor model IDs to run.
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
        student: Student spec dict (model, mode, thinking).
        scorer: Scorer spec dict (model, thinking).
        resolved_tutors: Dict[model_id -> kwargs dict] for resolved tutors.
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
    student: dict
    scorer: dict
    resolved_tutors: dict[str, dict]
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
        tutors: List of tutor model IDs (required).
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
        ValueError: If any tutor model_id is not in the roster.
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

    # Resolve tutors (check registry first, then roster)
    resolved_tutors = {}
    for tutor_id in tutors:
        if get_registered_tutor(tutor_id) is not None:
            # Registered tutor: no kwargs needed
            resolved_tutors[tutor_id] = {}
        else:
            # Resolve via roster
            resolved = resolve_model(tutor_id, config_path=config_path)
            resolved_tutors[tutor_id] = resolved["kwargs"]

    # Get student and scorer specs
    student = student_spec(config_path)
    scorer = scorer_spec(config_path)

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
        student=student,
        scorer=scorer,
        resolved_tutors=resolved_tutors,
        config_source=describe_config_source(config_path),
    )
