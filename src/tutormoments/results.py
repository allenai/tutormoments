"""Run-results store for tutormoments benchmark runs.

On-disk layout::

    results/<run_id>/
        config.json
        transcripts/<scenario_id>.json
        scores/<scenario_id>.json
        summary.json

All JSON is UTF-8.  Pass ``results_root`` to redirect to a tmp dir in tests.
"""

import json
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Run-id naming
# ---------------------------------------------------------------------------


def make_run_id(tutor: str, mode: str, dataset: str, date: str) -> str:
    """Build a self-documenting run id: ``{tutor}_{mode}_{dataset}_{date}``.

    ``/`` in *tutor* is replaced with ``_`` so the string is safe as a
    directory name.

    Args:
        tutor:   Tutor model id, e.g. ``"claude-opus-4-8"`` or
                 ``"deepseek-ai/DeepSeek-V4-Pro"``.
        mode:    Prompt mode, e.g. ``"plain"`` or ``"cot"``.
        dataset: Dataset label, e.g. ``"balanced_520"``.
        date:    Date string, e.g. ``"20260626"`` (caller supplies; not auto).
    """
    safe_tutor = tutor.replace("/", "_")
    return f"{safe_tutor}_{mode}_{dataset}_{date}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_dir(run_id: str, results_root: str = "results") -> Path:
    return Path(results_root) / run_id


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, data: dict) -> None:
    _ensure(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def write_config(run_id: str, config: dict, results_root: str = "results") -> None:
    """Write ``config.json`` for *run_id*."""
    _write_json(_run_dir(run_id, results_root) / "config.json", config)


def read_config(run_id: str, results_root: str = "results") -> Optional[dict]:
    """Read ``config.json`` for *run_id*; returns ``None`` if missing."""
    return _read_json(_run_dir(run_id, results_root) / "config.json")


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def write_transcript(
    run_id: str, scenario_id: str, transcript_dict: dict, results_root: str = "results"
) -> None:
    """Write ``transcripts/<scenario_id>.json`` for *run_id*."""
    path = _run_dir(run_id, results_root) / "transcripts" / f"{scenario_id}.json"
    _write_json(path, transcript_dict)


def read_transcript(
    run_id: str, scenario_id: str, results_root: str = "results"
) -> Optional[dict]:
    """Read transcript for *scenario_id*; returns ``None`` if missing."""
    path = _run_dir(run_id, results_root) / "transcripts" / f"{scenario_id}.json"
    return _read_json(path)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def write_score(
    run_id: str, scenario_id: str, annotation_dict: dict, results_root: str = "results"
) -> None:
    """Write ``scores/<scenario_id>.json`` for *run_id*."""
    path = _run_dir(run_id, results_root) / "scores" / f"{scenario_id}.json"
    _write_json(path, annotation_dict)


def read_score(
    run_id: str, scenario_id: str, results_root: str = "results"
) -> Optional[dict]:
    """Read score for *scenario_id*; returns ``None`` if missing."""
    path = _run_dir(run_id, results_root) / "scores" / f"{scenario_id}.json"
    return _read_json(path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_summary(run_id: str, summary: dict, results_root: str = "results") -> None:
    """Write ``summary.json`` for *run_id*."""
    _write_json(_run_dir(run_id, results_root) / "summary.json", summary)


def read_summary(run_id: str, results_root: str = "results") -> Optional[dict]:
    """Read ``summary.json`` for *run_id*; returns ``None`` if missing."""
    return _read_json(_run_dir(run_id, results_root) / "summary.json")


# ---------------------------------------------------------------------------
# Resume guard
# ---------------------------------------------------------------------------


def is_done(run_id: str, scenario_id: str, results_root: str = "results") -> bool:
    """Return True iff BOTH transcript AND score exist for *scenario_id*.

    A scenario is considered done only when its transcript has been saved
    AND scoring is present — meaning nothing needs to be re-run.
    """
    transcript_path = (
        _run_dir(run_id, results_root) / "transcripts" / f"{scenario_id}.json"
    )
    score_path = _run_dir(run_id, results_root) / "scores" / f"{scenario_id}.json"
    return transcript_path.exists() and score_path.exists()


# ---------------------------------------------------------------------------
# Listing runs
# ---------------------------------------------------------------------------


def list_runs(results_root: str = "results") -> list[str]:
    """Return run_ids present on disk under *results_root*.

    Only directories are included; stray files are ignored.
    """
    root = Path(results_root)
    if not root.exists():
        return []
    return [p.name for p in sorted(root.iterdir()) if p.is_dir()]
