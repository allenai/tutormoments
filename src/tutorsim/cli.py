"""CLI entry point for tutorsim benchmark runs.

Provides:
  run_cell(tutor, mode, run_cfg, *, date, results_root)
    -- single-cell pipeline: load -> conversation -> score -> store -> summary.
    Resumable: scenarios already on disk are skipped.
    Skip-on-error: one bad scenario logs + skips, never crashes the cell.
    trials>1: runs conversation+score N times per scenario; summary has mean+spread.

  expand_cells(tutors, modes) -> list[dict]
    -- expand (tutors x modes) into cell dicts with lane assignment by provider.

  run_sweep(cells, run_cfg, *, date, results_root, _run_cell_fn)
    -- schedule cells: lanes parallel (ThreadPoolExecutor), within-lane sequential.
    Returns list of all run_ids.

  main() / argparse "run" subcommand:
    tutorsim run --tutors X [--modes ...] [--dataset HF_ID | --data_path DIR]
                 [--dataset-revision REV] [--sample N]
                 [--trials N] [--max-turns N]
                 [--log-level LEVEL] [--log-file FILE]

No module-level SDK imports.
ASCII console output only (no Unicode / em-dash / box-drawing).
All file I/O is UTF-8.
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

# Fixed name (not __name__): under `python -m tutorsim.cli` this module is
# "__main__", which would fall outside the "tutorsim" package logger level.
logger = logging.getLogger("tutorsim.cli")


# ---------------------------------------------------------------------------
# Cell expansion + lane scheduling
# ---------------------------------------------------------------------------


def expand_cells(tutors: list[str], modes: list[str]) -> list[dict]:
    """Expand (tutors x modes) into one cell dict per combination.

    Each cell: {"tutor": str, "mode": str, "lane": str}
    Lane is inferred from provider (e.g. "anthropic", "gemini", "openai").
    Registered tutors (not in the API roster) get lane "default".

    Args:
        tutors: List of tutor model IDs.
        modes:  List of prompt mode strings.

    Returns:
        List of cell dicts in sweep order (tutor-major, mode-minor).
    """
    # Lazy import to avoid module-level SDK import
    from tutorsim.client import infer_provider
    from tutorsim.config import get_registered_tutor

    cells = []
    for tutor in tutors:
        # Registered (non-API) tutors -> "default" lane
        if get_registered_tutor(tutor) is not None:
            lane = "default"
        else:
            try:
                lane = infer_provider(tutor)
            except ValueError:
                lane = "default"
        for mode in modes:
            cells.append({"tutor": tutor, "mode": mode, "lane": lane})
    return cells


def run_sweep(
    cells: list[dict],
    run_cfg,
    *,
    date: str,
    results_root: str = "results",
    _run_cell_fn=None,
) -> list[str]:
    """Schedule cells: lanes run in parallel; cells within a lane run sequentially.

    Args:
        cells:           Output of expand_cells().
        run_cfg:         Pre-built RunConfig passed through to run_cell.
        date:            Date string for run_id (e.g. "20260626").
        results_root:    Root directory for results.
        _run_cell_fn:    Injectable for testing (default: run_cell from this module).

    Returns:
        List of run_ids (one per cell), in lane-then-within-lane order.
    """
    if _run_cell_fn is None:
        _run_cell_fn = run_cell

    # Group cells by lane, preserving expansion order
    by_lane: OrderedDict = OrderedDict()
    for cell in cells:
        by_lane.setdefault(cell["lane"], []).append(cell)

    n_lanes = len(by_lane)
    all_run_ids: list[str] = []

    def _run_lane(lane: str, lane_cells: list[dict]) -> list[str]:
        """Run one lane's cells sequentially. Returns run_ids in order."""
        lane_run_ids = []
        for idx, cell in enumerate(lane_cells, 1):
            with log_context(f"{cell['tutor']}/{cell['mode']}"):
                logger.info(
                    "Starting cell %d/%d on lane %s", idx, len(lane_cells), lane
                )
                rid = _run_cell_fn(
                    cell["tutor"],
                    cell["mode"],
                    run_cfg,
                    date=date,
                    results_root=results_root,
                )
                logger.info("Finished cell -> %s", rid)
            lane_run_ids.append(rid)
        return lane_run_ids

    # Lanes in parallel
    with ThreadPoolExecutor(max_workers=n_lanes) as pool:
        futs = {
            pool.submit(_run_lane, lane, lane_cells): lane
            for lane, lane_cells in by_lane.items()
        }
        for fut in as_completed(futs):
            all_run_ids.extend(fut.result())

    return all_run_ids


# ---------------------------------------------------------------------------
# Lazy module imports (no module-level SDK imports)
# ---------------------------------------------------------------------------


def _import_modules():
    """Import heavy modules lazily to avoid SDK import at module level."""
    from tutorsim import conversation, report, results, scoring
    from tutorsim.config import build_run_config
    from tutorsim.moments import load_moments

    return conversation, scoring, results, report, build_run_config, load_moments


# Expose at module level for monkeypatching in tests.
# These are imported at function call time, not at module load time,
# but we re-export references so patch targets resolve correctly.
import tutorsim.conversation as conversation
import tutorsim.report as report
import tutorsim.results as results
import tutorsim.scoring as scoring
from tutorsim.config import build_run_config
from tutorsim.logging_setup import (
    bind_worker_logging,
    log_context,
    logging_args_parent,
    per_run_log_file,
    setup_logging,
)
from tutorsim.moments import (
    DatasetNotFoundError,
    load_manifest,
    load_moments,
)


def _dataset_label(cfg) -> str:
    """Short, path-safe dataset label for run ids and logs."""
    if cfg.data_path:
        label = os.path.basename(os.path.normpath(str(cfg.data_path))) or "local"
    elif cfg.dataset:
        label = cfg.dataset.split("/")[-1]
    else:
        label = "dataset"
    return label.replace("/", "_")


def _json_sha256(data) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version() -> str | None:
    try:
        return version("tutorsim")
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _resource_hashes(root: str) -> dict[str, str]:
    """Hash packaged resource files under a package-relative directory."""
    base = files("tutorsim") / root
    out: dict[str, str] = {}

    def walk(node, rel: str) -> None:
        if node.is_file():
            out[rel] = hashlib.sha256(node.read_bytes()).hexdigest()
            return
        for child in sorted(node.iterdir(), key=lambda p: p.name):
            child_rel = f"{rel}/{child.name}" if rel else child.name
            walk(child, child_rel)

    walk(base, root)
    return out


# ---------------------------------------------------------------------------
# Trials aggregation helpers
# ---------------------------------------------------------------------------


def _mean_spread(values: list) -> tuple[float | None, float | None]:
    """Return (mean, std) for a list of numeric values; None for empty/all-None."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None
    n = len(nums)
    mean = sum(nums) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in nums) / n
    return mean, math.sqrt(variance)


def _aggregate_trials(trial_metrics: list[dict], n_trials: int) -> dict:
    """Compute mean+spread summary across N trial metric dicts.

    Convention:
      - "trials": N
      - "mean": aggregate dict (same shape as single-trial output)
      - "spread": matching dict with std values (None where metric was None everywhere)

    Numeric leaf fields from report.aggregate are:
      n_scenarios,
      overscaffold.{n_yes, n_total, rate},
      scaffold_calibrated.{n_clean_yes, n_overscaffold, n_total, score},
      rigor_calibrated.{n_clean_yes, n_total, score}.
    """

    def _field(dicts, *keys):
        """Extract a nested field from each dict in dicts."""
        vals = []
        for d in dicts:
            v = d
            for k in keys:
                if v is None:
                    break
                v = v.get(k) if isinstance(v, dict) else None
            vals.append(v)
        return vals

    def _sub_mean_spread(dicts, subkey, fields):
        mean_sub = {}
        spread_sub = {}
        for f in fields:
            m, s = _mean_spread(_field(dicts, subkey, f))
            mean_sub[f] = m
            spread_sub[f] = s
        return mean_sub, spread_sub

    # top-level scalar
    m_nscen, s_nscen = _mean_spread(_field(trial_metrics, "n_scenarios"))

    # sub-dicts
    m_over, s_over = _sub_mean_spread(
        trial_metrics, "overscaffold", ["n_yes", "n_total", "rate"]
    )

    # scaffold_calibrated has an extra field
    m_sc, s_sc = _sub_mean_spread(
        trial_metrics,
        "scaffold_calibrated",
        ["n_clean_yes", "n_overscaffold", "n_total", "score"],
    )
    m_rc, s_rc = _sub_mean_spread(
        trial_metrics,
        "rigor_calibrated",
        ["n_clean_yes", "n_total", "score"],
    )

    # "available" is boolean -- take first trial's value (invariant across trials)
    over_available = (trial_metrics[0].get("overscaffold") or {}).get(
        "available", False
    )

    mean_dict = {
        "n_scenarios": m_nscen,
        "overscaffold": {**m_over, "available": over_available},
        "scaffold_calibrated": m_sc,
        "rigor_calibrated": m_rc,
    }
    spread_dict = {
        "n_scenarios": s_nscen,
        "overscaffold": {**s_over},
        "scaffold_calibrated": s_sc,
        "rigor_calibrated": s_rc,
    }

    return {
        "trials": n_trials,
        "mean": mean_dict,
        "spread": spread_dict,
    }


# ---------------------------------------------------------------------------
# Public API: run_cell
# ---------------------------------------------------------------------------


def _classify_run_taxonomy(run_dir, annotations, scenarios, *, tutor, mode):
    """Run the always-on action-taxonomy classification for a completed cell.

    Returns the taxonomy summary dict, or ``{"error": ...}`` on any failure --
    a taxonomy failure (missing API key, classifier error) must never discard
    the run's primary metrics. `tutor`/`mode` are recorded on the facets; the
    classifier model comes from the `taxonomy` config block.
    """
    from tutorsim import taxonomy

    out_dir = os.path.join(run_dir, "taxonomy")
    try:
        return taxonomy.classify_run(
            annotations,
            scenarios,
            out_dir,
            model=tutor,
            mode=mode,
        )
    except Exception as e:  # best-effort: never fail the run on taxonomy
        logger.warning("Taxonomy classification failed (run metrics unaffected): %s", e)
        return {"error": str(e)}


def run_cell(
    tutor: str,
    mode: str,
    run_cfg,
    *,
    date: str,
    results_root: str = "results",
) -> str:
    """Run a single (tutor, mode) cell end-to-end.

    Sequence:
      1. build_run_config -> load_moments -> (slice sample)
      2. make_run_id + write_config
      3. For each scenario (skip if is_done):
           run_conversation -> score -> write_transcript + write_score
           On any exception: log SKIP <id>: <err> and continue
      4. aggregate over completed annotations -> write_summary
      5. Return run_id

    Args:
        tutor: Tutor model id (e.g. "claude-opus-4-8").
        mode: Prompt mode (e.g. "plain").
        run_cfg: Pre-built RunConfig or None (if None, built from config defaults).
        date: Date string for run_id (e.g. "20260626"). Caller supplies; not auto.
        results_root: Root directory for results (default "results").

    Returns:
        run_id string (e.g. "claude-opus-4-8_plain_balanced_520_20260626").
    """
    # Step 1: build config
    if run_cfg is None:
        cfg = build_run_config(tutors=[tutor], modes=[mode])
    else:
        cfg = run_cfg

    # Load moments from the released dataset (HF id or local release dir)
    all_scenarios, dataset_source = load_moments(
        dataset=cfg.dataset,
        data_path=cfg.data_path,
        revision=cfg.dataset_revision,
        config=cfg.dataset_config,
    )
    if cfg.sample is not None:
        all_scenarios = all_scenarios[: cfg.sample]

    dataset_label = _dataset_label(cfg)
    n_total = len(all_scenarios)
    if n_total == 0:
        raise RuntimeError(
            f"Dataset '{dataset_label}' yielded zero moments after sampling; "
            "refusing to write an empty benchmark run."
        )
    logger.info("Loaded %d moments from dataset '%s'", n_total, dataset_label)
    dataset_manifest = load_manifest(cfg.data_path) if cfg.data_path else None

    # Step 2: run_id + config on disk
    run_id = results.make_run_id(tutor, mode, dataset_label, date)

    # Everything below is also captured in the run's own log file,
    # kept next to config.json / summary.json for reproducibility.
    # The handler is keyed to this thread (plus replay-pool workers registered
    # via bind_worker_logging) so parallel lanes don't mix.
    # log_context tags records with [tutor/mode] for programmatic callers;
    # under run_sweep the lane already set the same tag.
    run_log_path = os.path.join(results_root, run_id, "run.log")
    with per_run_log_file(run_log_path) as run_log, log_context(f"{tutor}/{mode}"):
        config_dict = {
            "tutor": tutor,
            "mode": mode,
            "dataset": {
                "id": cfg.dataset,
                "revision": cfg.dataset_revision,
                "data_path": cfg.data_path,
                "config": cfg.dataset_config,
                "record_count": dataset_source["record_count"],
                "content_hash": dataset_source["content_hash"],
            },
            "sample": cfg.sample,
            "max_turns": cfg.max_turns,
            "trials": cfg.trials,
            # Informational only: replay concurrency does not affect results, so
            # it is deliberately kept out of reproducibility.config_hash below.
            "replay_concurrency": getattr(cfg, "replay_concurrency", None),
            "student": cfg.student,
            "scorer": cfg.scorer,
            "resolved_tutors": cfg.resolved_tutors,
            "config_source": getattr(cfg, "config_source", None),
            "reproducibility": {
                "tutorsim_version": _package_version(),
                "git_commit": _git_commit(),
                "config_hash": _json_sha256(
                    {
                        "student": cfg.student,
                        "scorer": cfg.scorer,
                        "resolved_tutors": cfg.resolved_tutors,
                        "defaults": {
                            "sample": cfg.sample,
                            "max_turns": cfg.max_turns,
                            "trials": cfg.trials,
                        },
                    }
                ),
                "prompt_hashes": _resource_hashes("prompts"),
                "dataset_manifest": dataset_manifest,
            },
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        results.write_config(run_id, config_dict, results_root=results_root)
        logger.info("Run id: %s", run_id)

        n_trials = getattr(cfg, "trials", 1) or 1
        replay_concurrency = getattr(cfg, "replay_concurrency", 1) or 1

        # Step 3: per-scenario loop
        # For trials>1: collect per-trial aggregate metrics; for trials=1: single pass.
        # Each trial runs all scenarios; we aggregate per-trial, then compute mean+spread.

        _EMPTY_METRICS = {
            "n_scenarios": 0,
            "overscaffold": {
                "n_yes": 0,
                "n_total": 0,
                "rate": None,
                "available": False,
            },
            "scaffold_calibrated": {
                "n_clean_yes": 0,
                "n_overscaffold": 0,
                "n_total": 0,
                "score": None,
            },
            "rigor_calibrated": {"n_clean_yes": 0, "n_total": 0, "score": None},
        }

        # ---------------------------------------------------------------------------
        # Latency / token helpers
        # ---------------------------------------------------------------------------

        def _latency_stats(samples: list) -> dict | None:
            """Mean / p50 / p95 over per-call latency samples. None on empty.

            Sorted-index percentile: p50 = s[n//2], p95_idx = max(0, min(n-1, round(0.95*n)-1)).
            """
            if not samples:
                return None
            s = sorted(samples)
            n = len(s)
            p50 = s[n // 2]
            p95_idx = max(0, min(n - 1, int(round(0.95 * n)) - 1))
            p95 = s[p95_idx]
            total = sum(samples)
            return {
                "n": n,
                "total_seconds": round(total, 3),
                "mean_seconds": round(total / n, 3),
                "p50_seconds": round(p50, 3),
                "p95_seconds": round(p95, 3),
            }

        def _run_trial(trial_idx: int) -> tuple:
            """Run all scenarios once (one trial): conversations, then pooled scoring.

            Phase 1 runs (or resumes) each moment's conversation; phase 2 scores
            every un-scored moment through ONE pooled scoring pipeline
            (scoring.score_batch), so the trial pays ~3 batch queue-waits total
            instead of 3 per moment. Per-moment score files keep resume
            granularity unchanged.

            Returns:
                (metrics_dict, transcripts_list)
                where transcripts_list are the Transcript objects for this trial.
            """
            completed_scenarios = []
            completed_annotations = []
            completed_transcripts = []
            counts = {
                "attempted": n_total,
                "succeeded": 0,
                "failed": 0,
                "resumed": 0,
            }
            failed_scenarios = []
            to_score = []  # (scenario, transcript, resume_sid) awaiting pooled scoring

            # ---- Phase 1: Replay -- generate conversations (with per-moment resume) ----
            # Three stages keep replay result-identical to the serial version while
            # overlapping only the (independent) LLM round-trips:
            #   1. sequential planning pass -- cheap disk reads; resolve resume vs
            #      needs-run; all counts/list mutation for resumed moments happens here.
            #   2. bounded ThreadPoolExecutor over needs-run moments -- workers call
            #      ONLY run_conversation (no shared state, no disk writes). The
            #      main thread writes each transcript as its future completes, so
            #      a killed run keeps everything already finished (per-moment
            #      resume durability, matching the serial version).
            #   3. single-threaded collection in original index order --
            #      counts/failed/to_score mutation, so no locks are needed and
            #      to_score is byte-identical regardless of completion order.
            logger.info(
                "Starting Replay (trial %d/%d): %d moments, student=%s (%s), max_turns=%s, concurrency=%d",
                trial_idx,
                n_trials,
                n_total,
                (cfg.student or {}).get("model"),
                (cfg.student or {}).get("mode", "oracle"),
                cfg.max_turns,
                replay_concurrency,
            )

            # ---- Stage 1: sequential planning pass (resume decisions) ----
            pending = []  # (i, scenario, resume_sid) needing a fresh conversation
            for i, scenario in enumerate(all_scenarios, 1):
                sid = scenario.id

                # Resume key includes trial index for trials>1
                if n_trials > 1:
                    resume_sid = f"{sid}_t{trial_idx}"
                else:
                    resume_sid = sid

                # Resume: skip if both transcript and score already exist
                if results.is_done(run_id, resume_sid, results_root=results_root):
                    logger.info(
                        "[trial %d][%d/%d] SKIP (already done): %s",
                        trial_idx,
                        i,
                        n_total,
                        sid,
                    )
                    score_dict = results.read_score(
                        run_id, resume_sid, results_root=results_root
                    )
                    if score_dict is not None:
                        from tutorsim.scoring import Annotation

                        try:
                            ann = Annotation(**score_dict)
                            completed_scenarios.append(scenario)
                            completed_annotations.append(ann)
                            counts["succeeded"] += 1
                            counts["resumed"] += 1
                            # No transcript object on resume -- latencies not available
                        except Exception as e:
                            counts["failed"] += 1
                            failed_scenarios.append(
                                {"id": sid, "error": str(e), "phase": "resume"}
                            )
                            logger.warning(
                                "[trial %d][%d/%d] Could not reload score for %s: %s",
                                trial_idx,
                                i,
                                n_total,
                                sid,
                                e,
                            )
                    continue

                # Interrupted-run resume: transcript on disk but no score yet --
                # skip the conversation and pool the moment for scoring.
                transcript_dict = results.read_transcript(
                    run_id, resume_sid, results_root=results_root
                )
                if transcript_dict is not None:
                    logger.info(
                        "[trial %d][%d/%d] RESUME replay (classification pending): %s",
                        trial_idx,
                        i,
                        n_total,
                        sid,
                    )
                    to_score.append(
                        (
                            scenario,
                            conversation.Transcript.from_dict(transcript_dict),
                            resume_sid,
                        )
                    )
                    continue

                pending.append((i, scenario, resume_sid))

            # ---- Stage 2: concurrent replay of needs-run moments ----
            # outcome[i] = ("ok", transcript) | ("err", exception); index-keyed so
            # Stage 3 can reassemble in canonical order.
            outcome: dict = {}

            def _replay_one(scenario):
                return conversation.run_conversation(
                    scenario,
                    tutor_id=tutor,
                    tutor_mode=mode if mode else None,
                    student_id=(cfg.student or {}).get("model"),
                    student_mode=(cfg.student or {}).get("mode", "oracle"),
                    max_turns=cfg.max_turns,
                )

            if pending:
                workers = max(1, min(replay_concurrency, len(pending)))
                cell_tag = f"{tutor}/{mode}"
                with ThreadPoolExecutor(
                    max_workers=workers,
                    # Workers must adopt this run's log file + [tutor/mode] tag:
                    # the run.log handler filters by registered thread ids, and
                    # contextvars don't cross thread boundaries.
                    initializer=bind_worker_logging,
                    initargs=(run_log, cell_tag),
                ) as pool:
                    fut_to_idx = {
                        pool.submit(_replay_one, scenario): (i, scenario.id, resume_sid)
                        for i, scenario, resume_sid in pending
                    }
                    for fut in as_completed(fut_to_idx):
                        idx, sid, resume_sid = fut_to_idx[fut]
                        try:
                            transcript = fut.result()
                        except Exception as e:  # noqa: BLE001 -- per-moment isolation
                            outcome[idx] = ("err", e)
                            continue
                        # Persist immediately (main thread), in completion order:
                        # transcripts are per-sid files, so write order doesn't
                        # affect the final file set, and a killed run keeps every
                        # conversation that already finished (resume durability).
                        # Written before scoring so a score failure doesn't lose it.
                        transcript_dict = (
                            transcript.to_dict()
                            if hasattr(transcript, "to_dict")
                            else dict(transcript)
                        )
                        results.write_transcript(
                            run_id,
                            resume_sid,
                            transcript_dict,
                            results_root=results_root,
                        )
                        outcome[idx] = ("ok", transcript)
                        logger.info(
                            "[trial %d][%d/%d] replay OK: %s",
                            trial_idx,
                            idx,
                            n_total,
                            sid,
                        )

            # ---- Stage 3: deterministic, single-threaded collection ----
            for i, scenario, resume_sid in pending:
                sid = scenario.id
                status, payload = outcome[i]
                if status == "err":
                    counts["failed"] += 1
                    failed_scenarios.append(
                        {"id": sid, "error": str(payload), "phase": "run"}
                    )
                    logger.error(
                        "[trial %d][%d/%d] SKIP %s: %s",
                        trial_idx,
                        i,
                        n_total,
                        sid,
                        payload,
                    )
                    continue

                to_score.append((scenario, payload, resume_sid))

            # ---- Phase 2: Classification -- pooled scoring (one 3-pass batch pipeline) ----
            if to_score:
                logger.info(
                    "Starting Classification (trial %d/%d): %d replays, 3 pooled batch passes",
                    trial_idx,
                    n_trials,
                    len(to_score),
                )
                try:
                    annotations_by_sid = scoring.score_batch(
                        [(s, t) for s, t, _ in to_score]
                    )
                except Exception as e:
                    # A pooled batch failure fails all pooled moments this trial.
                    annotations_by_sid = {}
                    for s, _, _ in to_score:
                        counts["failed"] += 1
                        failed_scenarios.append(
                            {"id": s.id, "error": str(e), "phase": "score"}
                        )
                    logger.error(
                        "[trial %d] Classification failed for all pooled replays: %s",
                        trial_idx,
                        e,
                    )

                for scenario, transcript, resume_sid in to_score:
                    annotation = annotations_by_sid.get(scenario.id)
                    if annotation is None:
                        if not annotations_by_sid:
                            continue  # whole-batch failure already recorded above
                        counts["failed"] += 1
                        failed_scenarios.append(
                            {
                                "id": scenario.id,
                                "error": "no annotation returned by score_batch",
                                "phase": "score",
                            }
                        )
                        continue

                    annotation_dict = (
                        annotation.to_dict()
                        if hasattr(annotation, "to_dict")
                        else dict(annotation)
                    )
                    results.write_score(
                        run_id, resume_sid, annotation_dict, results_root=results_root
                    )

                    completed_scenarios.append(scenario)
                    completed_annotations.append(annotation)
                    completed_transcripts.append(transcript)
                    counts["succeeded"] += 1
                    logger.info("[trial %d] classified OK: %s", trial_idx, scenario.id)

            if completed_scenarios:
                metrics = report.aggregate(completed_scenarios, completed_annotations)
            else:
                raise RuntimeError(
                    f"No scenarios completed for {tutor}/{mode} trial {trial_idx}; "
                    f"{counts['failed']} of {counts['attempted']} attempted scenarios failed."
                )
            metrics["run_counts"] = counts
            if failed_scenarios:
                metrics["failed_scenarios"] = failed_scenarios
            return (
                metrics,
                completed_transcripts,
                counts,
                completed_scenarios,
                completed_annotations,
            )

        # Run all trials
        trial_results = [_run_trial(t) for t in range(1, n_trials + 1)]
        trial_metrics = [m for m, _, _, _, _ in trial_results]
        trial_counts = [c for _, _, c, _, _ in trial_results]
        all_trial_transcripts = [t for _, ts, _, _, _ in trial_results for t in ts]
        # Pooled (scenario, annotation) pairs across trials for taxonomy
        # classification; dedup by statement text happens in classify_pool.
        all_trial_scenarios = [s for _, _, _, scs, _ in trial_results for s in scs]
        all_trial_annotations = [a for _, _, _, _, anns in trial_results for a in anns]

        # Build latency + token blocks from all completed transcripts
        tutor_lat_samples: list = []
        student_lat_samples: list = []
        tutor_input = tutor_output = 0
        student_input = student_output = 0
        for tx in all_trial_transcripts:
            tutor_lat_samples.extend(getattr(tx, "tutor_latencies", []) or [])
            student_lat_samples.extend(getattr(tx, "student_latencies", []) or [])
            tu = getattr(tx, "tutor_usage", {}) or {}
            su = getattr(tx, "student_usage", {}) or {}
            tutor_input += tu.get("input_tokens", 0) or 0
            tutor_output += tu.get("output_tokens", 0) or 0
            student_input += su.get("input_tokens", 0) or 0
            student_output += su.get("output_tokens", 0) or 0

        tutor_tokens = {
            "input_tokens": tutor_input,
            "output_tokens": tutor_output,
            "total_tokens": tutor_input + tutor_output,
        }
        student_tokens = {
            "input_tokens": student_input,
            "output_tokens": student_output,
            "total_tokens": student_input + student_output,
        }
        total_tokens = {
            "input_tokens": tutor_input + student_input,
            "output_tokens": tutor_output + student_output,
            "total_tokens": tutor_input + tutor_output + student_input + student_output,
        }
        latency_block = {
            "tutor": _latency_stats(tutor_lat_samples),
            "student": _latency_stats(student_lat_samples),
        }
        token_block = {
            "tutor": tutor_tokens,
            "student": student_tokens,
            "total": total_tokens,
        }

        # Step 4: aggregate + write summary
        if n_trials == 1:
            # trials=1: plain aggregate dict -- identical to Task-4 output
            metrics = trial_metrics[0]
        else:
            # trials>1: compute mean and spread (std) across trials for numeric leaf values
            metrics = _aggregate_trials(trial_metrics, n_trials)

        metrics["run_counts"] = {
            "attempted": sum(c["attempted"] for c in trial_counts),
            "succeeded": sum(c["succeeded"] for c in trial_counts),
            "failed": sum(c["failed"] for c in trial_counts),
            "resumed": sum(c["resumed"] for c in trial_counts),
        }

        # Merge latency + token blocks into summary (spec S7)
        metrics = dict(metrics)
        metrics["latency"] = latency_block
        metrics["tokens"] = token_block

        # Always-on action-taxonomy classification (LM side): a first-class run
        # output alongside the headline metrics. A failure here (missing API
        # key, classifier error) must never discard the run's primary metrics.
        run_dir = os.path.join(results_root, run_id)
        tax = _classify_run_taxonomy(
            run_dir,
            all_trial_annotations,
            all_trial_scenarios,
            tutor=tutor,
            mode=mode,
        )
        metrics["taxonomy"] = tax
        tax_usage = (tax or {}).get("usage") or {}
        if tax_usage:
            metrics["tokens"]["taxonomy"] = tax_usage
            total = metrics["tokens"]["total"]
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                total[k] = total.get(k, 0) + int(tax_usage.get(k, 0) or 0)

        results.write_summary(run_id, metrics, results_root=results_root)
        run_counts = metrics["run_counts"]
        logger.info(
            "Run complete: %d/%d moments succeeded (%d failed, %d resumed, "
            "%d trial(s)) -> %s",
            run_counts["succeeded"],
            run_counts["attempted"],
            run_counts["failed"],
            run_counts["resumed"],
            n_trials,
            os.path.join(results_root, run_id),
        )

        # Echo the per-run score summary to stdout. The metrics are already in
        # summary.json; this surfaces them without a separate `report` step
        # (which stays the cross-run leaderboard). Printed, not logged, so it
        # shows regardless of --log-level.
        print(
            report.format_run_summary(
                metrics, tutor_model=tutor, mode=mode, run_id=run_id
            )
        )

        return run_id


# ---------------------------------------------------------------------------
# CLI: argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tutorsim",
        description="Tutorsim benchmark runner",
    )
    subs = parser.add_subparsers(dest="command")
    log_parent = logging_args_parent()

    # -- run subcommand -------------------------------------------------------
    run_p = subs.add_parser(
        "run",
        help="Run one or more (tutor x mode) cells",
        parents=[log_parent],
    )
    run_p.add_argument(
        "--tutors",
        nargs="+",
        required=True,
        metavar="MODEL_ID",
        help="Tutor model id(s), e.g. claude-opus-4-8",
    )
    run_p.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Config file override (default: TUTORSIM_CONFIG, ./config.yaml, then packaged default)",
    )
    run_p.add_argument(
        "--modes",
        nargs="+",
        default=None,
        metavar="MODE",
        help="Prompt mode(s) (default: plain scaffolding_rigor)",
    )
    run_p.add_argument(
        "--dataset",
        default=None,
        metavar="HF_ID",
        help="Hugging Face dataset id for the released benchmark "
        "(default: dataset.id from config)",
    )
    run_p.add_argument(
        "--data_path",
        default=None,
        dest="data_path",
        metavar="DIR",
        help="Local release directory containing moments.jsonl "
        "(developer override; wins over --dataset)",
    )
    run_p.add_argument(
        "--dataset-revision",
        default=None,
        dest="dataset_revision",
        metavar="REV",
        help="Pinned dataset revision (default: dataset.revision from config)",
    )
    run_p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use first N moments from the dataset (default: all)",
    )
    run_p.add_argument(
        "--trials",
        type=int,
        default=None,
        metavar="N",
        help="Number of trials per cell (default: from config)",
    )
    run_p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        dest="max_turns",
        metavar="N",
        help="Max turns per conversation (default: from config)",
    )
    run_p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        dest="replay_concurrency",
        metavar="N",
        help="Concurrent per-moment replays within a cell (default: from config, "
        "typically 4). Result-preserving; lower it on smaller API tiers that "
        "hit rate limits.",
    )
    # -- report subcommand ----------------------------------------------------
    report_p = subs.add_parser(
        "report",
        help="Aggregate all run summaries into a leaderboard (md + csv)",
        parents=[log_parent],
    )
    report_p.add_argument(
        "--results-root",
        default="results",
        dest="results_root",
        metavar="DIR",
        help="Root directory containing run subdirectories (default: results)",
    )
    report_p.add_argument(
        "--out",
        default="leaderboard",
        metavar="PATH",
        help="Output file stem (default: leaderboard); writes .md and .csv",
    )

    # -- view subcommand ------------------------------------------------------
    view_p = subs.add_parser(
        "view",
        help="Build a self-contained HTML leaderboard viewer",
        parents=[log_parent],
    )
    view_p.add_argument(
        "--results-root",
        default="results",
        dest="results_root",
        metavar="DIR",
        help="Root directory containing run subdirectories (default: results)",
    )
    view_p.add_argument(
        "--out",
        default="viewer.html",
        metavar="FILE",
        help="Output HTML file (default: viewer.html)",
    )

    # taxonomy: standalone (re)generation of action-taxonomy data and headline
    # tables from a run dir or the ground-truth bundle. All args after
    # `taxonomy` are forwarded to tutorsim.taxonomy.cli_dispatch (which has its
    # own classify/headline/run subcommands).
    tax_p = subs.add_parser(
        "taxonomy",
        help="Action-taxonomy data: classify / headline / run (see 'taxonomy -h')",
        add_help=False,
    )
    tax_p.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def _cmd_report(args) -> None:
    """Implement the 'report' subcommand: read all run summaries -> leaderboard md+csv."""
    from pathlib import Path

    run_ids = results.list_runs(args.results_root)
    summaries = []
    for run_id in run_ids:
        summary = results.read_summary(run_id, results_root=args.results_root)
        if summary is None:
            logger.warning("No summary.json for run %s -- skipping", run_id)
            continue
        # Inject run_id-derived fields if not already present
        if "tutor_model" not in summary or not summary.get("tutor_model"):
            # Best-effort: parse tutor from run_id prefix
            parts = run_id.split("_")
            summary = dict(summary)
            summary.setdefault("tutor_model", parts[0] if parts else run_id)
        if "mode" not in summary or not summary.get("mode"):
            parts = run_id.split("_")
            summary = dict(summary)
            summary.setdefault("mode", parts[1] if len(parts) > 1 else "")
        summaries.append(summary)

    if not summaries:
        print("No run summaries found in: " + args.results_root)
        return

    markdown, csv_str = report.leaderboard(summaries)

    out_stem = args.out
    md_path = Path(out_stem + ".md")
    csv_path = Path(out_stem + ".csv")

    md_path.write_text(markdown, encoding="utf-8")
    csv_path.write_text(csv_str, encoding="utf-8")

    print("Leaderboard written:")
    print("  md : " + str(md_path))
    print("  csv: " + str(csv_path))
    print("Rows: " + str(len(summaries)))


def _cmd_view(args) -> None:
    """Implement the 'view' subcommand: read all run summaries -> HTML viewer."""
    from pathlib import Path

    run_ids = results.list_runs(args.results_root)
    summaries = []
    for run_id in run_ids:
        summary = results.read_summary(run_id, results_root=args.results_root)
        if summary is None:
            logger.warning("No summary.json for run %s -- skipping", run_id)
            continue
        summary = dict(summary)
        summary.setdefault("tutor_model", run_id.split("_")[0] if run_id else "")
        summary.setdefault(
            "mode", run_id.split("_")[1] if len(run_id.split("_")) > 1 else ""
        )
        summaries.append(summary)

    html = report.view(summaries)

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")

    print("Viewer written: " + str(out_path))
    print("Runs included: " + str(len(summaries)))


def main(argv=None) -> None:
    """Main entry point: parse args and dispatch subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # taxonomy delegates to its own dispatcher (with its own logging); it has
    # no shared --log-level/--log-file args, so handle it before setup_logging.
    if args.command == "taxonomy":
        from tutorsim import taxonomy

        sys.exit(taxonomy.cli_dispatch(args.args))

    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info(
        "Command: tutorsim %s", " ".join(argv if argv is not None else sys.argv[1:])
    )

    if args.command == "run":
        if args.config:
            os.environ["TUTORSIM_CONFIG"] = args.config
        cfg = build_run_config(
            tutors=args.tutors,
            modes=args.modes,
            dataset=args.dataset,
            data_path=args.data_path,
            dataset_revision=args.dataset_revision,
            sample=args.sample,
            trials=args.trials,
            max_turns=args.max_turns,
            replay_concurrency=args.replay_concurrency,
            config_path=args.config,
        )
        date = datetime.date.today().strftime("%Y%m%d")
        cells = expand_cells(cfg.tutors, cfg.modes)
        try:
            run_ids = run_sweep(
                cells=cells,
                run_cfg=cfg,
                date=date,
                results_root="results",
            )
        except DatasetNotFoundError as e:
            print("Error: " + str(e), file=sys.stderr)
            sys.exit(2)
        for run_id in run_ids:
            print("Completed run: " + run_id)

    elif args.command == "report":
        _cmd_report(args)

    elif args.command == "view":
        _cmd_view(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
