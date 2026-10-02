#!/usr/bin/env python3
"""Regenerate the site's chart data from a local allenai/tutormoments checkout.

The tutormoments repo gitignores its results, so the website keeps its own copies as
static JSON. Re-run this after benchmarking new models:

    python3 scripts/refresh-data.py /path/to/tutormoments

Reads, for the paper's models (same sources as the repo's
analysis/working-paper-20260630 scripts):
  results/benchmark/_full_combined/<model>__<prompt>/scores.json   -> leaderboard.json
  results/benchmark/<model>_v10_<prompt>_tutor_oracle_student*/exchanges/*.json
                                                                   -> latency.json (latency_s)
for every other model:
  results/<run_id>/{summary,config}.json (full `tutormoments run`s)  -> leaderboard.json,
                                                                      latency.json (latency_s)
and for all of them:
  results/<model>_<prompt>_latency_<date>/latency.json             -> latency.json (ttft_s)
  results/<run_id>/summary.json (full runs), else the probe above  -> cost.json
  data/taxonomy/{human,lm}/classified.csv (via tutormoments.taxonomy)  -> action_distribution.json

Writes to static/data/. Sections of the site hide automatically when their JSON
is absent, so partial refreshes are fine.

Run this with an interpreter that can ``import tutormoments`` -- the checkout's
own venv is the easy one:

    /path/to/tutormoments/.venv/bin/python scripts/refresh-data.py /path/to/tutormoments

The TTFT figures come with publishability rules (which providers' cache labels
can be trusted, how many hit samples support a percentile) that live in
`tutormoments.latency` and must not be reimplemented here -- an earlier version
of this script did reimplement them and got them wrong. Without that import the
script still refreshes everything else and says what it skipped.

The same goes for cost.json: the uncached per-response cost is priced by
`tutormoments.costing` from a model's full benchmark run -- or, where that run
predates usage capture, by `tutormoments.latency.probe_cost_figures` from a
probe -- at the checkout's *current* registry rates (tokens are the
measurement, prices a lookup), so a price change shows up on the next refresh
without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = SITE_ROOT / "static" / "data"

PROMPTS = {"plain": "plain", "scaffolding_rigor": "eval_aware"}

# Display label per site model id, in leaderboard row order (grouped by
# provider). Add new models here. A paper model's id is its directory prefix
# under results/benchmark/_full_combined; any other model's id is its
# benchmark arm name (the `tutor_model` its full run's summary.json records,
# "/" replaced by "_").
MODELS = [
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-fable-5-1", "Claude Fable 5.1"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("deepseek-ai_DeepSeek-V4-Pro", "DeepSeek V4 Pro"),
    ("deepseek-v4-pro-0813", "DeepSeek V4 Pro 0813"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("gemini-3.8-flash", "Gemini 3.8 Flash"),
    ("gemini-3.6-flash", "Gemini 3.6 Flash"),
    ("gpt-5.5-2026-04-23", "GPT 5.5"),
    ("gpt-5.5-2026-04-23-none", "GPT 5.5 (no reasoning)"),
    ("gpt-5.4-mini-2026-03-17", "GPT 5.4 mini"),
    ("gpt-6-astra", "GPT-6 Astra"),
    ("gpt-6-sol-none", "GPT-6 Sol"),
    ("gpt-6-luna-none", "GPT-6 Luna"),
]

# The working paper's models and the provider-native reasoning parameters
# they ran with, as the benchmark config states them. The paper's runs predate
# `benchmark_models:`, so these are the tutor roster of the first runtime
# config ("the 7 that ran", 97055d9) in today's key names -- the values the
# current default_config.yaml still carries for the six arms it keeps.
# DeepSeek V4 Pro sent no reasoning parameter (`{}`): it reasoned at the
# model's default. Their scores stay the paper's (Table 8): read from
# results/benchmark/_full_combined where present, carried forward from the
# committed leaderboard.json otherwise -- a later run of the same arm does not
# replace a published number. Every other model is scored from its full
# benchmark run, its parameters read off that run's config.json.
PAPER_THINKING = {
    "claude-opus-4-8": {"thinking": {"type": "adaptive"}, "effort": "xhigh"},
    "claude-sonnet-4-6": {"thinking": {"type": "adaptive"}, "effort": "high"},
    "deepseek-ai_DeepSeek-V4-Pro": {},
    "gemini-2.5-pro": {"include_thoughts": True, "thinking_budget": -1},
    "gemini-3.5-flash": {"include_thoughts": True, "thinking_budget": -1},
    "gpt-5.5-2026-04-23": {"reasoning": "high"},
    "gpt-5.4-mini-2026-03-17": {"reasoning": "high"},
}

# Shown when an arm sends no reasoning parameter at all.
NO_REASONING_PARAM = "none sent (model default)"

# Human reference scores from the paper (Table 8 caption context). Update if the
# scoring pipeline is re-run over the human transcripts.
HUMAN = {"scaffolding": 0.458, "rigor": 0.182, "avoids_over": 0.496}

# Short axis labels per action-taxonomy letter (full names live in the CSV's
# "name" column; letter M "Other" is dropped, matching the paper figure).
ACTION_LABELS = {
    "A": "Guiding questions",
    "B": "Breaking into steps",
    "C": "Explaining",
    "D": "Alternative representations",
    "E": "Hints",
    "F": "Supplying answers",
    "G": "Prompting justification",
    "H": "Independent work",
    "I": "Increasing complexity",
    "J": "Prompting self-assessment",
    "K": "Affirmations",
    "L": "Transitioning",
}

# Site model id -> column prefix in action_taxonomy_distribution.csv.
ACTION_CSV_MODELS = {
    "claude-opus-4-8": "claude_opus_4_8",
    "claude-sonnet-4-6": "claude_sonnet_4_6",
    "deepseek-ai_DeepSeek-V4-Pro": "deepseek_v4_pro",
    "gemini-2.5-pro": "gemini_2_5_pro",
    "gemini-3.5-flash": "gemini_3_5_flash",
    "gpt-5.5-2026-04-23": "gpt_5_5",
    "gpt-5.4-mini-2026-03-17": "gpt_5_4_mini",
}


def write_json(name: str, payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    try:
        shown = path.relative_to(SITE_ROOT)
    except ValueError:  # OUT_DIR redirected outside the site (tests)
        shown = path
    print(f"wrote {shown}")


def perf(bench: Path, model: str, prompt: str) -> dict | None:
    fp = bench / "_full_combined" / f"{model}__{prompt}" / "scores.json"
    if not fp.exists():
        print(f"  missing {fp} — skipping", file=sys.stderr)
        return None
    d = json.loads(fp.read_text("utf-8"))
    return {
        "scaffolding": round(d["scaffold_calibrated"]["score"], 3),
        "rigor": round(d["rigor_calibrated"]["score"], 3),
        "avoids_over": round(1.0 - d["overscaffold"]["rate"], 3),
        "n": d.get("n_scenarios"),
    }


def latency(bench: Path, model: str, prompt: str, ids: set[str]) -> float | None:
    """Mean tutor latency per turn, mirroring summarize_exchanges in the repo's
    analysis/working-paper-20260630/benchmark_perf_cost.py (filter to the
    balanced-520 ids, de-dupe by scenario_id, first wins).

    End-to-end seconds per turn. On streamed runs this equals time-to-last-token;
    on pre-streaming runs it is wall clock including server-side buffering."""
    needle = f"{model}_v10_{prompt}_tutor_oracle_student"
    seen: set[str] = set()
    lats: list[float] = []
    for run in bench.iterdir():
        if not (run.is_dir() and run.name.startswith(needle)):
            continue
        for fp in (run / "exchanges").rglob("*.json"):
            try:
                ex = json.loads(fp.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sid = ex.get("scenario_id")
            if sid not in ids or sid in seen:
                continue
            seen.add(sid)
            lats.extend(ex.get("tutor_latencies") or [])
    return round(statistics.mean(lats), 2) if lats else None


def load_latency_module(repo: Path):
    """Import `tutormoments.latency` from the checkout, or None if unavailable.

    The probe's rules about what may be published -- whether a provider's
    cache labels mean session warmth at all, how many hit samples support a
    percentile -- belong to the runtime, and this script's job is to read them
    out, not to restate them. An earlier version restated them and gated on
    cache hit *rate*, which the runtime deliberately rejects: the rate is fixed
    by --max-turns, so the threshold sat on the structural boundary, and a
    provider whose "hits" read back 256 tokens of shared system prompt sailed
    through it.

    Returns None (rather than falling back to a local copy of the rules) when
    the interpreter cannot import the package, so a bare `python3` run still
    refreshes the rest of the site and reports the gap.
    """
    src = repo / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from tutormoments import latency  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"cannot import tutormoments.latency ({exc}); skipping ttft_s. "
            f"Re-run with an interpreter that has the package installed, e.g. "
            f"{repo}/.venv/bin/python",
            file=sys.stderr,
        )
        return None
    return latency


def probe_ttft(repo: Path, probe_root: Path, prompt: str) -> tuple[dict, dict]:
    """TTFT figures per site model id, from `tutormoments latency` probe runs.

    Returns ``({site_id: {ttft_s, ttlt_s?, ttft_first_s?, ttft_later_s?}},
    provenance)``.

    Only probe runs are read. A benchmark run also records TTFT, but under
    --concurrency, which distorts it by a model-dependent amount and makes it
    incomparable across models -- exactly the comparison this chart invites.

    `ttft_s` is the pooled p50 over all samples. `ttft_first_s` /
    `ttft_later_s` split it by turn position -- the first message of a session
    against the later ones -- which the probe recorded itself, so every model
    with a probe run gets the split regardless of what its provider reports
    about caching.
    """
    lat_mod = load_latency_module(repo)
    if lat_mod is None:
        return {}, {}

    probes = lat_mod.probe_runs(str(probe_root))
    figures, measured, subsamples = {}, {}, set()
    for (tutor_model, mode), block in probes.items():
        if mode != prompt:
            continue
        # Site ids mirror result-directory names, which flatten the provider
        # slash (deepseek-ai/DeepSeek-V4-Pro -> deepseek-ai_DeepSeek-V4-Pro).
        site_id = tutor_model.replace("/", "_")
        figs = lat_mod.probe_figures(block)
        if figs["ttft_p50"] is None:
            continue
        env = block.get("measurement_environment") or {}
        row = {"ttft_s": round(figs["ttft_p50"], 2)}
        for src, key in (
            ("ttlt_p50", "ttlt_s"),
            ("ttft_first_p50", "ttft_first_s"),
            ("ttft_later_p50", "ttft_later_s"),
        ):
            if figs[src] is not None:
                row[key] = round(figs[src], 2)
        figures[site_id] = row
        measured[site_id] = env.get("measured_at")
        subsamples.add((block.get("subsample") or {}).get("subsample_id"))

    provenance = {
        "mode": prompt,
        "subsample_id": _single_subsample(subsamples),
        "measured_at": measured,
    }
    return figures, provenance


def _single_subsample(subsamples: set) -> str | None:
    """The one subsample id the figures share, or None (and say so) if mixed."""
    if len(subsamples) > 1:
        print(
            f"probe runs mix {len(subsamples)} latency subsamples "
            f"({', '.join(sorted(str(i) for i in subsamples))}); those figures "
            "were measured over different prompts and must not be charted "
            "together -- re-measure the roster against one subsample",
            file=sys.stderr,
        )
    return next(iter(subsamples)) if len(subsamples) == 1 else None


def _cost_row(figs: dict, source: dict) -> dict:
    """One cost.json row from `tutormoments.costing.uncached_cost_figures`."""
    rates = figs["rates"][figs["model"]]
    return {
        "uncached_cost_per_response_usd": round(figs["uncached_cost_per_call_usd"], 8),
        "prompt_tokens_per_response": round(figs["prompt_tokens_per_call"]),
        "output_tokens_per_response": round(figs["output_tokens_per_call"]),
        "n_calls": figs["n_calls"],
        "rates": {
            "input_per_mtok": rates["input"],
            "output_per_mtok": rates["output"],
            "as_of": rates["as_of"],
        },
        "source": source,
    }


def measured_cost(repo: Path, results_root: Path, prompt: str) -> tuple[dict, dict]:
    """Uncached tutor cost per response per site model id.

    Returns ``({site_id: row}, provenance)``. Two sources, in order, both
    priced at the checkout's current registry rates:

    1. **A full benchmark run** (`tutormoments.costing.costed_runs`): every
       run made since usage-vector capture carries its tutor tokens and call
       count, so a model's cost comes free with the run that scores it, over
       all its moments rather than a subsample. Token counts do not depend on
       replay concurrency, so -- unlike latency -- a run is a clean source.
    2. **The latency probe** (`tutormoments.latency.probe_cost_figures`), for
       models whose runs predate capture and so cannot be priced.

    A model with neither contributes no row -- "not measured", never 0. Each
    row's ``source`` says which one it came from.
    """
    lat_mod = load_latency_module(repo)
    if lat_mod is None:
        return {}, {}
    from tutormoments import costing  # noqa: PLC0415 -- importable once lat_mod is

    figures, versions = {}, set()
    for (tutor_model, mode), figs in costing.costed_runs(str(results_root)).items():
        if mode != prompt:
            continue
        source = {
            "kind": "run",
            "run_id": figs["run_id"],
            "n_conversations": figs["n_conversations"],
        }
        figures[tutor_model.replace("/", "_")] = _cost_row(figs, source)
        versions.add(figs["pricing_version"])

    subsamples = set()
    for (tutor_model, mode), block in lat_mod.probe_runs(str(results_root)).items():
        site_id = tutor_model.replace("/", "_")
        if mode != prompt or site_id in figures:
            continue
        figs = lat_mod.probe_cost_figures(block)
        if not figs or figs["uncached_cost_per_call_usd"] is None:
            continue
        sub_id = (block.get("subsample") or {}).get("subsample_id")
        source = {
            "kind": "probe",
            "measured_at": (block.get("measurement_environment") or {}).get(
                "measured_at"
            ),
            "subsample_id": sub_id,
        }
        figures[site_id] = _cost_row(figs, source)
        subsamples.add(sub_id)
        versions.add(figs["pricing_version"])

    # Probe-sourced rows must still share one subsample (run rows each cover
    # a whole dataset and need no such check).
    _single_subsample(subsamples)
    provenance = {
        "mode": prompt,
        "pricing_version": versions.pop() if len(versions) == 1 else None,
    }
    return figures, provenance


def cost_rows(rows: list, figures: dict) -> tuple[list, list]:
    """Join cost figures onto the latency chart's rows.

    Returns ``(models, omitted_names)``. The roster and the y-axis score
    come from the latency rows, so the two charts plot the same models at the
    same heights; a probe for a model the site does not list is ignored, and
    a listed model without a measured cost is named in ``omitted`` rather
    than plotted at 0. A run of an arm the site does not list is ignored the
    same way.
    """
    models, omitted = [], []
    for row in rows:
        fig = figures.get(row["id"])
        if fig is None:
            omitted.append(row["name"])
            continue
        models.append(
            {"id": row["id"], "name": row["name"], "score": row["score"], **fig}
        )
    return models, omitted


def write_cost(repo: Path, results_root: Path, rows: list) -> None:
    """Write cost.json for the latency chart's rows, or leave it alone."""
    figures, provenance = measured_cost(repo, results_root, "scaffolding_rigor")
    models, omitted = cost_rows(rows, figures)
    if not models:
        print(
            "no costable benchmark or probe runs for the site's models — "
            "skipping cost.json",
            file=sys.stderr,
        )
        return
    write_json(
        "cost.json",
        {
            "source": (
                "score as in latency.json. Cost is the uncached list cost per "
                "tutor response: every prompt token at the model's input rate, "
                "output and reasoning tokens at its output rate, divided by the "
                "number of tutor calls. Measured from the model's full "
                "benchmark run, or from the `tutormoments latency` probe where "
                "that run predates usage capture (see each row's source). A "
                "ceiling -- provider prompt caching can cut the input share "
                "substantially. See docs/cost.md."
            ),
            "cost": provenance,
            "omitted": omitted,
            "models": models,
        },
    )
    print(f"  cost measured for {len(models)} model(s); omitted: {omitted or 'none'}")


def apply_ttft(rows: list, figures: dict) -> int:
    """Write TTFT figures onto latency.json rows in place; returns how many.

    Absent figures are *removed* rather than left standing: a stale ttft_s from
    an earlier subsample next to a freshly measured one is the failure the
    subsample_id exists to prevent. Rows the chart plots on TTFT therefore only
    ever carry a figure from the run just read.
    """
    n = 0
    for row in rows:
        # ttft_cold_s / ttft_warm_s are legacy: the split published under
        # those names keyed on cache state and is superseded by the turn-based
        # first/later one. Popped so a refreshed row cannot carry both.
        for key in (
            "ttft_s",
            "ttlt_s",
            "ttft_first_s",
            "ttft_later_s",
            "ttft_cold_s",
            "ttft_warm_s",
        ):
            row.pop(key, None)
        figs = figures.get(row["id"])
        if figs:
            row.update(figs)
            n += 1
    return n


TTFT_KEYS = ("ttft_s", "ttlt_s", "ttft_first_s", "ttft_later_s")


def reasoning_label(thinking: dict) -> str:
    """The leaderboard's reasoning cell: an arm's provider-native thinking
    parameters as the config writes them, e.g. ``{"effort": "high",
    "thinking": {"type": "adaptive"}}`` -> ``"thinking: adaptive, effort:
    high"``. ``include_thoughts`` is left out: it asks for thought summaries
    back and does not change how the model reasons."""

    def fmt(v):
        return str(v).lower() if isinstance(v, bool) else str(v)

    parts = []
    for key, value in sorted(thinking.items(), key=lambda kv: kv[0] != "thinking"):
        if key == "include_thoughts":
            continue
        if key == "thinking" and isinstance(value, dict) and set(value) == {"type"}:
            value = value["type"]
        parts.append(f"{key}: {fmt(value)}")
    return ", ".join(parts) or NO_REASONING_PARAM


def full_runs(results_root: Path) -> dict:
    """Latest full benchmark run per ``(site_id, mode)``.

    Returns ``{(site_id, mode): {"run_id", "summary", "config"}}``. Eligible
    runs replayed the whole dataset (no ``--sample``) with no failed moments
    -- the same rule `tutormoments.costing.costed_runs` applies -- and among
    those the newest run id (date suffix) wins. Plain JSON reads, so this half
    of the refresh works under a bare ``python3``.
    """
    out, best = {}, {}
    if not results_root.is_dir():
        return out
    for run in sorted(results_root.iterdir()):
        try:
            summary = json.loads((run / "summary.json").read_text("utf-8"))
            config = json.loads((run / "config.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("sample") is not None:
            continue
        if (summary.get("run_counts") or {}).get("failed", 0):
            continue
        if not (summary.get("tutor_model") and summary.get("scaffold_calibrated")):
            continue
        cell = (summary["tutor_model"].replace("/", "_"), summary.get("mode", ""))
        rank = (run.name.rsplit("_", 1)[-1], run.name)
        if cell not in best or rank > best[cell]:
            best[cell] = rank
            out[cell] = {"run_id": run.name, "summary": summary, "config": config}
    return out


def run_perf(summary: dict) -> dict:
    """The leaderboard's three figures from a run summary (same fields, and
    same derivation, as `perf` reads from a _full_combined scores.json)."""
    return {
        "scaffolding": round(summary["scaffold_calibrated"]["score"], 3),
        "rigor": round(summary["rigor_calibrated"]["score"], 3),
        "avoids_over": round(1.0 - summary["overscaffold"]["rate"], 3),
    }


def _read_rows(name: str) -> tuple[dict, dict]:
    """The committed ``static/data/<name>``: ``(payload, {id: row})``."""
    fp = OUT_DIR / name
    if not fp.exists():
        return {}, {}
    payload = json.loads(fp.read_text("utf-8"))
    return payload, {row["id"]: row for row in payload.get("models") or []}


def _paper_rows(bench: Path, model: str, label: str, ids: set) -> tuple:
    """Leaderboard + latency rows for a paper model from _full_combined."""
    scores, n = {}, None
    for prompt, key in PROMPTS.items():
        p = perf(bench, model, prompt)
        if p:
            n = n or p.pop("n")
            p.pop("n", None)
            scores[key] = p
    if len(scores) != len(PROMPTS):
        return None, None, None
    lb = {
        "id": model,
        "name": label,
        "reasoning": reasoning_label(PAPER_THINKING[model]),
        "source": "paper",
        **scores,
    }
    lat = latency(bench, model, "scaffolding_rigor", ids)
    lat_row = None
    if lat is not None:
        lat_row = {
            "id": model,
            "name": label,
            "latency_s": lat,
            "latency_estimated": False,
        }
    return lb, lat_row, n


def _run_rows(runs: dict, model: str, label: str) -> tuple:
    """Leaderboard + latency rows for a non-paper model from its full runs."""
    cells = {key: runs.get((model, prompt)) for prompt, key in PROMPTS.items()}
    if not all(cells.values()):
        return None, None, None
    ea = cells["eval_aware"]
    tutor = (ea["config"].get("resolved_tutors") or {}).get(
        ea["config"].get("arm")
    ) or {}
    lb = {
        "id": model,
        "name": label,
        "reasoning": reasoning_label(tutor.get("thinking") or {}),
        "source": {key: cell["run_id"] for key, cell in cells.items()},
        **{key: run_perf(cell["summary"]) for key, cell in cells.items()},
    }
    # End-to-end seconds per tutor turn under the run's --concurrency: kept
    # for correspondence with the paper's latency_s, never charted.
    mean = ((ea["summary"].get("latency") or {}).get("tutor") or {}).get("mean_seconds")
    lat_row = None
    if mean is not None:
        lat_row = {
            "id": model,
            "name": label,
            "latency_s": round(mean, 2),
            "latency_estimated": False,
        }
    return lb, lat_row, ea["summary"].get("n_scenarios")


def build_benchmark_json(repo: Path, probe_root: Path) -> None:
    """Rebuild leaderboard.json and latency.json (and cost.json from them).

    Every figure is re-measured where this checkout has its source and
    carried forward from the committed JSON where it does not: a checkout
    with only some of the results (probes are cheap to re-run, a scored sweep
    is not; the paper's _full_combined is not in most checkouts) must not
    throw away the rows it cannot rebuild.
    """
    bench = repo / "results" / "benchmark"
    ids_fp = bench / "_balanced_520_scenario_ids.json"
    ids = set(json.loads(ids_fp.read_text("utf-8"))) if ids_fp.exists() else set()
    runs = full_runs(probe_root)
    prior_lb, prior_lb_rows = _read_rows("leaderboard.json")
    prior_lat, prior_lat_rows = _read_rows("latency.json")

    lb_models, lat_models, n_moments, carried = [], [], None, []
    for model, label in MODELS:
        if model in PAPER_THINKING:
            lb, lat_row, n = (
                _paper_rows(bench, model, label, ids)
                if bench.exists()
                else (None, None, None)
            )
        else:
            lb, lat_row, n = _run_rows(runs, model, label)
        if lb is None:
            lb = prior_lb_rows.get(model)
            lat_row = prior_lat_rows.get(model)
            if lb is None:
                print(f"  no scores for {model} — leaving it off", file=sys.stderr)
                continue
            carried.append(model)
            if model in PAPER_THINKING:
                # Always from PAPER_THINKING, so a fix there reaches the page.
                lb = {
                    **lb,
                    "reasoning": reasoning_label(PAPER_THINKING[model]),
                    "source": "paper",
                }
            rest = {k: v for k, v in lb.items() if k not in ("id", "name")}
            lb = {"id": model, "name": label, **rest}
        else:
            n_moments = n_moments or n
            # The paper's exchanges may be absent even where its scores are
            # not; keep the latency_s already published rather than drop it.
            lat_row = lat_row or prior_lat_rows.get(model)
        lb_models.append(lb)
        if lat_row is not None:
            ea = lb["eval_aware"]
            lat_models.append(
                {
                    **{k: v for k, v in lat_row.items() if k not in TTFT_KEYS},
                    "name": label,
                    "score": round((ea["scaffolding"] + ea["rigor"]) / 2, 4),
                }
            )
    if carried:
        print(f"  carried forward from the committed JSON: {', '.join(carried)}")

    if not lb_models:
        print("no scores to write — skipping leaderboard.json", file=sys.stderr)
        return
    write_json(
        "leaderboard.json",
        {
            "source": (
                "Paper models: Table 8 of the TutorMoments-Preview working paper "
                "(2026-06-30). Other models: their full `tutormoments run` "
                "benchmark runs, run ids in each row's source. Regenerate with "
                "scripts/refresh-data.py."
            ),
            "n_moments": n_moments or prior_lb.get("n_moments"),
            "human": HUMAN,
            "models": lb_models,
        },
    )

    # ttft_s is only present for models with a probe run -- omitted rather
    # than zero-filled, so the chart can tell "not measured" from "fast". A
    # checkout with no probe runs at all keeps the published figures.
    figures, provenance = probe_ttft(repo, probe_root, "scaffolding_rigor")
    if figures:
        apply_ttft(lat_models, figures)
    else:
        print("  no probe runs — keeping the published ttft figures", file=sys.stderr)
        provenance = prior_lat.get("ttft")
        for row in lat_models:
            prior = prior_lat_rows.get(row["id"]) or {}
            row.update({k: prior[k] for k in TTFT_KEYS if k in prior})
    write_json(
        "latency.json",
        {
            "source": (
                "score = mean of evaluation-aware Appropriate Scaffolding and "
                "Appropriate Rigor, from leaderboard.json. ttft_s / ttlt_s "
                "measured by `tutormoments latency`, a serial probe: ttft_s is "
                "time to the first *answer* token (reasoning excluded), which "
                "the site calls TTFAT. ttft_first_s / ttft_later_s split it by "
                "turn position (the first message of a session vs turns 3 and "
                "5). See the ttft block for mode, subsample and timestamps. "
                "latency_s is end-to-end seconds per tutor turn from benchmark "
                "runs under --concurrency, kept for correspondence with the "
                "paper's Figure 7 but not displayed; read off that figure to "
                "~±0.2s where latency_estimated is true."
            ),
            "ttft": provenance,
            "models": lat_models,
        },
    )
    write_cost(repo, probe_root, lat_models)


def build_action_distribution(csv_path: Path, source: str) -> None:
    """Convert the repo's action_taxonomy_distribution.csv export into the site's
    action_distribution.json. Column layout: letter,name,orientation, then
    human__{n_moments,macro_mean_pct,ci_low,ci_high}, then per model
    <model>__{plain,SR}__{n_moments,macro_mean_pct,ci_low,ci_high}.
    Letter M (Other) is dropped, matching the paper figure."""
    import csv  # noqa: PLC0415

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = {row["letter"]: row for row in csv.DictReader(fh)}

    missing = [letter for letter in ACTION_LABELS if letter not in rows]
    if missing:
        print(f"{csv_path} is missing letters {missing} — skipping", file=sys.stderr)
        return

    csv_prompts = {"plain": "plain", "SR": "eval_aware"}

    def cell(row: dict, prefix: str) -> dict:
        return {
            "pct": round(float(row[f"{prefix}__macro_mean_pct"]), 2),
            "ci": [
                round(float(row[f"{prefix}__ci_low"]), 2),
                round(float(row[f"{prefix}__ci_high"]), 2),
            ],
        }

    # category order = descending human rate, as in the paper figure
    letters = sorted(
        ACTION_LABELS, key=lambda L: -float(rows[L]["human__macro_mean_pct"])
    )
    categories = [
        {
            "key": letter,
            "label": ACTION_LABELS[letter],
            "orientation": rows[letter]["orientation"],
            "human": cell(rows[letter], "human"),
        }
        for letter in letters
    ]

    models = []
    for model, label in MODELS:
        # Only the paper's models have an export; later runs' taxonomy
        # classifications are not in it.
        col = ACTION_CSV_MODELS.get(model)
        if col is None:
            continue
        entry: dict = {"id": model, "name": label}
        for csv_prompt, key in csv_prompts.items():
            entry[key] = {
                letter: cell(rows[letter], f"{col}__{csv_prompt}") for letter in letters
            }
        models.append(entry)

    write_json(
        "action_distribution.json",
        {
            "source": source,
            "n_human_moments": int(rows["A"]["human__n_moments"]),
            "categories": categories,
            "models": models,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "tutormoments_repo",
        type=Path,
        nargs="?",
        help="path to a local allenai/tutormoments checkout with results",
    )
    ap.add_argument(
        "--probe-root",
        type=Path,
        default=None,
        help="results root holding `tutormoments latency` probe runs and, for "
        "cost.json, `tutormoments run` benchmark runs (default: "
        "<checkout>/results, where both write unless given --results-root)",
    )
    ap.add_argument(
        "--action-csv",
        type=Path,
        default=None,
        help="path to action_taxonomy_distribution.csv (overrides the copy in the checkout)",
    )
    args = ap.parse_args()

    if not (args.tutormoments_repo or args.action_csv):
        ap.error("pass a tutormoments checkout path and/or --action-csv")

    if args.tutormoments_repo:
        repo = args.tutormoments_repo.expanduser().resolve()
        if not repo.exists():
            ap.error(f"{repo} does not exist")
        probe_root = (
            args.probe_root.expanduser().resolve()
            if args.probe_root
            else repo / "results"
        )
        build_benchmark_json(repo, probe_root)

    csv_path = args.action_csv or (
        repo
        / "analysis"
        / "working-paper-20260630"
        / "action_taxonomy_distribution.csv"
    )
    if csv_path.exists():
        build_action_distribution(
            csv_path.resolve(),
            f"Generated by scripts/refresh-data.py from {csv_path.name}",
        )
    else:
        print(f"no {csv_path} — skipping action_distribution.json", file=sys.stderr)


if __name__ == "__main__":
    main()
