#!/usr/bin/env python3
"""Regenerate the site's chart data from a local allenai/tutormoments checkout.

The tutormoments repo gitignores its results, so the website keeps its own copies as
static JSON. Re-run this after benchmarking new models:

    python3 scripts/refresh-data.py /path/to/tutormoments

Reads (same sources as the repo's analysis/working-paper-20260630 scripts):
  results/benchmark/_full_combined/<model>__<prompt>/scores.json   -> leaderboard.json
  results/benchmark/<model>_v10_<prompt>_tutor_oracle_student*/exchanges/*.json
                                                                   -> latency.json (latency_s)
  results/<model>_<prompt>_latency_<date>/latency.json             -> latency.json (ttft_s)
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

# Display label per model dir prefix, in paper row order. Add new models here
# (id must match the directory prefix under results/benchmark/_full_combined and
# the model id used in taxonomy classified.csv with "/" replaced by "_").
MODELS = [
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("deepseek-ai_DeepSeek-V4-Pro", "DeepSeek V4 Pro"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("gpt-5.5-2026-04-23", "GPT 5.5"),
    ("gpt-5.4-mini-2026-03-17", "GPT 5.4 mini"),
]

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

    Returns ``({site_id: {ttft_s, ttft_cold_s?, ttft_warm_s?, measured_at}},
    provenance)``.

    Only probe runs are read. A benchmark run also records TTFT, but under
    --concurrency, which distorts it by a model-dependent amount and makes it
    incomparable across models -- exactly the comparison this chart invites.

    `ttft_s` is the pooled p50 over all samples: the only figure measured
    identically on all four providers, since Gemini reports no cache tokens and
    so has neither a cold nor a warm bucket. The cold/warm split is carried
    alongside it when the runtime says that provider's split is publishable.
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
        figs = lat_mod.probe_figures(block.get("tutor") or {})
        if figs["ttft_p50"] is None:
            continue
        env = block.get("measurement_environment") or {}
        row = {"ttft_s": round(figs["ttft_p50"], 2)}
        if figs["ttft_cold_p50"] is not None:
            row["ttft_cold_s"] = round(figs["ttft_cold_p50"], 2)
        if figs["ttft_warm_p50"] is not None:
            row["ttft_warm_s"] = round(figs["ttft_warm_p50"], 2)
        figures[site_id] = row
        measured[site_id] = env.get("measured_at")
        subsamples.add((block.get("subsample") or {}).get("subsample_id"))

    if len(subsamples) > 1:
        print(
            f"probe runs mix {len(subsamples)} latency subsamples "
            f"({', '.join(sorted(str(i) for i in subsamples))}); those figures "
            "were measured over different prompts and must not be charted "
            "together -- re-measure the roster against one subsample",
            file=sys.stderr,
        )
    provenance = {
        "mode": prompt,
        "subsample_id": subsamples.pop() if len(subsamples) == 1 else None,
        "measured_at": measured,
    }
    return figures, provenance


def apply_ttft(rows: list, figures: dict) -> int:
    """Write TTFT figures onto latency.json rows in place; returns how many.

    Absent figures are *removed* rather than left standing: a stale ttft_s from
    an earlier subsample next to a freshly measured one is the failure the
    subsample_id exists to prevent. Rows the chart plots on TTFT therefore only
    ever carry a figure from the run just read.
    """
    n = 0
    for row in rows:
        for key in ("ttft_s", "ttft_cold_s", "ttft_warm_s"):
            row.pop(key, None)
        figs = figures.get(row["id"])
        if figs:
            row.update(figs)
            n += 1
    return n


def refresh_ttft_only(repo: Path, probe_root: Path) -> None:
    """Update ttft_s in the existing latency.json, leaving the rest alone.

    The two figures in latency.json come from different places: latency_s from
    benchmark runs, ttft_s from probe runs. A checkout can easily have the
    probes without the benchmark results (probes are cheap to re-run; a full
    scored sweep is not), and in that case rebuilding the file wholesale would
    throw away the scores it already carries.
    """
    fp = OUT_DIR / "latency.json"
    if not fp.exists():
        print(f"no {fp} to update — skipping ttft_s", file=sys.stderr)
        return
    payload = json.loads(fp.read_text("utf-8"))
    figures, provenance = probe_ttft(repo, probe_root, "scaffolding_rigor")
    if not figures:
        return
    n = apply_ttft(payload.get("models") or [], figures)
    # Keep provenance above the rows it describes.
    payload = {
        "source": payload.get("source"),
        "ttft": provenance,
        "models": payload.get("models") or [],
    }
    write_json("latency.json", payload)
    print(f"  ttft_s updated for {n} model(s) from probe runs in {probe_root}")


def build_benchmark_json(repo: Path, probe_root: Path) -> None:
    bench = repo / "results" / "benchmark"
    if not bench.exists():
        print(
            f"no {bench} — skipping leaderboard.json scores and latency_s",
            file=sys.stderr,
        )
        refresh_ttft_only(repo, probe_root)
        return

    ids_fp = bench / "_balanced_520_scenario_ids.json"
    ids = set(json.loads(ids_fp.read_text("utf-8"))) if ids_fp.exists() else set()

    lb_models, lat_models, n_moments = [], [], None
    for model, label in MODELS:
        scores = {}
        for prompt, key in PROMPTS.items():
            p = perf(bench, model, prompt)
            if p:
                n_moments = n_moments or p.pop("n")
                p.pop("n", None)
                scores[key] = p
        if len(scores) != len(PROMPTS):
            continue
        lb_models.append({"id": model, "name": label, **scores})

        lat = latency(bench, model, "scaffolding_rigor", ids)
        ea = scores["eval_aware"]
        if lat is not None:
            row = {
                "id": model,
                "name": label,
                "latency_s": lat,
                "latency_estimated": False,
                "score": round((ea["scaffolding"] + ea["rigor"]) / 2, 4),
            }
            lat_models.append(row)

    if lb_models:
        write_json(
            "leaderboard.json",
            {
                "source": f"Generated by scripts/refresh-data.py from {repo}",
                "n_moments": n_moments,
                "human": HUMAN,
                "models": lb_models,
            },
        )
    if lat_models:
        # ttft_s is only present for models with a probe run -- omitted rather
        # than zero-filled, so the chart can tell "not measured" from "fast".
        figures, provenance = probe_ttft(repo, probe_root, "scaffolding_rigor")
        apply_ttft(lat_models, figures)
        write_json(
            "latency.json",
            {
                "source": f"Generated by scripts/refresh-data.py from {repo}",
                "ttft": provenance,
                "models": lat_models,
            },
        )


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
        col = ACTION_CSV_MODELS[model]
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
        help="results root holding `tutormoments latency` probe runs "
        "(default: <checkout>/results, where the probe writes unless given "
        "--results-root)",
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
