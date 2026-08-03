#!/usr/bin/env python3
"""Regenerate the site's chart data from a local allenai/tutorsim checkout.

The tutorsim repo gitignores its results, so the website keeps its own copies as
static JSON. Re-run this after benchmarking new models:

    python3 scripts/refresh-data.py /path/to/tutorsim

Reads (same sources as the repo's analysis/working-paper-20260630 scripts):
  results/benchmark/_full_combined/<model>__<prompt>/scores.json   -> leaderboard.json
  results/benchmark/<model>_v10_<prompt>_tutor_oracle_student*/exchanges/*.json
                                                                   -> latency.json
  data/taxonomy/{human,lm}/classified.csv (via tutorsim.taxonomy)  -> action_distribution.json

Writes to static/data/. Sections of the site hide automatically when their JSON
is absent, so partial refreshes are fine.
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
    print(f"wrote {path.relative_to(SITE_ROOT)}")


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
    balanced-520 ids, de-dupe by scenario_id, first wins)."""
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


def build_benchmark_json(repo: Path) -> None:
    bench = repo / "results" / "benchmark"
    if not bench.exists():
        print(f"no {bench} — skipping leaderboard.json and latency.json", file=sys.stderr)
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
            lat_models.append({
                "id": model, "name": label,
                "latency_s": lat, "latency_estimated": False,
                "score": round((ea["scaffolding"] + ea["rigor"]) / 2, 4),
            })

    if lb_models:
        write_json("leaderboard.json", {
            "source": f"Generated by scripts/refresh-data.py from {repo}",
            "n_moments": n_moments,
            "human": HUMAN,
            "models": lb_models,
        })
    if lat_models:
        write_json("latency.json", {
            "source": f"Generated by scripts/refresh-data.py from {repo}",
            "models": lat_models,
        })


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
            "ci": [round(float(row[f"{prefix}__ci_low"]), 2), round(float(row[f"{prefix}__ci_high"]), 2)],
        }

    # category order = descending human rate, as in the paper figure
    letters = sorted(ACTION_LABELS, key=lambda L: -float(rows[L]["human__macro_mean_pct"]))
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
            entry[key] = {letter: cell(rows[letter], f"{col}__{csv_prompt}") for letter in letters}
        models.append(entry)

    write_json("action_distribution.json", {
        "source": source,
        "n_human_moments": int(rows["A"]["human__n_moments"]),
        "categories": categories,
        "models": models,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tutorsim_repo", type=Path, nargs="?",
                    help="path to a local allenai/tutorsim checkout with results")
    ap.add_argument("--action-csv", type=Path, default=None,
                    help="path to action_taxonomy_distribution.csv (overrides the copy in the checkout)")
    args = ap.parse_args()

    if not (args.tutorsim_repo or args.action_csv):
        ap.error("pass a tutorsim checkout path and/or --action-csv")

    if args.tutorsim_repo:
        repo = args.tutorsim_repo.expanduser().resolve()
        if not repo.exists():
            ap.error(f"{repo} does not exist")
        build_benchmark_json(repo)

    csv_path = args.action_csv or (repo / "analysis" / "working-paper-20260630" / "action_taxonomy_distribution.csv")
    if csv_path.exists():
        build_action_distribution(csv_path.resolve(), f"Generated by scripts/refresh-data.py from {csv_path.name}")
    else:
        print(f"no {csv_path} — skipping action_distribution.json", file=sys.stderr)


if __name__ == "__main__":
    main()
