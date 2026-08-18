"""Benchmark results: paper table + perf-vs-cost figures, in one place.

Run:  python analysis/working-paper-20260630/benchmark_perf_cost.py

Reads canonical 520-scenario performance verbatim from
``results/benchmark/_full_combined/<model>__<prompt>/scores.json`` (the files
behind ``leaderboard.md``) and aggregates tutor latency/tokens from the per-run
exchange JSONs (filtered to the balanced-520 ids, de-duped by ``scenario_id``).

Writes to analysis/working-paper-20260630/figures/:
  results_table.tex   the paper table (plain vs eval-aware, 3 metrics each)
  results_table.md    full table incl. latency/token columns
  latency_vs_perf.{pdf,png}   latency vs. mean(Appropriate Scaffolding, Appropriate Rigor)

The y-axis of the latency plot = mean(scaffold_calibrated, rigor_calibrated),
i.e. the mean of the Appropriate-Scaffolding and Appropriate-Rigor rates in the
paper table. The token-usage plot was cut (cost tracking deferred); output-token
columns remain in results_table.md for reference.

`tutor_latencies` is end-to-end wall-clock seconds per tutor call and means the
same thing before and after the streaming work, so this figure stays comparable
across runs. It has never captured time to *first* token; for that (and for
cross-model-comparable latency generally) use `tutormoments latency` and see
docs/latency.md.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results" / "benchmark"
FIGDIR = Path(__file__).resolve().parent / "figures"

PROMPTS = ("plain", "scaffolding_rigor")

# Display label + paper row order (keyed by the model dir prefix).
MODELS = (
    ("claude-opus-4-8", "Claude Opus 4.8"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("deepseek-ai_DeepSeek-V4-Pro", "DeepSeek V4 Pro"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("gpt-5.5-2026-04-23", "GPT 5.5"),
    ("gpt-5.4-mini-2026-03-17", "GPT 5.4 mini"),
)

# Per-LM color + marker, shared with the action-distribution / dumbbell figures
# so a model reads the same across every figure in the paper. Okabe-Ito
# colorblind-safe palette plus one pink; circle 'o' reserved for the human row.
MODEL_COLORS = {
    "claude-opus-4-8": "#D55E00",
    "claude-sonnet-4-6": "#E69F00",
    "deepseek-ai_DeepSeek-V4-Pro": "#CC79A7",
    "gemini-2.5-pro": "#0072B2",
    "gemini-3.5-flash": "#56B4E9",
    "gpt-5.5-2026-04-23": "#009E73",
    "gpt-5.4-mini-2026-03-17": "#F0529C",
}
MODEL_MARKERS = {
    "claude-opus-4-8": "v",
    "claude-sonnet-4-6": "s",
    "deepseek-ai_DeepSeek-V4-Pro": "^",
    "gemini-2.5-pro": "D",
    "gemini-3.5-flash": "p",
    "gpt-5.5-2026-04-23": "P",
    "gpt-5.4-mini-2026-03-17": "X",
}


# --------------------------------------------------------------------------- #
# Pure aggregation (unit-tested in tests/analysis/)                           #
# --------------------------------------------------------------------------- #
def summarize_exchanges(exchanges: list[dict], id_set: set[str] | None = None) -> dict:
    """Aggregate tutor latency/tokens over exchange records.

    Filters to ``id_set`` (if given) and de-dupes by ``scenario_id`` (first wins).
    """
    seen: set[str] = set()
    turn_latencies: list[float] = []
    session_tokens: list[int] = []
    out_per_turn: list[float] = []
    for ex in exchanges:
        sid = ex.get("scenario_id")
        if (id_set is not None and sid not in id_set) or sid in seen:
            continue
        seen.add(sid)
        lats = ex.get("tutor_latencies") or []
        usage = ex.get("tutor_usage") or {}
        turn_latencies.extend(lats)
        if usage.get("total_tokens"):
            session_tokens.append(usage["total_tokens"])
        if lats and usage.get("output_tokens") is not None:
            out_per_turn.append(usage["output_tokens"] / len(lats))
    return {
        "n_scenarios": len(seen),
        "tutor_latency_mean_s": statistics.mean(turn_latencies)
        if turn_latencies
        else None,
        "tutor_output_tokens_per_turn": statistics.mean(out_per_turn)
        if out_per_turn
        else None,
    }


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def _perf(model: str, prompt: str) -> dict:
    d = json.loads(
        (BENCH / "_full_combined" / f"{model}__{prompt}" / "scores.json").read_text(
            "utf-8"
        )
    )
    scaf, rig = d["scaffold_calibrated"]["score"], d["rigor_calibrated"]["score"]
    return {
        "n": d["n_scenarios"],
        "scaffold_cal": scaf,
        "rigor_cal": rig,
        "avoid_over": 1.0 - d["overscaffold"]["rate"],
        "composite": (scaf + rig) / 2.0,
    }


def _cost(model: str, prompt: str, ids: set[str]) -> dict:
    exchanges = []
    needle = f"{model}_v10_{prompt}_tutor_oracle_student"
    for run in BENCH.iterdir():
        if not (run.is_dir() and run.name.startswith(needle)):
            continue
        for fp in (run / "exchanges").rglob("*.json"):
            try:
                exchanges.append(json.loads(fp.read_text("utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
    return summarize_exchanges(exchanges, ids)


def load_table() -> pd.DataFrame:
    ids = set(
        json.loads((BENCH / "_balanced_520_scenario_ids.json").read_text("utf-8"))
    )
    rows = []
    for model, label in MODELS:
        for prompt in PROMPTS:
            rows.append(
                {
                    "model": model,
                    "label": label,
                    "prompt": prompt,
                    **_perf(model, prompt),
                    **_cost(model, prompt, ids),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Outputs                                                                     #
# --------------------------------------------------------------------------- #
def paper_latex(df: pd.DataFrame) -> str:
    def cell(model, prompt, col):
        v = df[(df.model == model) & (df.prompt == prompt)][col].iloc[0]
        return f"{v:.3f}"

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \begin{tabular}{lccc ccc}",
        r"        \toprule",
        r"        & \multicolumn{3}{c}{\textbf{Plain prompt}} & \multicolumn{3}{c}{\textbf{Evaluation-aware prompt}} \\",
        r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"        \textbf{Tutor} & $\checkmark$Scaffold & $\checkmark$Rigor & $\neg$Over-scaffolding & $\checkmark$Scaffold & $\checkmark$Rigor & $\neg$Over-scaffolding \\",
        r"        \midrule",
    ]
    for model, label in MODELS:
        c = [
            cell(model, "plain", "scaffold_cal"),
            cell(model, "plain", "rigor_cal"),
            cell(model, "plain", "avoid_over"),
            cell(model, "scaffolding_rigor", "scaffold_cal"),
            cell(model, "scaffolding_rigor", "rigor_cal"),
            cell(model, "scaffolding_rigor", "avoid_over"),
        ]
        lines.append(f"        {label:<18} & " + " & ".join(c) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figures(df: pd.DataFrame) -> None:
    """Latency vs. score scatter (eval-aware prompt). Legend outside the axes.

    y = mean of the Appropriate-Scaffolding (S) and Appropriate-Rigor (R) rates
    from Table 7. x = mean tutor latency per turn (see caption).
    """
    sub = df[df.prompt == "scaffolding_rigor"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, label in MODELS:
        r = sub[sub.model == model]
        if not len(r):
            continue
        ax.scatter(
            r["tutor_latency_mean_s"],
            r["composite"],
            s=80,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label=label,
        )
    ax.set_xlabel("Mean tutor latency per turn (s)", fontsize=11)
    ax.set_ylabel("Mean of Appropriate Scaffolding & Appropriate Rigor", fontsize=11)
    ax.grid(axis="both", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(FIGDIR / "latency_vs_perf.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "latency_vs_perf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    df = load_table()
    tex = paper_latex(df)
    (FIGDIR / "results_table.tex").write_text(tex, encoding="utf-8")
    cols = [
        "label",
        "prompt",
        "n",
        "scaffold_cal",
        "rigor_cal",
        "avoid_over",
        "composite",
        "tutor_latency_mean_s",
        "tutor_output_tokens_per_turn",
    ]
    (FIGDIR / "results_table.md").write_text(
        df[cols].round(3).to_markdown(index=False), encoding="utf-8"
    )
    figures(df)
    print(tex)
    print("wrote: results_table.tex, results_table.md, latency_vs_perf.{pdf,png}")


if __name__ == "__main__":
    main()
