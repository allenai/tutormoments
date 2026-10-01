"""Compare a built action_taxonomy release against the paper's frozen numbers.

For co-author review before the release is published; the report itself is
not published. It sets the regenerated per-category macro distribution
beside analysis/working-paper-20260630/v1_action_taxonomy_distribution.csv
(human and every model x prompt cell) and the regenerated S-vs-R KL
divergences beside the locked kl_divergence_table.tex.

The KL math is the paper notebook's (kl_divergence_table.ipynb): per
situation, macro % -> pseudo-counts over that situation's moments, add-one
smoothing, KL in nats.

    python -m tutormoments_build.action_taxonomy_compare \
        [--release data/action_taxonomy_release]   # writes <release>/comparison.md

Requires pandas (the `analysis` extra).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tutormoments import taxonomy as tx
from tutormoments_build.action_taxonomy import (
    DEFAULT_OUT,
    MANIFEST_FILENAME,
    RELEASE_FILENAME,
    SOURCE_HUMAN,
)

PAPER_DIR = Path(__file__).resolve().parents[1] / "analysis" / "working-paper-20260630"
PAPER_DISTRIBUTION = PAPER_DIR / "v1_action_taxonomy_distribution.csv"
PAPER_KL_TEX = PAPER_DIR / "kl_divergence_table.tex"

PROMPTS = ("plain", "scaffolding_rigor")
_PAPER_PROMPT = {"plain": "plain", "scaffolding_rigor": "SR"}

# benchmark_520 tutor_model -> (paper CSV series stem, KL-table row label), in
# the paper's display order.
PAPER_MODELS = {
    "claude-opus-4-8": ("claude_opus_4_8", "Claude Opus 4.8"),
    "claude-sonnet-4-6": ("claude_sonnet_4_6", "Claude Sonnet 4.6"),
    "deepseek-ai_DeepSeek-V4-Pro": ("deepseek_v4_pro", "DeepSeek V4 Pro"),
    "gemini-2.5-pro": ("gemini_2_5_pro", "Gemini 2.5 Pro"),
    "gemini-3.5-flash": ("gemini_3_5_flash", "Gemini 3.5 Flash"),
    "gpt-5.5-2026-04-23": ("gpt_5_5", "GPT 5.5"),
    "gpt-5.4-mini-2026-03-17": ("gpt_5_4_mini", "GPT 5.4 mini"),
}
HUMAN_LABEL = "Human tutors"

PAPER_N_PER_SITUATION = 50


def load_release(release_jsonl: Path):
    """Kept (classified) rows as a DataFrame with taxonomy's column names."""
    import pandas as pd

    rows = []
    with Path(release_jsonl).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["category"] is None:
                continue
            rows.append(
                {
                    "source": r["source"],
                    "model": r["tutor_model"],
                    "prompt": r["prompt_mode"],
                    "moment_id": r["moment_id"],
                    "situation_label": r["situation_label"],
                    "category": r["category"],
                    "orientation": r["orientation"],
                }
            )
    return pd.DataFrame(rows)


# -- KL (kl_divergence_table.ipynb) -------------------------------------------


def smoothed_probs(
    pcts: list[float], n_moments: int, alpha: float = 1.0
) -> list[float]:
    counts = [p / 100.0 * n_moments + alpha for p in pcts]
    total = sum(counts)
    return [c / total for c in counts]


def kl_nats(p: list[float], q: list[float]) -> float:
    return sum(pi * (math.log(pi) - math.log(qi)) for pi, qi in zip(p, q))


def _letter_pcts(dist) -> list[float]:
    return (
        dist.set_index("letter")["mean_pct"]
        .reindex(tx.CATEGORY_LETTERS, fill_value=0.0)
        .tolist()
    )


def kl_s_r(df, n_per_situation: int | None = None) -> tuple[float, float]:
    """(KL(S||R), KL(R||S)) between scaffolding and rigor moments of `df`.

    `n_per_situation` overrides the moment count the add-one smoothing
    pseudo-counts are scaled by -- the smoothing's pull toward uniform shrinks
    as n grows, so KL is only comparable at equal n.
    """
    dists = tx.macro_distribution(df, group_keys=("situation_label",))
    out = {}
    for sit in ("scaffolding", "rigor"):
        sub = dists[dists["situation_label"] == sit]
        if sub.empty:
            return (float("nan"), float("nan"))
        n = n_per_situation or int(sub["n_moments"].iloc[0])
        out[sit] = smoothed_probs(_letter_pcts(sub), n)
    s, r = out["scaffolding"], out["rigor"]
    return kl_nats(s, r), kl_nats(r, s)


def parse_kl_tex(tex: str) -> dict[str, tuple[float, ...]]:
    """Row label -> (plain S||R, plain R||S, eval S||R, eval R||S)."""
    out = {}
    for line in tex.splitlines():
        vals = re.findall(r"\\gradientwo\{([0-9.]+)\}", line)
        if len(vals) == 4:
            out[line.split("&")[0].strip()] = tuple(float(v) for v in vals)
    return out


# -- Report -------------------------------------------------------------------


def _series(df):
    """[(label, paper_series, frame)] for human then every paper cell."""
    out = [(HUMAN_LABEL, "human", df[df["source"] == SOURCE_HUMAN])]
    for model, (stem, label) in PAPER_MODELS.items():
        for prompt in PROMPTS:
            cell = df[(df["model"] == model) & (df["prompt"] == prompt)]
            out.append(
                (f"{label} / {prompt}", f"{stem}__{_PAPER_PROMPT[prompt]}", cell)
            )
    return out


def _paper_n_moments(paper_csv, series: str) -> int:
    import pandas as pd

    return int(pd.read_csv(paper_csv)[f"{series}__n_moments"].iloc[0])


def _f(x: float, nd: int = 3) -> str:
    return "--" if x != x else f"{x:.{nd}f}"


def build_report(
    release_dir: Path,
    paper_csv: Path = PAPER_DISTRIBUTION,
    kl_tex: Path = PAPER_KL_TEX,
) -> str:
    release_dir = Path(release_dir)
    upload = release_dir / "upload"
    manifest = json.loads((upload / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    df = load_release(upload / RELEASE_FILENAME)
    series = _series(df)

    new_dist, paper_dist = {}, {}
    for label, key, frame in series:
        d = tx.macro_distribution(frame)
        new_dist[label] = (_letter_pcts(d), int(d["n_moments"].iloc[0]))
        p = tx.read_paper_distribution(paper_csv, key)["mean_pct"]
        paper_dist[label] = (
            p.reindex(tx.CATEGORY_LETTERS).tolist(),
            _paper_n_moments(paper_csv, key),
        )

    lines = [
        "# Action-taxonomy release vs the paper's frozen numbers",
        "",
        "For co-author review before publishing; not itself published.",
        "",
        f"- Release: `{manifest['name']}` v{manifest['version']}, "
        f"{manifest['record_count']:,} rows, scheme `{manifest['scheme_version']}`, "
        f"file_sha256 `{manifest['file_sha256'][:12]}…`",
        f"- Inputs: `{manifest['inputs']['repo_id']}` @ "
        f"`{manifest['inputs']['revision']}`",
        f"- Classifier: `{manifest['classifier']['model']}`, thinking "
        f"`{json.dumps(manifest['classifier']['thinking'])}`, batch size "
        f"{manifest['classifier']['batch_size']}, "
        f"{manifest['classifier']['pools']}",
        f"- Paper side: `{paper_csv.name}` (Fig 4) and `{kl_tex.name}` (locked)",
        "",
        "## Why the numbers differ",
        "",
        "- **Different pools.** The paper classified 514 human moments and 100 "
        "moments per LM cell due to data availability at the time. This release "
        "covers the full release: every human scaffolding moment with a kept facet "
        f"({new_dist[HUMAN_LABEL][1]:,}) and every benchmark_520 replay "
        "(~520 per cell).",
        "- **A fresh LLM pass.** Same model, prompt and scheme, but a new "
        "classification run with different batch composition (separate human "
        "and LM pools), so individual labels can differ.",
        "- **KL grows with sample size.** The paper's KL adds one pseudo-count "
        "per category to (macro % × moments). With ~50 moments per situation "
        "(the paper's ~100 per cell) that pulls both distributions strongly "
        "toward uniform; with ~260 (this release) much less, so KL rises even "
        "if the underlying distributions are unchanged. The KL section below "
        f"also shows the new values smoothed as if n = {PAPER_N_PER_SITUATION} "
        "per situation.",
        "- Distributions are macro means over moments (kept facets only), "
        "as in the paper.",
        "",
        "## Summary per series",
        "",
        "JS = Jensen-Shannon divergence (base 2, 0-1) between the regenerated "
        "and paper distributions; max |Δ| is the largest per-category "
        "difference in percentage points.",
        "",
        "| Series | moments (paper → new) | JS | max \\|Δ\\| pp | letter |",
        "|---|---|---|---|---|",
    ]
    for label, _, _ in series:
        (new, n_new), (old, n_old) = new_dist[label], paper_dist[label]
        deltas = [a - b for a, b in zip(new, old)]
        i = max(range(len(deltas)), key=lambda k: abs(deltas[k]))
        js = tx.js_divergence([v / 100 for v in new], [v / 100 for v in old])
        lines.append(
            f"| {label} | {n_old} → {n_new} | {js:.3f} | "
            f"{deltas[i]:+.1f} | {tx.CATEGORY_LETTERS[i]} |"
        )

    (h_new, _), (h_old, _) = new_dist[HUMAN_LABEL], paper_dist[HUMAN_LABEL]
    lines += [
        "",
        "## Human distribution (macro %)",
        "",
        "| | Category | Paper | New | Δ pp |",
        "|---|---|---|---|---|",
    ]
    for letter, old, new in zip(tx.CATEGORY_LETTERS, h_old, h_new):
        lines.append(
            f"| {letter} | {tx.NAME_BY_LETTER[letter]} | {old:.1f} | {new:.1f} | "
            f"{new - old:+.1f} |"
        )

    cells = [s for s in series if s[0] != HUMAN_LABEL]
    lines += [
        "",
        "## LM cells: Δ pp (new − paper) per category",
        "",
        "| Cell | " + " | ".join(tx.CATEGORY_LETTERS) + " |",
        "|---|" + "---|" * len(tx.CATEGORY_LETTERS),
    ]
    for label, _, _ in cells:
        new, old = new_dist[label][0], paper_dist[label][0]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{a - b:+.1f}" for a, b in zip(new, old))
            + " |"
        )

    locked = parse_kl_tex(Path(kl_tex).read_text(encoding="utf-8"))
    lines += [
        "",
        "## KL divergence, scaffolding (S) vs rigor (R) moments (nats)",
        "",
        f"Each cell is paper → new (new at n = {PAPER_N_PER_SITUATION} per situation).",
        "",
        "| Tutor | Plain KL(S‖R) | Plain KL(R‖S) | Eval-aware KL(S‖R) | Eval-aware KL(R‖S) |",
        "|---|---|---|---|---|",
    ]

    def kl_row(frame, both_prompts: bool):
        full = kl_s_r(frame)
        at_n = kl_s_r(frame, PAPER_N_PER_SITUATION)
        return (full + full, at_n + at_n) if both_prompts else (full, at_n)

    table = [(HUMAN_LABEL, *kl_row(df[df["source"] == SOURCE_HUMAN], True))]
    for model, (_, label) in PAPER_MODELS.items():
        full, at_n = (), ()
        for prompt in PROMPTS:
            f, a = kl_row(df[(df["model"] == model) & (df["prompt"] == prompt)], False)
            full, at_n = full + f, at_n + a
        table.append((label, full, at_n))
    for label, full, at_n in table:
        old = locked.get(label, (float("nan"),) * 4)
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{_f(o)} → {_f(n)} ({_f(a)})" for o, n, a in zip(old, full, at_n)
            )
            + " |"
        )
    lines += [
        "",
        "Human S and R use `situation_label_agg` ∈ {scaffolding, rigor}; mixed / "
        "neither / both moments are not in either side (as in the paper).",
        "",
        "Reading the table: at matched n the LM values land near the paper's, "
        "so for LMs the rise is mostly the smoothing. The human gap is not: the "
        "regenerated human pool separates S from R well beyond the paper's "
        "value at any n, and the human distribution itself shifts (see above). "
        "The paper's human sample was the balanced_520 benchmark subset (514 of "
        "those 520 moments, ~257 per situation), so the gap is a change of "
        "population rather than of sample size.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", default=DEFAULT_OUT, metavar="DIR")
    args = parser.parse_args(argv)
    out = Path(args.release) / "comparison.md"
    out.write_text(build_report(Path(args.release)), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
