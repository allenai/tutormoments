"""Per-set aggregate rate math for tutormoments benchmark results.

Computes the three metrics the paper reports:

  Appropriate Scaffolding   scaffold_calibrated.score  = clean scaffolds / scaffolding-gold moments. Higher better.
  Appropriate Rigor         rigor_calibrated.score     = clean rigor pushes / rigor-gold moments.    Higher better.
  Avoids Over-Scaffolding   1 - overscaffold.rate (computed at report time).                         Higher better.

  "clean" = right action direction AND no over-scaffold facets emitted.

The collapsed action label decomposes via _ACTION_LABEL_TO_DIMENSIONS:
  both        -> scaffolding=yes, rigor=yes
  scaffolding -> scaffolding=yes, rigor=no
  rigor       -> scaffolding=no,  rigor=yes
  neither     -> scaffolding=no,  rigor=no

Inputs (Scenario + Annotation objects):
  scenario.dimension                  gold dimension (= scenario.rubric["gold"], set at build time)
  annotation.action_label             the first and only Annotation per scenario
  annotation.overscaffold_decomposed  non-empty list means over-scaffold detected; the
                                      field always exists on the Annotation dataclass,
                                      so it is treated as always available.

No module-level SDK imports.
"""
# ---------------------------------------------------------------------------
# Constants + helpers
# ---------------------------------------------------------------------------

_ACTION_LABEL_TO_DIMENSIONS = {
    "both": ("yes", "yes"),
    "scaffolding": ("yes", "no"),
    "rigor": ("no", "yes"),
    "neither": ("no", "no"),
}


def _to_dims(label):
    return _ACTION_LABEL_TO_DIMENSIONS.get(label)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate(scenarios: list, annotations: list) -> dict:
    """Compute the paper's three metrics from scored annotations.

    outcome_pos_rate and the did-rates are not computed (both dropped
    from the paper).

    Args:
        scenarios: list of tutormoments.moments.Moment objects (canonical order).
        annotations: list of tutormoments.scoring.Annotation objects, parallel to scenarios.

    Returns:
        {
          "n_scenarios": int,
          "overscaffold":    {"n_yes": int, "n_total": int, "rate": float|None, "available": bool},
          "scaffold_calibrated": {"n_clean_yes": int, "n_overscaffold": int, "n_total": int, "score": float|None},
          "rigor_calibrated":    {"n_clean_yes": int, "n_total": int, "score": float|None},
        }

    scaffold_calibrated.score = scaffolding-gold scenarios where the LM took the
                            right action (action_label in {scaffolding, both})
                            AND emitted no over-scaffold facets, divided by all
                            scaffolding-gold scenarios. (Appropriate Scaffolding)
    rigor_calibrated.score  = analogous for rigor-gold scenarios. (Appropriate Rigor)
    overscaffold.rate       = scenarios with non-empty overscaffold_decomposed
                            divided by total scenarios. Reported to readers as
                            Avoids Over-Scaffolding = 1 - rate (see leaderboard).
                            available=False if the overscaffold_decomposed field
                            was never present (not applicable for the new
                            Annotation dataclass -- field always exists, so
                            available is always True when any scenario is
                            processed).

    Scenarios where gold is mixed/unknown/neither/unclear are excluded from
    BOTH calibrated denominators -- they don't carry a clean direction.
    """
    scaf_total = 0
    rig_total = 0
    over_yes = 0
    any_overscaffold_field = False
    n_total = 0

    # Calibrated scoring:
    #   scaffold_calibrated = n_scaffolded_cleanly / n_scaffold_moments
    #   rigor_calibrated    = n_rigor_pushed_cleanly / n_rigor_moments
    # "cleanly" = right action direction AND no over-scaffold facets.
    # Both axes are symmetric: count clean moments / total moments. Range [0, 1].
    #
    # NOTE: an earlier version subtracted n_over_scaffolded from the scaffold
    # numerator. That double-penalized -- a moment that over-scaffolds is
    # already excluded from n_clean_yes (clean requires no over-scaffold), so
    # subtracting it again counted it twice. The subtraction was removed;
    # scaf_over_yes is kept as a reported component only (not in the score).
    scaf_clean_yes = 0  # scaffold-gold + action right + no over-scaffold
    scaf_over_yes = 0  # scaffold-gold + over-scaffold emitted (reported, not scored)
    rig_clean_yes = 0  # rigor-gold + action right + no over-scaffold

    for scenario, annotation in zip(scenarios, annotations):
        n_total += 1

        # scenario.dimension is set from situation_label_agg at build time.
        gt_label = scenario.dimension

        pred_label = annotation.action_label
        # Treat empty/None as absent.
        if not isinstance(pred_label, str) or not pred_label:
            pred_label = None
        pred_dims = _to_dims(pred_label)

        # annotation.overscaffold_decomposed is always present as a list field;
        # non-empty means over-scaffold detected.
        has_over = bool(annotation.overscaffold_decomposed)

        if gt_label == "scaffolding":
            scaf_total += 1
            action_right = pred_dims is not None and pred_dims[0] == "yes"
            if action_right and not has_over:
                scaf_clean_yes += 1
            if has_over:
                scaf_over_yes += 1
        elif gt_label == "rigor":
            rig_total += 1
            action_right = pred_dims is not None and pred_dims[1] == "yes"
            if action_right and not has_over:
                rig_clean_yes += 1
        # mixed / both / neither / unknown / unclear -> excluded from both
        # calibrated denominators. Still counts toward overscaffold.

        # The Annotation dataclass always has the overscaffold_decomposed field,
        # so any_overscaffold_field becomes True as soon as we process any
        # scenario (available=True whenever n_total > 0).
        any_overscaffold_field = True  # field always present on Annotation dataclass
        if has_over:
            over_yes += 1

    def _rate(yes, total):
        return (yes / total) if total else None

    return {
        "n_scenarios": n_total,
        "overscaffold": {
            "n_yes": over_yes,
            "n_total": n_total,
            "rate": _rate(over_yes, n_total),
            "available": any_overscaffold_field,
        },
        # Calibrated scores -- subsume did-rate + over-scaffold into one
        # number per axis. Components exposed so other formulas can be
        # recomputed from the same data without re-running annotation.
        "scaffold_calibrated": {
            "n_clean_yes": scaf_clean_yes,
            "n_overscaffold": scaf_over_yes,
            "n_total": scaf_total,
            "score": _rate(scaf_clean_yes, scaf_total),
        },
        "rigor_calibrated": {
            "n_clean_yes": rig_clean_yes,
            "n_total": rig_total,
            "score": _rate(rig_clean_yes, rig_total),
        },
    }


def leaderboard_row(summary_or_metrics: dict) -> dict:
    """Add avoids_overscaffold = 1 - overscaffold["rate"] (None-safe) to a metrics dict.

    avoids_overscaffold is higher=better (inverse of overscaffold rate).
    Returns a new dict with all original keys plus "avoids_overscaffold".
    """
    row = dict(summary_or_metrics)
    over = summary_or_metrics.get("overscaffold", {})
    rate = over.get("rate") if isinstance(over, dict) else None
    row["avoids_overscaffold"] = (1.0 - rate) if rate is not None else None
    return row


# ---------------------------------------------------------------------------
# Leaderboard columns -- the paper's three metrics, under the paper's names,
# plus identity and latency/token diagnostics.
# ---------------------------------------------------------------------------

_LEADERBOARD_COLS = [
    # (summary_key, header_label)
    ("tutor_model", "tutor_model"),
    ("condition", "condition"),
    ("mode", "mode"),
    ("n", "n"),
    ("appropriate_scaffolding", "appropriate_scaffolding"),
    ("appropriate_rigor", "appropriate_rigor"),
    ("avoids_overscaffold", "avoids_overscaffold"),
    # From `tutormoments latency` (serial probe), joined on (tutor_model,
    # mode). ttft_p50 is the headline: pooled over all samples, one number per
    # model. first/later split on turn position -- the first message of a
    # session against turns 3 and 5 -- which is ground truth on every
    # provider, unlike cache state.
    ("ttft_p50", "ttft_p50"),
    ("ttft_first_p50", "ttft_first_p50"),
    ("ttft_later_p50", "ttft_later_p50"),
    # From the benchmark run: end-to-end wall clock under --concurrency, so
    # comparable within a model over time but not across models.
    ("tutor_lat_p50", "tutor_lat_p50"),
    ("tutor_lat_p95", "tutor_lat_p95"),
    ("tokens_total", "tokens_total"),
]


def _extract_row(summary: dict, probes: dict | None = None) -> dict:
    """Pull the leaderboard-column values out of a run summary dict.

    *probes* is ``{(tutor_model, mode): latency.json block}`` from
    `tutormoments.latency.probe_runs`; a cell with no probe gets ``None`` TTFT
    columns rather than a figure borrowed from the run, which was gathered
    under concurrency and means something else.
    """
    cal_s = summary.get("scaffold_calibrated") or {}
    cal_r = summary.get("rigor_calibrated") or {}
    over = summary.get("overscaffold") or {}
    lat = (summary.get("latency") or {}).get("tutor") or {}
    tokens = (summary.get("tokens") or {}).get("total") or {}

    over_rate = over.get("rate")
    avoids = (1.0 - over_rate) if isinstance(over_rate, (int, float)) else None

    tutor_model = summary.get("tutor_model", "")
    mode = summary.get("mode", "")
    # Lazy: this module keeps no module-level SDK imports, and the latency
    # module reaches the provider clients.
    from tutormoments.latency import probe_figures  # noqa: PLC0415

    ttft = probe_figures((probes or {}).get((tutor_model, mode)) or {})

    return {
        "tutor_model": tutor_model,
        "condition": summary.get("condition", ""),
        "mode": mode,
        "n": summary.get("n_scenarios", 0),
        "appropriate_scaffolding": cal_s.get("score"),
        "appropriate_rigor": cal_r.get("score"),
        "avoids_overscaffold": avoids,
        # Absent unless this cell has a probe run: a benchmark run records
        # TTFT too, but under --concurrency, which distorts it by a
        # model-dependent amount. See docs/latency.md.
        "ttft_p50": ttft["ttft_p50"],
        "ttft_first_p50": ttft["ttft_first_p50"],
        "ttft_later_p50": ttft["ttft_later_p50"],
        # End-to-end wall-clock seconds per tutor turn; unaffected by
        # streaming, so historical and new runs stay comparable.
        "tutor_lat_p50": lat.get("p50_seconds"),
        "tutor_lat_p95": lat.get("p95_seconds"),
        "tokens_total": tokens.get("total_tokens"),
    }


def _fmt_md(v) -> str:
    """Format a value for a Markdown table cell."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def _fmt_csv(v) -> str:
    """Format a value for CSV (None -> empty string)."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def leaderboard(runs: list, probes: dict | None = None) -> tuple:
    """Build a leaderboard Markdown table and CSV from a list of run summaries.

    Args:
        runs: list of run summary dicts (as returned by aggregate / read from
              summary.json).  Each dict must contain the standard keys written
              by the benchmark pipeline.
        probes: optional ``{(tutor_model, mode): latency.json block}`` from
              `tutormoments.latency.probe_runs`, supplying the TTFT columns.
              Cells with no probe show "-".

    Returns:
        (markdown_table: str, csv_str: str)

    Columns (the paper's three metrics under the paper's names, plus
    identity, TTFT, and latency/token diagnostics):
        tutor_model, mode, n, appropriate_scaffolding, appropriate_rigor,
        avoids_overscaffold, ttft_p50, ttft_first_p50, ttft_later_p50,
        tutor_lat_p50, tutor_lat_p95, tokens_total

    Two different latency measurements sit side by side here, deliberately:

    - ``ttft_p50`` and its first/later-message split come from `tutormoments
      latency`, which runs strictly serially. These are the
      cross-model-comparable figures, and they are joined in from a probe's
      own run directory rather than read off the benchmark run.
    - ``tutor_lat_p50`` / ``tutor_lat_p95`` are end-to-end wall clock per
      tutor call from the benchmark run itself, gathered under
      ``--concurrency``, so they compare a model against its own history but
      not against another model.

    See docs/latency.md.

    Rows sorted descending by appropriate_scaffolding (None last).
    avoids_overscaffold = 1 - overscaffold["rate"]  (higher is better).
    Floats formatted to 3 decimal places; None -> "-" (md) / "" (csv).
    """
    rows = [_extract_row(s, probes) for s in runs]

    def _sort_key(r):
        v = r.get("appropriate_scaffolding")
        return (v is None, -(v if isinstance(v, (int, float)) else 0.0))

    rows.sort(key=_sort_key)

    col_keys = [k for k, _ in _LEADERBOARD_COLS]
    col_heads = [h for _, h in _LEADERBOARD_COLS]

    # --- Markdown ---
    header = "| " + " | ".join(col_heads) + " |"
    sep = "|" + "|".join("---" for _ in col_heads) + "|"
    md_lines = [header, sep]
    for r in rows:
        cells = [_fmt_md(r.get(k)) for k in col_keys]
        md_lines.append("| " + " | ".join(cells) + " |")
    markdown = "\n".join(md_lines)

    # --- CSV ---
    import csv as _csv
    import io

    buf = io.StringIO()
    writer = _csv.writer(buf)
    writer.writerow(col_heads)
    for r in rows:
        writer.writerow([_fmt_csv(r.get(k)) for k in col_keys])
    csv_str = buf.getvalue()

    return markdown, csv_str


def format_run_summary(
    metrics: dict,
    *,
    tutor_model: str = "",
    mode: str = "",
    run_id: str = "",
) -> str:
    """Render a compact, human-readable summary of a single run's metrics.

    Accepts either a single-trial aggregate dict (as written to summary.json
    when trials==1) or the trials>1 {trials, mean, spread} shape. Uses the same
    reader-facing metric derivations as the leaderboard (appropriate
    scaffolding/rigor, avoids-over-scaffolding = 1 - rate), so the end-of-run
    terminal summary matches what `report` later tabulates. For trials>1 each
    score is shown as ``mean ± std``.
    """
    trials = metrics.get("trials", 1)
    core = metrics.get("mean", metrics)  # trials>1 nests the aggregate under "mean"
    spread = metrics.get("spread") or {}

    row = _extract_row(
        {
            **core,
            "latency": metrics.get("latency"),
            "tokens": metrics.get("tokens"),
            "tutor_model": tutor_model,
            "mode": mode,
        }
    )

    def _sub(spread_dict, subkey, field):
        sub = spread_dict.get(subkey) if isinstance(spread_dict, dict) else None
        return sub.get(field) if isinstance(sub, dict) else None

    def _fmt_metric(value, std):
        if value is None:
            return "-"
        text = f"{value:.3f}"
        if trials > 1 and isinstance(std, (int, float)):
            text += f" ± {std:.3f}"
        return text

    n = row.get("n") or 0
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    counts = metrics.get("run_counts") or {}

    lines = ["Run summary" + (f": {run_id}" if run_id else "")]
    ident = f"  tutor={tutor_model or '-'}  mode={mode or '-'}  moments={n}"
    if trials > 1:
        ident += f"  trials={trials}"
    lines.append(ident)
    lines.append("  " + "-" * 44)

    metric_rows = [
        (
            "Appropriate Scaffolding",
            row.get("appropriate_scaffolding"),
            _sub(spread, "scaffold_calibrated", "score"),
        ),
        (
            "Appropriate Rigor",
            row.get("appropriate_rigor"),
            _sub(spread, "rigor_calibrated", "score"),
        ),
        (
            "Avoids Over-Scaffolding",
            row.get("avoids_overscaffold"),
            _sub(spread, "overscaffold", "rate"),
        ),
    ]
    for label, value, std in metric_rows:
        lines.append(f"  {label:<26} {_fmt_metric(value, std)}")

    if counts:
        lines.append(
            f"  {'Moments succeeded':<26} "
            f"{counts.get('succeeded', 0)}/{counts.get('attempted', 0)} "
            f"(failed {counts.get('failed', 0)}, resumed {counts.get('resumed', 0)})"
        )

    # Action taxonomy (present when a run classified its tutor actions). Skip
    # cleanly if it was disabled/failed (no n_facets) -- never break the summary.
    tax = metrics.get("taxonomy") or {}
    n_facets = tax.get("n_facets")
    if isinstance(n_facets, int) and n_facets > 0:
        lines.append(
            f"  {'Actions classified':<26} {n_facets} ({tax.get('excluded', 0)} excluded)"
        )
        orient = tax.get("orientation") or {}
        parts = []
        for key in ("scaffolding", "rigor", "neutral"):
            pct = 100.0 * orient.get(key, 0) / n_facets
            parts.append(f"{key} {pct:.0f}%")
        lines.append(f"  {'Action mix':<26} {'  '.join(parts)}")

    if row.get("tutor_lat_p50") is not None or row.get("tutor_lat_p95") is not None:
        lines.append(
            f"  {'Tutor latency p50/p95 (s)':<26} "
            f"{_fmt_md(row.get('tutor_lat_p50'))} / {_fmt_md(row.get('tutor_lat_p95'))}"
        )
    if row.get("tokens_total") is not None:
        lines.append(f"  {'Total tokens':<26} {_fmt_md(row.get('tokens_total'))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------


def view(runs: list) -> str:
    """Build a self-contained HTML viewer for a list of run summaries.

    The viewer embeds the run data as JSON and renders model/mode selectors
    plus score blocks for each run.

    Args:
        runs: list of run summary dicts.

    Returns:
        self-contained HTML string (utf-8 safe, no external dependencies).
    """
    import json as _json

    def _safe(v, places=3):
        if v is None:
            return None
        if isinstance(v, float):
            return round(v, places)
        return v

    # Build a compact payload for each run
    payload_runs = []
    for s in runs:
        row = _extract_row(s)
        payload_runs.append(
            {
                "tutor_model": row["tutor_model"],
                "mode": row["mode"],
                "n": row["n"],
                "appropriate_scaffolding": _safe(row["appropriate_scaffolding"]),
                "appropriate_rigor": _safe(row["appropriate_rigor"]),
                "avoids_overscaffold": _safe(row["avoids_overscaffold"]),
                "tutor_lat_p50": _safe(row["tutor_lat_p50"]),
                "tutor_lat_p95": _safe(row["tutor_lat_p95"]),
                "tokens_total": row["tokens_total"],
            }
        )

    blob = _json.dumps(payload_runs, ensure_ascii=True)

    html = (
        r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tutormoments Leaderboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f6fa; color: #333; padding: 24px; }
h1 { font-size: 20px; margin-bottom: 16px; }
.controls { display: flex; gap: 12px; margin-bottom: 20px; align-items: center; flex-wrap: wrap; }
.controls label { font-size: 12px; font-weight: 700; color: #666;
  text-transform: uppercase; letter-spacing: 0.4px; margin-right: 4px; }
.controls select { padding: 6px 10px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 13px; background: #fff; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
th { background: #f0f1f5; font-size: 11px; font-weight: 700; color: #555;
     text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 12px;
     text-align: left; border-bottom: 2px solid #ddd; white-space: nowrap; }
td { padding: 9px 12px; font-size: 13px; border-bottom: 1px solid #eee; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f9f9fc; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.top { background: #f0fff4 !important; }
.none-val { color: #bbb; }
.score-high { color: #155724; font-weight: 700; }
.score-mid  { color: #856404; }
.score-low  { color: #721c24; }
</style>
</head>
<body>
<h1>Tutormoments Leaderboard</h1>
<div class="controls">
  <span><label>Filter model</label>
  <select id="model-filter"><option value="">All models</option></select></span>
  <span><label>Filter condition</label>
  <select id="condition-filter"><option value="">All conditions</option></select></span>
  <span><label>Filter mode</label>
  <select id="mode-filter"><option value="">All modes</option></select></span>
</div>
<table id="lb-table">
  <thead>
    <tr>
      <th>tutor_model</th>
      <th>condition</th>
      <th>mode</th>
      <th class="num">n</th>
      <th class="num">appropriate_scaffolding</th>
      <th class="num">appropriate_rigor</th>
      <th class="num">avoids_overscaffold</th>
      <th class="num">tutor_lat_p50</th>
      <th class="num">tutor_lat_p95</th>
      <th class="num">tokens_total</th>
    </tr>
  </thead>
  <tbody id="lb-body"></tbody>
</table>

<script>
const RUNS = """
        + blob
        + r""";

function fmt(v, places) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return v.toFixed(places !== undefined ? places : 3);
  return String(v);
}

function scoreClass(v) {
  if (v === null || v === undefined) return '';
  if (v >= 0.7) return 'score-high';
  if (v >= 0.4) return 'score-mid';
  return 'score-low';
}

function buildRow(r, isTop) {
  function cell(v, cls, places) {
    const fv = fmt(v, places);
    const sc = (typeof v === 'number') ? scoreClass(v) : '';
    const classes = ['num', cls, sc].filter(Boolean).join(' ');
    if (fv === null) return '<td class="num none-val">-</td>';
    return '<td class="' + classes + '">' + fv + '</td>';
  }
  const rowCls = isTop ? ' class="top"' : '';
  return (
    '<tr' + rowCls + '>' +
    '<td>' + (r.tutor_model || '') + '</td>' +
    '<td>' + (r.condition || '') + '</td>' +
    '<td>' + (r.mode || '') + '</td>' +
    cell(r.n, '', 0) +
    cell(r.appropriate_scaffolding, '') +
    cell(r.appropriate_rigor, '') +
    cell(r.avoids_overscaffold, '') +
    cell(r.tutor_lat_p50, '') +
    cell(r.tutor_lat_p95, '') +
    cell(r.tokens_total, '', 0) +
    '</tr>'
  );
}

function render() {
  const mf = document.getElementById('model-filter').value;
  const cf = document.getElementById('condition-filter').value;
  const pf = document.getElementById('mode-filter').value;
  const filtered = RUNS.filter(r =>
    (!mf || r.tutor_model === mf) &&
    (!cf || r.condition === cf) &&
    (!pf || r.mode === pf)
  );
  // Sorted desc by appropriate_scaffolding (null last) -- already sorted
  // server-side, but re-sort in JS to survive filter.
  filtered.sort((a, b) => {
    const av = a.appropriate_scaffolding, bv = b.appropriate_scaffolding;
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return bv - av;
  });
  const body = document.getElementById('lb-body');
  body.innerHTML = filtered.map((r, i) => buildRow(r, i === 0 && filtered.length > 1)).join('');
}

// Populate filter dropdowns
const models = [...new Set(RUNS.map(r => r.tutor_model).filter(Boolean))].sort();
const conditions = [...new Set(RUNS.map(r => r.condition).filter(Boolean))].sort();
const modes  = [...new Set(RUNS.map(r => r.mode).filter(Boolean))].sort();
models.forEach(m => {
  document.getElementById('model-filter').add(new Option(m, m));
});
conditions.forEach(c => {
  document.getElementById('condition-filter').add(new Option(c, c));
});
modes.forEach(m => {
  document.getElementById('mode-filter').add(new Option(m, m));
});
document.getElementById('model-filter').addEventListener('change', render);
document.getElementById('condition-filter').addEventListener('change', render);
document.getElementById('mode-filter').addEventListener('change', render);

render();
</script>
</body>
</html>"""
    )
    return html
