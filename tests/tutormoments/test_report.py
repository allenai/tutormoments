"""Golden test: pins the exact numeric output of report.aggregate.

Fixture: 6 (Scenario, Annotation) pairs, covering all branches in the port:
  1. gold=scaffolding, action=scaffolding (clean hit, no over-scaffold)
  2. gold=scaffolding, action=both       (right action, but over-scaffold -> not clean)
  3. gold=rigor,       action=rigor      (clean hit)
  4. gold=rigor,       action=neither    (miss)
  5. gold=both         (excluded from did-rate denominators), action=both, no over
  6. gold=scaffolding, action=neither    (miss)

Expected values computed BY HAND from score.py's formulas:

  scaf_total    = 3   (pairs 1, 2, 6)
  scaf_yes      = 2   (pairs 1, 2 -- action_label in {scaffolding, both} -> pred_dims[0]=="yes")
  scaf_clean_yes= 1   (pair 1 only -- pair 2 has over-scaffold so excluded from clean)
  scaf_over_yes = 1   (pair 2)

  rig_total     = 2   (pairs 3, 4)
  rig_yes       = 1   (pair 3 -- action=rigor -> pred_dims[1]=="yes")
  rig_clean_yes = 1   (pair 3 -- no over-scaffold)

  over_yes      = 1   (pair 2)
  n_total       = 6

Paper metrics (the only reported scores):
  scaffold_calibrated.score = 1/3   (Appropriate Scaffolding)
  rigor_calibrated.score    = 1/2   (Appropriate Rigor)
  overscaffold.rate         = 1/6   (component of Avoids Over-Scaffolding)

leaderboard_row:
  avoids_overscaffold = 1 - 1/6 = 5/6
"""

from types import SimpleNamespace

import pytest

from tutormoments.report import aggregate, leaderboard_row

# ---------------------------------------------------------------------------
# Minimal stubs -- no SDK imports, no I/O
# ---------------------------------------------------------------------------


def _make_scenario(dimension: str) -> SimpleNamespace:
    """Minimal Scenario stub: only .dimension is read by aggregate."""
    return SimpleNamespace(dimension=dimension)


def _make_annotation(
    action_label: str,
    overscaffold_decomposed: list,
) -> SimpleNamespace:
    """Minimal Annotation stub: only .action_label and
    .overscaffold_decomposed are read by aggregate."""
    return SimpleNamespace(
        action_label=action_label,
        overscaffold_decomposed=overscaffold_decomposed,
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def six_pairs():
    """6 (scenario, annotation) pairs covering all scoring branches."""
    scenarios = [
        _make_scenario("scaffolding"),  # pair 1: clean scaf hit
        _make_scenario("scaffolding"),  # pair 2: right action but over-scaffold
        _make_scenario("rigor"),  # pair 3: clean rigor hit, pos outcome
        _make_scenario("rigor"),  # pair 4: rigor miss
        _make_scenario("both"),  # pair 5: excluded from did-rate denominators
        _make_scenario("scaffolding"),  # pair 6: scaf miss, pos outcome
    ]
    annotations = [
        _make_annotation("scaffolding", []),  # pair 1
        _make_annotation("both", ["gave answer"]),  # pair 2: over-scaffold
        _make_annotation("rigor", []),  # pair 3
        _make_annotation("neither", []),  # pair 4
        _make_annotation("both", []),  # pair 5
        _make_annotation("neither", []),  # pair 6
    ]
    return scenarios, annotations


# ---------------------------------------------------------------------------
# Golden test
# ---------------------------------------------------------------------------


def test_aggregate_golden(six_pairs):
    """aggregate() returns the EXACT metric dict for the golden fixture."""
    scenarios, annotations = six_pairs
    result = aggregate(scenarios, annotations)

    # --- n_scenarios ---
    assert result["n_scenarios"] == 6

    # --- did-rates dropped: only the 3 paper metrics are reported ---
    assert "scaffolding_did" not in result
    assert "rigor_did" not in result

    # --- overscaffold ---
    ov = result["overscaffold"]
    assert ov["n_yes"] == 1
    assert ov["n_total"] == 6
    assert ov["rate"] == pytest.approx(1 / 6)
    assert ov["available"] is True  # Annotation dataclass always has the field

    # --- outcome_pos_rate dropped from the paper; must not be reported ---
    assert "outcome_pos_rate" not in result

    # --- scaffold_calibrated ---
    sc = result["scaffold_calibrated"]
    assert sc["n_clean_yes"] == 1  # pair 1 only (pair 2 has over-scaffold)
    assert sc["n_overscaffold"] == 1  # pair 2
    assert sc["n_total"] == 3
    assert sc["score"] == pytest.approx(1 / 3)

    # --- rigor_calibrated ---
    rc = result["rigor_calibrated"]
    assert rc["n_clean_yes"] == 1
    assert rc["n_total"] == 2
    assert rc["score"] == pytest.approx(1 / 2)


def test_aggregate_golden_full_dict(six_pairs):
    """Full dict equality check (all keys and values match expected)."""
    scenarios, annotations = six_pairs
    result = aggregate(scenarios, annotations)

    expected = {
        "n_scenarios": 6,
        "overscaffold": {
            "n_yes": 1,
            "n_total": 6,
            "rate": 1 / 6,
            "available": True,
        },
        "scaffold_calibrated": {
            "n_clean_yes": 1,
            "n_overscaffold": 1,
            "n_total": 3,
            "score": 1 / 3,
        },
        "rigor_calibrated": {
            "n_clean_yes": 1,
            "n_total": 2,
            "score": 1 / 2,
        },
    }

    # Check keys match exactly
    assert set(result.keys()) == set(expected.keys())

    # Check integer counts exactly
    assert result["n_scenarios"] == expected["n_scenarios"]
    for key in ("scaffold_calibrated", "rigor_calibrated"):
        assert result[key]["n_clean_yes"] == expected[key]["n_clean_yes"]
        assert result[key]["n_total"] == expected[key]["n_total"]
    assert (
        result["scaffold_calibrated"]["n_overscaffold"]
        == expected["scaffold_calibrated"]["n_overscaffold"]
    )
    assert result["overscaffold"]["n_yes"] == expected["overscaffold"]["n_yes"]
    assert result["overscaffold"]["n_total"] == expected["overscaffold"]["n_total"]
    assert result["overscaffold"]["available"] == expected["overscaffold"]["available"]

    # Check floating-point values approximately
    assert result["overscaffold"]["rate"] == pytest.approx(
        expected["overscaffold"]["rate"]
    )
    assert result["scaffold_calibrated"]["score"] == pytest.approx(
        expected["scaffold_calibrated"]["score"]
    )
    assert result["rigor_calibrated"]["score"] == pytest.approx(
        expected["rigor_calibrated"]["score"]
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_aggregate_empty():
    """aggregate([],[]) returns all zeros and None rates."""
    result = aggregate([], [])
    assert result["n_scenarios"] == 0
    assert result["overscaffold"]["rate"] is None
    assert result["overscaffold"]["available"] is False
    assert "outcome_pos_rate" not in result
    assert "scaffolding_did" not in result
    assert "rigor_did" not in result
    assert result["scaffold_calibrated"]["score"] is None
    assert result["rigor_calibrated"]["score"] is None


def test_aggregate_neither_gold_excluded():
    """gold=neither is excluded from both calibrated denominators."""
    scenarios = [_make_scenario("neither")]
    annotations = [_make_annotation("scaffolding", [])]
    result = aggregate(scenarios, annotations)
    assert result["n_scenarios"] == 1
    assert result["scaffold_calibrated"]["n_total"] == 0
    assert result["scaffold_calibrated"]["score"] is None
    assert result["rigor_calibrated"]["n_total"] == 0
    assert result["rigor_calibrated"]["score"] is None


def test_leaderboard_row_adds_avoids_overscaffold():
    """leaderboard_row adds avoids_overscaffold = 1 - overscaffold.rate."""
    metrics = {
        "n_scenarios": 6,
        "overscaffold": {"n_yes": 1, "n_total": 6, "rate": 1 / 6, "available": True},
    }
    row = leaderboard_row(metrics)
    assert "avoids_overscaffold" in row
    assert row["avoids_overscaffold"] == pytest.approx(5 / 6)
    # Original keys preserved
    assert row["n_scenarios"] == 6


def test_leaderboard_row_none_safe():
    """leaderboard_row returns avoids_overscaffold=None when rate is None."""
    metrics = {
        "overscaffold": {"n_yes": 0, "n_total": 0, "rate": None, "available": False},
    }
    row = leaderboard_row(metrics)
    assert row["avoids_overscaffold"] is None


# ---------------------------------------------------------------------------
# leaderboard() and view() tests
# ---------------------------------------------------------------------------

from tutormoments.report import leaderboard, view


def _make_run_summary(
    tutor_model: str,
    mode: str,
    n: int,
    scaffold_cal: float | None,
    rigor_cal: float | None,
    overscaffold_rate: float | None,
    tutor_lat_p50: float | None = None,
    tutor_lat_p95: float | None = None,
    tokens_total: int | None = None,
) -> dict:
    """Build a fake run summary dict matching the shape report.leaderboard expects."""
    return {
        "tutor_model": tutor_model,
        "mode": mode,
        "n_scenarios": n,
        "scaffold_calibrated": {
            "score": scaffold_cal,
            "n_clean_yes": 0,
            "n_total": n,
            "n_overscaffold": 0,
        },
        "rigor_calibrated": {"score": rigor_cal, "n_clean_yes": 0, "n_total": n},
        "overscaffold": {
            "rate": overscaffold_rate,
            "n_yes": 0,
            "n_total": n,
            "available": overscaffold_rate is not None,
        },
        "latency": {
            "tutor": {
                "p50_seconds": tutor_lat_p50,
                "p95_seconds": tutor_lat_p95,
                "mean_seconds": 1.0,
                "n": n,
            }
        },
        "tokens": {
            "total": {
                "total_tokens": tokens_total,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        },
    }


SUMMARY_A = _make_run_summary(
    tutor_model="model-alpha",
    mode="scaffolding_rigor",
    n=100,
    scaffold_cal=0.75,
    rigor_cal=0.60,
    overscaffold_rate=0.10,
    tutor_lat_p50=1.234,
    tutor_lat_p95=3.456,
    tokens_total=500000,
)

SUMMARY_B = _make_run_summary(
    tutor_model="model-beta",
    mode="scaffolding_rigor",
    n=100,
    scaffold_cal=0.50,
    rigor_cal=0.40,
    overscaffold_rate=0.20,
    tutor_lat_p50=2.000,
    tutor_lat_p95=5.000,
    tokens_total=600000,
)

SUMMARY_NONE = _make_run_summary(
    tutor_model="model-gamma",
    mode="scaffolding_only",
    n=50,
    scaffold_cal=None,
    rigor_cal=None,
    overscaffold_rate=None,
    tutor_lat_p50=None,
    tutor_lat_p95=None,
    tokens_total=None,
)


# Reader-facing column names match the paper's three metrics.
EXPECTED_COLUMNS = [
    "tutor_model",
    "mode",
    "n",
    "appropriate_scaffolding",
    "appropriate_rigor",
    "avoids_overscaffold",
    "tutor_lat_p50",
    "tutor_lat_p95",
    "tokens_total",
]


def test_leaderboard_returns_both_formats():
    """leaderboard() returns a (markdown, csv) 2-tuple."""
    result = leaderboard([SUMMARY_A, SUMMARY_B])
    assert isinstance(result, tuple)
    assert len(result) == 2
    md, csv_str = result
    assert isinstance(md, str) and len(md) > 0
    assert isinstance(csv_str, str) and len(csv_str) > 0


def test_leaderboard_emits_both_rows():
    """Both run summaries appear in the markdown table."""
    md, csv_str = leaderboard([SUMMARY_A, SUMMARY_B])
    assert "model-alpha" in md
    assert "model-beta" in md
    assert "model-alpha" in csv_str
    assert "model-beta" in csv_str


def test_leaderboard_column_order_in_header():
    """Markdown header contains exactly the expected columns in order."""
    md, _ = leaderboard([SUMMARY_A, SUMMARY_B])
    header_line = md.split("\n")[0]
    positions = [header_line.find(col) for col in EXPECTED_COLUMNS]
    # Each column must appear in the header
    for col, pos in zip(EXPECTED_COLUMNS, positions):
        assert pos != -1, f"Column '{col}' not found in header: {header_line}"
    # Columns must be in ascending order of position
    assert positions == sorted(positions), (
        f"Columns out of order: {list(zip(EXPECTED_COLUMNS, positions))}"
    )
    # Dropped from the paper; must not resurface
    assert "outcome_pos" not in header_line
    assert "did_scaf" not in header_line
    assert "did_rig" not in header_line
    assert "scaffold_cal " not in header_line and "| scaffold_cal |" not in header_line


def test_leaderboard_avoids_overscaffold_formula():
    """avoids_overscaffold == 1 - overscaffold_rate (to 3 decimal places)."""
    md, _ = leaderboard([SUMMARY_A])
    # SUMMARY_A has overscaffold_rate=0.10 => avoids_overscaffold=0.900
    assert "0.900" in md


def test_leaderboard_sorted_desc_by_appropriate_scaffolding():
    """Rows sorted descending by appropriate_scaffolding (model-alpha first)."""
    md, _ = leaderboard([SUMMARY_B, SUMMARY_A])  # pass in reverse order
    lines = [
        l
        for l in md.split("\n")
        if "|" in l and "---" not in l and "tutor_model" not in l
    ]
    assert lines[0].count("model-alpha") >= 1 or "0.750" in lines[0]
    assert lines[1].count("model-beta") >= 1 or "0.500" in lines[1]
    # More direct: first data row has higher appropriate_scaffolding value
    assert "0.750" in lines[0]
    assert "0.500" in lines[1]


def test_leaderboard_none_formatting():
    """None values render as '-' in md and '' in csv."""
    md, csv_str = leaderboard([SUMMARY_NONE])
    # appropriate_scaffolding is None => should appear as '-' in md
    data_lines = [l for l in md.split("\n") if "model-gamma" in l]
    assert len(data_lines) == 1
    assert (
        "| - |" in data_lines[0]
        or "|-|" in data_lines[0]
        or data_lines[0].count("| - ") > 0
    )

    # CSV: None fields should be empty strings (not 'None' or '-')
    csv_lines = [l for l in csv_str.split("\n") if "model-gamma" in l]
    assert len(csv_lines) == 1
    assert "None" not in csv_lines[0]


def test_leaderboard_none_sorted_last():
    """Rows with appropriate_scaffolding=None sort after rows with a value."""
    md, _ = leaderboard([SUMMARY_NONE, SUMMARY_A])
    lines = [
        l
        for l in md.split("\n")
        if "|" in l and "---" not in l and "tutor_model" not in l
    ]
    assert "model-alpha" in lines[0]
    assert "model-gamma" in lines[1]


def test_leaderboard_float_precision():
    """Float values are formatted to 3 decimal places."""
    md, _ = leaderboard([SUMMARY_A])
    # appropriate_scaffolding=0.75 -> "0.750"
    assert "0.750" in md
    # tutor_lat_p50=1.234 -> "1.234"
    assert "1.234" in md


def test_view_returns_nonempty_html():
    """view() returns a non-empty string starting with <!DOCTYPE html>."""
    html = view([SUMMARY_A, SUMMARY_B])
    assert isinstance(html, str) and len(html) > 0
    assert "<!DOCTYPE html>" in html or "<html" in html


def test_view_embeds_model_ids():
    """HTML output contains both model IDs."""
    html = view([SUMMARY_A, SUMMARY_B])
    assert "model-alpha" in html
    assert "model-beta" in html


def test_view_embeds_score_values():
    """HTML output contains score values from the runs."""
    html = view([SUMMARY_A])
    # appropriate_scaffolding=0.75 should appear somewhere in the HTML
    assert "0.75" in html or "0.750" in html
    # Paper metric names in the viewer, no did-rate columns
    assert "appropriate_scaffolding" in html
    assert "appropriate_rigor" in html
    assert "did_scaf" not in html and "did_rig" not in html


# ---------------------------------------------------------------------------
# Spec S7: leaderboard lat/tokens columns show non-"-" when summary carries blocks
# ---------------------------------------------------------------------------


def test_leaderboard_latency_and_tokens_non_dash():
    """A summary with latency.tutor p50/p95 and tokens.total.total_tokens
    produces non-'-' columns in the leaderboard markdown (spec S7).

    SUMMARY_A has tutor_lat_p50=1.234, tutor_lat_p95=3.456, tokens_total=500000.
    """
    md, csv_str = leaderboard([SUMMARY_A])

    # tutor_lat_p50 and tutor_lat_p95 must appear as numbers, not as '-'
    data_lines = [l for l in md.split("\n") if "model-alpha" in l]
    assert len(data_lines) == 1, (
        f"Expected exactly one data row for model-alpha, got: {data_lines}"
    )
    row = data_lines[0]

    # '1.234' and '3.456' must appear in the row (not replaced by '-')
    assert "1.234" in row, f"tutor_lat_p50=1.234 missing from leaderboard row: {row}"
    assert "3.456" in row, f"tutor_lat_p95=3.456 missing from leaderboard row: {row}"
    assert "500000" in row, f"tokens_total=500000 missing from leaderboard row: {row}"

    # Verify '- ' does not cover those columns (None check: model-gamma has None)
    md_none, _ = leaderboard([SUMMARY_NONE])
    none_lines = [l for l in md_none.split("\n") if "model-gamma" in l]
    assert len(none_lines) == 1
    # None values -> '-' in markdown
    assert "| - |" in none_lines[0] or none_lines[0].count("| - ") >= 2, (
        f"Expected '-' for None lat/tokens in: {none_lines[0]}"
    )


# ---------------------------------------------------------------------------
# format_run_summary() -- the end-of-run terminal summary
# ---------------------------------------------------------------------------

from tutormoments.report import format_run_summary


def test_format_run_summary_single_trial():
    """Single-trial metrics render the paper's three reader metrics + counts."""
    metrics = dict(SUMMARY_A)
    metrics["run_counts"] = {
        "attempted": 100,
        "succeeded": 98,
        "failed": 2,
        "resumed": 1,
    }
    out = format_run_summary(
        metrics,
        tutor_model="model-alpha",
        mode="scaffolding_rigor",
        run_id="model-alpha_scaffolding_rigor_x",
    )

    assert "Run summary: model-alpha_scaffolding_rigor_x" in out
    assert "tutor=model-alpha" in out and "mode=scaffolding_rigor" in out
    assert "moments=100" in out
    # Reader-facing metrics, 3dp, same derivation as the leaderboard.
    assert "Appropriate Scaffolding    0.750" in out
    assert "Appropriate Rigor          0.600" in out
    assert "Avoids Over-Scaffolding    0.900" in out  # 1 - 0.10
    assert "98/100" in out and "failed 2" in out and "resumed 1" in out
    assert "500000" in out
    # No spread markers for a single trial.
    assert "±" not in out
    assert "trials=" not in out


def test_format_run_summary_renders_taxonomy_block():
    """A taxonomy block renders the count + orientation-mix lines."""
    metrics = dict(SUMMARY_A)
    metrics["run_counts"] = {
        "attempted": 100,
        "succeeded": 100,
        "failed": 0,
        "resumed": 0,
    }
    metrics["taxonomy"] = {
        "scheme_version": "lm_extended_v1",
        "counts": {"A": 10},
        "orientation": {"scaffolding": 250, "rigor": 90, "neutral": 60},
        "n_facets": 400,
        "excluded": 38,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    out = format_run_summary(metrics, tutor_model="model-alpha", mode="plain")
    assert "Actions classified         400 (38 excluded)" in out
    # 250/400=62%, 90/400=22%, 60/400=15%
    assert "scaffolding 62%" in out and "rigor 22%" in out and "neutral 15%" in out


def test_format_run_summary_skips_taxonomy_when_absent_or_failed():
    """No taxonomy lines when the block is missing or carries only an error."""
    metrics = dict(SUMMARY_A)
    metrics["run_counts"] = {"attempted": 1, "succeeded": 1, "failed": 0, "resumed": 0}
    assert "Action mix" not in format_run_summary(metrics)
    metrics["taxonomy"] = {"error": "no API key"}
    assert "Action mix" not in format_run_summary(metrics)


def test_format_run_summary_trials_shows_mean_and_spread():
    """trials>1 uses the mean/spread shape and renders mean ± std."""
    metrics = {
        "trials": 3,
        "mean": {
            "n_scenarios": 100,
            "scaffold_calibrated": {
                "score": 0.75,
                "n_clean_yes": 0,
                "n_total": 100,
                "n_overscaffold": 0,
            },
            "rigor_calibrated": {"score": 0.60, "n_clean_yes": 0, "n_total": 100},
            "overscaffold": {
                "rate": 0.10,
                "n_yes": 0,
                "n_total": 100,
                "available": True,
            },
        },
        "spread": {
            "scaffold_calibrated": {"score": 0.02},
            "rigor_calibrated": {"score": 0.03},
            "overscaffold": {"rate": 0.01},
        },
        "latency": {"tutor": {"p50_seconds": 1.2, "p95_seconds": 3.4}},
        "tokens": {"total": {"total_tokens": 500000}},
        "run_counts": {"attempted": 300, "succeeded": 300, "failed": 0, "resumed": 0},
    }
    out = format_run_summary(
        metrics, tutor_model="model-alpha", mode="scaffolding_rigor"
    )

    assert "trials=3" in out
    assert "0.750 ± 0.020" in out
    assert "0.600 ± 0.030" in out
    assert "0.900 ± 0.010" in out  # avoids = 1 - mean rate; std carries through


def test_format_run_summary_none_metrics_render_dash():
    """None scores/latency/tokens render as '-' rather than crashing."""
    out = format_run_summary(
        SUMMARY_NONE, tutor_model="model-gamma", mode="scaffolding_only"
    )
    assert "Appropriate Scaffolding    -" in out
    assert "Appropriate Rigor          -" in out
    assert "Avoids Over-Scaffolding    -" in out


# ---------------------------------------------------------------------------
# Legacy latency columns
# ---------------------------------------------------------------------------


def test_leaderboard_tolerates_pre_streaming_summaries():
    """Runs recorded before the streaming work have no timing block at all;
    their end-to-end latency columns must still render."""
    md, _ = leaderboard([SUMMARY_A])
    row = next(line for line in md.split("\n") if "model-alpha" in line)
    assert "1.234" in row, "existing latency columns still render"
