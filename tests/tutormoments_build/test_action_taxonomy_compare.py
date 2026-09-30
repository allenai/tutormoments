"""Tests for the co-author comparison report (action_taxonomy_compare)."""

import json
import math

import pytest

pd = pytest.importorskip("pandas")

from tutormoments import taxonomy as tx  # noqa: E402
from tutormoments_build import action_taxonomy_compare as cmp  # noqa: E402
from tutormoments_build.action_taxonomy import (  # noqa: E402
    MANIFEST_FILENAME,
    RELEASE_FILENAME,
)


def test_paper_series_and_labels_exist():
    """Every mapped model has CSV columns and a locked KL row: no silent drops."""
    header = cmp.PAPER_DISTRIBUTION.read_text().splitlines()[0].split(",")
    locked = cmp.parse_kl_tex(cmp.PAPER_KL_TEX.read_text())
    assert "human__macro_mean_pct" in header
    assert cmp.HUMAN_LABEL in locked
    for stem, label in cmp.PAPER_MODELS.values():
        for p in ("plain", "SR"):
            assert f"{stem}__{p}__macro_mean_pct" in header
        assert label in locked


def test_parse_locked_kl_table():
    locked = cmp.parse_kl_tex(cmp.PAPER_KL_TEX.read_text())
    assert len(locked) == 8
    assert locked["Human tutors"] == (0.179, 0.177, 0.179, 0.177)
    assert locked["GPT 5.5"] == (0.048, 0.050, 0.119, 0.128)


def _df(rows):
    return pd.DataFrame(
        [{"moment_id": m, "situation_label": s, "category": c} for m, s, c in rows]
    )


def test_kl_matches_notebook_formula():
    # S: two moments, all A; R: two moments, half A half G.
    df = _df(
        [
            ("s1", "scaffolding", "A"),
            ("s2", "scaffolding", "A"),
            ("r1", "rigor", "A"),
            ("r1", "rigor", "G"),
            ("r2", "rigor", "G"),
        ]
    )
    k = len(tx.CATEGORY_LETTERS)
    # Pseudo-counts: S: A=2; R: A=0.5, G=1.5; add-one smoothing.
    s = [1.0] * k
    s[0] = 3.0
    r = [1.0] * k
    r[0], r[6] = 1.5, 2.5
    s = [v / sum(s) for v in s]
    r = [v / sum(r) for v in r]
    expected = (
        sum(a * math.log(a / b) for a, b in zip(s, r)),
        sum(b * math.log(b / a) for a, b in zip(s, r)),
    )
    assert cmp.kl_s_r(df) == pytest.approx(expected)


def test_kl_smoothing_shrinks_with_sample_size():
    df = _df(
        [(f"s{i}", "scaffolding", "A") for i in range(4)]
        + [(f"r{i}", "rigor", "G") for i in range(4)]
    )
    small, large = cmp.kl_s_r(df, 5), cmp.kl_s_r(df, 500)
    assert large[0] > small[0] and large[1] > small[1]
    assert cmp.kl_s_r(df) == cmp.kl_s_r(df, 4)


def test_kl_is_nan_without_both_situations():
    kl = cmp.kl_s_r(_df([("s1", "scaffolding", "A")]))
    assert all(v != v for v in kl)


def _release(tmp_path):
    """A release covering human + every paper cell, S and R moments each."""
    rows = []

    def add(source, model, prompt, mid, sit, cat):
        rows.append(
            {
                "source": source,
                "tutor_model": model,
                "prompt_mode": prompt,
                "moment_id": mid,
                "situation_label": sit,
                "category": cat,
                "orientation": tx.ORIENTATION_BY_LETTER[cat] if cat else None,
            }
        )

    for i, (sit, cat) in enumerate(
        [("scaffolding", "A"), ("rigor", "G"), ("mixed", "C")]
    ):
        add("human", None, None, f"h{i}", sit, cat)
    add("human", None, None, "h9", "rigor", None)  # excluded: ignored
    for model in cmp.PAPER_MODELS:
        for prompt in cmp.PROMPTS:
            add("benchmark_520", model, prompt, "m1", "scaffolding", "B")
            add("benchmark_520", model, prompt, "m2", "rigor", "H")
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / RELEASE_FILENAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {
        "name": "action_taxonomy",
        "version": "0",
        "record_count": len(rows),
        "scheme_version": tx.SCHEME_VERSION,
        "file_sha256": "0" * 64,
        "inputs": {"repo_id": "r", "revision": "a" * 40},
        "classifier": {
            "model": "m",
            "thinking": {},
            "batch_size": 50,
            "pools": "separate",
        },
    }
    (upload / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return tmp_path


def test_build_report_end_to_end(tmp_path):
    report = cmp.build_report(_release(tmp_path))
    # Human: 3 kept moments (the all-excluded one is not a moment); paper had 514.
    assert "| Human tutors | 514 → 3 |" in report
    assert "| Claude Opus 4.8 / plain | 100 → 2 |" in report
    # Every KL row renders paper → new, with both values numeric.
    kl_rows = [ln for ln in report.splitlines() if " → " in ln and "0.179" in ln]
    assert kl_rows and "--" not in kl_rows[0]
    for _, label in cmp.PAPER_MODELS.values():
        assert f"| {label} | " in report
