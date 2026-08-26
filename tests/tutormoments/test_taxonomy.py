"""Tests for tutormoments.taxonomy.

Covers the pure functions (filter regexes, normal-approx CI, JS divergence),
the input adapters, pool + CSV round-trips, the LLM classifier with a fake
client (resume safety, JSON-error handling, total_tokens recording), and
atomic CSV writes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tutormoments import taxonomy as tx

# ---------------------------------------------------------------------------
# 1. filter_statement: the four reason codes + keep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement, expected_bucket, expected_reason",
    [
        # KEEPs (real tutor actions with a real verb)
        ("The tutor asks guiding questions about the next step.", "keep", ""),
        ("After confirmation, the tutor offers a hint.", "keep", ""),
        ("The tutor scaffolds by introducing a different representation.", "keep", ""),
        # non_tutor_actor: subject is the student / interaction / etc.
        ("The student answers correctly.", "strip", "non_tutor_actor"),
        ("There is no withdrawal of support.", "strip", "non_tutor_actor"),
        (
            "No attempt is made to review the student's work.",
            "strip",
            "non_tutor_actor",
        ),
        (
            "The interaction consists entirely of casual banter.",
            "strip",
            "non_tutor_actor",
        ),
        # pure_stance: stance phrase + only filler
        ("The tutor is scaffolding.", "strip", "pure_stance"),
        ("The tutor pushes for rigor.", "strip", "pure_stance"),
        ("The tutor uses a mix of scaffolding and rigor.", "strip", "pure_stance"),
        # stance_negation: stance phrase + negation
        ("The tutor does not push for rigor.", "strip", "stance_negation"),
        ("The tutor is not scaffolding.", "strip", "stance_negation"),
        # non_action: tutor verb but the verb is a negation
        ("The tutor does not ask the student to explain.", "strip", "non_action"),
        ("The tutor takes no substantive pedagogical action.", "strip", "non_action"),
        (
            "The tutor fails to acknowledge the student's correct answer.",
            "strip",
            "non_action",
        ),
    ],
)
def test_filter_statement(statement, expected_bucket, expected_reason):
    bucket, reason, _ = tx.filter_statement(statement)
    assert bucket == expected_bucket, statement
    assert reason == expected_reason, statement


def test_filter_statement_stance_prefixed_flag():
    """`stance_prefixed` is True iff a stance phrase appears, regardless of bucket."""
    _, _, sp = tx.filter_statement("The tutor scaffolds by asking a guiding question.")
    assert sp is True
    _, _, sp = tx.filter_statement("The tutor asks a guiding question.")
    assert sp is False


# ---------------------------------------------------------------------------
# 2. Scheme integrity
# ---------------------------------------------------------------------------


def test_frozen_scheme_shape():
    assert len(tx.CATEGORIES) == 13
    assert tx.CATEGORY_LETTERS == list("ABCDEFGHIJKLM")
    assert tx.LAST_LETTER == "M"
    # L = Transitions, M = Other (paper convention; tested explicitly because
    # an L<->M swap is easy to introduce and hard to spot in review).
    assert "Transitioning" in tx.NAME_BY_LETTER["L"]
    assert tx.NAME_BY_LETTER["M"] == "Other"


def test_categories_have_required_fields():
    for c in tx.CATEGORIES:
        assert set(c) >= {"letter", "name", "orientation", "definition", "examples"}
        assert c["orientation"] in ("scaffolding", "rigor", "neutral")
        assert isinstance(c["examples"], list) and c["examples"]


# ---------------------------------------------------------------------------
# 3. Pool + CSV round-trip
# ---------------------------------------------------------------------------


def _mk_facet(stmt: str, **overrides) -> tx.Facet:
    base = dict(
        moment_id="m0",
        transcript_id="t0",
        turn_start=1,
        turn_end=2,
        statement_index=0,
        statement=stmt,
        annotation_type="scaffolding",
        situation_label="scaffolding",
        source="canonical",
    )
    base.update(overrides)
    return tx.Facet(**base)


def test_build_pool_partitions_kept_and_excluded():
    facets = [
        _mk_facet("The tutor asks a guiding question."),
        _mk_facet("The tutor is scaffolding."),
        _mk_facet("The student answers correctly."),
    ]
    kept, excluded = tx.build_pool(facets)
    assert len(kept) == 1
    assert kept[0].statement == "The tutor asks a guiding question."
    reasons = sorted(r for _, r in excluded)
    assert reasons == ["non_tutor_actor", "pure_stance"]


def test_pool_csv_round_trip(tmp_path):
    facets = [
        _mk_facet("The tutor asks a guiding question."),
        _mk_facet("The tutor scaffolds by asking a guiding question."),
        _mk_facet("The student answers correctly."),
    ]
    kept, excluded = tx.build_pool(facets)
    tx.write_pool_csv(kept, excluded, tmp_path)
    assert (tmp_path / "pool.csv").exists()
    assert (tmp_path / "excluded.csv").exists()
    k2, e2 = tx.read_pool_csv(tmp_path)
    assert len(k2) == len(kept)
    assert {f.statement for f in k2} == {f.statement for f in kept}
    # stance_prefixed bool round-trips
    sp = {f.statement: f.stance_prefixed for f in k2}
    assert sp["The tutor scaffolds by asking a guiding question."] is True
    assert sp["The tutor asks a guiding question."] is False
    # reasons preserved on excluded rows
    e_reasons = {f.statement: r for f, r in e2}
    assert e_reasons["The student answers correctly."] == "non_tutor_actor"


def test_atomic_write_leaves_no_tmp_after_success(tmp_path):
    facets = [_mk_facet("The tutor asks a guiding question.")]
    kept, _ = tx.build_pool(facets)
    tx.write_pool_csv(kept, [], tmp_path)
    # no orphan .tmp files on disk
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 4. Adapters
# ---------------------------------------------------------------------------


def _write_key_moments_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_key_moments_jsonl(tmp_path):
    src = tmp_path / "kmoments.jsonl"
    _write_key_moments_jsonl(
        src,
        [
            {
                "conversation_id": "conv-1",
                "num_turns": 100,
                "key_moments": [
                    # scaffolding moment we should pull in
                    {
                        "annotation_type": "scaffolding",
                        "turn_start": 5,
                        "turn_end": 7,
                        "situation_label_agg": "scaffolding",
                        "annotator_id": "ann-A",
                        "action_decomposed": [
                            "The tutor asks a guiding question.",
                            "The tutor offers a hint.",
                        ],
                    },
                    # rapport moment we should SKIP (annotation_type filter)
                    {
                        "annotation_type": "rapport",
                        "turn_start": 20,
                        "turn_end": 22,
                        "annotator_id": "ann-A",
                        "action_decomposed": ["The tutor expresses warmth."],
                    },
                ],
            },
        ],
    )
    facets = list(tx.load_key_moments_jsonl(src))
    assert len(facets) == 2
    assert {f.statement for f in facets} == {
        "The tutor asks a guiding question.",
        "The tutor offers a hint.",
    }
    # moment_id encodes the annotator id (per-annotator macro/moment averaging)
    assert all("ann-A" in f.moment_id for f in facets)
    # situation_label propagated from situation_label_agg
    assert all(f.situation_label == "scaffolding" for f in facets)


def test_load_canonical_jsonl_round_trip(tmp_path):
    src = tmp_path / "canon.jsonl"
    facets = [
        _mk_facet("The tutor asks a guiding question.", moment_id="m1"),
        _mk_facet("The tutor offers a hint.", moment_id="m2"),
    ]
    with src.open("w") as f:
        for fc in facets:
            f.write(json.dumps(asdict(fc)) + "\n")
    loaded = list(tx.load_canonical_jsonl(src))
    assert {f.statement for f in loaded} == {f.statement for f in facets}


def test_load_tutormoments_results_reads_config_and_scenarios(tmp_path):
    # Stage a minimal tutormoments run-results directory + matching scenarios.jsonl.
    run_dir = tmp_path / "run-x"
    (run_dir / "scores").mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"tutor": "fake-model", "mode": "plain"})
    )
    (run_dir / "scores" / "scen-1.json").write_text(
        json.dumps(
            {
                "scenario_id": "scen-1",
                "annotation_type": "scaffolding",
                "turn_start": 10,
                "turn_end": 14,
                "action_decomposed": ["The tutor asks a guiding question."],
                "action_label": "scaffolding",
                "result_label": "pos",
            }
        )
    )
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        json.dumps({"id": "scen-1", "dimension": "scaffolding"}) + "\n"
    )

    facets = list(tx.load_tutormoments_results(run_dir, scenarios))
    assert len(facets) == 1
    f = facets[0]
    assert f.model == "fake-model"
    assert f.prompt == "plain"
    assert f.situation_label == "scaffolding"
    assert f.statement == "The tutor asks a guiding question."


# ---------------------------------------------------------------------------
# 5. Headline math (pure, no LLM)
# ---------------------------------------------------------------------------


def test_normal_ci_basic():
    mean, lo, hi = tx._normal_ci([0.5, 0.5, 0.5, 0.5])
    assert mean == 0.5 and lo == 0.5 and hi == 0.5
    mean, lo, hi = tx._normal_ci([0.0, 1.0])
    assert mean == 0.5
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_normal_ci_handles_empty_and_single():
    assert tx._normal_ci([]) == (0.0, 0.0, 0.0)
    assert tx._normal_ci([0.7]) == (0.7, 0.7, 0.7)


def test_js_divergence_symmetric_and_self_zero():
    p = [0.2, 0.5, 0.3]
    q = [0.5, 0.3, 0.2]
    assert tx.js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    js_pq = tx.js_divergence(p, q)
    js_qp = tx.js_divergence(q, p)
    assert js_pq == pytest.approx(js_qp, abs=1e-12)
    assert 0.0 < js_pq < 1.0  # bounded in [0, 1] for base-2 log


def test_macro_distribution_averages_per_moment(monkeypatch):
    pytest.importorskip("pandas")
    # two moments: m1 -> {A:2, B:0} (so A=100%); m2 -> {A:0, B:1} (so B=100%)
    # macro mean: A = (1.0 + 0.0)/2 = 0.5; B = (0.0 + 1.0)/2 = 0.5
    facets = [
        _mk_facet("a1", moment_id="m1", category="A"),
        _mk_facet("a2", moment_id="m1", category="A"),
        _mk_facet("b1", moment_id="m2", category="B"),
    ]
    df = tx.facets_to_dataframe(facets)
    dist = tx.macro_distribution(df)
    by_letter = {r["letter"]: r["mean_pct"] for _, r in dist.iterrows()}
    assert by_letter["A"] == pytest.approx(50.0)
    assert by_letter["B"] == pytest.approx(50.0)
    assert by_letter["C"] == 0.0


# ---------------------------------------------------------------------------
# 6. Classifier wiring + resume + json-error handling + total_tokens
# ---------------------------------------------------------------------------


class _FakeModelResponse:
    def __init__(self, text: str, usage: dict):
        self.text = text
        self.usage = usage
        self.latency_seconds = 0.0


class _FakeClient:
    """Records every generate() call; returns a queued payload or a canned
    default. Mimics tutormoments.client.ModelClient.generate()."""

    def __init__(self, payloads: list[str] | None = None):
        self.calls: list[dict] = []
        self._payloads = list(payloads or [])

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        payload = (
            self._payloads.pop(0)
            if self._payloads
            else '{"assignments": [{"id":1,"category":"A"}]}'
        )
        return _FakeModelResponse(
            payload,
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )


def test_classifier_records_total_tokens(tmp_path):
    facets = [_mk_facet("The tutor asks a guiding question.")]
    client = _FakeClient()
    tx.classify_pool(facets, tmp_path, client=client, max_batches=1)
    usage_log = tmp_path / "usage_log.jsonl"
    assert usage_log.exists()
    records = [json.loads(line) for line in usage_log.read_text().splitlines() if line]
    assert records, "expected at least one usage record"
    for rec in records:
        assert {"input_tokens", "output_tokens", "total_tokens"} <= set(rec)
        assert rec["total_tokens"] == rec["input_tokens"] + rec["output_tokens"]


def test_classifier_handles_unparseable_json(tmp_path, caplog):
    """Bad JSON from the model should NOT crash the run; the batch
    re-asks via the retry loop and eventually forces to the last letter."""
    facets = [_mk_facet("The tutor asks a guiding question.")]
    # Every response is junk -> retries exhausted -> statement forced to LAST_LETTER.
    client = _FakeClient(payloads=["not valid json"] * 8)
    with caplog.at_level("WARNING", logger="tutormoments.taxonomy"):
        assignments = tx.classify_pool(facets, tmp_path, client=client, max_batches=1)
    assert assignments[facets[0].statement.strip()] == tx.LAST_LETTER
    assert any("unparseable JSON" in rec.message for rec in caplog.records)


def test_classifier_resume_skips_completed_batches(tmp_path):
    facets = [
        _mk_facet("The tutor asks a guiding question.", moment_id="m1"),
        _mk_facet("The tutor offers a hint.", moment_id="m2"),
    ]
    # First pass: both classified.
    first = _FakeClient(
        payloads=[
            '{"assignments":[{"id":1,"category":"A"},{"id":2,"category":"E"}]}',
        ]
    )
    tx.classify_pool(facets, tmp_path, client=first, max_batches=1)
    assert len(first.calls) == 1

    # Second pass: nothing pending; the classifier should make no API calls.
    second = _FakeClient()
    tx.classify_pool(facets, tmp_path, client=second, max_batches=1)
    assert second.calls == []


# ---------------------------------------------------------------------------
# 7. CLI extras guard
# ---------------------------------------------------------------------------


def test_require_analysis_extras_passes_when_pandas_present():
    pytest.importorskip("pandas")
    tx._require_analysis_extras()  # should NOT raise


def test_require_analysis_extras_message_mentions_install(monkeypatch):
    """If pandas import raises, the error message tells the user how to install."""
    import importlib

    real = importlib.import_module

    def fake(name, *a, **kw):
        if name == "pandas":
            raise ImportError("pretend pandas isn't installed")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(ImportError) as ei:
        tx._require_analysis_extras()
    assert "tutormoments[analysis]" in str(ei.value)


# ---------------------------------------------------------------------------
# 8. In-run classification: facets_from_annotations + classify_run
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _ann(**kw):
    base = dict(
        scenario_id="c1__0_2",
        annotation_type="scaffolding",
        action_decomposed=["The tutor asks a guiding question."],
        turn_start=0,
        turn_end=2,
        action_label="scaffolding",
        result="ok",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _moment(id="c1__0_2", dimension="scaffolding"):
    return SimpleNamespace(id=id, dimension=dimension)


def test_facets_from_annotations_uses_moment_dimension():
    facets = list(
        tx.facets_from_annotations(
            [_ann()],
            [_moment(dimension="rigor")],
            model="claude-sonnet-5",
            mode="plain",
        )
    )
    assert len(facets) == 1
    f = facets[0]
    assert f.situation_label == "rigor"  # from Moment.dimension
    assert f.result_label == "ok"  # from Annotation.result (not result_label)
    assert f.model == "claude-sonnet-5" and f.prompt == "plain"
    assert f.annotation_type == "scaffolding"


def test_facets_from_annotations_skips_non_scaffolding():
    facets = list(
        tx.facets_from_annotations(
            [_ann(annotation_type="rapport")],
            [_moment()],
            model="m",
            mode="plain",
        )
    )
    assert facets == []


def test_classify_run_writes_classified_and_counts(tmp_path):
    anns = [
        _ann(
            scenario_id="c1__0_2",
            action_decomposed=["The tutor asks a guiding question."],
        ),
        _ann(scenario_id="c2__0_2", action_decomposed=["The tutor offers a hint."]),
    ]
    moments = [_moment(id="c1__0_2"), _moment(id="c2__0_2")]
    client = _FakeClient(
        payloads=[
            '{"assignments":[{"id":1,"category":"A"},{"id":2,"category":"E"}]}',
        ]
    )
    out = tmp_path / "taxonomy"
    summary = tx.classify_run(
        anns,
        moments,
        out,
        model="claude-sonnet-5",
        mode="plain",
        client=client,
    )
    assert (out / "classified.csv").exists()
    assert summary["scheme_version"] == tx.SCHEME_VERSION
    assert summary["n_facets"] == 2
    assert summary["counts"] == {"A": 1, "E": 1}
    # A and E are both scaffolding-oriented categories
    assert summary["orientation"] == {"scaffolding": 2, "rigor": 0, "neutral": 0}
    assert summary["usage"]["total_tokens"] > 0


def test_taxonomy_spec_reads_config():
    from tutormoments import config as cfgmod
    from tutormoments.models import ThinkingLevel

    cfgmod._reset_config_cache()
    spec = cfgmod.taxonomy_spec()
    assert spec.model == "claude-opus-4-8"
    assert spec.thinking == ThinkingLevel.NONE
    assert spec.batch_size == 50


def test_read_paper_distribution_selects_series(tmp_path):
    pytest.importorskip("pandas")
    csv = tmp_path / "dist.csv"
    csv.write_text(
        "letter,name,orientation,"
        "human__n_moments,human__macro_mean_pct,human__ci_low,human__ci_high,"
        "claude_opus_4_8__plain__n_moments,claude_opus_4_8__plain__macro_mean_pct,"
        "claude_opus_4_8__plain__ci_low,claude_opus_4_8__plain__ci_high\n"
        "A,Guiding,scaffolding,514,10.18,8.60,11.75,100,20.27,16.59,23.94\n"
        "B,Steps,scaffolding,514,2.01,1.38,2.63,100,7.83,5.46,10.21\n",
        encoding="utf-8",
    )
    human = tx.read_paper_distribution(csv, "human")
    assert human.loc["A", "mean_pct"] == pytest.approx(10.18)
    assert human.loc["A", "orientation"] == "scaffolding"
    assert set(human.columns) >= {
        "name",
        "orientation",
        "mean_pct",
        "ci_low",
        "ci_high",
    }
    # A model series selects its own column group
    model = tx.read_paper_distribution(csv, "claude_opus_4_8__plain")
    assert model.loc["A", "mean_pct"] == pytest.approx(20.27)
    with pytest.raises(KeyError):
        tx.read_paper_distribution(csv, "nonexistent_series")
