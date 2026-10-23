"""Tests for the action-taxonomy release build (tutormoments_build.action_taxonomy).

Offline: a stub client stands in for the classifier model; fixtures are tiny
ground_truth / benchmark_520 / moments files written to tmp_path.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tutormoments import taxonomy as tx
from tutormoments.moments import file_sha256, records_content_hash
from tutormoments_build import action_taxonomy as at

REVISION = "66058b4c7ef5e7631b3c4d39a1dfa8172e36d8bc"
SPEC = SimpleNamespace(
    model="claude-opus-4-8", thinking={"type": "disabled"}, batch_size=2
)

U1 = "11111111-1111-5111-8111-111111111111"
U2 = "22222222-2222-5222-8222-222222222222"
CONV = f"{U1}_{U2}_{U1}"  # benchmark conv_id composite; trailing UUID = U1
SCEN = f"{CONV}__hum_10_20"

GROUND_TRUTH = [
    {
        "conversation_id": U2,
        "num_turns": 50,
        "key_moments": [
            {"annotation_type": "rapport", "turn_start": 1, "turn_end": 2},
            {
                "annotation_type": "scaffolding",
                "turn_start": 5,
                "turn_end": 8,
                "annotator_id": "ann-1",
                "situation_label_agg": "rigor",
                "action_direction_agg": "scaffolding",
                "student_outcome_agg": "pos",
                "action": "free text",
                "result": "free text result",
                "action_decomposed": [
                    "The tutor asks a guiding question.",
                    "The student answers correctly.",
                    "The tutor does not push for rigor.",
                ],
            },
            # Same (conversation, turns, annotator) as above: shares a moment_id.
            {
                "annotation_type": "scaffolding",
                "turn_start": 5,
                "turn_end": 8,
                "annotator_id": "ann-1",
                "situation_label_agg": "rigor",
                "action_direction_agg": "scaffolding",
                "student_outcome_agg": "pos",
                "action_decomposed": ["The tutor asks a guiding question."],
            },
            {
                "annotation_type": "scaffolding",
                "turn_start": 30,
                "turn_end": 31,
                "annotator_id": "ann-2",
                "situation_label_agg": "neither",
                "action_direction_agg": "neither",
                "student_outcome_agg": "no_evidence",
                "action_decomposed": ["There is no scaffolding."],
            },
            {
                "annotation_type": "scaffolding",
                "turn_start": 40,
                "turn_end": 41,
                "situation_label_agg": "scaffolding",
                "action_decomposed": [],
            },
        ],
    }
]

MOMENTS = [{"id": f"balanced_520:{SCEN}", "dimension": "scaffolding"}]


def _bench_row(model, mode, statements):
    return {
        "cell": f"{model}__{mode}",
        "tutor_model": model,
        "prompt_mode": mode,
        "scenario_id": SCEN,
        "conv_id": CONV,
        "detection": {"situation_label_agg": "scaffolding"},
        "annotation": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "turn_start": 11,
                    "turn_end": 15,
                    "action": "free text",
                    "result": "The tutor's strategy is effective. A paragraph.",
                    "action_decomposed": statements,
                    "action_label": "both",
                    "result_label": "neg",
                }
            ]
        },
    }


BENCHMARK = [
    _bench_row(
        "model-b",
        "plain",
        ["The tutor is scaffolding throughout.", "The tutor offers a hint."],
    ),
    _bench_row(
        "model-a",
        "scaffolding_rigor",
        ["The tutor offers a hint.", "The tutor asks a guiding question."],
    ),
]

KNOWN = {
    "The tutor asks a guiding question.": "A",
    "The tutor offers a hint.": "E",
}


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    d = tmp_path / "inputs"
    d.mkdir()
    return {
        "ground_truth.jsonl": _write_jsonl(d / "ground_truth.jsonl", GROUND_TRUTH),
        "benchmark_520.jsonl": _write_jsonl(d / "benchmark_520.jsonl", BENCHMARK),
        "moments.jsonl": _write_jsonl(d / "moments.jsonl", MOMENTS),
    }


class _Resp:
    def __init__(self, text, usage):
        self.text = text
        self.usage = usage


class _StubClient:
    """Labels every known statement in the prompt's numbered block."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        assignments = [
            {"id": int(m.group(1)), "category": KNOWN[m.group(2)]}
            for m in re.finditer(r"^(\d+)\. (.+)$", prompt, re.MULTILINE)
            if m.group(2) in KNOWN
        ]
        usage = {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "input_uncached": 100,
            "output": 10,
            "total": 110,
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "endpoint": "sync",
        }
        return _Resp(json.dumps({"assignments": assignments}), usage)


class _NoCallClient:
    def generate(self, *a, **kw):
        raise AssertionError("classifier must not be called")


def _build(inputs, out, client, **kw):
    return at.run_build(
        inputs, REVISION, out, spec=SPEC, client=client, created="2026-10-01", **kw
    )


def _rows(out: Path) -> list[dict]:
    path = out / "upload" / at.RELEASE_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

_SHARED = (
    "moment_id",
    "transcript_id",
    "turn_start",
    "turn_end",
    "statement_index",
    "statement",
    "annotation_type",
    "situation_label",
    "model",
    "prompt",
)


def test_human_facets_mirror_runtime_loader_except_labels(inputs):
    ours = list(at.human_facets(inputs["ground_truth.jsonl"]))
    runtime = list(tx.load_key_moments_jsonl(inputs["ground_truth.jsonl"]))
    assert [[getattr(f, k) for k in _SHARED] for f in ours] == [
        [getattr(f, k) for k in _SHARED] for f in runtime
    ]
    # Labels come from the release's aggregate fields, which the runtime
    # loader does not read.
    assert {(f.action_label, f.result_label) for f in runtime} == {(None, None)}
    assert (ours[0].action_label, ours[0].result_label) == ("scaffolding", "pos")
    assert [f.key_moment_index for f in ours] == [1, 1, 1, 2, 3]
    assert ours[0].moment_id == f"{U2}__5_8__ann-1"


def test_lm_facets_mirror_runtime_adapter_except_result_label(inputs):
    dims = at.moment_dimensions(inputs["moments.jsonl"])
    ours = list(at.lm_facets(inputs["benchmark_520.jsonl"], dims))

    row = BENCHMARK[0]
    ann = SimpleNamespace(scenario_id=None, **row["annotation"]["annotations"][0])
    moment = SimpleNamespace(id=f"balanced_520:{SCEN}", dimension="scaffolding")
    runtime = list(
        tx.facets_from_annotations(
            [ann], [moment], model=row["tutor_model"], mode=row["prompt_mode"]
        )
    )
    assert [[getattr(f, k) for k in _SHARED] for f in ours[:2]] == [
        [getattr(f, k) for k in _SHARED] for f in runtime
    ]
    f = ours[0]
    assert f.moment_id == f"balanced_520:{SCEN}"
    assert f.transcript_id == U1
    assert f.situation_label == "scaffolding"  # from moments.dimension
    assert (f.action_label, f.result_label) == ("both", "neg")
    # The runtime adapter puts the free-text result paragraph here.
    assert runtime[0].result_label.startswith("The tutor's strategy")
    assert f.key_moment_index is None


def test_lm_facets_require_moment_join(inputs):
    with pytest.raises(ValueError, match="no moments.jsonl row"):
        list(at.lm_facets(inputs["benchmark_520.jsonl"], {}))


def test_lm_facets_require_single_annotation(tmp_path):
    row = _bench_row("m", "plain", ["The tutor offers a hint."])
    row["annotation"]["annotations"].append(row["annotation"]["annotations"][0])
    path = _write_jsonl(tmp_path / "b.jsonl", [row])
    with pytest.raises(ValueError, match="expected exactly 1"):
        list(at.lm_facets(path, {f"balanced_520:{SCEN}": "rigor"}))


def test_revision_must_be_full_commit_sha(inputs, tmp_path):
    for bad in ("main", "66058b4", ""):
        with pytest.raises(ValueError, match="40-character"):
            at.run_build(inputs, bad, tmp_path / "out", spec=SPEC, dry_run=True)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_makes_no_calls_and_writes_nothing(inputs, tmp_path, capsys):
    out = tmp_path / "out"
    assert _build(inputs, out, _NoCallClient(), dry_run=True) is None
    assert not out.exists()
    report = capsys.readouterr().out
    assert "unique statements 1 (0 already assigned, 1 pending)" in report
    assert "unique statements 2 (0 already assigned, 2 pending)" in report


# ---------------------------------------------------------------------------
# Build + assembly
# ---------------------------------------------------------------------------


def test_partial_classification_does_not_write_release(inputs, tmp_path):
    out = tmp_path / "out"
    spec = SimpleNamespace(**{**vars(SPEC), "batch_size": 1})
    result = at.run_build(
        inputs, REVISION, out, spec=spec, client=_StubClient(), max_batches=1
    )
    assert result is None
    assert (at.pool_dir(out, "benchmark_520") / "assignments.jsonl").exists()
    assert not (out / "upload").exists()


def test_build_is_deterministic_from_fixed_assignments(inputs, tmp_path):
    first = tmp_path / "first"
    _build(inputs, first, _StubClient())

    # Re-assemble from copies of the sidecars alone: no classifier calls, and
    # byte-identical output.
    second = tmp_path / "second"
    for source in at.SOURCES:
        dst = at.pool_dir(second, source)
        dst.mkdir(parents=True)
        for name in ("assignments.jsonl", "usage_log.jsonl"):
            (dst / name).write_bytes((at.pool_dir(first, source) / name).read_bytes())
    _build(inputs, second, _NoCallClient())

    for name in (at.RELEASE_FILENAME, at.MANIFEST_FILENAME, at.SCHEMA_FILENAME):
        assert (first / "upload" / name).read_bytes() == (
            second / "upload" / name
        ).read_bytes()


def test_release_rows(inputs, tmp_path):
    out = tmp_path / "out"
    _build(inputs, out, _StubClient())
    rows = _rows(out)

    # Every action_decomposed item is a row, excluded ones included.
    assert len(rows) == 5 + 4
    keys = [
        (
            r["source"],
            r["tutor_model"],
            r["prompt_mode"],
            r["moment_id"],
            r["key_moment_index"],
            r["statement_index"],
        )
        for r in rows
    ]
    assert len(set(keys)) == len(keys)
    assert rows == sorted(rows, key=at._row_sort_key)
    assert [r["source"] for r in rows] == ["human"] * 5 + ["benchmark_520"] * 4
    assert [r["tutor_model"] for r in rows[5:]] == ["model-a"] * 2 + ["model-b"] * 2

    by_stmt = {(r["source"], r["statement"]): r for r in rows}
    hint = by_stmt[("benchmark_520", "The tutor offers a hint.")]
    assert (hint["category"], hint["category_name"], hint["orientation"]) == (
        "E",
        tx.NAME_BY_LETTER["E"],
        "scaffolding",
    )
    assert hint["excluded_reason"] is None

    expected_reasons = {
        ("human", "The student answers correctly."): "non_tutor_actor",
        ("human", "The tutor does not push for rigor."): "stance_negation",
        ("human", "There is no scaffolding."): "non_tutor_actor",
        ("benchmark_520", "The tutor is scaffolding throughout."): "pure_stance",
    }
    for key, reason in expected_reasons.items():
        r = by_stmt[key]
        assert (r["category"], r["category_name"], r["orientation"]) == (None,) * 3
        assert r["excluded_reason"] == reason


def test_release_rows_validate_against_schema(inputs, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    out = tmp_path / "out"
    _build(inputs, out, _StubClient())
    schema = json.loads((out / "upload" / at.SCHEMA_FILENAME).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for row in _rows(out):
        assert not list(validator.iter_errors(row)), row

    # The schema is enforced, not documentation: inconsistent rows are refused.
    good = next(r for r in _rows(out) if r["source"] == "human" and r["category"])
    for bad in (
        {**good, "excluded_reason": "pure_stance"},  # excluded row with a category
        {**good, "category": None},  # kept row without a category
        {**good, "tutor_model": "x"},  # human row with a model
        {**good, "category": "Z"},
        {**good, "extra": 1},
    ):
        with pytest.raises(ValueError, match="violates"):
            at.validate_rows([bad])


def test_manifest_matches_output(inputs, tmp_path):
    out = tmp_path / "out"
    manifest = _build(inputs, out, _StubClient())
    upload = out / "upload"
    assert json.loads((upload / at.MANIFEST_FILENAME).read_text()) == manifest

    rows = _rows(out)
    assert manifest["record_count"] == len(rows)
    assert manifest["file_sha256"] == file_sha256(upload / at.RELEASE_FILENAME)
    assert manifest["content_hash"] == records_content_hash(rows)
    assert manifest["inputs"]["revision"] == REVISION
    assert manifest["inputs"]["file_sha256"] == {
        name: file_sha256(p) for name, p in inputs.items()
    }
    assert manifest["classifier"]["model"] == SPEC.model
    assert manifest["scheme_version"] == tx.SCHEME_VERSION

    human, lm = manifest["counts"]["human"], manifest["counts"]["benchmark_520"]
    # Two key-moment records share a moment_id; the empty-actions one yields
    # no rows; the "There is no scaffolding." moment has no kept facet.
    assert (human["facets"], human["kept"], human["moments"]) == (5, 2, 2)
    assert human["moments_with_kept"] == 1
    assert human["excluded_by_reason"]["non_tutor_actor"] == 2
    assert human["by_category"]["A"] == 2
    # Each LM replay is its own moment even though the moment id repeats.
    assert (lm["moments"], lm["unique_kept_statements"]) == (2, 2)
    assert set(lm["by_cell"]) == {"model-a__scaffolding_rigor", "model-b__plain"}
    assert lm["by_situation"]["scaffolding"]["facets"] == 4

    assert manifest["usage"]["benchmark_520"]["input_tokens"] > 0
    assert manifest["cost_usd"]["total"] is not None


def test_pools_are_classified_separately(inputs, tmp_path):
    """The shared statement is classified once per pool, in its own sidecar."""
    out = tmp_path / "out"
    _build(inputs, out, _StubClient())
    shared = "The tutor asks a guiding question."
    for source in at.SOURCES:
        assigned = tx._read_assignments(at.pool_dir(out, source) / "assignments.jsonl")
        assert shared in assigned


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_dispatches(tmp_path):
    from tutormoments_build.cli import main

    with patch(
        "tutormoments_build.action_taxonomy._cli_build_action_taxonomy"
    ) as mock_build:
        main(
            [
                "dataset",
                "build-action-taxonomy",
                "--revision",
                REVISION,
                "--out",
                str(tmp_path / "out"),
                "--dry-run",
                "--max-batches",
                "2",
            ]
        )
    args = mock_build.call_args[0][0]
    assert (args.revision, args.dry_run, args.max_batches) == (REVISION, True, 2)
    assert args.out == str(tmp_path / "out")
    assert not (tmp_path / "out").exists()  # dry run writes no build.log


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

# The anchors update_dataset_card edits, as they appear in the Hub card at the
# pinned revision.
CARD = """---
configs:
  - config_name: benchmark_520
    data_files:
      - split: train
        path: benchmark_520.jsonl
---

| `benchmark_520` | 7,280 | one AI-tutor replay (520 moments × 7 models × 2 prompts) |

Fields are defined by the bundled JSON Schemas (`moments.schema.json`,
`benchmark_520.schema.json`). Common join key.

## Deidentification

Models: and GPT 5.5 and 5.4 mini (provided by OpenAI). 
"""


def test_build_writes_updated_dataset_card(inputs, tmp_path):
    (tmp_path / "README.md").write_text(CARD, encoding="utf-8")
    out = tmp_path / "out"
    manifest = _build(
        {**inputs, at.CARD_FILENAME: tmp_path / "README.md"}, out, _StubClient()
    )
    card = (out / "upload" / "README.md").read_text(encoding="utf-8")
    assert (
        "  - config_name: action_taxonomy\n    data_files:\n      - split: train\n"
        "        path: action_taxonomy.jsonl\n---\n"
    ) in card
    assert "| `action_taxonomy` | 9 | one decomposed tutor action" in card
    assert "`benchmark_520.schema.json`, `action_taxonomy.schema.json`)" in card
    assert card.index("## action_taxonomy") < card.index("## Deidentification")
    assert REVISION in card and "claude-opus-4-8" in card
    with pytest.raises(ValueError, match="already has"):
        at.update_dataset_card(card, manifest)


def test_dataset_card_anchor_drift_raises(inputs, tmp_path):
    manifest = _build(inputs, tmp_path / "out", _StubClient())
    with pytest.raises(ValueError, match="anchor found 0 times"):
        at.update_dataset_card(
            CARD.replace("## Deidentification", "## Privacy"), manifest
        )
