"""Tests for the v2 doubly-annotated ground-truth build."""

import json
import os

import pytest

from tutormoments_build.v2 import build_ground_truth as G


def _payload(
    *,
    scaffolding_appropriate=False,
    rigor_appropriate=False,
    scaffolding_present=False,
    rigor_present=False,
    scaffolding_amount=None,
    meta=None,
):
    payload = {
        "situation": {
            "scaffolding_appropriate": scaffolding_appropriate,
            "rigor_appropriate": rigor_appropriate,
            "why": "",
        },
        "action": {
            "scaffolding_present": scaffolding_present,
            "rigor_present": rigor_present,
            "scaffolding_amount": scaffolding_amount,
            "over_scaffolding_reasons": [],
        },
        "result": {},
    }
    if meta is not None:
        payload["meta"] = meta
    return payload


def _annotation(role="selector", name="Paul", **kw):
    return {
        "annotator_id": f"id-{name.lower()}",
        "annotator_name": name,
        "role": role,
        "revision": 1,
        "payload": _payload(**kw),
    }


def _moment(moment_id="m1", **overrides):
    moment = {
        "moment_id": moment_id,
        "start_turn": 10,
        "end_turn": 20,
        "cut_turn": 15,
        "start_index": 12,
        "end_index": 24,
        "cut_index": 18,
        "dialogue_turns": 10,
        "status": "reannotated",
        "created_at": "2026-08-03T18:28:56Z",
    }
    moment.update(overrides)
    return moment


def _row(transcript_id="t1", moment_id="m1", annotations=None, **moment_kw):
    return {
        "transcript_id": transcript_id,
        "moment": _moment(moment_id, **moment_kw),
        "annotations": annotations if annotations is not None else [_annotation()],
    }


# ---------------------------------------------------------------------------
# label resolution -- union across annotators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,kwargs",
    [
        ("scaffolding_appropriate", {"scaffolding_appropriate": True}),
        ("rigor_appropriate", {"rigor_appropriate": True}),
        ("scaffolding_present", {"scaffolding_present": True}),
        ("rigor_present", {"rigor_present": True}),
    ],
)
def test_single_annotator_true_makes_the_field_true(field, kwargs):
    labels, agreement, _ = G.resolve_labels(
        [_annotation(**kwargs), _annotation(role="reannotator", name="Anita")]
    )
    assert labels[field] is True, "union must take a label only one annotator gave"
    assert agreement[field] is False


def test_union_is_symmetric_in_annotator_order():
    a = _annotation(scaffolding_present=True)
    b = _annotation(role="reannotator", name="Anita", rigor_present=True)
    assert G.resolve_labels([a, b])[0] == G.resolve_labels([b, a])[0]


def test_all_false_stays_false_and_agrees():
    labels, agreement, _ = G.resolve_labels(
        [_annotation(), _annotation(role="reannotator", name="Anita")]
    )
    assert not any(labels.values())
    assert all(agreement.values())


def test_agreement_is_true_when_both_say_true():
    labels, agreement, _ = G.resolve_labels(
        [
            _annotation(rigor_appropriate=True),
            _annotation(role="reannotator", name="Anita", rigor_appropriate=True),
        ]
    )
    assert labels["rigor_appropriate"] is True
    assert agreement["rigor_appropriate"] is True


def test_union_covers_three_annotators():
    labels, _, _ = G.resolve_labels(
        [
            _annotation(scaffolding_appropriate=True),
            _annotation(name="Anita", rigor_appropriate=True),
            _annotation(role="reannotator", name="Kelly", rigor_present=True),
        ]
    )
    assert labels["scaffolding_appropriate"]
    assert labels["rigor_appropriate"]
    assert labels["rigor_present"]


# ---------------------------------------------------------------------------
# over-scaffolding derivation
# ---------------------------------------------------------------------------


def test_over_scaffolding_comes_from_scaffolding_amount():
    labels, _, _ = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=True,
                scaffolding_present=True,
                scaffolding_amount="over_scaffolding",
            ),
            _annotation(
                role="reannotator",
                name="Anita",
                scaffolding_appropriate=True,
                scaffolding_present=True,
                scaffolding_amount="appropriate",
            ),
        ]
    )
    assert labels["over_scaffolding_present"] is True
    assert labels["scaffolding_present"] is True


@pytest.mark.parametrize(
    "amount", ["appropriate", "under_scaffolding", "unclear", None]
)
def test_other_scaffolding_amounts_are_not_over_scaffolding(amount):
    labels, _, _ = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=True,
                scaffolding_present=True,
                scaffolding_amount=amount,
            )
        ]
    )
    assert labels["over_scaffolding_present"] is False


def test_scaffolding_where_it_was_not_appropriate_counts_as_over_scaffolding():
    # Supporting a student who did not need supporting is over-scaffolding
    # whatever amount the annotator picked.
    labels, _, inferred = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=False,
                scaffolding_present=True,
                scaffolding_amount="appropriate",
            )
        ]
    )
    assert labels["over_scaffolding_present"] is True
    assert inferred is True


@pytest.mark.parametrize("amount", ["appropriate", "under_scaffolding", "unclear"])
def test_inference_overrides_whatever_amount_was_declared(amount):
    labels, _, _ = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=False,
                scaffolding_present=True,
                scaffolding_amount=amount,
            )
        ]
    )
    assert labels["over_scaffolding_present"] is True


def test_inference_needs_the_tutor_to_actually_scaffold():
    # Scaffolding inappropriate and none given is the well-behaved case, not
    # over-scaffolding.
    labels, _, inferred = G.resolve_labels(
        [_annotation(scaffolding_appropriate=False, scaffolding_present=False)]
    )
    assert labels["over_scaffolding_present"] is False
    assert inferred is False


def test_inference_is_per_annotator_not_post_union():
    # One annotator saw scaffolding where they judged none was needed. Post-union
    # the situation label is True (union of appropriate), so the rule would never
    # fire there; applied per annotator it does.
    labels, _, inferred = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=False,
                scaffolding_present=True,
                scaffolding_amount="appropriate",
            ),
            _annotation(
                role="reannotator",
                name="Anita",
                scaffolding_appropriate=True,
                scaffolding_present=False,
            ),
        ]
    )
    assert labels["scaffolding_appropriate"] is True
    assert labels["over_scaffolding_present"] is True
    assert inferred is True


def test_declared_over_scaffolding_is_not_marked_inferred():
    _, _, inferred = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=True,
                scaffolding_present=True,
                scaffolding_amount="over_scaffolding",
            )
        ]
    )
    assert inferred is False


def test_inferred_flag_reaches_the_record():
    row = _row(
        annotations=[
            _annotation(
                scaffolding_appropriate=False,
                scaffolding_present=True,
                scaffolding_amount="appropriate",
            ),
            _annotation(role="reannotator", name="Anita"),
        ]
    )
    record = G.build_record(row, "iterate", {})
    assert record["labels"]["over_scaffolding_present"] is True
    assert record["over_scaffolding_inferred"] is True


def test_over_scaffolding_implies_scaffolding_present():
    # The interface only offers an amount once scaffolding is marked present, so
    # the union of the two fields can never contradict itself.
    labels, _, _ = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=True,
                scaffolding_present=True,
                scaffolding_amount="over_scaffolding",
            ),
            _annotation(role="reannotator", name="Anita", scaffolding_present=False),
        ]
    )
    assert labels["over_scaffolding_present"] and labels["scaffolding_present"]


# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------


def test_boundaries_unchanged_when_reannotator_redrew_nothing():
    moment = _moment()
    reann = _annotation(role="reannotator", meta={"redrew_cut_point": False})
    boundaries, original = G.effective_boundaries(moment, reann)
    assert original is None
    assert boundaries["cut_turn"] == 15 and boundaries["end_turn"] == 20


def test_partial_redraw_keeps_untouched_fields():
    # Real rows carry a new cut turn with null start/end; those must not be
    # overwritten with null.
    moment = _moment()
    reann = _annotation(
        role="reannotator",
        meta={
            "redrew_cut_point": True,
            "new_cut_turn": 17,
            "new_start_turn": None,
            "new_end_turn": None,
        },
    )
    boundaries, original = G.effective_boundaries(moment, reann)
    assert boundaries["cut_turn"] == 17
    assert boundaries["start_turn"] == 10 and boundaries["end_turn"] == 20
    assert original["cut_turn"] == 15


def test_redrawn_boundaries_are_reported_on_the_record():
    row = _row(
        annotations=[
            _annotation(),
            _annotation(
                role="reannotator",
                name="Anita",
                meta={"redrew_cut_point": True, "new_cut_turn": 17},
            ),
        ]
    )
    record = G.build_record(row, "iterate", {})
    assert record["cut_turn"] == 17
    assert record["boundaries_redrawn"] is True
    assert record["cut_point_redrawn"] is True
    assert record["original_boundaries"]["cut_turn"] == 15


def test_original_boundaries_absent_when_nothing_moved():
    row = _row(
        annotations=[_annotation(), _annotation(role="reannotator", name="Anita")]
    )
    assert "original_boundaries" not in G.build_record(row, "iterate", {})


# ---------------------------------------------------------------------------
# de-identification
# ---------------------------------------------------------------------------


def test_annotator_name_is_replaced_by_label():
    assert G.annotator_label(_annotation(name="Paul"), {"paul": "A02"}) == "A02"


def test_multiword_name_is_normalised():
    assert (
        G.annotator_label(_annotation(name="Jessica-Lyn"), {"jessica-lyn": "A16"})
        == "A16"
    )


def test_unmapped_annotator_falls_back_to_id_not_name():
    label = G.annotator_label(_annotation(name="Nobody"), {"paul": "A02"})
    assert label == "id-nobody"
    assert "Nobody" not in label


# ---------------------------------------------------------------------------
# build: filtering and routing
# ---------------------------------------------------------------------------


def _fixture(tmp_path, rows, assignments):
    ann = tmp_path / "annotations.jsonl"
    ann.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 1,
                "heldout_fraction": 0.5,
                "balance_by": "transcripts",
                "batches": [{"batch": 1}],
                "assignments": {t: {"split": s} for t, s in assignments.items()},
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"paul": "A02", "anita": "A01"}), encoding="utf-8")
    return str(ann), str(splits), str(labels)


def _both(name="Anita", **kw):
    return [_annotation(), _annotation(role="reannotator", name=name, **kw)]


def test_splits_route_to_iteration_and_test(tmp_path):
    rows = [
        _row("t1", "m1", _both()),
        _row("t2", "m2", _both()),
    ]
    out, _ = G.build(*_fixture(tmp_path, rows, {"t1": "iterate", "t2": "heldout"}))
    assert [r["moment_id"] for r in out["iteration"]] == ["m1"]
    assert [r["moment_id"] for r in out["test"]] == ["m2"]
    assert out["iteration"][0]["split"] == "iteration"


def test_singly_annotated_moments_are_dropped(tmp_path):
    rows = [_row("t1", "m1", [_annotation()]), _row("t1", "m2", _both())]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    assert [r["moment_id"] for r in out["iteration"]] == ["m2"]
    assert dropped["not_doubly_annotated"] == 1


def test_moment_without_reannotator_is_dropped(tmp_path):
    rows = [_row("t1", "m1", [_annotation(), _annotation(name="Anita")])]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    assert out["iteration"] == []
    assert dropped["not_doubly_annotated"] == 1


def test_thrown_out_moments_are_dropped_by_default(tmp_path):
    rows = [
        _row(
            "t1", "m1", _both(meta={"throw_out": True, "throw_out_reason": "no math"})
        ),
        _row("t1", "m2", _both()),
    ]
    args = _fixture(tmp_path, rows, {"t1": "iterate"})
    out, dropped = G.build(*args)
    assert [r["moment_id"] for r in out["iteration"]] == ["m2"]
    assert dropped["thrown_out"] == 1

    kept, _ = G.build(*args, keep_thrown_out=True)
    assert {r["moment_id"] for r in kept["iteration"]} == {"m1", "m2"}
    assert [r for r in kept["iteration"] if r["thrown_out"]][0]["moment_id"] == "m1"


def test_no_key_moment_rows_are_skipped(tmp_path):
    rows = [
        {
            "transcript_id": "t1",
            "no_key_moments_record": {"payload": {"no_key_moments": True}},
        },
        _row("t1", "m1", _both()),
    ]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    assert len(out["iteration"]) == 1
    assert dropped["no_key_moments_record"] == 1


def test_transcript_missing_from_splits_is_skipped(tmp_path):
    rows = [_row("t1", "m1", _both()), _row("unknown", "m2", _both())]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    assert len(out["iteration"]) == 1
    assert dropped["transcript_not_in_splits"] == 1


def test_empty_split_manifest_is_an_error(tmp_path):
    rows = [_row("t1", "m1", _both())]
    with pytest.raises(ValueError, match="no split assignments"):
        G.build(*_fixture(tmp_path, rows, {}))


def test_records_are_sorted_deterministically(tmp_path):
    rows = [
        _row("t2", "m2", _both()),
        _row("t1", "m1", _both()),
        _row("t1", "m0", _both()),
    ]
    out, _ = G.build(*_fixture(tmp_path, rows, {"t1": "iterate", "t2": "iterate"}))
    assert [r["moment_id"] for r in out["iteration"]] == ["m0", "m1", "m2"]


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def test_record_carries_all_five_labels_and_moment_metadata(tmp_path):
    rows = [_row("t1", "m1", _both(scaffolding_present=True))]
    out, _ = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    record = out["iteration"][0]

    assert set(record["labels"]) == set(G.LABEL_FIELDS)
    assert set(record["agreement"]) == set(G.LABEL_FIELDS)
    assert all(isinstance(v, bool) for v in record["labels"].values())
    for field in (
        "moment_id",
        "transcript_id",
        "split",
        "cut_turn",
        "start_turn",
        "end_turn",
        "dialogue_turns",
        "n_annotators",
        "annotators",
    ):
        assert field in record
    assert record["annotators"] == ["A02", "A01"]
    assert record["annotator_roles"] == ["selector", "reannotator"]


def test_cli_writes_both_files(tmp_path):
    rows = [_row("t1", "m1", _both()), _row("t2", "m2", _both())]
    ann, splits, labels = _fixture(tmp_path, rows, {"t1": "iterate", "t2": "heldout"})
    out_dir = tmp_path / "ground_truth"
    code = G.main(
        [
            "--annotations",
            ann,
            "--splits",
            splits,
            "--annotator-labels",
            labels,
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    for stem in ("iteration", "test"):
        path = out_dir / f"{stem}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["split"] == stem


def test_cli_dry_run_writes_nothing(tmp_path):
    rows = [_row("t1", "m1", _both())]
    ann, splits, labels = _fixture(tmp_path, rows, {"t1": "iterate"})
    out_dir = tmp_path / "ground_truth"
    assert (
        G.main(
            [
                "--annotations",
                ann,
                "--splits",
                splits,
                "--annotator-labels",
                labels,
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not os.path.exists(out_dir)


def test_rerun_is_byte_identical(tmp_path):
    rows = [_row("t1", "m1", _both()), _row("t1", "m2", _both(rigor_present=True))]
    ann, splits, labels = _fixture(tmp_path, rows, {"t1": "iterate"})
    out_dir = tmp_path / "ground_truth"
    argv = [
        "--annotations",
        ann,
        "--splits",
        splits,
        "--annotator-labels",
        labels,
        "--out-dir",
        str(out_dir),
    ]
    G.main(argv)
    first = (out_dir / "iteration.jsonl").read_bytes()
    G.main(argv)
    assert (out_dir / "iteration.jsonl").read_bytes() == first
