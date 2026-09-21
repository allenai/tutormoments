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
    labels, agreement, _, _ = G.resolve_labels(
        [_annotation(**kwargs), _annotation(role="reannotator", name="Anita")]
    )
    assert labels[field] is True, "union must take a label only one annotator gave"
    assert agreement[field] is False


def test_union_is_symmetric_in_annotator_order():
    a = _annotation(scaffolding_present=True)
    b = _annotation(role="reannotator", name="Anita", rigor_present=True)
    assert G.resolve_labels([a, b])[0] == G.resolve_labels([b, a])[0]


def test_all_false_stays_false_and_agrees():
    labels, agreement, _, _ = G.resolve_labels(
        [_annotation(), _annotation(role="reannotator", name="Anita")]
    )
    assert not any(labels.values())
    assert all(agreement.values())


def test_agreement_is_true_when_both_say_true():
    labels, agreement, _, _ = G.resolve_labels(
        [
            _annotation(rigor_appropriate=True),
            _annotation(role="reannotator", name="Anita", rigor_appropriate=True),
        ]
    )
    assert labels["rigor_appropriate"] is True
    assert agreement["rigor_appropriate"] is True


def test_with_no_adjudicator_both_halves_go_by_majority():
    # The two rules only diverge through the adjudicators. With none on the
    # moment, situation and action both count every vote, so one voice out of
    # three loses on either side.
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(scaffolding_appropriate=True),
            _annotation(name="Anita", rigor_present=True),
            _annotation(role="reannotator", name="Kelly"),
        ]
    )
    assert labels["scaffolding_appropriate"] is False
    assert labels["rigor_present"] is False
    assert how["action_decided_by"] == "annotators"


def test_situation_majority_carries_without_a_tie():
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(scaffolding_appropriate=True),
            _annotation(name="Anita", scaffolding_appropriate=True),
            _annotation(role="reannotator", name="Kelly"),
        ]
    )
    assert labels["scaffolding_appropriate"] is True
    assert how["situation_ties"] == {}, "a decided majority is not a tie"


# ---------------------------------------------------------------------------
# over-scaffolding derivation
# ---------------------------------------------------------------------------


def test_over_scaffolding_comes_from_scaffolding_amount():
    labels, _, _, _ = G.resolve_labels(
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
    labels, _, _, _ = G.resolve_labels(
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
    labels, _, inferred, _ = G.resolve_labels(
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
    labels, _, _, _ = G.resolve_labels(
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
    labels, _, inferred, _ = G.resolve_labels(
        [_annotation(scaffolding_appropriate=False, scaffolding_present=False)]
    )
    assert labels["over_scaffolding_present"] is False
    assert inferred is False


def test_inference_reads_the_resolved_labels_not_one_annotator():
    # One annotator saw scaffolding where they judged none was needed; the other
    # judged it called for. Union resolves scaffolding_appropriate to True, so
    # the rule's premise does not hold and it must not fire -- a label inferred
    # here would contradict the moment's own resolved situation label.
    labels, _, inferred, _ = G.resolve_labels(
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
    assert labels["over_scaffolding_present"] is False
    assert inferred is False


def test_inference_fires_when_both_annotators_say_not_appropriate():
    # Agreed "not called for", and one of them saw scaffolding delivered: the
    # rule's premise survives the union, so it still fires.
    labels, _, inferred, _ = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=False,
                scaffolding_present=True,
                scaffolding_amount="appropriate",
            ),
            _annotation(
                role="reannotator",
                name="Anita",
                scaffolding_appropriate=False,
                scaffolding_present=False,
            ),
        ]
    )
    assert labels["over_scaffolding_present"] is True
    assert inferred is True


def test_a_declared_amount_still_wins_whatever_the_situation_labels_say():
    # The inference only ever adds; an annotator who declared over-scaffolding
    # outright is never overridden by resolving the situation labels to
    # "appropriate".
    labels, _, inferred, _ = G.resolve_labels(
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
    assert inferred is False


# ===========================================================================
# Staff exclusion
# ===========================================================================


@pytest.mark.parametrize(
    "name", ["Lucy", "lucy", "Lucy Li", "REBECCA", "Albert", "Kajal"]
)
def test_staff_annotations_are_excluded(name):
    assert G.is_excluded({"annotator_name": name}) is True


@pytest.mark.parametrize("name", ["Carla", "Emily", "Jessica-Lyn", "", "Lucinda"])
def test_everyone_else_is_kept(name):
    assert G.is_excluded({"annotator_name": name}) is False


def test_a_missing_name_is_not_excluded():
    assert G.is_excluded({}) is False


def test_declared_over_scaffolding_is_not_marked_inferred():
    _, _, inferred, _ = G.resolve_labels(
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
    labels, _, _, _ = G.resolve_labels(
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
    boundaries, original, source = G.effective_boundaries(moment, [reann])
    assert original is None
    assert source == "selector"
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
    boundaries, original, source = G.effective_boundaries(moment, [reann])
    assert source == "reannotator"
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


def test_a_moment_annotated_by_staff_and_one_teacher_is_dropped(tmp_path):
    """Removing the staff pass leaves one annotator, so the moment falls out
    through the doubly-annotated requirement rather than being resolved alone."""
    rows = [
        _row("t1", "m1", [_annotation(), _annotation(role="reannotator", name="Lucy")]),
        _row("t1", "m2", _both()),
    ]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))

    assert [r["moment_id"] for r in out["iteration"]] == ["m2"]
    assert dropped["staff_annotations_removed"] == 1
    assert dropped["not_doubly_annotated"] == 1


def test_a_staff_pass_alongside_two_teachers_is_dropped_but_the_moment_survives(
    tmp_path,
):
    rows = [
        _row(
            "t1",
            "m1",
            [
                _annotation(),
                _annotation(role="reannotator", name="Anita"),
                _annotation(role="reannotator", name="Rebecca"),
            ],
        )
    ]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))

    assert [r["moment_id"] for r in out["iteration"]] == ["m1"]
    assert dropped["staff_annotations_removed"] == 1
    assert out["iteration"][0]["n_annotators"] == 2


def test_staff_labels_do_not_reach_the_resolved_moment(tmp_path):
    """The staff pass is the only True on rigor_present; it must not survive the
    union, which would otherwise carry any single annotator's True through."""
    rows = [
        _row(
            "t1",
            "m1",
            [
                _annotation(rigor_present=False),
                _annotation(role="reannotator", name="Anita", rigor_present=False),
                _annotation(role="reannotator", name="Albert", rigor_present=True),
            ],
        )
    ]
    out, _ = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))

    assert out["iteration"][0]["labels"]["rigor_present"] is False


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


# ===========================================================================
# Reading annotations: staff, re-saves, adjudicators, abstentions
# ===========================================================================


def _resolved(field):
    """The label name an action field resolves to on the record."""
    return (
        "over_scaffolding_present"
        if field == "over_scaffolding_declared"
        else field
    )


def _action(field, value):
    """``_payload`` keyword setting one action field, by its resolved name."""
    if field == "over_scaffolding_declared":
        return {
            "scaffolding_present": True,
            "scaffolding_amount": "over_scaffolding" if value else "appropriate",
        }
    return {field: value}


def _adjudication(name="Kelly", threw_out=False, boundaries=None, **kw):
    """An adjudicator's annotation, whose answers live in payload["final"]."""
    payload = {"final": None if threw_out else _payload(**kw)}
    if threw_out:
        payload["meta"] = {"throw_out": True}
    for field, value in (boundaries or {}).items():
        payload[f"final_{field}"] = value
    return {
        "annotator_id": f"id-{name.lower()}",
        "annotator_name": name,
        "role": "adjudicator",
        "revision": 1,
        "payload": payload,
    }


def test_a_resave_is_collapsed_to_the_highest_revision():
    first = _annotation(rigor_present=True)
    second = _annotation(rigor_present=False)
    second["revision"] = 2
    kept = G.latest_annotations([first, second])
    assert len(kept) == 1
    assert kept[0]["revision"] == 2


def test_a_resave_does_not_count_as_a_second_opinion():
    # The earlier draft said yes and the later one no. Left in, it would union
    # into a True and be recorded as a disagreement with its own author.
    first = _annotation(rigor_present=True)
    second = _annotation(rigor_present=False)
    second["revision"] = 2
    annotations = G.latest_annotations(
        [first, second, _annotation(role="reannotator", name="Anita")]
    )
    labels, agreement, _, _ = G.resolve_labels(annotations)
    assert labels["rigor_present"] is False
    assert agreement["rigor_present"] is True


def test_the_same_person_in_two_roles_is_kept_twice():
    # The key is (annotator, role): someone who selected a moment and later
    # reannotated it gave two judgments, not one saved twice.
    annotations = G.latest_annotations(
        [_annotation(), _annotation(role="reannotator")]
    )
    assert [a["role"] for a in annotations] == ["selector", "reannotator"]


def test_latest_annotations_drops_staff():
    assert G.latest_annotations([_annotation(name="Lucy"), _annotation(name="Paul")]) == [
        _annotation(name="Paul")
    ]


def test_an_adjudicator_is_read_from_the_final_block():
    # The adjudicator's answers are the only True here. Read from the top level
    # of their payload they would be invisible, and the person brought in to
    # settle the split would be scored as voting no to everything.
    labels, _, _, _ = G.resolve_labels(
        [
            _annotation(),
            _annotation(role="reannotator", name="Anita"),
            _adjudication(rigor_present=True, rigor_appropriate=True),
        ]
    )
    assert labels["rigor_present"] is True


def test_an_adjudicator_who_threw_the_moment_out_abstains():
    # A null final is "this does not belong in the benchmark", not five noes;
    # counting it would drag every situation majority towards False.
    labels, agreement, _, _ = G.resolve_labels(
        [
            _annotation(scaffolding_appropriate=True),
            _annotation(role="reannotator", name="Anita", scaffolding_appropriate=True),
            _adjudication(threw_out=True),
        ]
    )
    assert labels["scaffolding_appropriate"] is True
    assert agreement["scaffolding_appropriate"] is True


def test_a_half_filled_payload_abstains():
    annotation = _annotation(rigor_present=True)
    annotation["payload"]["action"] = None
    assert G.answered_payload(annotation) is None


# ===========================================================================
# Situation by majority, action by union
# ===========================================================================


@pytest.mark.parametrize("field", G.SITUATION_FIELDS)
def test_a_tie_with_no_adjudicator_resolves_true(field):
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(**{field: True}),
            _annotation(role="reannotator", name="Anita", **{field: False}),
        ]
    )
    assert labels[field] is True
    assert how["situation_ties"][field] == G.TIE_DEFAULT_TRUE


@pytest.mark.parametrize("field", G.SITUATION_FIELDS)
@pytest.mark.parametrize("call", [True, False])
def test_the_adjudicator_breaks_a_tied_situation_vote(field, call):
    # The two passes below agreed and the two adjudicators agreed on the
    # opposite, so the four votes split two-two. The adjudicators were brought
    # in for exactly this, and their call settles it in either direction --
    # including overturning a label the two passes below both marked.
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(**{field: not call}),
            _annotation(role="reannotator", name="Anita", **{field: not call}),
            _adjudication(name="Kelly", **{field: call}),
            _adjudication(name="Dana", **{field: call}),
        ]
    )
    assert labels[field] is call
    assert how["situation_ties"][field] == G.TIE_ADJUDICATOR


@pytest.mark.parametrize("field", G.SITUATION_FIELDS)
def test_adjudicators_who_split_fall_back_to_true(field):
    # Two-two overall and one-one among the adjudicators: nobody carries the
    # vote, so the default stands.
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(**{field: True}),
            _annotation(role="reannotator", name="Anita", **{field: False}),
            _adjudication(name="Kelly", **{field: True}),
            _adjudication(name="Dana", **{field: False}),
        ]
    )
    assert labels[field] is True
    assert how["situation_ties"][field] == G.TIE_DEFAULT_TRUE


@pytest.mark.parametrize("field", G.SITUATION_FIELDS)
def test_a_lone_adjudicator_never_breaks_a_tie(field):
    # Three votes cannot tie, so the tie-break is unreachable with a single
    # adjudicator -- their vote counts, it just counts as one of three.
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(**{field: True}),
            _annotation(role="reannotator", name="Anita", **{field: False}),
            _adjudication(name="Kelly", **{field: False}),
        ]
    )
    assert labels[field] is False
    assert how["situation_ties"] == {}


@pytest.mark.parametrize("field", G.ACTION_FIELDS)
def test_the_adjudicators_decide_the_action_alone(field):
    # Both passes below saw the move and both adjudicators, ruling on the same
    # excerpt afterwards, say it is not there. Action is theirs to settle, so
    # the two votes below do not dilute it.
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(**_action(field, True)),
            _annotation(role="reannotator", name="Anita", **_action(field, True)),
            _adjudication(name="Kelly"),
            _adjudication(name="Dana"),
        ]
    )
    assert labels[_resolved(field)] is False
    assert how["action_decided_by"] == "adjudicator"


@pytest.mark.parametrize("field", G.ACTION_FIELDS)
def test_a_lone_adjudicator_decides_the_action_alone(field):
    labels, _, _, _ = G.resolve_labels(
        [
            _annotation(**_action(field, True)),
            _annotation(role="reannotator", name="Anita", **_action(field, True)),
            _adjudication(name="Kelly"),
        ]
    )
    assert labels[_resolved(field)] is False


@pytest.mark.parametrize("field", G.ACTION_FIELDS)
def test_adjudicators_who_split_on_the_action_union(field):
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(),
            _annotation(role="reannotator", name="Anita"),
            _adjudication(name="Kelly", **_action(field, True)),
            _adjudication(name="Dana"),
        ]
    )
    resolved = _resolved(field)
    assert labels[resolved] is True
    assert how["action_ties"][resolved] == G.TIE_ADJUDICATOR_UNION


@pytest.mark.parametrize("field", G.ACTION_FIELDS)
def test_annotators_who_split_on_the_action_union(field):
    labels, _, _, how = G.resolve_labels(
        [
            _annotation(**_action(field, True)),
            _annotation(role="reannotator", name="Anita"),
        ]
    )
    resolved = _resolved(field)
    assert labels[resolved] is True
    assert how["action_ties"][resolved] == G.TIE_UNION


def test_an_adjudicator_can_add_an_action_the_passes_below_both_missed():
    labels, _, _, _ = G.resolve_labels(
        [
            _annotation(),
            _annotation(role="reannotator", name="Anita"),
            _adjudication(name="Kelly", rigor_present=True),
        ]
    )
    assert labels["rigor_present"] is True


def test_declared_over_scaffolding_follows_the_action_rule():
    # The selector declared it and the adjudicator did not. Over-scaffolding is
    # an action field, so the adjudicator's read is the one that ships.
    labels, _, inferred, _ = G.resolve_labels(
        [
            _annotation(
                scaffolding_appropriate=True,
                scaffolding_present=True,
                scaffolding_amount="over_scaffolding",
            ),
            _annotation(role="reannotator", name="Anita", scaffolding_appropriate=True),
            _adjudication(
                name="Kelly", scaffolding_appropriate=True, scaffolding_present=True
            ),
        ]
    )
    assert labels["over_scaffolding_present"] is False
    assert inferred is False


def test_an_adjudicator_can_declare_over_scaffolding_alone():
    labels, _, inferred, _ = G.resolve_labels(
        [
            _annotation(scaffolding_appropriate=True, scaffolding_present=True),
            _annotation(
                role="reannotator",
                name="Anita",
                scaffolding_appropriate=True,
                scaffolding_present=True,
            ),
            _adjudication(
                name="Kelly",
                scaffolding_appropriate=True,
                scaffolding_present=True,
                scaffolding_amount="over_scaffolding",
            ),
        ]
    )
    assert labels["over_scaffolding_present"] is True
    assert inferred is False


def test_the_inference_reads_the_resolved_situation_after_the_vote():
    # Two of three said scaffolding was not called for, so the resolved
    # situation is "not appropriate"; the adjudicator settles that it was
    # nonetheless delivered, and the rule fires on the two resolved labels.
    labels, _, inferred, _ = G.resolve_labels(
        [
            _annotation(scaffolding_appropriate=True, scaffolding_present=True),
            _annotation(role="reannotator", name="Anita"),
            _adjudication(name="Kelly", scaffolding_present=True),
        ]
    )
    assert labels["scaffolding_appropriate"] is False
    assert labels["scaffolding_present"] is True
    assert labels["over_scaffolding_present"] is True
    assert inferred is True


# ===========================================================================
# Boundaries: the strictest adjudicator draft wins
# ===========================================================================


def _draft(start, end, cut):
    return {
        "start_turn": start,
        "end_turn": end,
        "cut_turn": cut,
        "start_index": start + 2,
        "end_index": end + 4,
        "cut_index": cut + 3,
        "dialogue_turns": end - start,
    }


def test_the_adjudicator_draft_that_ends_earliest_wins():
    annotations = [
        _annotation(),
        _annotation(role="reannotator", name="Anita"),
        _adjudication(name="Kelly", boundaries=_draft(10, 22, 15)),
        _adjudication(name="Dana", boundaries=_draft(11, 18, 15)),
    ]
    boundaries, original, source = G.effective_boundaries(_moment(), annotations)
    assert source == "adjudicator"
    assert (boundaries["start_turn"], boundaries["end_turn"]) == (11, 18)
    assert original["end_turn"] == 20


def test_a_tie_on_the_end_takes_the_later_start():
    annotations = [
        _annotation(role="reannotator", name="Anita"),
        _adjudication(name="Kelly", boundaries=_draft(10, 18, 15)),
        _adjudication(name="Dana", boundaries=_draft(13, 18, 15)),
    ]
    boundaries, _, _ = G.effective_boundaries(_moment(), annotations)
    assert boundaries["start_turn"] == 13


def test_the_winning_draft_is_taken_whole_not_stitched():
    # Crossing spans: Kelly starts earlier and ends earlier, Dana the reverse.
    # Per-field tightening would ship 13-18, a span neither of them drew.
    annotations = [
        _annotation(role="reannotator", name="Anita"),
        _adjudication(name="Kelly", boundaries=_draft(10, 18, 15)),
        _adjudication(name="Dana", boundaries=_draft(13, 25, 16)),
    ]
    boundaries, _, _ = G.effective_boundaries(_moment(), annotations)
    assert (boundaries["start_turn"], boundaries["end_turn"], boundaries["cut_turn"]) == (
        10,
        18,
        15,
    )


def test_the_adjudicator_draft_overrides_the_second_pass_redraw():
    annotations = [
        _annotation(
            role="reannotator",
            name="Anita",
            meta={"new_start_turn": 11, "new_end_turn": 19},
        ),
        _adjudication(name="Kelly", boundaries=_draft(12, 17, 15)),
    ]
    boundaries, _, source = G.effective_boundaries(_moment(), annotations)
    assert source == "adjudicator"
    assert (boundaries["start_turn"], boundaries["end_turn"]) == (12, 17)


def test_an_adjudicator_who_drew_nothing_leaves_the_second_pass_standing():
    annotations = [
        _annotation(role="reannotator", name="Anita", meta={"new_end_turn": 19}),
        _adjudication(name="Kelly"),
    ]
    boundaries, _, source = G.effective_boundaries(_moment(), annotations)
    assert source == "reannotator"
    assert boundaries["end_turn"] == 19


def test_a_thrown_out_adjudication_draws_no_boundaries():
    assert G.adjudicator_draft(_adjudication(threw_out=True)) is None


# ---------------------------------------------------------------------------
# boundaries_source names whose values ship, not who last looked at them
# ---------------------------------------------------------------------------


def test_an_adjudicator_who_redraws_the_original_span_is_not_the_source():
    # Most adjudicators' drafts reproduce the span they were handed. Crediting
    # the draft rather than the values would report 123 of this export's 155
    # adjudicated moments as adjudicator boundaries that are the selector's own.
    annotations = [
        _annotation(role="reannotator", name="Anita"),
        _adjudication(name="Kelly", boundaries=_draft(10, 20, 15)),
    ]
    boundaries, original, source = G.effective_boundaries(_moment(), annotations)
    assert source == "selector"
    assert original is None
    assert (boundaries["start_turn"], boundaries["end_turn"]) == (10, 20)


def test_an_adjudicator_who_endorses_the_second_pass_redraw_is_not_the_source():
    # The draft has to match the redraw on every boundary field, indices
    # included -- an adjudicator who agrees about the turns but hands back
    # different indices has drawn their own span.
    redraw = _draft(11, 19, 15)
    annotations = [
        _annotation(
            role="reannotator",
            name="Anita",
            meta={f"new_{field}": value for field, value in redraw.items()},
        ),
        _adjudication(name="Kelly", boundaries=redraw),
    ]
    boundaries, original, source = G.effective_boundaries(_moment(), annotations)
    assert source == "reannotator"
    assert (boundaries["start_turn"], boundaries["end_turn"]) == (11, 19)
    assert original["end_turn"] == 20


def test_an_adjudicator_who_reverts_the_second_pass_leaves_the_selector_standing():
    # The adjudicator moved something -- back to where it started. What ships is
    # the selector's span, so that is what the record says.
    annotations = [
        _annotation(
            role="reannotator",
            name="Anita",
            meta={"new_start_turn": 11, "new_end_turn": 19},
        ),
        _adjudication(name="Kelly", boundaries=_draft(10, 20, 15)),
    ]
    boundaries, original, source = G.effective_boundaries(_moment(), annotations)
    assert source == "selector"
    assert original is None
    assert (boundaries["start_turn"], boundaries["end_turn"]) == (10, 20)


def test_the_selector_source_and_an_unredrawn_record_are_the_same_statement(tmp_path):
    # The two fields cannot contradict each other: a record that names a later
    # pass as its source is exactly a record whose boundaries moved.
    rows = [
        _row("t1", "m1", _both()),
        _row(
            "t2",
            "m2",
            _both() + [_adjudication(name="Kelly", boundaries=_draft(11, 18, 15))],
        ),
        _row(
            "t3",
            "m3",
            [
                _annotation(),
                _annotation(role="reannotator", name="Anita", meta={"new_end_turn": 19}),
            ],
        ),
    ]
    out, _ = G.build(
        *_fixture(tmp_path, rows, {"t1": "iterate", "t2": "iterate", "t3": "iterate"})
    )
    records = out["iteration"]
    assert [r["boundaries_source"] for r in records] == [
        "selector",
        "adjudicator",
        "reannotator",
    ]
    for record in records:
        assert (record["boundaries_source"] == "selector") is (
            not record["boundaries_redrawn"]
        )


# ===========================================================================
# build: the new filters
# ===========================================================================


def test_retracted_moments_are_dropped(tmp_path):
    rows = [
        _row("t1", "m1", _both(), status="retracted"),
        _row("t1", "m2", _both()),
    ]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    assert [r["moment_id"] for r in out["iteration"]] == ["m2"]
    assert dropped["retracted"] == 1


def test_a_moment_an_adjudicator_threw_out_is_dropped(tmp_path):
    rows = [
        _row("t1", "m1", _both() + [_adjudication(threw_out=True)]),
        _row("t1", "m2", _both()),
    ]
    args = _fixture(tmp_path, rows, {"t1": "iterate"})
    out, dropped = G.build(*args)
    assert [r["moment_id"] for r in out["iteration"]] == ["m2"]
    assert dropped["thrown_out"] == 1

    kept, _ = G.build(*args, keep_thrown_out=True)
    assert {r["moment_id"] for r in kept["iteration"]} == {"m1", "m2"}


def test_cut_point_redrawn_moments_are_dropped(tmp_path):
    rows = [
        _row("t1", "m1", _both(meta={"redrew_cut_point": True, "new_cut_turn": 17})),
        _row("t1", "m2", _both()),
    ]
    args = _fixture(tmp_path, rows, {"t1": "iterate"})
    out, dropped = G.build(*args)
    assert [r["moment_id"] for r in out["iteration"]] == ["m2"]
    assert dropped["cut_point_redrawn"] == 1

    kept, _ = G.build(*args, keep_cut_point_redrawn=True)
    assert {r["moment_id"] for r in kept["iteration"]} == {"m1", "m2"}


def test_a_resave_does_not_satisfy_the_doubly_annotated_requirement(tmp_path):
    second = _annotation(rigor_present=True)
    second["revision"] = 2
    rows = [_row("t1", "m1", [_annotation(), second])]
    out, dropped = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    assert out["iteration"] == []
    assert dropped["resaves_collapsed"] == 1
    assert dropped["not_doubly_annotated"] == 1


def test_the_record_reports_how_it_was_resolved(tmp_path):
    rows = [
        _row(
            "t1",
            "m1",
            _both(scaffolding_appropriate=True)
            + [_adjudication(name="Kelly", boundaries=_draft(11, 18, 15))],
        )
    ]
    out, _ = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    record = out["iteration"][0]

    assert record["boundaries_source"] == "adjudicator"
    assert record["n_votes"] == 3
    assert record["cut_point_redrawn"] is False
    assert record["situation_ties"] == {}
    assert record["end_turn"] == 18


# ===========================================================================
# The five-way situation split
# ===========================================================================


def _situation(*pairs):
    """Votes as (scaffolding_appropriate, rigor_appropriate) pairs."""
    return [
        (None, {"scaffolding_appropriate": s, "rigor_appropriate": r})
        for s, r in pairs
    ]


def test_scaffold_only_needs_everyone_on_one_side_and_nobody_on_the_other():
    assert G.situation_type(_situation((True, False), (True, False))) == "scaffold_only"


def test_one_annotator_calling_for_both_drops_it_to_a_majority():
    # Unanimous for scaffolding, but somebody also called for rigor, so the
    # strict row no longer holds.
    assert G.situation_type(_situation((True, False), (True, True))) == "scaffold_maj"


def test_rigor_only_is_the_mirror_of_scaffold_only():
    assert G.situation_type(_situation((False, True), (False, True))) == "rigor_only"


def test_equal_counts_are_an_even_split():
    assert G.situation_type(_situation((True, False), (False, True))) == "fifty_fifty"


def test_a_moment_calling_for_neither_is_not_a_side_winning():
    # Arithmetically an even split at 0-0. It must not be reported as one side
    # carrying the moment; the build's report counts these separately.
    assert G.situation_type(_situation((False, False), (False, False))) == "fifty_fifty"


def test_no_votes_has_no_situation_type():
    assert G.situation_type([]) is None


def test_situation_type_counts_the_adjudicator_as_a_vote():
    votes = G.votes(
        [
            _annotation(scaffolding_appropriate=True),
            _annotation(role="reannotator", name="Anita", scaffolding_appropriate=True),
            _adjudication(name="Kelly", rigor_appropriate=True),
        ]
    )
    assert G.situation_type(votes) == "scaffold_maj"


def test_situation_type_reaches_the_record(tmp_path):
    rows = [
        _row(
            "t1",
            "m1",
            [
                _annotation(scaffolding_appropriate=True),
                _annotation(
                    role="reannotator", name="Anita", scaffolding_appropriate=True
                ),
            ],
        )
    ]
    out, _ = G.build(*_fixture(tmp_path, rows, {"t1": "iterate"}))
    assert out["iteration"][0]["situation_type"] == "scaffold_only"
