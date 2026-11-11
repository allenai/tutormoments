"""Tests for the v2 ground-truth build (the annotation_stats.ipynb rules)."""

import json

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
        },
        "action": {
            "scaffolding_present": scaffolding_present,
            "rigor_present": rigor_present,
            "scaffolding_amount": scaffolding_amount,
        },
    }
    if meta is not None:
        payload["meta"] = meta
    return payload


def _annotation(role="selector", name="Grace", revision=1, **kw):
    return {
        "annotator_id": f"id-{name.lower()}",
        "annotator_name": name,
        "role": role,
        "revision": revision,
        "created_at": f"2026-09-0{revision}T00:00:00Z",
        "payload": _payload(**kw),
    }


def _adjudicator(name, *, final_turns=None, meta=None, **kw):
    """An adjudicator whose answers sit in payload["final"]."""
    annotation = {
        "annotator_id": f"id-{name.lower()}",
        "annotator_name": name,
        "role": "adjudicator",
        "revision": 1,
        "created_at": "2026-09-09T00:00:00Z",
        "payload": {"final": _payload(**kw), "meta": meta or {}},
    }
    for field, value in (final_turns or {}).items():
        annotation["payload"][f"final_{field}"] = value
    return annotation


def _moment(moment_id="m1", **overrides):
    moment = {
        "moment_id": moment_id,
        "start_turn": 10,
        "end_turn": 20,
        "cut_turn": 15,
        "status": "reannotated",
        "created_at": "2026-08-03T18:28:56Z",
    }
    moment.update(overrides)
    return moment


def _votes(*annotations):
    return G.votes(list(annotations))


# ---------------------------------------------------------------------------
# situation type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answers,expected",
    [
        ([(True, False), (True, False)], "scaffold_only"),
        ([(False, True), (False, True)], "rigor_only"),
        # One annotator ticking both drops a unanimous moment to a majority.
        ([(True, False), (True, True)], "scaffold_maj"),
        ([(False, True), (True, True), (False, True)], "rigor_maj"),
        ([(True, False), (False, True)], "fifty_fifty"),
        ([(True, True), (True, True)], "fifty_fifty"),
    ],
)
def test_situation_type_counts_each_checkbox_as_a_vote(answers, expected):
    annotations = [
        _annotation(name=f"a{i}", scaffolding_appropriate=s, rigor_appropriate=r)
        for i, (s, r) in enumerate(answers)
    ]
    assert G.situation_type(G.votes(annotations)) == expected


def test_situation_type_is_none_without_votes():
    assert G.situation_type([]) is None


# ---------------------------------------------------------------------------
# action labels
# ---------------------------------------------------------------------------


def test_action_tie_resolves_true_and_is_recorded():
    labels, ties = G.resolve_labels(
        _votes(
            _annotation(scaffolding_present=True),
            _annotation(role="reannotator", name="Bob"),
        )
    )
    assert labels["scaffolding_present"] is True
    assert ties == ["scaffolding_present"]


def test_action_majority_overrides_a_lone_true():
    labels, ties = G.resolve_labels(
        _votes(
            _annotation(rigor_present=True),
            _annotation(role="reannotator", name="Bob"),
            _adjudicator("Ben"),
        )
    )
    assert labels["rigor_present"] is False
    assert ties == []


def test_adjudicator_counts_as_one_vote_not_the_decider():
    # Two passes say scaffolded; the adjudicator alone does not overturn them.
    labels, _ = G.resolve_labels(
        _votes(
            _annotation(scaffolding_present=True),
            _annotation(role="reannotator", name="Bob", scaffolding_present=True),
            _adjudicator("Ben", scaffolding_present=False),
        )
    )
    assert labels["scaffolding_present"] is True


def test_over_scaffolding_is_declared_only_never_inferred():
    # Scaffolding present but not appropriate used to infer over-scaffolding.
    labels, _ = G.resolve_labels(
        _votes(
            _annotation(scaffolding_present=True),
            _annotation(role="reannotator", name="Bob", scaffolding_present=True),
        )
    )
    assert labels["over_scaffolding_present"] is False


def test_over_scaffolding_majority_with_tie_true():
    over = {"scaffolding_amount": G.OVER_SCAFFOLDING}
    labels, ties = G.resolve_labels(
        _votes(_annotation(**over), _annotation(role="reannotator", name="Bob"))
    )
    assert labels["over_scaffolding_present"] is True
    assert "over_scaffolding_present" in ties


def test_thrown_out_and_blank_payloads_abstain():
    blank = _annotation(role="reannotator", name="Bob")
    blank["payload"]["action"] = None
    adjudicator_no_final = _adjudicator("Ben")
    adjudicator_no_final["payload"]["final"] = None
    assert len(_votes(_annotation(), blank, adjudicator_no_final)) == 1


def test_latest_annotations_drops_staff_and_keeps_highest_revision():
    kept = G.latest_annotations(
        [
            _annotation(name="Grace", revision=1),
            _annotation(name="Grace", revision=2, rigor_present=True),
            _annotation(role="reannotator", name="Frank"),
        ],
        frozenset({"id-frank"}),
    )
    assert len(kept) == 1
    assert kept[0]["revision"] == 2


# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------


def test_boundaries_default_to_the_selector():
    moment = _moment()
    annotations = [_annotation(), _annotation(role="reannotator", name="Bob")]
    boundaries = G.adjudicated_boundaries(moment, annotations)
    assert boundaries == {"start_turn": 10, "end_turn": 20, "cut_turn": 15}
    assert G.boundaries_source(moment, annotations, boundaries) == "selector"


def test_reannotator_redraw_stands_without_adjudication():
    moment = _moment()
    annotations = [
        _annotation(),
        _annotation(role="reannotator", name="Bob", meta={"new_end_turn": 18}),
    ]
    boundaries = G.adjudicated_boundaries(moment, annotations)
    assert boundaries["end_turn"] == 18
    assert G.boundaries_source(moment, annotations, boundaries) == "reannotator"


def test_adjudicator_blank_keeps_what_they_were_shown():
    moment = _moment()
    annotations = [
        _annotation(),
        _annotation(role="reannotator", name="Bob", meta={"new_start_turn": 12}),
        _adjudicator("Ben", final_turns={"end_turn": 17}),
    ]
    boundaries = G.adjudicated_boundaries(moment, annotations)
    assert boundaries == {"start_turn": 12, "end_turn": 17, "cut_turn": 15}
    assert G.boundaries_source(moment, annotations, boundaries) == "adjudicator"


def test_disagreeing_adjudicators_take_the_earliest_end_and_the_voted_cut():
    moment = _moment()
    annotations = [
        _annotation(),
        _annotation(role="reannotator", name="Bob"),
        _adjudicator("Ben", final_turns={"end_turn": 18, "cut_turn": 14}),
        _adjudicator("Cy", final_turns={"end_turn": 20, "cut_turn": 15}),
    ]
    # Ben's span ends first, but selector, second pass and Cy all cut at 15.
    assert G.adjudicated_boundaries(moment, annotations) == {
        "start_turn": 10,
        "end_turn": 18,
        "cut_turn": 15,
    }


def test_voted_cut_outside_the_strictest_span_is_not_used():
    moment = _moment()
    annotations = [
        _annotation(),
        _annotation(role="reannotator", name="Bob"),
        _adjudicator(
            "Ben", final_turns={"start_turn": 16, "end_turn": 18, "cut_turn": 17}
        ),
        _adjudicator("Cy", final_turns={"end_turn": 20}),
    ]
    assert G.adjudicated_boundaries(moment, annotations)["cut_turn"] == 17


# ---------------------------------------------------------------------------
# build: filtering and uniqueness
# ---------------------------------------------------------------------------


def _write(tmp_path, rows, assignments=None):
    path = tmp_path / "tutoring_provider_x_annotations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "assignments": assignments or {"t1": {"split": "iterate"}},
            }
        )
    )
    return [str(path)], str(splits)


def _row(moment_id="m1", annotations=None, **moment_kw):
    return {
        "transcript_id": "t1",
        "moment": _moment(moment_id, **moment_kw),
        "annotations": annotations
        if annotations is not None
        else [_annotation(), _annotation(role="reannotator", name="Bob")],
    }


def _no_label_files(tmp_path):
    """Label map and staff list paths that don't exist."""
    return str(tmp_path / "missing.json"), str(tmp_path / "missing.json")


def test_build_filters_like_the_notebook(tmp_path):
    rows = [
        _row("keep"),
        _row("retracted", status="retracted"),
        _row(
            "thrown",
            annotations=[
                _annotation(),
                _annotation(role="reannotator", name="Bob"),
                _adjudicator("Ben", meta={"throw_out": True}),
            ],
        ),
        _row(
            "cut",
            annotations=[
                _annotation(),
                _annotation(
                    role="reannotator", name="Bob", meta={"redrew_cut_point": True}
                ),
            ],
        ),
        _row("single", annotations=[_annotation()]),
        {"transcript_id": "t1", "no_key_moments_record": {}},
    ]
    paths, splits = _write(tmp_path, rows)
    out, dropped = G.build(paths, splits, *_no_label_files(tmp_path))
    assert [r["moment_id"] for r in out["iteration"]] == ["keep"]
    assert out["iteration"][0]["source"] == "tutoring_provider_x"
    assert dropped == {
        "retracted": 1,
        "thrown_out": 1,
        "cut_point_redrawn": 1,
        "fewer_than_two_votes": 1,
        "no_key_moments_record": 1,
    }


def test_build_labels_and_excludes_annotators_by_id(tmp_path):
    staff = _annotation(role="reannotator", name="Frank")
    rows = [_row("m1", annotations=[*_row()["annotations"], staff])]
    paths, splits = _write(tmp_path, rows)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"id-grace": "A01", "id-bob": "A02"}))
    excluded = tmp_path / "excluded.json"
    excluded.write_text(json.dumps(["id-frank"]))

    out, dropped = G.build(paths, splits, str(labels), str(excluded))
    assert out["iteration"][0]["annotators"] == ["A01", "A02"]
    assert dropped["staff_annotations_removed"] == 1


def test_build_rejects_a_repeated_moment_id(tmp_path):
    paths, splits = _write(tmp_path, [_row("m1"), _row("m1")])
    with pytest.raises(ValueError, match="m1"):
        G.build(paths, splits, *_no_label_files(tmp_path))


def test_record_carries_one_situation_and_one_action_set(tmp_path):
    paths, splits = _write(tmp_path, [_row("m1")])
    out, _ = G.build(paths, splits, *_no_label_files(tmp_path))
    record = out["iteration"][0]
    assert record["situation_type"] in G.SITUATION_TYPES
    assert set(record["labels"]) == set(G.ACTION_FIELDS)
    assert "original_boundaries" not in record
