"""Moments writer: released transcripts, annotations and labels -> ``benchmark/moments.jsonl``.

Rows are synthetic, built from the transcript and annotation writers' fixtures. Builds
run every writer in order against a local store, so each output also passes the
packaged schemas and the name check.
"""

import copy
import json

import pytest

pytest.importorskip("jsonschema")

from tests.tutormoments_build.test_release_annotations import (
    ADJUDICATOR,
    CLEAN,
    LABELS,
    NOW,
    SELECTION_ROWS,
    U,
    adjudicator,
    raw_a,
    raw_c,
    reannotator,
    selector,
    store_with,
)
from tests.tutormoments_build.test_release_annotations import (
    INPUTS as ANNOTATION_INPUTS,
)
from tests.tutormoments_build.test_release_benchmark import ASSIGNMENTS
from tests.tutormoments_build.test_release_transcripts import (
    INPUTS as TRANSCRIPT_INPUTS,
)
from tests.tutormoments_build.test_release_transcripts import bucket_schemas, jsonl
from tutormoments_build.release import benchmark, moments
from tutormoments_build.release import build as rb
from tutormoments_build.release.annotations import write_annotations
from tutormoments_build.release.benchmark import write_benchmark_labels
from tutormoments_build.release.snapshot import create_snapshot
from tutormoments_build.release.transcripts import write_transcripts

WRITERS = (write_transcripts, write_annotations, write_benchmark_labels)


@pytest.fixture(autouse=True)
def splits(tmp_path, monkeypatch):
    """raw_a's transcript is in dev, raw_c's in test (the labels writer's fixture)."""
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"schema_version": "1.0", "assignments": ASSIGNMENTS}))
    monkeypatch.setattr(benchmark, "SPLITS", str(path))


def turn_a(n, role):
    return {
        "turn_number": n,
        "start_seconds": 10.0 * n,
        "end_seconds": 10.0 * n + 5,
        "role": role,
        "text": f"{role} line {n}",
    }


def event(n, before_turn, kind="screen_interaction"):
    return {
        "enrichment_number": n,
        "type": kind,
        "start_seconds": 10.0 * before_turn - 3,
        "end_seconds": 10.0 * before_turn - 2,
        "before_turn": before_turn,
        "label": "",
        "content": f"Event {n}.",
    }


# raw_a's transcript (U(1)): seven turns. The tool showed event 1 on its own row before
# turn 4 and event 2 folded into turn 4 (SELECTION_ROWS); event 2's before_turn is 5.
TURNS_A = [turn_a(n, "Tutor" if n % 2 else "Student") for n in range(1, 8)]
EVENTS_A = [event(1, 4), event(2, 5, "pause")]


def transcript_a():
    return {
        "transcript_id": U(1),
        "source": "tutoring_provider_a",
        "batch": "2025-10-16",
        "has_video": True,
        "turns": copy.deepcopy(TURNS_A),
        "session": {
            "session_id": U(20),
            "tutor_id": U(21),
            "student_id": U(22),
            "transcription_by": "LS",
            "primary_language": "en",
        },
        "eedi_content_link": [],
        "enrichments": copy.deepcopy(EVENTS_A),
    }


PROBLEMS_C = [
    {
        "slide_id": 7,
        "slide_position": 1,
        "learning_objective": {"id": 900, "title": "Adding money.", "year": 4},
        "problem_text": "Add the coins.",
    }
]


def transcript_c():
    return {
        "transcript_id": U(3),
        "source": "tutoring_provider_c",
        "batch": "2026-07-06",
        "has_video": False,
        "turns": [
            {"turn_number": n, "start_seconds": 4.0 * n, "role": role, "text": f"c{n}"}
            for n, role in zip(
                range(1, 8),
                [
                    "Tutor",
                    "Student",
                    "Tutor",
                    "Unattributed",
                    "Tutor",
                    "Student",
                    "Tutor",
                ],
                strict=True,
            )
        ],
        "session": {
            "session_id": U(30),
            "tutor_id": U(31),
            "student_id": U(32),
            "program_id": U(33),
        },
        "problems": copy.deepcopy(PROBLEMS_C),
    }


def full_store(tmp_path, a_rows=None, labels=LABELS):
    store = store_with(tmp_path, a_rows or [raw_a()], [raw_c()], labels)
    store.write_bytes(TRANSCRIPT_INPUTS[0], jsonl(transcript_a()))
    store.write_bytes(TRANSCRIPT_INPUTS[1], jsonl(transcript_c()))
    for name, schema in bucket_schemas().items():
        store.write_bytes(f"schemas/{name}.schema.json", json.dumps(schema).encode())
    return store


def build(store, writers=(*WRITERS, moments.write_moments)):
    create_snapshot(
        store, snapshot_id="s1", inputs=(*ANNOTATION_INPUTS, *TRANSCRIPT_INPUTS)
    )
    return rb.run_build(store, "s1", writers=writers, code=CLEAN, now=NOW)


def released(store, manifest):
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    lines = store.read_bytes(prefix + moments.PATH).splitlines()
    return {(row := json.loads(line))["moment_id"]: row for line in lines}


def view_rows(*sides):
    """SELECTION_ROWS as released, with the given in_situation per row."""
    return [
        {
            "turn_number": r["turn_number"],
            "enrichment_numbers": r["enrichment_numbers"],
            "in_situation": side,
        }
        for r, side in zip(SELECTION_ROWS, sides, strict=True)
    ]


def test_a_row_carries_the_session_boundaries_situation_and_view(tmp_path):
    store = full_store(tmp_path)
    manifest = build(store)
    row = released(store, manifest)[U(2)]

    assert {k: row[k] for k in row if k not in ("transcript", "annotator_view")} == {
        "moment_id": U(2),
        "source": "tutoring_provider_a",
        "transcript_id": U(1),
        "split": "dev",
        "start_turn": 3,
        "cut_turn": 4,
        "end_turn": 6,
        "boundaries_source": "reannotator",
        "situation_type": "scaffold_only",
        "situation_votes": {"scaffolding": 3, "rigor": 0},
        "problems": None,
    }
    # The whole session, every object unchanged, events before their before_turn.
    t = TURNS_A
    assert row["transcript"] == [
        {"turn": t[0]},
        {"turn": t[1]},
        {"turn": t[2]},
        {"enrichment": EVENTS_A[0]},
        {"turn": t[3]},
        {"enrichment": EVENTS_A[1]},
        *({"turn": turn} for turn in t[4:]),
    ]
    # The reannotator drew these boundaries, so theirs is the view; event 2, which
    # before_turn places after the cut, was shown before it.
    assert row["annotator_view"] == {
        "annotator": "A02",
        "role": "reannotator",
        "interface_version": "2.34.5",
        "rows": view_rows(True, True, True, False),
    }
    assert manifest["files"][moments.PATH]["schemas"] == [moments.SCHEMA]
    assert manifest["files"][moments.PATH]["rows"] == 2


def test_c_row_carries_its_problems_and_no_view(tmp_path):
    store = full_store(tmp_path)
    row = released(store, build(store))[U(4)]
    assert row["split"] == "test"
    assert row["problems"] == PROBLEMS_C
    assert row["annotator_view"] is None
    assert [item["turn"] for item in row["transcript"]] == transcript_c()["turns"]


def test_view_follows_a_cut_drawn_after_a_screen_event_row(tmp_path):
    # The reannotator cut after event 1's row; the tool converted that to turn 4,
    # the turn after it. The view keeps the cut where it was drawn.
    rows = copy.deepcopy(SELECTION_ROWS)
    for r in rows:
        r["is_cut"] = r["turn_number"] is None
    second = reannotator()
    second["transcript_selection"] = {"rows": rows}
    store = full_store(
        tmp_path, [raw_a(annotations=[selector(), second, adjudicator()])]
    )
    row = released(store, build(store))[U(2)]
    assert row["cut_turn"] == 4
    assert row["annotator_view"]["rows"] == view_rows(True, True, False, False)


def adjudicator_drawing(annotator_id, start, cut, end, cut_row_turn):
    entry = adjudicator()
    entry["annotator_id"] = annotator_id
    entry["payload"].update(
        final_start_turn=start, final_cut_turn=cut, final_end_turn=end
    )
    rows = copy.deepcopy(SELECTION_ROWS)
    for r in rows:
        r["is_cut"] = r["turn_number"] == cut_row_turn
    entry["transcript_selection"] = {"rows": rows}
    return entry


def test_a_majority_cut_is_placed_after_the_row_holding_cut_turn(tmp_path):
    # Two adjudicators disagree. The span that ends first (3-5) is used, with the cut
    # most annotators chose (4: selector, reannotator, second adjudicator), which is
    # not the first adjudicator's own (5). Their view is used with the cut after turn 4.
    first = adjudicator_drawing(ADJUDICATOR, 3, 5, 5, cut_row_turn=5)
    second = adjudicator_drawing(U(13), 3, 4, 6, cut_row_turn=4)
    second["created_at"] = "2026-10-02T12:00:00Z"
    store = full_store(
        tmp_path,
        [raw_a(annotations=[selector(), reannotator(), first, second])],
        {**LABELS, U(13): "A04"},
    )
    row = released(store, build(store))[U(2)]
    assert (row["start_turn"], row["cut_turn"], row["end_turn"]) == (3, 4, 5)
    assert row["boundaries_source"] == "adjudicator"
    assert row["annotator_view"]["annotator"] == "A03"
    assert row["annotator_view"]["rows"] == view_rows(True, True, True, False)


def test_a_view_naming_an_event_the_transcript_lacks_fails(tmp_path):
    rows = copy.deepcopy(SELECTION_ROWS)
    rows[1]["enrichment_numbers"] = [9]
    second = reannotator()
    second["transcript_selection"] = {"rows": rows}
    store = full_store(
        tmp_path, [raw_a(annotations=[selector(), second, adjudicator()])]
    )
    with pytest.raises(SystemExit, match="9"):
        build(store)


def test_moments_need_the_files_they_are_built_from(tmp_path):
    store = full_store(tmp_path)
    with pytest.raises(SystemExit, match="transcripts/"):
        build(
            store,
            writers=(write_annotations, write_benchmark_labels, moments.write_moments),
        )


def test_copied_provider_definitions_match_the_provider_schemas():
    schemas = {name: json.loads(data) for name, data in rb.packaged_schemas().items()}
    moments.check_copied_definitions(schemas)

    drifted = copy.deepcopy(schemas)
    drifted["tutoring_provider_a.schema.json"]["$defs"]["Turn"]["required"].remove(
        "end_seconds"
    )
    with pytest.raises(SystemExit, match="TurnProviderA"):
        moments.check_copied_definitions(drifted)


def test_moments_are_a_default_release_output():
    assert rb.default_writers()[-1] is moments.write_moments
