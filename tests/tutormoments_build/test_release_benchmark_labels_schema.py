"""The approved benchmark labels schema (``benchmark/dev.jsonl``, ``test.jsonl``).

All rows are synthetic. The rejections pin the rules the writer relies on: the
release split names, annotators as labels each paired with their role, fields
dropped from the working ground truth, and the inclusion rule of two votes.
"""

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from tutormoments_build.release import build as rb
from tutormoments_build.release import checks

REPO = Path(__file__).resolve().parents[2]
TID = "00000000-0000-5000-8000-000000000001"
MID = "00000000-0000-5000-8000-000000000002"


@pytest.fixture(scope="module")
def schema():
    return json.loads(rb.packaged_schemas()["benchmark_labels.schema.json"])


@pytest.fixture(scope="module")
def validator(schema):
    return checks.load_validator(schema)


def label_row():
    return {
        "moment_id": MID,
        "source": "tutoring_provider_c",
        "transcript_id": TID,
        "split": "dev",
        "start_turn": 10,
        "end_turn": 18,
        "cut_turn": 14,
        "boundaries_source": "selector",
        "original_boundaries": None,
        "situation_type": "rigor_maj",
        "situation_votes": {"scaffolding": 1, "rigor": 3},
        "labels": {
            "scaffolding_present": False,
            "rigor_present": True,
            "over_scaffolding_present": False,
        },
        "action_ties": [],
        "n_votes": 4,
        "annotators": [
            {"annotator": "A04", "role": "selector"},
            {"annotator": "A11", "role": "reannotator"},
            {"annotator": "A05", "role": "adjudicator"},
            {"annotator": "A08", "role": "adjudicator"},
        ],
    }


def mutated(row, change):
    row = copy.deepcopy(row)
    change(row)
    return row


def test_accepts_dev_row(validator):
    assert list(validator.iter_errors(label_row())) == []


def test_accepts_moved_boundaries_and_a_tie(validator):
    row = label_row()
    row.update(
        split="test",
        boundaries_source="adjudicator",
        original_boundaries={"start_turn": 10, "end_turn": 16, "cut_turn": 14},
        situation_type="fifty_fifty",
        situation_votes={"scaffolding": 2, "rigor": 2},
        action_ties=["scaffolding_present", "over_scaffolding_present"],
    )
    assert list(validator.iter_errors(row)) == []


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda r: r.update(split="iteration"), id="working split name"),
        pytest.param(
            lambda r: r.update(annotator_roles=["selector"] * 4),
            id="parallel roles array",
        ),
        pytest.param(
            lambda r: r.update(annotators=["A04", "A11", "A05", "A08"]),
            id="annotators without roles",
        ),
        pytest.param(
            lambda r: r["annotators"][0].pop("role"), id="annotator entry without role"
        ),
        pytest.param(
            lambda r: r["annotators"][0].update(annotator=MID), id="raw annotator id"
        ),
        pytest.param(
            lambda r: r["annotators"][0].update(annotator_name="Pat"),
            id="annotator name",
        ),
        pytest.param(
            lambda r: r.update(moment_status="adjudicated"), id="moment_status"
        ),
        pytest.param(
            lambda r: r.update(moment_created_at="2026-10-01T12:00:00Z"),
            id="moment_created_at",
        ),
        pytest.param(lambda r: r.pop("original_boundaries"), id="boundaries omitted"),
        pytest.param(lambda r: r.update(n_votes=1), id="one vote"),
        pytest.param(
            lambda r: r.update(action_ties=["rigor_present", "rigor_present"]),
            id="repeated tie",
        ),
        pytest.param(
            lambda r: r.update(action_ties=["rigor_appropriate"]), id="unknown tie"
        ),
        pytest.param(lambda r: r["labels"].update(rigor_present=None), id="null label"),
        pytest.param(
            lambda r: r.update(situation_type="scaffolding"), id="unknown situation"
        ),
        pytest.param(
            lambda r: r.update(boundaries_source="consensus"),
            id="unknown boundaries source",
        ),
    ],
)
def test_rejects(validator, change):
    assert list(validator.iter_errors(mutated(label_row(), change)))


def test_files_the_description_names_exist(schema):
    """The description tells researchers how to reproduce the files; keep it true."""
    for path in (
        "tutormoments_build/v2/build_ground_truth.py",
        "tutormoments_build/v2/splits.json",
    ):
        assert f"`{path}`" in schema["description"]
        assert (REPO / path).is_file()
