"""The approved annotations release schema: what it accepts and what it must reject.

All rows are synthetic. The rejections pin the de-identification and shape rules the
writer relies on: no names, labels instead of raw annotator IDs, no retracted moments,
and role-specific answer shapes.
"""

import copy
import json
from importlib import resources

import pytest

pytest.importorskip("jsonschema")

from tutormoments_build.release import build as rb
from tutormoments_build.release import checks

TID = "00000000-0000-5000-8000-000000000001"
MID = "00000000-0000-5000-8000-000000000002"


@pytest.fixture(scope="module")
def schema():
    return json.loads(rb.packaged_schemas()["annotations.schema.json"])


@pytest.fixture(scope="module")
def validator(schema):
    return checks.load_validator(schema)


def sar():
    return {
        "situation": {
            "scaffolding_appropriate": False,
            "rigor_appropriate": True,
            "why": "Three right in a row.",
        },
        "action": {
            "scaffolding_present": False,
            "scaffolding_amount": None,
            "scaffolding_strategies": [],
            "scaffolding_strategy_other": "",
            "over_scaffolding_reasons": [],
            "rigor_present": True,
            "rigor_amount": "strong",
            "rigor_strategies": ["justify_explain"],
            "rigor_strategy_other": "",
            "explanation": "Asks why the method works.",
        },
        "result": {
            "problem_success": "meaningful_success",
            "cognitive_engagement": "high_or_good",
            "explanation": "Explains it.",
        },
    }


def entry(annotator, role, payload):
    return {
        "annotator": annotator,
        "role": role,
        "created_at": "2026-10-01T12:00:00Z",
        "interface_version": "2.34.5",
        "schema_version": "2.7",
        "payload": payload,
    }


def moment_row():
    return {
        "transcript_id": TID,
        "source": "tutoring_provider_c",
        "purpose": "rigor_scaffolding",
        "batch": "v2",
        "moment": {
            "moment_id": MID,
            "start_turn": 10,
            "cut_turn": 14,
            "end_turn": 18,
            "status": "adjudicated",
            "selected_by": "A01",
            "created_at": "2026-10-01T11:00:00Z",
            "in_v1_rigor_region": None,
        },
        "v1_rigor_hinted": False,
        "v1_rigor_regions": [],
        "annotations": [
            entry("A01", "selector", {**sar(), "other_observations": ""}),
            entry(
                "A02",
                "reannotator",
                {
                    **sar(),
                    "other_observations": "",
                    "meta": {
                        "throw_out": False,
                        "throw_out_reason": "",
                        "changed_boundaries": False,
                        "new_start_turn": None,
                        "new_end_turn": None,
                        "redrew_cut_point": False,
                        "new_cut_turn": None,
                    },
                },
            ),
            entry(
                "A03",
                "adjudicator",
                {
                    "final": {**sar(), "other_observations": ""},
                    "decisions": dict.fromkeys(
                        [
                            "situation",
                            "action",
                            "result",
                            "boundaries",
                            "other_observations",
                        ],
                        "agreed",
                    ),
                    "final_start_turn": 10,
                    "final_cut_turn": 14,
                    "final_end_turn": 18,
                    "rationale": "Both agree.",
                    "reverted_throw_out": False,
                    "meta": {
                        "throw_out": False,
                        "throw_out_category": None,
                        "throw_out_reason": "",
                    },
                },
            ),
        ],
    }


def no_key_row():
    return {
        "transcript_id": TID,
        "source": "tutoring_provider_a",
        "purpose": "rigor_scaffolding",
        "batch": "v2",
        "moment": None,
        "v1_rigor_hinted": False,
        "v1_rigor_regions": [],
        "annotations": [
            entry(
                "A04",
                "selector",
                {"no_key_moments": True, "note": "Reading lesson, no math."},
            )
        ],
    }


def test_accepts_adjudicated_moment(validator):
    assert list(validator.iter_errors(moment_row())) == []


def test_accepts_no_key_moments_record(validator):
    assert list(validator.iter_errors(no_key_row())) == []


def test_accepts_reannotator_throw_out_with_no_answers(validator):
    row = moment_row()
    row["moment"]["status"] = "thrown_out"
    payload = row["annotations"][1]["payload"]
    payload.update(situation=None, action=None, result=None)
    payload["meta"].update(throw_out=True, throw_out_reason="Not math.")
    assert list(validator.iter_errors(row)) == []


def test_accepts_provider_a_extras(validator):
    row = moment_row()
    row["source"] = "tutoring_provider_a"
    row["v1_rigor_hinted"] = True
    row["v1_rigor_regions"] = [[12, 20]]
    row["moment"]["in_v1_rigor_region"] = True
    row["annotations"][0]["transcript_selection"] = {
        "rows": [
            {"turn_number": 12, "enrichment_numbers": [3], "in_situation": True},
            {"turn_number": None, "enrichment_numbers": [4], "in_situation": False},
        ]
    }
    assert list(validator.iter_errors(row)) == []


def test_rejects_the_dropped_flat_enrichment_list(validator):
    row = moment_row()
    row["source"] = "tutoring_provider_a"
    row["annotations"][0]["enrichment_numbers"] = [3, 4]
    assert list(validator.iter_errors(row)) != []


def test_rejects_the_tools_cut_flag(validator):
    row = moment_row()
    row["source"] = "tutoring_provider_a"
    row["annotations"][0]["transcript_selection"] = {
        "rows": [{"turn_number": 12, "enrichment_numbers": [], "is_cut": True}]
    }
    assert list(validator.iter_errors(row)) != []


def mutated(row, change):
    row = copy.deepcopy(row)
    change(row)
    return row


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda r: r["annotations"][0].update(annotator_name="Pat"),
            id="annotator name",
        ),
        pytest.param(
            lambda r: r["moment"].update(created_by_name="Pat"),
            id="moment creator name",
        ),
        pytest.param(
            lambda r: r["annotations"][0].update(annotator=MID), id="raw annotator id"
        ),
        pytest.param(
            lambda r: r["moment"].update(selected_by=MID), id="raw selector id"
        ),
        pytest.param(
            lambda r: r["moment"].update(status="retracted"), id="retracted moment"
        ),
        pytest.param(
            lambda r: r["annotations"][0].update(revision=2), id="revision field"
        ),
        pytest.param(lambda r: r["moment"].update(dialogue_turns=9), id="turn count"),
        pytest.param(lambda r: r.update(purpose="key_moments"), id="unknown purpose"),
        pytest.param(
            lambda r: r["annotations"][0].update(
                payload=r["annotations"][2]["payload"]
            ),
            id="selector with adjudicator answers",
        ),
        pytest.param(
            lambda r: r["annotations"][0]["payload"]["action"].update(
                rigor_amount="medium"
            ),
            id="unknown rigor amount",
        ),
    ],
)
def test_rejects_moment_row(validator, change):
    assert list(validator.iter_errors(mutated(moment_row(), change)))


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda r: r["annotations"].append(copy.deepcopy(r["annotations"][0])),
            id="two entries",
        ),
        pytest.param(
            lambda r: r["annotations"][0].update(role="reannotator"),
            id="not a selector",
        ),
        pytest.param(
            lambda r: r["annotations"][0]["payload"].update(no_key_moments=False),
            id="no_key_moments false",
        ),
    ],
)
def test_rejects_no_key_row(validator, change):
    assert list(validator.iter_errors(mutated(no_key_row(), change)))


def test_annotation_guide_ships_where_the_schema_points(schema):
    assert "`docs/annotation_guide.md`" in schema["description"]
    guide = (
        resources.files("tutormoments_build.release") / "docs" / "annotation_guide.md"
    )
    assert guide.is_file()
