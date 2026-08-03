"""moments.schema.json is load-bearing, not documentation.

write_release jsonschema-validates every record it publishes, and these
conformance tests validate the shipped fixtures and the local release so a
schema/record drift can never ship silently again.
"""

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from tutorsim.moments import Moment
from tutorsim_build.moments_build import write_release

REPO = Path(__file__).resolve().parents[2]
MINI_RELEASE = REPO / "tests/tutorsim/fixtures/mini_release"
LOCAL_RELEASE = REPO / "data/balanced_520_release"


def _shipped_schema() -> dict:
    from importlib.resources import files

    return json.loads(
        (files("tutorsim_build") / "moments.schema.json").read_text(encoding="utf-8")
    )


def _validate_records(jsonl_path: Path):
    schema = _shipped_schema()
    validator = jsonschema.Draft202012Validator(schema)
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            errors = sorted(validator.iter_errors(rec), key=str)
            assert not errors, (
                f"record {i} ({rec.get('id')}) violates moments.schema.json: "
                f"{errors[0].message} at {'/'.join(str(p) for p in errors[0].absolute_path)}"
            )


def _full_moment(mid="t:c__hum_1_2"):
    return Moment(
        id=mid,
        context=[{"turn_number": 1, "role": "tutor", "text": "Q"}],
        dimension="rigor",
        student={
            "mode": "oracle",
            "reference": "",
            "context": "",
            "trait": {
                "persona": "The student hedges.",
                "trait_mode": "joined-3",
                "generator_model": "claude-opus-4-6",
                "generated_at": "2026-06-18T00:00:00",
            },
        },
        rubric={"gold": "rigor", "hint": ""},
        provenance={
            "conv_id": "c",
            "cut_turn": 1,
            "turn_start": 1,
            "turn_end": 2,
            "moment_id": None,
            "annotator_id": "a-1",
            "chosen_cut_turn": 1,
            "cut_votes": {"1": 1},
            "cluster_size": 1,
        },
    )


def test_write_release_output_conforms_to_shipped_schema(tmp_path):
    """emit records -> validate against the shipped schema (keeps it honest)."""
    write_release([_full_moment()], tmp_path, set_name="t", created="2026-07-03")
    _validate_records(tmp_path / "moments.jsonl")


def test_write_release_refuses_nonconforming_record(tmp_path):
    """A record violating the schema (provenance missing required fields)
    cannot be released, independent of the trait check."""
    m = _full_moment()
    m.provenance = {
        "conv_id": "c",
        "cut_turn": 1,
        "cut_votes": {},
    }  # missing 6 required fields
    with pytest.raises(ValueError, match="schema"):
        write_release([m], tmp_path, set_name="t", created="2026-07-03")


def test_mini_release_fixture_conforms():
    """The runtime test fixture must satisfy the schema it claims to model."""
    _validate_records(MINI_RELEASE / "moments.jsonl")


@pytest.mark.skipif(
    not (LOCAL_RELEASE / "moments.jsonl").exists(),
    reason="local balanced_520 release not present",
)
def test_local_balanced_520_release_conforms():
    """The real release dir (when present locally) satisfies the shipped schema."""
    _validate_records(LOCAL_RELEASE / "moments.jsonl")
