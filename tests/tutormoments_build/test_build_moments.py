"""Hermetic tests for build_scenarios using fixture data.

Fixture layout:
  tests/tutormoments_build/fixtures/build_src/
    ground_truth/  conv-aaa.json  conv-bbb.json  conv-ccc.json  conv-ddd.json
    transcripts/   conv-aaa.json  conv-bbb.json  conv-ccc.json
    transcripts.jsonl  (normalized JSONL records; contains conv-ddd only)

conv-aaa: cluster (3,5), votes=[4,4,5] -> modal=4, STUDENT turn -> cut stays 4
conv-bbb: cluster (1,3), votes=[2,3]   -> modal=2 (tie->min), STUDENT turn -> cut stays 2
conv-ccc: cluster (2,4), votes=[3]     -> modal=3, TUTOR turn -> cut decrements to 2
conv-ddd: cluster (1,3), votes=[2]     -> JSONL-only conv; STUDENT turn -> cut stays 2
"""

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path("tests/tutormoments_build/fixtures/build_src")
GT_DIR = FIXTURE_DIR / "ground_truth"
TX_DIR = FIXTURE_DIR / "transcripts"
TX_JSONL = str(FIXTURE_DIR / "transcripts.jsonl")


def _ids_for(*conv_ts_te_pairs):
    """Build scenario id strings from (conv_id, ts, te) tuples."""
    return [f"{conv}__hum_{ts}_{te}" for conv, ts, te in conv_ts_te_pairs]


# ---------------------------------------------------------------------------
# Import guard — fails loudly if build_scenarios isn't implemented yet
# ---------------------------------------------------------------------------


def _import_build():
    from tutormoments_build.moments_build import build_moments as build_scenarios

    return build_scenarios


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildScenarios:
    """All tests are fully hermetic -- fixture dirs only, no real data."""

    def _build(self, ids, set_name="test"):
        build_scenarios = _import_build()
        return build_scenarios(
            set_name=set_name,
            ids=ids,
            ground_truth_dir=str(GT_DIR),
            transcripts_dir=str(TX_DIR),
        )

    def test_field_mapping_dimension_and_gold(self):
        """dimension and rubric.gold both come from situation_label_agg."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.dimension == "scaffolding"
        assert s.rubric["gold"] == "scaffolding"

    def test_field_mapping_hint_from_situation(self):
        """rubric.hint comes from the situation field."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        assert s.rubric["hint"] == "Student gave a vague guess about fractions"

    def test_field_mapping_id(self):
        """Scenario id = set_name:conv_id__hum_ts_te."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids, set_name="myset")
        assert scenarios[0].id == "myset:conv-aaa__hum_3_5"

    def test_context_is_prefix_turns(self):
        """context contains turns up to and including cut_turn as [{turn_number, role, text}]."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        # conv-aaa cut=4 (STUDENT at 4, stays). Prefix: turns 1-4
        assert len(s.context) == 4
        assert s.context[0] == {
            "turn_number": 1,
            "role": "tutor",
            "text": "What is half of 6?",
        }
        assert s.context[1] == {
            "turn_number": 2,
            "role": "student",
            "text": "I think it might be 3?",
        }
        assert s.context[2] == {
            "turn_number": 3,
            "role": "tutor",
            "text": "Can you explain how you got that?",
        }
        assert s.context[3] == {
            "turn_number": 4,
            "role": "student",
            "text": "I split it into two groups",
        }
        # Roles must be lowercased
        for turn in s.context:
            assert turn["role"] in ("tutor", "student")
        # turn_numbers must be real (from source transcript)
        assert [t["turn_number"] for t in s.context] == [1, 2, 3, 4]

    def test_student_reference_is_post_cut(self):
        """student.reference contains only turns AFTER cut_turn."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        # cut=4, post-cut turns are 5,6,7
        ref = s.student["reference"]
        assert "Turn 5." in ref
        assert "Turn 6." in ref
        assert "Turn 7." in ref
        # Pre-cut turns must NOT appear
        assert "Turn 4." not in ref
        assert "Turn 3." not in ref

    def test_student_context_field(self):
        """student.context comes from conversation context."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        s = scenarios[0]
        assert s.student["context"] == "Grade 5, Mathematics, fractions"

    def test_student_mode_is_oracle(self):
        """student.mode is always 'oracle'."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        assert scenarios[0].student["mode"] == "oracle"

    def test_provenance_fields(self):
        """provenance includes all expected fields."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        p = scenarios[0].provenance
        assert p["conv_id"] == "conv-aaa"
        assert p["cut_turn"] == 4  # final adjusted cut
        assert p["turn_start"] == 3
        assert p["turn_end"] == 5
        assert p["moment_id"] == "m-001"
        assert "annotator_id" in p
        assert p["chosen_cut_turn"] == 4  # pre-adjustment modal vote
        assert "cut_votes" in p
        assert "cluster_size" in p

    def test_cut_logic_modal_vote_with_tie_picks_smallest(self):
        """conv-bbb: votes=[2,3] -> tie -> smallest=2. STUDENT at 2 -> stays 2."""
        ids = _ids_for(("conv-bbb", 1, 3))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.provenance["chosen_cut_turn"] == 2
        assert s.provenance["cut_turn"] == 2

    def test_cut_logic_tutor_role_decrements(self):
        """conv-ccc: vote=3, turn 3 is TUTOR -> cut decrements to 2."""
        ids = _ids_for(("conv-ccc", 2, 4))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        s = scenarios[0]
        assert s.provenance["chosen_cut_turn"] == 3  # modal before adjustment
        assert s.provenance["cut_turn"] == 2  # after TUTOR decrement
        # context should be prefix up to cut=2 (turns 1,2)
        assert len(s.context) == 2

    def test_ids_order_is_preserved(self):
        """Output order matches the requested ids order, not file-system order."""
        # Request bbb before aaa intentionally
        ids = _ids_for(("conv-bbb", 1, 3), ("conv-aaa", 3, 5), ("conv-ccc", 2, 4))
        scenarios = self._build(ids)
        assert len(scenarios) == 3
        assert scenarios[0].provenance["conv_id"] == "conv-bbb"
        assert scenarios[1].provenance["conv_id"] == "conv-aaa"
        assert scenarios[2].provenance["conv_id"] == "conv-ccc"

    def test_unknown_ids_skipped(self):
        """IDs that don't match any ground-truth cluster are silently skipped."""
        ids = _ids_for(("conv-aaa", 3, 5), ("conv-zzz", 9, 11))
        scenarios = self._build(ids)
        assert len(scenarios) == 1
        assert scenarios[0].provenance["conv_id"] == "conv-aaa"

    def test_cut_votes_structure(self):
        """conv-aaa: votes=[4,4,5] -> modal=4, cut_votes={4:2, 5:1}."""
        ids = _ids_for(("conv-aaa", 3, 5))
        scenarios = self._build(ids)
        p = scenarios[0].provenance
        assert p["cluster_size"] == 3
        votes = p["cut_votes"]
        assert votes.get(4) == 2
        assert votes.get(5) == 1


class TestBuildScenariosJsonl:
    """Tests that JSONL transcripts are loaded and hydrated correctly."""

    def _build(self, ids, tutoring_provider_a_jsonl=None, set_name="test"):
        from tutormoments_build.moments_build import build_moments as build_scenarios

        return build_scenarios(
            set_name=set_name,
            ids=ids,
            ground_truth_dir=str(GT_DIR),
            transcripts_dir=str(TX_DIR),
            tutoring_provider_a_jsonl=tutoring_provider_a_jsonl,
        )

    def test_jsonl_only_conv_hydrates_with_jsonl_param(self):
        """conv-ddd exists only in JSONL; with tutoring_provider_a_jsonl it should hydrate fully."""
        ids = _ids_for(("conv-ddd", 1, 3))
        scenarios = self._build(ids, tutoring_provider_a_jsonl=TX_JSONL)
        assert len(scenarios) == 1
        s = scenarios[0]
        # context = turns up to cut=2 (STUDENT at 2, stays)
        assert len(s.context) == 2
        assert s.context[0]["role"] == "tutor"
        assert s.context[1]["role"] == "student"
        # turn_number must be present and correct
        assert "turn_number" in s.context[0]
        assert "turn_number" in s.context[1]
        # student.reference has post-cut turns (3, 4)
        ref = s.student["reference"]
        assert "Turn 3." in ref
        assert "Turn 4." in ref
        assert "Turn 2." not in ref
        # student.context comes from demographics (Grade 7, Algebra)
        ctx = s.student["context"]
        assert "7" in ctx
        assert "Algebra" in ctx
        # provenance
        assert s.provenance["conv_id"] == "conv-ddd"
        assert s.provenance["cut_turn"] == 2

    def test_jsonl_only_conv_not_hydrated_without_jsonl_param(self):
        """conv-ddd exists only in JSONL; without tutoring_provider_a_jsonl it must be skipped."""
        ids = _ids_for(("conv-ddd", 1, 3))
        scenarios = self._build(ids, tutoring_provider_a_jsonl=None)
        assert len(scenarios) == 0

    def test_jsonl_conv_and_json_conv_both_hydrate(self):
        """When JSONL and per-file JSON convs are both requested, both hydrate."""
        ids = _ids_for(("conv-aaa", 3, 5), ("conv-ddd", 1, 3))
        scenarios = self._build(ids, tutoring_provider_a_jsonl=TX_JSONL)
        assert len(scenarios) == 2
        conv_ids = [s.provenance["conv_id"] for s in scenarios]
        assert "conv-aaa" in conv_ids
        assert "conv-ddd" in conv_ids


# ---------------------------------------------------------------------------
# build_moments_from_reference_run: reconstruct the frozen set from a
# published benchmark run (the only surviving record of the benchmark-time
# detections when later ground-truth rebuilds have drifted).
# ---------------------------------------------------------------------------

_STUB_TRAIT = {
    "persona": "The student is careful but second-guesses correct first answers.",
    "trait_mode": "joined-3",
    "generator_model": "claude-opus-4-6",
    "generated_at": "2026-06-18T06:28:40",
}


def _reference_run_record(
    cell,
    scenario_id,
    conv_id,
    cut_turn,
    *,
    dimension="scaffolding",
    ts=3,
    te=5,
    situation="Ref-run situation text",
    student_trait=_STUB_TRAIT,
):
    rec = {
        "cell": cell,
        "tutor_model": cell.split("__")[0],
        "prompt_mode": cell.split("__")[1],
        "scenario_id": scenario_id,
        "conv_id": conv_id,
        "cut_turn": cut_turn,
        "detection": {
            "turn_start": ts,
            "turn_end": te,
            "annotation_type": "scaffolding",
            "situation": situation,
            "situation_label_agg": dimension,
            "moment_id": "m-ref-001",
            "annotator_id": "a-ref-001",
            "chosen_cut_turn": cut_turn,
            "cut_votes": {str(cut_turn): 2},
            "cluster_size": 2,
        },
        "exchange": {"generated_turns": []},
        "annotation": {},
    }
    if student_trait is not None:
        rec["student_trait"] = student_trait
    return rec


class TestBuildMomentsFromReferenceRun:
    """Hermetic tests against the build_src transcript fixtures."""

    def _build(self, records, tmp_path, set_name="test"):
        from tutormoments_build.moments_build import build_moments_from_reference_run

        run_file = tmp_path / "reference_run.jsonl"
        with open(run_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return build_moments_from_reference_run(
            set_name=set_name,
            reference_run_jsonl=str(run_file),
            transcripts_dir=str(TX_DIR),
            tutoring_provider_a_jsonl=TX_JSONL,
        )

    def test_dedupes_cells_and_sorts_by_scenario_id(self, tmp_path):
        """One Moment per unique scenario_id, sorted by id, across many cells."""
        records = [
            _reference_run_record("m1__plain", "conv-aaa__hum_3_5", "conv-aaa", 4),
            _reference_run_record(
                "m1__scaffolding_rigor", "conv-aaa__hum_3_5", "conv-aaa", 4
            ),
            _reference_run_record(
                "m2__plain", "conv-ddd__hum_1_3", "conv-ddd", 2, ts=1, te=3
            ),
        ]
        moments = self._build(records, tmp_path)
        assert [m.id for m in moments] == [
            "test:conv-aaa__hum_3_5",
            "test:conv-ddd__hum_1_3",
        ]

    def test_detection_payload_becomes_moment_fields(self, tmp_path):
        """dimension/gold/hint/provenance come from the run's frozen detection."""
        records = [
            _reference_run_record("m1__plain", "conv-aaa__hum_3_5", "conv-aaa", 4)
        ]
        (m,) = self._build(records, tmp_path)
        assert m.dimension == "scaffolding"
        assert m.rubric["gold"] == "scaffolding"
        assert m.rubric["hint"] == "Ref-run situation text"
        p = m.provenance
        assert p["conv_id"] == "conv-aaa"
        assert p["cut_turn"] == 4
        assert p["turn_start"] == 3 and p["turn_end"] == 5
        assert p["moment_id"] == "m-ref-001"
        assert p["annotator_id"] == "a-ref-001"
        assert p["chosen_cut_turn"] == 4
        assert p["cut_votes"] == {"4": 2}
        assert p["cluster_size"] == 2

    def test_context_and_reference_derived_from_transcript(self, tmp_path):
        """context = prefix turns <= cut_turn; reference = post-cut turns."""
        records = [
            _reference_run_record("m1__plain", "conv-aaa__hum_3_5", "conv-aaa", 4)
        ]
        (m,) = self._build(records, tmp_path)
        assert [t["turn_number"] for t in m.context] == [1, 2, 3, 4]
        assert m.context[3] == {
            "turn_number": 4,
            "role": "student",
            "text": "I split it into two groups",
        }
        ref = m.student["reference"]
        assert "Turn 5." in ref and "Turn 7." in ref and "Turn 4." not in ref
        assert m.student["mode"] == "oracle"
        assert m.student["context"] == "Grade 5, Mathematics, fractions"

    def test_jsonl_only_conv_resolves(self, tmp_path):
        """A conv present only in the normalized JSONL transcript pool builds fine."""
        records = [
            _reference_run_record(
                "m1__plain", "conv-ddd__hum_1_3", "conv-ddd", 2, ts=1, te=3
            )
        ]
        (m,) = self._build(records, tmp_path)
        assert m.id == "test:conv-ddd__hum_1_3"
        assert [t["turn_number"] for t in m.context] == [1, 2]

    def test_missing_transcript_skipped(self, tmp_path):
        """A scenario whose conv has no transcript is skipped, not fatal."""
        records = [
            _reference_run_record("m1__plain", "conv-aaa__hum_3_5", "conv-aaa", 4),
            _reference_run_record(
                "m1__plain", "conv-zzz__hum_1_2", "conv-zzz", 1, ts=1, te=2
            ),
        ]
        moments = self._build(records, tmp_path)
        assert [m.id for m in moments] == ["test:conv-aaa__hum_3_5"]

    def test_student_trait_embedded_from_run(self, tmp_path):
        """The run record's frozen student_trait lands verbatim in student.trait."""
        records = [
            _reference_run_record("m1__plain", "conv-aaa__hum_3_5", "conv-aaa", 4)
        ]
        (m,) = self._build(records, tmp_path)
        assert m.student["trait"] == _STUB_TRAIT

    def test_missing_student_trait_raises(self, tmp_path):
        """A run record without student_trait must fail the build loudly:
        the release contract is that every moment carries the paper's frozen
        persona, and a silent skip would ship a non-reproducible set."""
        records = [
            _reference_run_record(
                "m1__plain", "conv-aaa__hum_3_5", "conv-aaa", 4, student_trait=None
            )
        ]
        with pytest.raises(ValueError, match="student_trait"):
            self._build(records, tmp_path)


def test_write_release_emits_flat_manifest_and_schema(tmp_path):
    """write_release writes moments.jsonl + moments.manifest.json +
    moments.schema.json (flat {thing}.{kind} release convention)."""
    from tutormoments.moments import Moment, validate_dataset
    from tutormoments_build.moments_build import write_release

    m = Moment(
        id="t:c__hum_1_2",
        context=[{"turn_number": 1, "role": "tutor", "text": "Q"}],
        dimension="rigor",
        student={
            "mode": "oracle",
            "reference": "",
            "context": "",
            "trait": dict(_STUB_TRAIT),
        },
        rubric={"gold": "rigor", "hint": ""},
        provenance={
            "conv_id": "c",
            "cut_turn": 1,
            "turn_start": 1,
            "turn_end": 2,
            "moment_id": None,
            "annotator_id": "a",
            "chosen_cut_turn": 1,
            "cut_votes": {"1": 1},
            "cluster_size": 1,
        },
    )
    write_release([m], tmp_path, set_name="t", created="2026-07-02")

    assert (tmp_path / "moments.jsonl").exists()
    assert (tmp_path / "moments.manifest.json").exists()
    assert not (tmp_path / "manifest.json").exists()
    schema = json.loads((tmp_path / "moments.schema.json").read_text(encoding="utf-8"))
    assert schema["title"].startswith("moments.jsonl")
    assert "trait" in schema["properties"]["student"]["required"]
    # the release dir round-trips through the runtime validator
    report = validate_dataset(tmp_path)
    assert report["record_count"] == 1


def test_write_release_refuses_traitless_moment(tmp_path):
    """A moment without a student.trait persona cannot be released."""
    from tutormoments.moments import Moment
    from tutormoments_build.moments_build import write_release

    m = Moment(
        id="t:c__hum_1_2",
        context=[{"turn_number": 1, "role": "tutor", "text": "Q"}],
        dimension="rigor",
        student={"mode": "oracle", "reference": "", "context": ""},
        rubric={"gold": "rigor", "hint": ""},
        provenance={
            "conv_id": "c",
            "cut_turn": 1,
            "turn_start": 1,
            "turn_end": 2,
            "moment_id": None,
            "annotator_id": "a",
            "chosen_cut_turn": 1,
            "cut_votes": {"1": 1},
            "cluster_size": 1,
        },
    )
    with pytest.raises(ValueError, match="trait"):
        write_release([m], tmp_path, set_name="t", created="2026-07-02")


def test_validate_dataset_requires_traits(tmp_path):
    """validate_dataset rejects a release whose records lack personas, even
    if hashes are self-consistent (guards hand-rolled v2 datasets)."""
    from tutormoments.moments import Moment, validate_dataset
    from tutormoments_build.moments_build import write_release

    m = Moment(
        id="t:c__hum_1_2",
        context=[{"turn_number": 1, "role": "tutor", "text": "Q"}],
        dimension="rigor",
        student={
            "mode": "oracle",
            "reference": "",
            "context": "",
            "trait": dict(_STUB_TRAIT),
        },
        rubric={"gold": "rigor", "hint": ""},
        provenance={
            "conv_id": "c",
            "cut_turn": 1,
            "turn_start": 1,
            "turn_end": 2,
            "moment_id": None,
            "annotator_id": "a",
            "chosen_cut_turn": 1,
            "cut_votes": {"1": 1},
            "cluster_size": 1,
        },
    )
    write_release([m], tmp_path, set_name="t", created="2026-07-02")

    # Strip the persona from the written record, rewrite manifest hashes to match.
    import json as _json

    from tutormoments.moments import (
        _read_moments_jsonl,
        file_sha256,
        records_content_hash,
    )

    rec = _json.loads((tmp_path / "moments.jsonl").read_text())
    rec["student"].pop("trait")
    (tmp_path / "moments.jsonl").write_text(_json.dumps(rec) + "\n", encoding="utf-8")
    manifest = _json.loads((tmp_path / "moments.manifest.json").read_text())
    loaded = _read_moments_jsonl(tmp_path / "moments.jsonl")
    manifest["content_hash"] = records_content_hash([x.to_dict() for x in loaded])
    manifest["file_sha256"] = file_sha256(tmp_path / "moments.jsonl")
    (tmp_path / "moments.manifest.json").write_text(
        _json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="trait"):
        validate_dataset(tmp_path)
