import pytest

from tutormoments.moments import (
    DatasetNotFoundError,
    Moment,
    load_moments,
    records_content_hash,
    validate_dataset,
)

_FIXTURE_RELEASE = "tests/tutormoments/fixtures/mini_release"


def test_moment_roundtrip():
    m = Moment(
        id="balanced_520:conv__hum_5_7",
        context=[
            {"turn_number": 5, "role": "tutor", "text": "Hi"},
            {"turn_number": 7, "role": "student", "text": "ok"},
        ],
        dimension="scaffolding",
        student={
            "mode": "oracle",
            "reference": "Turn 8. STUDENT: ...",
            "context": "Grade 5, fractions",
        },
        rubric={"gold": "scaffolding", "hint": "The student gave a guess."},
        provenance={"conv_id": "conv", "cut_turn": 7, "turn_start": 5, "turn_end": 7},
    )
    d = m.to_dict()
    assert Moment.from_dict(d) == m
    assert d["dimension"] == "scaffolding"
    assert d["rubric"]["gold"] == "scaffolding"
    assert d["context"][0]["role"] == "tutor"
    assert d["context"][0]["turn_number"] == 5


def test_from_dict_normalizes_cut_votes_forms():
    """Released (list-of-pairs) and internal (dict) cut_votes load identically."""
    base = {
        "id": "s:c__hum_1_2",
        "context": [{"turn_number": 1, "role": "tutor", "text": "Q"}],
        "dimension": "rigor",
        "student": {"mode": "oracle", "reference": "", "context": ""},
        "rubric": {"gold": "rigor", "hint": ""},
    }
    released = dict(
        base,
        provenance={
            "conv_id": "c",
            "cut_turn": 27,
            "cut_votes": [{"cut_turn": 27, "votes": 7}, {"cut_turn": 32, "votes": 1}],
        },
    )
    internal = dict(
        base,
        provenance={
            "conv_id": "c",
            "cut_turn": 27,
            "cut_votes": {"27": 7, "32": 1},
        },
    )
    int_keyed = dict(
        base,
        provenance={
            "conv_id": "c",
            "cut_turn": 27,
            "cut_votes": {27: 7, 32: 1},
        },
    )
    expected = {"27": 7, "32": 1}
    assert Moment.from_dict(released).provenance["cut_votes"] == expected
    assert Moment.from_dict(internal).provenance["cut_votes"] == expected
    assert Moment.from_dict(int_keyed).provenance["cut_votes"] == expected
    # And the hash over normalized records is form-independent
    h = records_content_hash
    assert h([Moment.from_dict(released).to_dict()]) == h(
        [Moment.from_dict(internal).to_dict()]
    )


def test_load_moments_data_path_reads_in_file_order():
    """load_moments(data_path=...) reads moments.jsonl in file order."""
    moments, source = load_moments(data_path=_FIXTURE_RELEASE)
    assert len(moments) == 2
    assert moments[0].id == "mini:conv1__hum_1_2"
    assert moments[1].dimension == "rigor"
    assert [m.dimension for m in moments] == ["scaffolding", "rigor"]
    assert source["data_path"] == _FIXTURE_RELEASE
    assert source["record_count"] == 2
    assert len(source["content_hash"]) == 64


def test_load_moments_missing_raises():
    with pytest.raises(DatasetNotFoundError):
        load_moments(data_path="tests/tutormoments/fixtures/does_not_exist")


def test_load_moments_no_source_raises():
    with pytest.raises(ValueError):
        load_moments()


def test_load_moments_hf_path_normalizes_datetimes(monkeypatch):
    """HF loading normalizes Arrow datetime fields to ISO strings.

    datasets.load_dataset returns timestamp columns (student.trait.generated_at)
    as datetime objects. The runtime must normalize them so records stay
    JSON-serializable for records_content_hash and hash identically to the
    equivalent jsonl (string) form.
    """
    import datasets

    from datetime import datetime

    trait = {
        "persona": "curious",
        "trait_mode": "frozen",
        "generator_model": "claude-opus-4-6",
        "generated_at": datetime(2026, 6, 18, 6, 28, 40),
    }
    row = {
        "id": "s:c__hum_1_2",
        "context": [{"turn_number": 1, "role": "tutor", "text": "Q"}],
        "dimension": "rigor",
        "student": {"mode": "oracle", "reference": "", "context": "", "trait": trait},
        "rubric": {"gold": "rigor", "hint": ""},
        "provenance": {"conv_id": "c", "cut_turn": 27},
    }

    def fake_load_dataset(dataset, name=None, revision=None, split=None):
        assert dataset == "org/ds"
        assert name == "moments"
        assert split == "train"
        return [row]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    moments, source = load_moments(dataset="org/ds")

    assert len(moments) == 1
    gen = moments[0].to_dict()["student"]["trait"]["generated_at"]
    assert gen == "2026-06-18T06:28:40"  # datetime -> ISO string, not a datetime
    assert source["dataset_id"] == "org/ds"

    # Hash parity: the HF (datetime) form hashes identically to the jsonl form
    # that carries the same timestamp as an ISO string.
    string_trait = dict(trait, generated_at="2026-06-18T06:28:40")
    string_row = dict(
        row, student=dict(row["student"], trait=string_trait)
    )
    expected = records_content_hash([Moment.from_dict(string_row).to_dict()])
    assert source["content_hash"] == expected


def test_validate_dataset_reads_manifest_and_hashes_fixture():
    report = validate_dataset(_FIXTURE_RELEASE)
    assert report["record_count"] == 2
    assert report["manifest"]["version"] == "fixture"
    assert len(report["content_hash"]) == 64
    # content_hash must match what load_moments computes (record-level)
    _, source = load_moments(data_path=_FIXTURE_RELEASE)
    assert source["content_hash"] == report["content_hash"]
