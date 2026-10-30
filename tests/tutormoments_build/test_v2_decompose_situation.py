"""Tests for decomposing adjudicators' situation descriptions into student facets."""

import json

import pytest

from tutormoments_build.v2 import decompose_situation as D
from tutormoments_build.v2.classify_excerpts import V2Spec

CFG = V2Spec(
    model="claude-opus-5",
    provider="anthropic",
    thinking={"thinking": {"type": "adaptive"}},
    poll_interval=1,
)

SITUATION = "The student solved it correctly, so they are ready for a challenge."


def _account(adjudicator, text):
    return {"adjudicator": adjudicator, "text": text}


def _gt(moment_id, split="iteration"):
    return {
        "moment_id": moment_id,
        "transcript_id": f"t-{moment_id}",
        "source": "tutoring_provider_a",
        "split": split,
    }


# --- loading ---------------------------------------------------------------


def _adjudication(annotator_id, why, *, revision=1, throw_out=False):
    return {
        "annotator_id": annotator_id,
        "role": "adjudicator",
        "revision": revision,
        "created_at": f"2026-10-0{revision}T00:00:00Z",
        "payload": {
            "meta": {"throw_out": throw_out},
            "final": {"situation": {"why": why}, "action": {}},
        },
    }


def test_load_adjudicator_descriptions_reads_each_adjudicators_final_account(
    tmp_path,
):
    row = {
        "transcript_id": "t",
        "moment": {"moment_id": "m"},
        "annotations": [
            {
                "annotator_id": "sel",
                "role": "selector",
                "payload": {"situation": {"why": "selector text"}, "action": {}},
            },
            _adjudication("adj-1", "first draft"),
            _adjudication("adj-1", "  final account  ", revision=2),
            _adjudication("adj-2", "   "),
            _adjudication("adj-3", "threw it out", throw_out=True),
            _adjudication("staff", "staff text"),
        ],
    }
    path = tmp_path / "provider_x_annotations.jsonl"
    path.write_text(
        json.dumps(row) + "\n" + json.dumps({"transcript_id": "u"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "excluded.json").write_text(json.dumps(["staff"]), encoding="utf-8")
    (tmp_path / "labels.json").write_text(
        json.dumps({"adj-1": "J07"}), encoding="utf-8"
    )

    found = D.load_adjudicator_descriptions(
        [str(path)], str(tmp_path / "excluded.json"), str(tmp_path / "labels.json")
    )

    assert found == {"m": [{"adjudicator": "J07", "text": "final account"}]}


# --- merging ---------------------------------------------------------------


def test_single_account_is_used_as_written():
    merged = D.merge_descriptions("m1", [_account("A01", SITUATION)])
    assert merged == {
        "situation": SITUATION,
        "method": "single",
        "similarity": None,
        "adjudicators": ["A01"],
    }


def test_similar_accounts_keep_the_longer():
    longer = SITUATION + " Clearly."
    merged = D.merge_descriptions(
        "m1", [_account("A01", SITUATION), _account("A02", longer)]
    )
    assert merged["method"] == "longer"
    assert merged["situation"] == longer
    assert merged["adjudicators"] == ["A02"]
    assert merged["similarity"] >= D.SIMILARITY_THRESHOLD


def test_different_accounts_are_concatenated_in_save_order():
    other = "Tutor rambles; no idea what the learner thinks."
    merged = D.merge_descriptions(
        "m1", [_account("A01", SITUATION), _account("A02", other)]
    )
    assert merged["method"] == "concatenated"
    assert merged["situation"] == SITUATION + D.CONCAT_SEPARATOR + other
    assert merged["adjudicators"] == ["A01", "A02"]
    assert merged["similarity"] < D.SIMILARITY_THRESHOLD


def test_length_tie_is_seeded_by_moment_id():
    a = SITUATION
    b = SITUATION.replace("solved", "SOLVED")
    candidates = [_account("A01", a), _account("A02", b)]
    picks = {
        D.merge_descriptions(f"m{i}", candidates)["adjudicators"][0] for i in range(40)
    }
    # Random across moments, but the same moment always gets the same pick,
    # whatever order its accounts arrive in.
    assert picks == {"A01", "A02"}
    for i in range(10):
        first = D.merge_descriptions(f"m{i}", candidates)
        assert D.merge_descriptions(f"m{i}", candidates) == first


# --- prompt ----------------------------------------------------------------


def test_fill_does_not_search_the_situation_for_slots():
    filled = D.fill("Situation: {situation}", "the student wrote {situation}")
    assert filled == "Situation: the student wrote {situation}"


def test_fill_raises_on_an_unknown_slot():
    with pytest.raises(ValueError, match="other"):
        D.fill("{situation} {other}", SITUATION)


def test_shipped_prompt_fills():
    from tutormoments_build.resources import resource_text

    filled = D.fill(resource_text(D.PROMPT_PATH), SITUATION)
    assert filled.rstrip().endswith(SITUATION)


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            '["The student is stuck.", " They guessed. "]',
            ["The student is stuck.", "They guessed."],
        ),
        ("[]", []),
        ('{"facets": ["The student is stuck."]}', ["The student is stuck."]),
        (
            'Here you go:\n```json\n["The student is stuck."]\n```',
            ["The student is stuck."],
        ),
    ],
)
def test_parse_facets(text, expected):
    assert D.parse_facets(text) == (expected, False)


def test_unparseable_response_is_none_not_empty():
    assert D.parse_facets("I cannot help with that.") == (None, True)
    assert D.parse_facets("") == (None, True)


# --- end to end ------------------------------------------------------------


@pytest.fixture
def gt_dir(tmp_path):
    for stem, ids in (("iteration", ["m1", "m2"]), ("test", ["m3"])):
        with open(tmp_path / f"{stem}.jsonl", "w") as fh:
            for moment_id in ids:
                fh.write(json.dumps(_gt(moment_id, stem)) + "\n")
    return tmp_path


@pytest.fixture
def run(monkeypatch):
    descriptions = {
        "m1": [_account("A01", SITUATION)],
        "m3": [_account("A02", SITUATION), _account("A03", "Something else entirely.")],
    }
    sent = {}

    def fake_run_entries(entries, cfg, batch_id=None, display_name=None):
        sent["entries"] = entries
        return {
            "state__m1": {
                "text": '["The student solved it correctly."]',
                "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            },
            "state__m3": {"text": "not json"},
        }

    monkeypatch.setattr(D, "load_adjudicator_descriptions", lambda: descriptions)
    monkeypatch.setattr(D, "phase_config", lambda model=None: CFG)
    monkeypatch.setattr(D, "run_entries", fake_run_entries)
    return sent


def _read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def test_main_writes_one_record_per_ground_truth_moment(gt_dir, run):
    assert (
        D.main(
            [
                "--model",
                "claude-opus-5",
                "--gt-dir",
                str(gt_dir),
                "--out",
                str(gt_dir / "student_state.jsonl"),
            ]
        )
        == 0
    )

    records = _read(gt_dir / "student_state.jsonl")
    assert [(r["moment_id"], r["split"]) for r in records] == [
        ("m1", "iteration"),
        ("m2", "iteration"),
        ("m3", "test"),
    ]
    by_id = {r["moment_id"]: r for r in records}

    assert by_id["m1"]["facets"] == ["The student solved it correctly."]
    assert by_id["m1"]["usage"]["total_tokens"] == 7

    # Undescribed: written, not sent.
    assert by_id["m2"]["skipped"] == D.NO_DESCRIPTION
    assert by_id["m2"]["facets"] is None
    assert {e["key"] for e in run["entries"]} == {"state__m1", "state__m3"}

    assert by_id["m3"]["situation_method"] == "concatenated"
    assert by_id["m3"]["adjudicators"] == ["A02", "A03"]
    assert by_id["m3"]["parse_error"] is True
    assert by_id["m3"]["facets"] is None


def test_main_refuses_to_overwrite_before_any_call(gt_dir, run):
    (gt_dir / "student_state.jsonl").write_text("existing\n")
    with pytest.raises(FileExistsError):
        D.main(
            [
                "--model",
                "claude-opus-5",
                "--gt-dir",
                str(gt_dir),
                "--out",
                str(gt_dir / "student_state.jsonl"),
            ]
        )
    assert "entries" not in run
    assert (gt_dir / "student_state.jsonl").read_text() == "existing\n"


def test_splits_and_limit_narrow_the_moments(gt_dir, run):
    D.main(
        [
            "--model",
            "claude-opus-5",
            "--gt-dir",
            str(gt_dir),
            "--out",
            str(gt_dir / "student_state.jsonl"),
            "--splits",
            "iteration",
            "--limit",
            "1",
        ]
    )
    assert [r["moment_id"] for r in _read(gt_dir / "student_state.jsonl")] == ["m1"]


def test_dry_run_prints_an_example_prompt_and_writes_nothing(gt_dir, run, capsys):
    assert (
        D.main(
            [
                "--gt-dir",
                str(gt_dir),
                "--out",
                str(gt_dir / "student_state.jsonl"),
                "--dry-run",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "===== moment m" in out
    assert "Situation: " + SITUATION in out
    assert "entries" not in run
    assert not (gt_dir / "student_state.jsonl").exists()
