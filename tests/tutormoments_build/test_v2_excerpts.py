"""Tests for v2 moment-excerpt rendering."""

import json

import pytest

from tutormoments_build.v2 import excerpts as E


def _dialogue(index, turn_number, role, text):
    return {
        "index": index,
        "role": role,
        "text": f"[{role}] {text}",
        "type": "dialogue",
        "timestamp": "",
        "turn_number": turn_number,
    }


def _enrichment(index, kind, text, role="TUTOR"):
    return {
        "index": index,
        "role": role,
        "text": f"[{kind}] {text}",
        "type": f"[{kind}]",
        "timestamp": "",
        "turn_number": None,
    }


def _rows(raw):
    """Normalise raw export rows the way TranscriptIndex does before rendering."""
    return [
        {
            "turn_index": row["index"],
            "turn_number": row["turn_number"],
            "role": row["role"],
            "text": row["text"],
            "type": row["type"],
            "timestamp": row["timestamp"],
        }
        for row in raw
    ]


# rows 0-8: dialogue turns 1-5 with screen activity interleaved after turn 3.
SAMPLE = _rows(
    [
        _dialogue(0, 1, "TUTOR", "What is half of six?"),
        _dialogue(1, 2, "STUDENT", "Three?"),
        _dialogue(2, 3, "TUTOR", "How did you get that?"),
        _enrichment(3, "SCREEN INTERACTION", "Student selects 3.", role="STUDENT"),
        _enrichment(4, "SCREEN UPDATE", "The option turns green."),
        _dialogue(5, 4, "STUDENT", "I split it in two."),
        _enrichment(6, "PAUSE", "The student is thinking."),
        _dialogue(7, 5, "TUTOR", "Nice work."),
        _enrichment(8, "PROBLEM CHANGE", "A new problem loads."),
    ]
)


def _boundaries(start_index, cut_index, end_index, start_turn, cut_turn, end_turn):
    return {
        "start_index": start_index,
        "cut_index": cut_index,
        "end_index": end_index,
        "start_turn": start_turn,
        "cut_turn": cut_turn,
        "end_turn": end_turn,
    }


# ===========================================================================
# Lead-up window
# ===========================================================================


class TestContextStart:
    def test_counts_dialogue_turns_not_rows(self):
        """Two turns of lead-up before row 7 reaches turn 3, carrying its enrichments."""
        assert E.context_start(SAMPLE, 7, 2) == 2

    def test_zero_turns_opens_on_the_moment(self):
        assert E.context_start(SAMPLE, 7, 0) == 7

    def test_opens_on_a_turn_not_mid_enrichment_run(self):
        """Row 5 (turn 4) is preceded by two enrichments; one turn back is turn 3."""
        assert E.context_start(SAMPLE, 5, 1) == 2

    def test_runs_out_of_transcript(self):
        assert E.context_start(SAMPLE, 5, 99) == 0

    def test_max_context_rows_clamps(self):
        assert E.context_start(SAMPLE, 7, 5, max_context_rows=2) == 5

    def test_max_context_rows_does_not_widen(self):
        """The cap is a ceiling; it never reaches further back than the turns do."""
        assert E.context_start(SAMPLE, 7, 1, max_context_rows=99) == 5


# ===========================================================================
# Row formatting
# ===========================================================================


class TestFormatRow:
    def test_dialogue_drops_the_repeated_role_tag(self):
        assert E.format_row(SAMPLE[0]) == "Turn 1. TUTOR: What is half of six?"

    def test_enrichment_keeps_its_tag_and_gets_no_turn_number(self):
        assert E.format_row(SAMPLE[3]) == "[SCREEN INTERACTION] Student selects 3."

    def test_disagreeing_tag_is_left_visible(self):
        row = dict(SAMPLE[0], role="STUDENT")
        assert E.format_row(row) == "Turn 1. STUDENT: [TUTOR] What is half of six?"


# ===========================================================================
# Excerpt rendering
# ===========================================================================


class TestRenderExcerpt:
    def test_window_runs_from_lead_up_to_end_and_stops(self):
        text, first = E.render_excerpt(
            SAMPLE, _boundaries(5, 6, 7, 4, 4, 5), context_turns=1
        )
        assert first == 2
        assert text.splitlines() == [
            "[... 2 earlier row(s) omitted ...]",
            "",
            "Turn 3. TUTOR: How did you get that?",
            "[SCREEN INTERACTION] Student selects 3.",
            "[SCREEN UPDATE] The option turns green.",
            ">>> MOMENT START (row 5, turn 4) <<<",
            "Turn 4. STUDENT: I split it in two.",
            "[PAUSE] The student is thinking.",
            ">>> CUT POINT (row 6, turn 4) <<<",
            "Turn 5. TUTOR: Nice work.",
            ">>> MOMENT END (row 7, turn 5) <<<",
        ]

    def test_nothing_after_the_end_row(self):
        """Row 8 exists but is never rendered -- there is no trailing window."""
        text, _ = E.render_excerpt(
            SAMPLE, _boundaries(5, 6, 7, 4, 4, 5), context_turns=1
        )
        assert "PROBLEM CHANGE" not in text
        assert text.endswith(">>> MOMENT END (row 7, turn 5) <<<")

    def test_no_omission_header_when_context_reaches_the_start(self):
        text, first = E.render_excerpt(
            SAMPLE, _boundaries(5, 5, 7, 4, 4, 5), context_turns=99
        )
        assert first == 0
        assert text.startswith("Turn 1. TUTOR:")

    def test_enrichment_only_moment_survives(self):
        """The all-screen-activity case: every turn number collapses onto one
        dialogue turn, and only the index span carries the moment's extent."""
        text, _ = E.render_excerpt(
            SAMPLE, _boundaries(3, 3, 4, 3, 3, 3), context_turns=0
        )
        assert text.splitlines() == [
            "[... 3 earlier row(s) omitted ...]",
            "",
            ">>> MOMENT START (row 3, turn 3) <<<",
            "[SCREEN INTERACTION] Student selects 3.",
            ">>> CUT POINT (row 3, turn 3) <<<",
            "[SCREEN UPDATE] The option turns green.",
            ">>> MOMENT END (row 4, turn 3) <<<",
        ]

    def test_cut_at_the_end_emits_both_markers_in_order(self):
        text, _ = E.render_excerpt(
            SAMPLE, _boundaries(5, 7, 7, 4, 5, 5), context_turns=0
        )
        assert text.splitlines()[-2:] == [
            ">>> CUT POINT (row 7, turn 5) <<<",
            ">>> MOMENT END (row 7, turn 5) <<<",
        ]

    def test_out_of_range_boundaries_raise(self):
        with pytest.raises(IndexError, match="outside the transcript"):
            E.render_excerpt(SAMPLE, _boundaries(5, 6, 99, 4, 4, 5))


# ===========================================================================
# Record assembly
# ===========================================================================


class TestBuildRecord:
    def _moment(self, **kw):
        moment = {
            "moment_id": "m1",
            "transcript_id": "t1",
            "split": "iteration",
            "labels": {"scaffolding_present": True},
            **_boundaries(5, 6, 7, 4, 4, 5),
        }
        moment.update(kw)
        return moment

    def test_counts_and_labels(self):
        record = E.build_record(self._moment(), SAMPLE, "conv-1", context_turns=1)
        assert record["conversation_id"] == "conv-1"
        assert record["context_start_index"] == 2
        assert record["context_rows"] == 3
        assert record["moment_rows"] == 3
        assert record["moment_dialogue_rows"] == 2
        assert record["moment_enrichment_rows"] == 1
        assert record["labels"] == {"scaffolding_present": True}
        assert record["excerpt"].startswith("[... 2 earlier row(s) omitted ...]")

    def test_context_turns_recorded(self):
        record = E.build_record(self._moment(), SAMPLE, "conv-1", context_turns=3)
        assert record["context_turns"] == 3


# ===========================================================================
# Turn-span loss detection
# ===========================================================================


class TestTurnSpanDisagrees:
    def test_flags_a_span_the_turn_numbers_do_not_cover(self):
        moment = {"start_turn": 3, "end_turn": 3, "start_index": 3, "end_index": 4}
        assert E._turn_span_disagrees(moment, SAMPLE) is True

    def test_quiet_when_the_turn_span_covers_the_rows(self):
        moment = {"start_turn": 1, "end_turn": 5, "start_index": 0, "end_index": 7}
        assert E._turn_span_disagrees(moment, SAMPLE) is False


# ===========================================================================
# Loading and writing
# ===========================================================================


class TestLoadGroundTruth:
    def test_reads_both_splits(self, tmp_path):
        (tmp_path / "iteration.jsonl").write_text('{"moment_id": "a"}\n\n')
        (tmp_path / "test.jsonl").write_text('{"moment_id": "b"}\n')
        loaded = E.load_ground_truth(str(tmp_path))
        assert loaded == {
            "iteration": [{"moment_id": "a"}],
            "test": [{"moment_id": "b"}],
        }

    def test_missing_split_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "test.jsonl").write_text('{"moment_id": "b"}\n')
        assert list(E.load_ground_truth(str(tmp_path))) == ["test"]

    def test_no_splits_at_all_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_ground_truth"):
            E.load_ground_truth(str(tmp_path))


def test_write_split_is_atomic_and_roundtrips(tmp_path):
    out = tmp_path / "excerpts"
    path = E.write_split(str(out), "iteration", [{"moment_id": "a", "excerpt": "x"}])
    assert not (out / "iteration.jsonl.tmp").exists()
    with open(path, encoding="utf-8") as fh:
        assert [json.loads(line) for line in fh] == [{"moment_id": "a", "excerpt": "x"}]
