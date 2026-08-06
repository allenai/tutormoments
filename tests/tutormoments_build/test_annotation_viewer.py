"""Tests for the annotation review site's case building and transcript windowing."""

import json

import pytest

from tutormoments_build.annotation_viewer import (
    action_tags,
    annotated_moments,
    build_cases,
    build_site,
    latest_revisions,
    moment_turns,
    paired_moments,
)
from tutormoments_build.annotation_viewer.transcripts import TranscriptIndex


def annotation(annotator, role, *, revision=1, created_at="2026-08-01T00:00:00Z", **payload):
    return {
        "annotator_id": annotator,
        "role": role,
        "revision": revision,
        "created_at": created_at,
        "payload": payload,
    }


def situation(scaffolding=True, rigor=False, why="because"):
    return {
        "scaffolding_appropriate": scaffolding,
        "rigor_appropriate": rigor,
        "why": why,
    }


def action(**over):
    base = {
        "scaffolding_present": True,
        "scaffolding_amount": "appropriate",
        "rigor_present": False,
        "rigor_amount": None,
        "scaffolding_strategies": ["guiding_questions"],
        "scaffolding_strategy_other": "",
        "over_scaffolding_reasons": [],
        "rigor_strategies": [],
        "rigor_strategy_other": "",
        "explanation": "did a thing",
    }
    return {**base, **over}


def result(success="meaningful_success", engagement="high_or_good"):
    return {
        "problem_success": success,
        "cognitive_engagement": engagement,
        "explanation": "outcome",
    }


def row(index, text="hello", *, role="TUTOR", kind="dialogue"):
    """One rendered transcript row, shaped like the v2 transcripts export."""
    prefix = kind if kind != "dialogue" else f"[{role}]"
    return {
        "index": index,
        "role": role,
        "text": f"{prefix} {text}",
        "type": kind,
        "timestamp": "00:01-00:02",
        "turn_number": index + 1 if kind == "dialogue" else None,
    }


def transcript(rows, transcript_id="t1"):
    return {"transcript_id": transcript_id, "conversation_id": "c1", "turns": rows}


def record(annotations, *, moment_id="m1", status="reannotated", start=10, cut=12, end=20):
    return {
        "transcript_id": "t1",
        "moment": {
            "id": moment_id,
            "status": status,
            "start_turn": start,
            "cut_turn": cut,
            "end_turn": end,
            "created_by": "ann",
        },
        "annotations": annotations,
    }


class TestLatestRevisions:
    def test_keeps_highest_revision_per_annotator(self):
        anns = [
            annotation("ann", "selector", revision=1, situation=situation(why="first")),
            annotation("ann", "selector", revision=2, situation=situation(why="second")),
        ]
        kept = latest_revisions(anns)
        assert len(kept) == 1
        assert kept[0]["payload"]["situation"]["why"] == "second"

    def test_breaks_ties_on_created_at(self):
        anns = [
            annotation("ann", "selector", created_at="2026-08-01T00:00:00Z",
                       situation=situation(why="early")),
            annotation("ann", "selector", created_at="2026-08-02T00:00:00Z",
                       situation=situation(why="late")),
        ]
        assert latest_revisions(anns)[0]["payload"]["situation"]["why"] == "late"

    def test_keeps_both_annotators(self):
        anns = [annotation("a", "selector"), annotation("b", "reannotator")]
        assert len(latest_revisions(anns)) == 2


class TestPairedMoments:
    def test_requires_both_passes(self):
        only_first = record([annotation("a", "selector", situation=situation())])
        assert paired_moments([only_first]) == []

    def test_pairs_first_and_second(self):
        r = record([
            annotation("a", "selector", situation=situation()),
            annotation("b", "reannotator", situation=situation()),
        ])
        [paired] = paired_moments([r])
        assert paired["first"]["annotator_id"] == "a"
        assert paired["second"]["annotator_id"] == "b"

    def test_drops_retracted_moments(self):
        r = record([
            annotation("a", "selector", situation=situation()),
            annotation("b", "reannotator", situation=situation()),
        ], status="retracted")
        assert paired_moments([r]) == []

    def test_excluded_annotator_can_break_a_pair(self):
        r = record([
            annotation("a", "selector", situation=situation()),
            annotation("b", "reannotator", situation=situation()),
        ])
        assert paired_moments([r], excluded={"b"}) == []

    def test_ignores_no_key_moment_reports(self):
        assert paired_moments([{"no_key_moments_record": {"annotator_id": "a"}}]) == []


class TestBuildCases:
    def _cases(self, first_payload, second_payload):
        r = record([
            annotation("a", "selector", **first_payload),
            annotation("b", "reannotator", **second_payload),
        ])
        return {c["construct"]: c for c in build_cases(paired_moments([r]))}

    def test_identical_judgments_agree(self):
        cases = self._cases(
            {"situation": situation(), "action": action(), "result": result()},
            {"situation": situation(why="different prose"), "action": action(),
             "result": result()},
        )
        assert {c["outcome"] for c in cases.values()} == {"agreement"}

    def test_free_text_alone_does_not_make_a_disagreement(self):
        cases = self._cases(
            {"situation": situation(why="one")}, {"situation": situation(why="two")}
        )
        assert cases["situation"]["outcome"] == "agreement"

    def test_differing_field_is_a_disagreement(self):
        cases = self._cases(
            {"result": result(success="meaningful_success")},
            {"result": result(success="no_success")},
        )
        assert cases["result"]["outcome"] == "disagreement"
        flags = {f["name"]: f["agrees"] for f in cases["result"]["first"]["fields"]}
        assert flags == {"problem_success": False, "cognitive_engagement": True}

    def test_constructs_are_judged_independently(self):
        cases = self._cases(
            {"situation": situation(), "result": result()},
            {"situation": situation(), "result": result(engagement="low")},
        )
        assert cases["situation"]["outcome"] == "agreement"
        assert cases["result"]["outcome"] == "disagreement"

    def test_thrown_out_second_pass_is_not_a_disagreement(self):
        cases = self._cases({"situation": situation()}, {"situation": None})
        case = cases["situation"]
        assert case["outcome"] == "no second judgment"
        # Nothing to differ from, so no field is flagged as differing.
        assert all(f["agrees"] for f in case["first"]["fields"])

    def test_thrown_out_pass_is_marked_unjudged_not_merely_silent(self):
        # An empty panel would otherwise read as "reviewed it and had no comment".
        cases = self._cases({"situation": situation()}, {"situation": None})
        assert cases["situation"]["second"]["judged"] is False
        assert cases["situation"]["first"]["judged"] is True

    def test_a_judgment_without_prose_is_still_judged(self):
        cases = self._cases({"situation": situation()}, {"situation": situation(why="")})
        assert cases["situation"]["second"]["judged"] is True
        assert cases["situation"]["second"]["text"] == ""

    def test_first_pass_without_a_judgment_yields_no_case(self):
        cases = self._cases({"situation": None}, {"situation": situation()})
        assert "situation" not in cases

    def test_multi_selects_mark_shared_and_unshared_boxes(self):
        cases = self._cases(
            {"action": action(scaffolding_strategies=["guiding_questions", "re_explain"])},
            {"action": action(scaffolding_strategies=["guiding_questions"])},
        )
        [group] = cases["action"]["first"]["multi"]
        assert group["caption"] == "scaffolding strategies"
        assert {b["box"]: b["shared"] for b in group["boxes"]} == {
            "guiding_questions": True,
            "re_explain": False,
        }

    def test_multi_selects_do_not_decide_agreement(self):
        cases = self._cases(
            {"action": action(scaffolding_strategies=["guiding_questions"])},
            {"action": action(scaffolding_strategies=["co_solve"])},
        )
        assert cases["action"]["outcome"] == "agreement"

    def test_other_free_text_joins_its_checkbox_group(self):
        cases = self._cases(
            {"action": action(scaffolding_strategy_other="improvised")}, {"action": action()}
        )
        boxes = [b["box"] for b in cases["action"]["first"]["multi"][0]["boxes"]]
        assert 'other: "improvised"' in boxes


class TestMomentTurns:
    def _turns(self, count):
        return {n: {"turn_number": n, "role": "Tutor", "text": f"turn {n}"}
                for n in range(1, count + 1)}

    def test_uncapped_by_default(self):
        moment = {"start_turn": 1, "cut_turn": 50, "end_turn": 100}
        shown, before, after, missing = moment_turns(self._turns(100), moment)
        assert len(shown) == 100
        assert (before, after, missing) == (0, 0, 0)

    def test_short_span_is_shown_whole(self):
        moment = {"start_turn": 3, "cut_turn": 4, "end_turn": 7}
        shown, before, after, missing = moment_turns(self._turns(20), moment, max_turns=24)
        assert shown == [3, 4, 5, 6, 7]
        assert (before, after, missing) == (0, 0, 0)

    def test_long_span_is_windowed_on_the_cut_turn(self):
        moment = {"start_turn": 1, "cut_turn": 50, "end_turn": 100}
        shown, before, after, missing = moment_turns(self._turns(100), moment, max_turns=10)
        assert len(shown) == 10
        assert moment["cut_turn"] in shown
        assert before + len(shown) + after == 100
        assert missing == 0

    def test_turns_past_the_end_are_counted_not_hidden(self):
        moment = {"start_turn": 8, "cut_turn": 9, "end_turn": 15}
        shown, _, _, missing = moment_turns(self._turns(10), moment, max_turns=24)
        assert shown == [8, 9, 10]
        assert missing == 5

    def test_span_entirely_past_the_end(self):
        moment = {"start_turn": 30, "cut_turn": 31, "end_turn": 34}
        shown, _, _, missing = moment_turns(self._turns(10), moment, max_turns=24)
        assert shown == []
        assert missing == 5


class TestBuildSite:
    @pytest.fixture
    def paths(self, tmp_path):
        annotations = tmp_path / "annotations.jsonl"
        annotations.write_text(json.dumps(record([
            annotation("a", "selector", situation=situation(), action=action(), result=result()),
            annotation("b", "reannotator", situation=situation(rigor=True), action=action(),
                       result=result()),
        ])) + "\n")
        transcripts = tmp_path / "transcripts.jsonl"
        transcripts.write_text(
            json.dumps(transcript([row(n, f"row {n}") for n in range(25)])) + "\n")
        return annotations, transcripts, tmp_path / "out" / "index.html"

    def test_writes_a_self_contained_page(self, paths):
        annotations, transcripts, out = paths
        payload = build_site(annotations, transcripts, out)
        html = out.read_text()
        assert "__VIEWER_DATA__" not in html
        assert len(payload["cases"]) == 3
        assert payload["annotators"] == ["a", "b"]
        assert payload["moments"]["m1"]["found"] is True

    def test_transcript_text_cannot_close_the_script_block(self, tmp_path, paths):
        annotations, _, out = paths
        transcripts = tmp_path / "hostile.jsonl"
        # A row inside the moment's span (10-20) whose text would end the data block.
        transcripts.write_text(
            json.dumps(transcript([row(12, "</script><b>pwned</b>")])) + "\n")
        build_site(annotations, transcripts, out)

        # The browser ends the block at the first "</script>", so parsing the same way it does
        # must still yield the whole payload with the hostile text intact inside it.
        block = out.read_text().split('<script id="data" type="application/json">')[1]
        data = json.loads(block.split("</script>")[0])
        assert data["moments"]["m1"]["turns"][0]["text"] == "</script><b>pwned</b>"

    def test_missing_transcript_is_reported_not_crashed(self, tmp_path, paths):
        annotations, _, out = paths
        transcripts = tmp_path / "other.jsonl"
        transcripts.write_text(json.dumps(transcript([], "somewhere-else")) + "\n")
        payload = build_site(annotations, transcripts, out)
        assert payload["moments"]["m1"]["found"] is False


class TestTranscriptIndex:
    def test_reads_only_the_wanted_lines(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text("\n".join(
            json.dumps(transcript([row(1, f"hello {i}")], f"t{i}")) for i in range(5)
        ) + "\n")
        index = TranscriptIndex(path, {"t1", "t3"})
        assert len(index) == 2
        assert "t1" in index and "t0" not in index
        assert index.turns("t3")[1]["text"] == "hello 3"


class TestSinglePassMoments:
    def _records(self):
        paired = record([
            annotation("a", "selector", situation=situation()),
            annotation("b", "reannotator", situation=situation()),
        ], moment_id="paired")
        alone = record([annotation("c", "selector", situation=situation())],
                       moment_id="alone", status="selected")
        return [paired, alone]

    def test_excluded_by_default(self):
        assert {r["moment"]["id"] for r in annotated_moments(self._records())} == {"paired"}

    def test_included_on_request(self):
        kept = annotated_moments(self._records(), include_single_pass=True)
        assert {r["moment"]["id"] for r in kept} == {"paired", "alone"}
        assert next(r for r in kept if r["moment"]["id"] == "alone")["second"] is None

    def test_single_pass_case_has_no_second_side(self):
        kept = annotated_moments(self._records(), include_single_pass=True)
        cases = {c["moment_id"]: c for c in build_cases(kept) if c["construct"] == "situation"}
        assert cases["alone"]["outcome"] == "single pass"
        assert cases["alone"]["second"] is None
        # Nothing to compare against, so the only pass is never flagged as differing.
        assert all(f["agrees"] for f in cases["alone"]["first"]["fields"])

    def test_single_pass_is_distinct_from_a_thrown_out_review(self):
        thrown_out = record([
            annotation("a", "selector", situation=situation()),
            annotation("b", "reannotator", situation=None),
        ], moment_id="thrown")
        kept = annotated_moments(self._records() + [thrown_out], include_single_pass=True)
        outcomes = {c["moment_id"]: c["outcome"]
                    for c in build_cases(kept) if c["construct"] == "situation"}
        assert outcomes["alone"] == "single pass"
        assert outcomes["thrown"] == "no second judgment"

    def test_annotators_with_no_reviewed_work_still_appear(self, tmp_path):
        annotations = tmp_path / "a.jsonl"
        annotations.write_text("\n".join(json.dumps(r) for r in self._records()) + "\n")
        transcripts = tmp_path / "t.jsonl"
        transcripts.write_text(
            json.dumps(transcript([row(n, f"row {n}") for n in range(25)])) + "\n")
        out = tmp_path / "index.html"
        assert build_site(annotations, transcripts, out)["annotators"] == ["a", "b", "c"]
        pairs_only = build_site(annotations, transcripts, out, include_single_pass=False)
        assert pairs_only["annotators"] == ["a", "b"]


class TestRowSchema:
    def _index(self, tmp_path, rows):
        path = tmp_path / "t.jsonl"
        path.write_text(json.dumps(transcript(rows)) + "\n")
        return TranscriptIndex(path, {"t1"})

    def test_rows_are_keyed_by_index_not_turn_number(self, tmp_path):
        # Enrichments occupy indices too, so index and turn_number diverge -- keying on the
        # wrong one is exactly what mis-locates a moment.
        rows = [row(0), row(1, kind="[PAUSE]"), row(2, "after the pause")]
        turns = self._index(tmp_path, rows).turns("t1")
        assert sorted(turns) == [0, 1, 2]
        assert turns[2]["text"] == "after the pause"

    def test_speaker_prefix_is_stripped_from_text(self, tmp_path):
        turns = self._index(tmp_path, [row(0, "Hi there", role="STUDENT")]).turns("t1")
        assert turns[0]["text"] == "Hi there"
        assert turns[0]["role"] == "Student"

    def test_enrichment_rows_are_marked_and_labelled_by_type(self, tmp_path):
        turns = self._index(tmp_path, [row(0, "The screen froze", kind="[SCREEN UPDATE]")]).turns("t1")
        assert turns[0]["enrichment"] is True
        assert turns[0]["role"] == "screen update"
        assert turns[0]["text"] == "The screen froze"

    def test_dialogue_rows_are_not_marked_as_enrichments(self, tmp_path):
        turns = self._index(tmp_path, [row(0)]).turns("t1")
        assert turns[0]["enrichment"] is False

    def test_export_without_index_is_rejected(self, tmp_path):
        path = tmp_path / "old.jsonl"
        path.write_text(json.dumps({
            "transcript_id": "t1",
            "turns": [{"turn_number": 1, "role": "Tutor", "text": "hi"}],
        }) + "\n")
        index = TranscriptIndex(path, {"t1"})
        with pytest.raises(ValueError, match="no 'index'"):
            index.turns("t1")


class TestActionTags:
    def _tags(self, first_action, second_action=None):
        anns = [annotation("a", "selector", action=first_action)]
        if second_action is not None:
            anns.append(annotation("b", "reannotator", action=second_action))
        [r] = annotated_moments([record(anns)], include_single_pass=True)
        return set(action_tags(r))

    def test_amounts_and_strategies_are_tagged(self):
        tags = self._tags(action(scaffolding_amount="over_scaffolding",
                                 scaffolding_strategies=["hint", "co_solve"]))
        assert "scaffolding_amount:over_scaffolding" in tags
        assert {"scaffolding_strategies:hint", "scaffolding_strategies:co_solve"} <= tags

    def test_tags_are_namespaced_by_field(self):
        # "unclear" is a value of both amount fields; the field prefix keeps them apart.
        tags = self._tags(action(scaffolding_amount="unclear", rigor_present=True,
                                 rigor_amount="unclear"))
        assert tags >= {"scaffolding_amount:unclear", "rigor_amount:unclear"}

    def test_either_pass_contributes(self):
        tags = self._tags(action(rigor_amount="weak"), action(rigor_amount="strong"))
        assert {"rigor_amount:weak", "rigor_amount:strong"} <= tags

    def test_single_pass_moment_has_no_second_side_to_read(self):
        assert self._tags(action(rigor_amount="strong")) >= {"rigor_amount:strong"}

    def test_empty_and_null_values_are_not_tagged(self):
        tags = self._tags(action(rigor_amount=None, rigor_strategies=[],
                                 scaffolding_amount="appropriate"))
        assert tags == {"scaffolding_amount:appropriate",
                        "scaffolding_strategies:guiding_questions"}

    def test_a_thrown_out_review_contributes_nothing(self):
        tags = self._tags(action(scaffolding_amount="appropriate"), None)
        assert all(t.startswith(("scaffolding_amount", "scaffolding_strategies")) for t in tags)


class TestActionFilterOptions:
    def test_options_are_grouped_counted_and_ordered(self, tmp_path):
        records = [
            record([annotation("a", "selector", action=action(scaffolding_amount="over_scaffolding"))],
                   moment_id="m1", status="selected"),
            record([annotation("b", "selector", action=action(scaffolding_amount="over_scaffolding"))],
                   moment_id="m2", status="selected"),
            record([annotation("c", "selector", action=action(scaffolding_amount="appropriate"))],
                   moment_id="m3", status="selected"),
        ]
        annotations = tmp_path / "a.jsonl"
        annotations.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        transcripts = tmp_path / "t.jsonl"
        transcripts.write_text(
            json.dumps(transcript([row(n, f"row {n}") for n in range(25)])) + "\n")
        payload = build_site(annotations, transcripts, tmp_path / "index.html")

        groups = {g["caption"]: g["options"] for g in payload["action_filters"]}
        assert groups["Scaffolding amount"][0] == {
            "tag": "scaffolding_amount:over_scaffolding", "label": "over scaffolding", "count": 2}
        # Only values actually present are offered, so no filter can come back empty.
        assert "Rigor amount" not in groups
