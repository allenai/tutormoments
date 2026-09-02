"""Tests for the annotation review site's axes, moment building and payload assembly.

The page recomputes filters and analytics in the browser, but only from what these
modules put in the payload: the coarse axes each pass reduces to, the rater-vs-rater
value pairs kappa is pooled from, and where a moment sits once a reannotator has moved
it. Those are what is tested here.
"""

import json

import pytest

from tutormoments_build.annotation_viewer import (
    AXIS_KEYS,
    REPORTED_AXIS_KEYS,
    TranscriptIndex,
    annotator_roster,
    axis_pairs,
    build_moments,
    build_payload,
    build_site,
    coarse_axes,
    diff_axes,
    latest_passes,
    locate_span,
    no_key_moment_verdicts,
    sar_of,
)


def annotation(
    annotator,
    role,
    *,
    revision=1,
    created_at="2026-08-01T00:00:00Z",
    name=None,
    **payload,
):
    """One saved pass. `name` is left off unless asked for: exports predating the
    name field are the case worth having as the default."""
    row = {
        "annotator_id": annotator,
        "role": role,
        "revision": revision,
        "created_at": created_at,
        "is_test": 0,
        "payload": payload,
    }
    if name is not None:
        row["annotator_name"] = name
    return row


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


def judged(**over):
    """A complete pass: all three constructs recorded."""
    return {"situation": situation(), "action": action(), "result": result(), **over}


def row(index, text="hello", *, role="TUTOR", kind="dialogue", turn_number=None):
    """One rendered transcript row, shaped like the v2 transcripts export.

    Enrichment rows carry no turn number: they sit between turns, which is exactly why
    row positions and turn numbers cannot be read as each other.
    """
    prefix = kind if kind != "dialogue" else f"[{role}]"
    return {
        "index": index,
        "role": role,
        "text": f"{prefix} {text}",
        "type": kind,
        "timestamp": "00:01-00:02",
        "turn_number": turn_number if kind == "dialogue" else None,
    }


#: Enrichment rows placed before the first dialogue turn, so every fixture's row
#: positions run this far ahead of its turn numbers. Never zero: numberings that
#: happened to coincide would hide every confusion between the two.
ENRICHMENTS = 2


def rendered_rows(last_turn=30):
    """A transcript's rows as the pane draws them: enrichments, then dialogue."""
    rows = [row(i, "screen", kind="[SCREEN UPDATE]") for i in range(ENRICHMENTS)]
    return rows + [
        row(ENRICHMENTS + n - 1, f"turn {n}", turn_number=n)
        for n in range(1, last_turn + 1)
    ]


def transcript(rows, transcript_id="t1", conversation_id="c1"):
    return {
        "transcript_id": transcript_id,
        "conversation_id": conversation_id,
        "turns": rows,
    }


def record(
    annotations,
    *,
    moment_id="m1",
    status="reannotated",
    start=10,
    cut=12,
    end=20,
    transcript_id="t1",
):
    """One exported moment. `start`/`cut`/`end` are dialogue turn numbers, which is all
    the export records -- where they land in the rendered rows is the transcript's to
    say."""
    return {
        "transcript_id": transcript_id,
        "moment": {
            "moment_id": moment_id,
            "status": status,
            "start_turn": start,
            "cut_turn": cut,
            "end_turn": end,
            "created_by": "ann",
            "created_by_name": "Jessica",
            "created_at": "2026-08-01T00:00:00Z",
            "is_test": 0,
        },
        "annotations": annotations,
    }


def no_key_moments(
    annotator="ann", *, note="nothing to see", transcript_id="t9", name=None
):
    record = {
        "transcript_id": transcript_id,
        "no_key_moments_record": {
            "annotator_id": annotator,
            "role": "selector",
            "revision": 1,
            "created_at": "2026-08-02T00:00:00Z",
            "is_test": 0,
            "payload": {"no_key_moments": True, "note": note},
        },
    }
    if name is not None:
        record["no_key_moments_record"]["annotator_name"] = name
    return record


class TestSarOf:
    def test_a_thrown_out_review_recorded_no_judgment(self):
        assert sar_of("reannotator", {"meta": {"throw_out": True}}) is None

    def test_an_adjudicator_judgment_lives_under_final(self):
        payload = {"final": {"situation": situation()}, "meta": {}}
        assert (
            sar_of("adjudicator", payload)["situation"]["scaffolding_appropriate"]
            is True
        )

    def test_a_partial_judgment_still_counts(self):
        assert sar_of("selector", {"situation": situation()})["action"] is None


class TestCoarseAxes:
    def test_reduces_a_judgment_to_booleans(self):
        axes = coarse_axes(judged())
        assert axes["scaffolding_appropriate"] is True
        assert axes["rigor_appropriate"] is False
        assert axes["meaningful_success"] is True
        assert axes["high_engagement"] is True

    def test_over_scaffolding_is_unjudged_when_no_scaffolding_was_seen(self):
        axes = coarse_axes(
            judged(action=action(scaffolding_present=False, scaffolding_amount=None))
        )
        assert axes["scaffolding_present"] is False
        assert axes["over_scaffolding"] is None

    def test_over_scaffolding_is_the_amount_the_rater_chose(self):
        axes = coarse_axes(judged(action=action(scaffolding_amount="over_scaffolding")))
        assert axes["over_scaffolding"] is True

    def test_no_judgment_leaves_every_axis_unjudged(self):
        assert coarse_axes(None) == dict.fromkeys(AXIS_KEYS)


class TestDiffAxes:
    def test_finds_the_axes_two_raters_judged_differently(self):
        first = coarse_axes(judged())
        second = coarse_axes(judged(situation=situation(rigor=True)))
        assert diff_axes(first, second) == {"rigor_appropriate"}

    def test_over_scaffolding_is_skipped_unless_both_saw_scaffolding(self):
        saw = coarse_axes(judged(action=action(scaffolding_amount="over_scaffolding")))
        did_not = coarse_axes(
            judged(action=action(scaffolding_present=False, scaffolding_amount=None))
        )
        # They differ about whether scaffolding happened at all, which is one disagreement --
        # not two, with "was there too much of it" counted a second time.
        assert diff_axes(saw, did_not) == {"scaffolding_present"}

    def test_an_unjudged_axis_is_not_a_disagreement(self):
        assert diff_axes(coarse_axes(judged()), coarse_axes(None)) == set()

    def test_an_unreported_axis_is_still_a_disagreement(self):
        # The result axes get no row in the agreement table, but they are still compared:
        # a card whose only split is on engagement must still read as a disagreement.
        first = coarse_axes(judged())
        second = coarse_axes(judged(result=result(engagement="low")))
        assert diff_axes(first, second) == {"high_engagement"}
        assert "high_engagement" not in REPORTED_AXIS_KEYS


class TestAxisPairs:
    def test_pairs_only_the_axes_both_raters_judged(self):
        saw = coarse_axes(judged(action=action(scaffolding_amount="over_scaffolding")))
        did_not = coarse_axes(
            judged(action=action(scaffolding_present=False, scaffolding_amount=None))
        )
        pairs = axis_pairs(saw, did_not)
        assert "over_scaffolding" not in pairs
        assert pairs["scaffolding_present"] == [True, False]


def whos_who(passes):
    """Each pass as (role, rater), which is what the columns of a card come down to."""
    return [(p["role"], p["annotator_id"]) for p in passes]


class TestLatestPasses:
    def test_keeps_the_highest_revision_of_each_rater(self):
        passes = latest_passes(
            [
                annotation(
                    "a", "selector", revision=1, situation=situation(why="first")
                ),
                annotation(
                    "a", "selector", revision=2, situation=situation(why="second")
                ),
                annotation("b", "reannotator", revision=1),
            ]
        )
        assert passes[0]["payload"]["situation"]["why"] == "second"
        assert whos_who(passes) == [("selector", "a"), ("reannotator", "b")]

    def test_breaks_ties_on_save_time(self):
        passes = latest_passes(
            [
                annotation(
                    "a",
                    "selector",
                    created_at="2026-08-01T00:00:00Z",
                    situation=situation(why="early"),
                ),
                annotation(
                    "a",
                    "selector",
                    created_at="2026-08-02T00:00:00Z",
                    situation=situation(why="late"),
                ),
            ]
        )
        assert passes[0]["payload"]["situation"]["why"] == "late"

    def test_two_adjudicators_stand_side_by_side_in_the_order_they_ruled(self):
        # Keying on the role alone made the later adjudicator replace the earlier one,
        # which took a real second opinion out of the page without saying so.
        passes = latest_passes(
            [
                annotation("tara", "adjudicator", created_at="2026-08-05T00:00:00Z"),
                annotation("a", "selector"),
                annotation("erika", "adjudicator", created_at="2026-08-03T00:00:00Z"),
            ]
        )
        assert whos_who(passes) == [
            ("selector", "a"),
            ("adjudicator", "erika"),
            ("adjudicator", "tara"),
        ]

    def test_one_adjudicators_revisions_do_not_displace_the_other(self):
        passes = latest_passes(
            [
                annotation(
                    "erika",
                    "adjudicator",
                    revision=1,
                    created_at="2026-08-03T00:00:00Z",
                    rationale="first go",
                ),
                annotation(
                    "erika",
                    "adjudicator",
                    revision=2,
                    created_at="2026-08-06T00:00:00Z",
                    rationale="revised",
                ),
                annotation("tara", "adjudicator", created_at="2026-08-05T00:00:00Z"),
            ]
        )
        assert whos_who(passes) == [("adjudicator", "erika"), ("adjudicator", "tara")]
        assert passes[0]["payload"]["rationale"] == "revised"

    def test_ignores_roles_the_viewer_does_not_render(self):
        assert latest_passes([annotation("a", "spectator")]) == []


class TestBuildMoments:
    def test_pairs_the_passes_and_calls_the_outcome(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation(
                            "b",
                            "reannotator",
                            **judged(situation=situation(rigor=True)),
                        ),
                    ]
                )
            ]
        )
        assert len(moments) == 1
        moment = moments[0]
        assert [p["role"] for p in moment["passes"]] == ["selector", "reannotator"]
        assert moment["outcome"] == "disagreement"
        assert moment["diff"] == ["rigor_appropriate"]
        assert moment["pairs"]["rigor_appropriate"] == [False, True]

    def test_matching_judgments_agree(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation("b", "reannotator", **judged()),
                    ]
                )
            ]
        )
        assert moments[0]["outcome"] == "agreement"
        assert moments[0]["diff"] == []

    def test_a_single_pass_has_nothing_to_compare(self):
        moments = build_moments(
            [record([annotation("a", "selector", **judged())], status="selected")]
        )
        assert moments[0]["outcome"] == "single pass"
        assert moments[0]["pairs"] == {}

    def test_a_thrown_out_review_is_not_an_agreement(self):
        # The reviewer left no judgment, so there is nothing to agree with -- counting it
        # as agreement would inflate every rate on the analytics panel.
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation(
                            "b",
                            "reannotator",
                            meta={"throw_out": True, "throw_out_reason": "off task"},
                        ),
                    ]
                )
            ]
        )
        assert moments[0]["outcome"] == "thrown out on review"
        assert moments[0]["thrown_out"] is True
        assert moments[0]["pairs"] == {}
        assert moments[0]["passes"][1]["axes"] is None

    def test_retracted_moments_are_dropped(self):
        assert (
            build_moments(
                [record([annotation("a", "selector", **judged())], status="retracted")]
            )
            == []
        )

    def test_excluded_annotators_are_dropped_with_their_passes(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation("b", "reannotator", **judged()),
                    ]
                )
            ],
            excluded={"b"},
        )
        assert [p["annotator_id"] for p in moments[0]["passes"]] == ["a"]
        assert moments[0]["outcome"] == "single pass"

    def test_a_moment_whose_every_pass_was_excluded_disappears(self):
        assert (
            build_moments(
                [record([annotation("a", "selector", **judged())])], excluded={"a"}
            )
            == []
        )

    def test_a_redrawn_span_is_where_the_moment_now_sits(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation(
                            "b",
                            "reannotator",
                            **judged(
                                meta={
                                    "changed_boundaries": True,
                                    "new_start_turn": 8,
                                    "new_end_turn": 18,
                                    "redrew_cut_point": True,
                                    "new_cut_turn": 11,
                                }
                            ),
                        ),
                    ]
                )
            ]
        )
        assert moments[0]["boundaries"] == {
            "start_turn": 8,
            "end_turn": 18,
            "cut_turn": 11,
        }
        assert moments[0]["original_boundaries"] == {
            "start_turn": 10,
            "end_turn": 20,
            "cut_turn": 12,
        }

    def test_an_untouched_span_has_no_earlier_version(self):
        moments = build_moments([record([annotation("a", "selector", **judged())])])
        assert moments[0]["boundaries"] == {
            "start_turn": 10,
            "end_turn": 20,
            "cut_turn": 12,
        }
        assert moments[0]["original_boundaries"] is None

    def test_an_edge_the_reannotator_left_alone_keeps_its_own_number(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation(
                            "b",
                            "reannotator",
                            **judged(
                                meta={"redrew_cut_point": True, "new_cut_turn": 11}
                            ),
                        )
                    ]
                )
            ]
        )
        b = moments[0]["boundaries"]
        assert (b["start_turn"], b["cut_turn"], b["end_turn"]) == (10, 11, 20)

    def test_carries_the_dates_the_filters_run_on(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation(
                            "a",
                            "selector",
                            created_at="2026-08-03T18:28:56Z",
                            **judged(),
                        ),
                    ]
                )
            ]
        )
        assert moments[0]["passes"][0]["date"] == "2026-08-03"

    def test_the_moment_is_credited_to_the_cutter_by_name(self):
        moments = build_moments([record([annotation("a", "selector", **judged())])])
        assert moments[0]["created_by"] == "Jessica"

    def test_falls_back_to_the_id_when_the_export_carries_no_name(self):
        row = record([annotation("a", "selector", **judged())])
        del row["moment"]["created_by_name"]
        assert build_moments([row])[0]["created_by"] == "ann"

    def test_a_pass_is_bylined_with_the_raters_name(self):
        moments = build_moments(
            [record([annotation("a", "selector", name="Ada", **judged())])]
        )
        assert moments[0]["passes"][0]["annotator_name"] == "Ada"

    def test_an_unnamed_rater_is_bylined_with_their_id(self):
        moments = build_moments([record([annotation("a", "selector", **judged())])])
        assert moments[0]["passes"][0]["annotator_name"] == "a"


class TestAdjudicatorPasses:
    """An adjudicator files their judgment under `final`, so a pass shape that reads
    the payload root finds nothing on them and renders an empty column."""

    def _adjudicated(self, **payload):
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation("b", "reannotator", **judged()),
                        annotation("c", "adjudicator", **payload),
                    ]
                )
            ]
        )
        return {p["role"]: p for p in moments[0]["passes"]}["adjudicator"]

    def test_the_judgment_is_unwrapped_for_the_page(self):
        pass_ = self._adjudicated(
            final=judged(situation=situation(why="the adjudicator's reading")),
            meta={},
        )
        assert pass_["sar"]["situation"]["why"] == "the adjudicator's reading"
        assert pass_["sar"]["action"]["explanation"] == "did a thing"
        assert pass_["sar"]["result"]["explanation"] == "outcome"
        assert pass_["axes"]["scaffolding_present"] is True

    def test_observations_are_unwrapped_too(self):
        pass_ = self._adjudicated(
            final={**judged(), "other_observations": "worth noting"}, meta={}
        )
        assert pass_["observations"] == "worth noting"

    def test_a_selectors_judgment_is_left_where_it_is(self):
        moments = build_moments([record([annotation("a", "selector", **judged())])])
        pass_ = moments[0]["passes"][0]
        assert pass_["sar"]["situation"]["why"] == "because"
        assert pass_["observations"] == ""

    def test_how_they_resolved_the_two_passes_is_carried_through(self):
        decisions = {"situation": "agreed", "action": "reannotator"}
        pass_ = self._adjudicated(
            final=judged(), rationale="  took the second read on the action  ",
            decisions=decisions, meta={},
        )
        assert pass_["rationale"] == "took the second read on the action"
        assert pass_["decisions"] == decisions

    def test_no_rationale_recorded_is_empty_not_missing(self):
        pass_ = self._adjudicated(final=judged(), rationale="", meta={})
        assert pass_["rationale"] == ""
        assert pass_["decisions"] is None

    def test_an_adjudicator_who_threw_the_moment_out_recorded_no_judgment(self):
        pass_ = self._adjudicated(final={}, meta={"throw_out": True})
        assert pass_["sar"] is None
        assert pass_["axes"] is None

    def test_a_mixed_export_partitions_by_how_far_review_got(self):
        """What the page's adjudication filter slices on: whether a moment has an
        adjudicator pass at all, and whether that pass left labels or removed it."""
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation("c", "adjudicator", final=judged(), meta={}),
                    ],
                    moment_id="labelled",
                ),
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation(
                            "c", "adjudicator", final={}, meta={"throw_out": True}
                        ),
                    ],
                    moment_id="thrown",
                ),
                record([annotation("a", "selector", **judged())], moment_id="untouched"),
            ]
        )
        adjudicator = {
            m["id"]: next(
                (p for p in m["passes"] if p["role"] == "adjudicator"), None
            )
            for m in moments
        }
        assert adjudicator["untouched"] is None
        assert adjudicator["labelled"]["axes"] is not None
        assert adjudicator["thrown"]["axes"] is None

    def test_a_re_saving_adjudicator_is_one_adjudicator_not_two(self):
        """What the page's singly/doubly adjudicated counts slice on: how many
        adjudicators ruled, which is a count of raters and not of saved passes.

        Doubly adjudicated moments are the only place two raters answered the same
        questions in the same role, so they are what adjudicator agreement is measured
        on. Counting one rater's revision as a second adjudicator would put a moment
        nobody disagreed on into that set.
        """
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation(
                            "erika",
                            "adjudicator",
                            revision=1,
                            created_at="2026-08-03T00:00:00Z",
                            final=judged(),
                            meta={},
                        ),
                        annotation(
                            "erika",
                            "adjudicator",
                            revision=2,
                            created_at="2026-08-03T00:05:00Z",
                            final=judged(),
                            meta={},
                        ),
                    ],
                    moment_id="re-saved",
                ),
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation(
                            "erika",
                            "adjudicator",
                            created_at="2026-08-03T00:00:00Z",
                            final=judged(),
                            meta={},
                        ),
                        annotation(
                            "tara",
                            "adjudicator",
                            created_at="2026-08-04T00:00:00Z",
                            final=judged(),
                            meta={},
                        ),
                    ],
                    moment_id="two-raters",
                ),
            ]
        )
        adjudicators = {
            m["id"]: [p for p in m["passes"] if p["role"] == "adjudicator"]
            for m in moments
        }
        assert len(adjudicators["re-saved"]) == 1
        assert adjudicators["re-saved"][0]["revision"] == 2
        assert len(adjudicators["two-raters"]) == 2


class TestTwoAdjudicators:
    """Most adjudicated moments went to two adjudicators, so the card is four columns
    wide: the two passes, then each adjudicator's own call."""

    def _moments(self, second_adjudicator):
        return build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation("b", "reannotator", **judged()),
                        annotation(
                            "erika",
                            "adjudicator",
                            created_at="2026-08-03T00:00:00Z",
                            final=judged(),
                            rationale="kept the first read",
                            meta={},
                        ),
                        second_adjudicator,
                    ]
                )
            ]
        )

    def _tara(self, **payload):
        return annotation(
            "tara", "adjudicator", created_at="2026-08-04T00:00:00Z", **payload
        )

    def test_both_adjudicators_get_a_column_in_the_order_they_ruled(self):
        moment = self._moments(self._tara(final=judged(), meta={}))[0]
        assert whos_who(moment["passes"]) == [
            ("selector", "a"),
            ("reannotator", "b"),
            ("adjudicator", "erika"),
            ("adjudicator", "tara"),
        ]

    def test_each_adjudicator_keeps_their_own_judgment_and_rationale(self):
        # The point of showing both: where two adjudicators split, the page has to say
        # so rather than picking one of them to be the final call.
        moment = self._moments(
            self._tara(
                final=judged(situation=situation(rigor=True)),
                rationale="read the rigor the other way",
                meta={},
            )
        )[0]
        erika, tara = moment["passes"][2], moment["passes"][3]
        assert erika["axes"]["rigor_appropriate"] is False
        assert tara["axes"]["rigor_appropriate"] is True
        assert erika["rationale"] == "kept the first read"
        assert tara["rationale"] == "read the rigor the other way"

    def test_agreement_is_still_measured_between_the_first_two_passes(self):
        # Adjudication settles a moment; it is not a third rater to pool into kappa.
        moment = self._moments(
            self._tara(final=judged(situation=situation(rigor=True)), meta={})
        )[0]
        assert moment["outcome"] == "agreement"
        assert moment["diff"] == []
        assert moment["pairs"]["rigor_appropriate"] == [False, False]

    def test_either_adjudicator_can_mark_the_moment_for_removal(self):
        moment = self._moments(self._tara(final={}, meta={"throw_out": True}))[0]
        assert moment["thrown_out"] is True
        assert moment["passes"][3]["axes"] is None

    def test_the_second_adjudicator_can_be_the_one_who_moved_the_boundaries(self):
        moment = self._moments(
            self._tara(
                final=judged(), meta={"changed_boundaries": True, "new_end_turn": 18}
            )
        )[0]
        assert moment["boundaries"]["end_turn"] == 18
        assert moment["original_boundaries"]["end_turn"] == 20


class TestNoKeyMomentVerdicts:
    def test_reads_the_verdict_and_its_note(self):
        verdicts = no_key_moment_verdicts(
            [no_key_moments("erika", note="no math here", name="Erika")]
        )
        assert verdicts == [
            {
                "transcript_id": "t9",
                "annotator_id": "erika",
                "annotator_name": "Erika",
                "created_at": "2026-08-02T00:00:00Z",
                "date": "2026-08-02",
                "is_test": 0,
                "note": "no math here",
            }
        ]

    def test_excluded_annotators_are_dropped(self):
        assert (
            no_key_moment_verdicts([no_key_moments("erika")], excluded={"erika"}) == []
        )


class TestAnnotatorRoster:
    def test_counts_passes_by_role_and_verdicts(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation("b", "reannotator", **judged()),
                    ]
                )
            ]
        )
        roster = annotator_roster(
            moments, no_key_moment_verdicts([no_key_moments("b")])
        )
        by_id = {r["id"]: r for r in roster}
        assert by_id["a"]["selector"] == 1
        assert by_id["b"]["reannotator"] == 1
        assert by_id["b"]["no_key_moments"] == 1
        assert by_id["b"]["total"] == 2

    def test_carries_the_name_the_filter_menu_lists(self):
        moments = build_moments(
            [record([annotation("a", "selector", name="Ada", **judged())])]
        )
        roster = annotator_roster(moments, [])
        assert roster == [
            {
                "id": "a",
                "name": "Ada",
                "selector": 1,
                "reannotator": 0,
                "adjudicator": 0,
                "no_key_moments": 0,
                "total": 1,
                "is_test": 0,
            }
        ]

    def test_lists_people_in_name_order(self):
        moments = build_moments(
            [
                record(
                    [
                        annotation("z", "selector", name="Ada", **judged()),
                        annotation("a", "reannotator", name="Zoe", **judged()),
                    ]
                )
            ]
        )
        assert [r["name"] for r in annotator_roster(moments, [])] == ["Ada", "Zoe"]


class TestTranscriptIndex:
    @pytest.fixture
    def path(self, tmp_path):
        out = tmp_path / "t.jsonl"
        out.write_text(
            "\n".join(
                json.dumps(
                    transcript([row(1, f"hello {i}", turn_number=2)], f"t{i}", f"c{i}")
                )
                for i in range(5)
            )
            + "\n"
        )
        return out

    def test_reads_only_the_wanted_lines(self, path):
        index = TranscriptIndex(path, {"t1", "t3"})
        assert len(index) == 2
        assert "t1" in index and "t0" not in index
        assert index.rows("t3")[0]["text"] == "[TUTOR] hello 3"

    def test_counts_every_transcript_in_the_file(self, path):
        # The overview says how many transcripts were never annotated, which only the
        # whole file knows -- the wanted set is the annotated ones.
        assert TranscriptIndex(path, {"t1"}).total == 5

    def test_rows_keep_the_position_the_pane_draws_them_at(self, path):
        assert TranscriptIndex(path, {"t2"}).rows("t2")[0]["turn_index"] == 1

    def test_rows_also_keep_the_turn_number_the_annotator_saw(self, path):
        assert TranscriptIndex(path, {"t2"}).rows("t2")[0]["turn_number"] == 2

    def test_an_enrichment_row_has_no_turn_number(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(
            json.dumps(transcript([row(3, "screen", kind="[SCREEN UPDATE]")])) + "\n"
        )
        assert TranscriptIndex(path, {"t1"}).rows("t1")[0]["turn_number"] is None

    def test_turn_rows_span_every_row_a_turn_number_covers(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(json.dumps(transcript(rendered_rows(3))) + "\n")
        # Turn 1 is the third row drawn: two enrichments precede it.
        assert TranscriptIndex(path, {"t1"}).turn_rows("t1")[1] == (2, 2)

    def test_an_export_without_index_is_refused(self, tmp_path):
        path = tmp_path / "old.jsonl"
        turn = {
            "role": "TUTOR",
            "text": "[TUTOR] hi",
            "type": "dialogue",
            "turn_number": 1,
        }
        path.write_text(json.dumps(transcript([turn])) + "\n")
        index = TranscriptIndex(path, {"t1"})
        with pytest.raises(ValueError, match="carry no 'index'"):
            index.rows("t1")


class TestLocateSpan:
    """Turn numbers, which is all the export records, onto the rows the pane draws."""

    def turn_rows(self, rows):
        return {
            r["turn_number"]: (r["index"], r["index"])
            for r in rows
            if r["turn_number"] is not None
        }

    def test_resolves_each_edge_past_the_enrichments_before_it(self):
        span = {"start_turn": 10, "cut_turn": 12, "end_turn": 20}
        located = locate_span(self.turn_rows(rendered_rows()), span)
        assert located == {
            "start_row": 10 + ENRICHMENTS - 1,
            "cut_row": 12 + ENRICHMENTS - 1,
            "end_row": 20 + ENRICHMENTS - 1,
        }

    def test_a_turn_drawn_over_two_rows_is_kept_whole(self):
        # The v2 numbering sometimes gives two rows one number: the start opens on the
        # first of them and the cut and the end close on the last, so neither edge
        # slices a turn in half.
        turn_rows = {5: (7, 9)}
        span = {"start_turn": 5, "cut_turn": 5, "end_turn": 5}
        assert locate_span(turn_rows, span) == {
            "start_row": 7,
            "cut_row": 9,
            "end_row": 9,
        }

    def test_an_edge_naming_a_skipped_number_lands_inside_the_span(self):
        # Wherever that numbering repeats a number it skips the next, so an edge can
        # name a turn no row carries. It falls to the nearest row on its own side
        # rather than being dropped.
        turn_rows = {4: (4, 4), 5: (5, 6), 7: (7, 7)}
        span = {"start_turn": 6, "cut_turn": 6, "end_turn": 6}
        assert locate_span(turn_rows, span) == {
            "start_row": 7,
            "cut_row": 6,
            "end_row": 6,
        }

    def test_a_transcript_with_no_dialogue_places_nothing(self):
        span = {"start_turn": 10, "cut_turn": 12, "end_turn": 20}
        assert locate_span({}, span) == {
            "start_row": None,
            "cut_row": None,
            "end_row": None,
        }


class TestBuildPayload:
    @pytest.fixture
    def paths(self, tmp_path):
        annotations = tmp_path / "annotations.jsonl"
        annotations.write_text(
            "\n".join(
                [
                    json.dumps(
                        record(
                            [
                                annotation("a", "selector", **judged()),
                                annotation(
                                    "b",
                                    "reannotator",
                                    **judged(situation=situation(rigor=True)),
                                ),
                            ]
                        )
                    ),
                    json.dumps(
                        record(
                            [annotation("a", "selector", **judged())],
                            moment_id="m2",
                            status="selected",
                            start=2,
                            cut=3,
                            end=6,
                        )
                    ),
                    json.dumps(no_key_moments("c")),
                ]
            )
            + "\n"
        )
        transcripts = tmp_path / "transcripts.jsonl"
        transcripts.write_text(
            "\n".join(
                [
                    json.dumps(transcript(rendered_rows(23))),
                    json.dumps(
                        transcript([row(0, "elsewhere", turn_number=1)], "t9", "c9")
                    ),
                    json.dumps(
                        transcript(
                            [row(0, "unannotated", turn_number=1)],
                            "t-other",
                            "c-other",
                        )
                    ),
                ]
            )
            + "\n"
        )
        return annotations, transcripts, tmp_path / "out" / "index.html"

    def test_groups_moments_and_verdicts_under_their_transcripts(self, paths):
        annotations, transcripts, _ = paths
        payload = build_payload(annotations, transcripts)
        by_id = {t["transcript_id"]: t for t in payload["transcripts"]}
        assert by_id["t1"]["n_moments"] == 2
        assert by_id["t1"]["found"] is True
        # The annotations export keys everything by transcript; the id the annotation
        # tool showed comes off the transcripts file.
        assert by_id["t1"]["conversation_id"] == "c1"
        assert by_id["t9"]["conversation_id"] == "c9"
        assert len(by_id["t1"]["turns"]) == 25
        assert by_id["t9"]["n_no_key_moments"] == 1
        # The unannotated transcript is counted but not embedded: it has nothing to show
        # and the file it comes from runs to tens of megabytes.
        assert "t-other" not in by_id
        assert payload["transcripts_total"] == 3

    def test_a_missing_transcript_is_reported_not_crashed(self, tmp_path, paths):
        annotations, _, _ = paths
        transcripts = tmp_path / "other.jsonl"
        transcripts.write_text(
            json.dumps(transcript([], "somewhere-else", "c-else")) + "\n"
        )
        payload = build_payload(annotations, transcripts)
        assert {t["found"] for t in payload["transcripts"]} == {False}
        # With no transcript to name it, the transcript id stands in for the one the
        # tool showed, so the picker still has something to sort and label by.
        assert {t["conversation_id"] for t in payload["transcripts"]} == {"t1", "t9"}

    def test_lists_everyone_who_annotated(self, paths):
        annotations, transcripts, _ = paths
        payload = build_payload(annotations, transcripts)
        assert [a["id"] for a in payload["annotators"]] == ["a", "b", "c"]

    def test_places_each_moment_in_the_rows_the_pane_draws(self, paths):
        # The moment is recorded in turn numbers; the pane highlights row positions, and
        # they run ENRICHMENTS apart. Reading one as the other draws the wrong stretch.
        annotations, transcripts, _ = paths
        payload = build_payload(annotations, transcripts)
        moment = next(m for m in payload["moments"] if m["id"] == "m1")
        assert moment["boundaries"] == {
            "start_turn": 10,
            "cut_turn": 12,
            "end_turn": 20,
            "start_row": 10 + ENRICHMENTS - 1,
            "cut_row": 12 + ENRICHMENTS - 1,
            "end_row": 20 + ENRICHMENTS - 1,
        }

    def test_a_moment_whose_transcript_is_missing_is_placed_nowhere(
        self, tmp_path, paths
    ):
        annotations, _, _ = paths
        transcripts = tmp_path / "other.jsonl"
        transcripts.write_text(
            json.dumps(transcript([], "somewhere-else", "c-else")) + "\n"
        )
        payload = build_payload(annotations, transcripts)
        moment = next(m for m in payload["moments"] if m["id"] == "m1")
        assert moment["boundaries"]["start_turn"] == 10
        assert moment["boundaries"]["start_row"] is None


class TestBuildSite:
    @pytest.fixture
    def paths(self, tmp_path):
        annotations = tmp_path / "annotations.jsonl"
        annotations.write_text(
            json.dumps(
                record(
                    [
                        annotation("a", "selector", **judged()),
                        annotation("b", "reannotator", **judged()),
                    ]
                )
            )
            + "\n"
        )
        transcripts = tmp_path / "transcripts.jsonl"
        transcripts.write_text(json.dumps(transcript(rendered_rows(23))) + "\n")
        return annotations, transcripts, tmp_path / "out" / "index.html"

    def test_writes_a_self_contained_page(self, paths):
        annotations, transcripts, out = paths
        payload = build_site(annotations, transcripts, out)
        html = out.read_text()
        assert "__VIEWER_DATA__" not in html
        assert len(payload["moments"]) == 1
        assert '<script id="data" type="application/json">' in html

    def test_transcript_text_cannot_close_the_script_block(self, tmp_path, paths):
        annotations, _, out = paths
        transcripts = tmp_path / "hostile.jsonl"
        transcripts.write_text(
            json.dumps(transcript([row(12, "</script><b>pwned</b>", turn_number=13)]))
            + "\n"
        )
        build_site(annotations, transcripts, out)

        # The browser ends the block at the first "</script>", so parsing the same way it
        # does must still yield the whole payload with the hostile text intact inside it.
        block = out.read_text().split('<script id="data" type="application/json">')[1]
        data = json.loads(block.split("</script>")[0])
        assert (
            data["transcripts"][0]["turns"][0]["text"]
            == "[TUTOR] </script><b>pwned</b>"
        )
