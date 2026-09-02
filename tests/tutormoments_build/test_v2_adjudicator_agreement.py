"""Tests for Cohen's kappa between adjudicators."""

import json

import pytest

from tutormoments_build.v2 import adjudicator_agreement as A


def _thrown_out(name="Tara", reason="not a good moment"):
    """An adjudication that threw the moment out: no final call, so no labels."""
    return {
        "annotator_id": f"id-{name.lower()}",
        "annotator_name": name,
        "role": "adjudicator",
        "revision": 1,
        "payload": {
            "final": None,
            "rationale": "",
            "meta": {"throw_out": True, "throw_out_reason": reason},
        },
    }


def _payload(
    *,
    scaffolding_appropriate=False,
    rigor_appropriate=False,
    scaffolding_present=False,
    rigor_present=False,
    scaffolding_amount=None,
    why="",
    explanation="",
    result_explanation="",
    other_observations="",
):
    """The label and free-text shape every role fills in.

    An adjudicator answers the same questions as the annotators below them, one
    level down under ``final``, so both sides are built from this.
    """
    return {
        "situation": {
            "scaffolding_appropriate": scaffolding_appropriate,
            "rigor_appropriate": rigor_appropriate,
            "why": why,
        },
        "action": {
            "scaffolding_present": scaffolding_present,
            "rigor_present": rigor_present,
            "scaffolding_amount": scaffolding_amount,
            "explanation": explanation,
        },
        "result": {"explanation": result_explanation},
        "other_observations": other_observations,
    }


def _annotation(name, role="selector", *, revision=1, **fields):
    """A selector's or reannotator's pass: the payload at the top level."""
    return {
        "annotator_id": f"id-{name.lower()}",
        "annotator_name": name,
        "role": role,
        "revision": revision,
        "payload": _payload(**fields),
    }


def _adjudication(name="Tara", *, role="adjudicator", rationale="", **fields):
    return {
        "annotator_id": f"id-{name.lower()}",
        "annotator_name": name,
        "role": role,
        "revision": 1,
        "payload": {"final": _payload(**fields), "rationale": rationale},
    }


def _row(moment_id, annotations):
    return {
        "transcript_id": "t1",
        "moment": {"moment_id": moment_id, "status": "adjudicated"},
        "annotations": annotations,
    }


def _write(tmp_path, rows):
    path = tmp_path / "annotations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------


def test_perfect_agreement_is_one():
    pairs = [(True, True), (False, False), (True, True), (False, False)]
    assert A.cohens_kappa(pairs) == pytest.approx(1.0)


def test_chance_level_agreement_is_zero():
    # Each rater says True half the time and they line up half the time, which is
    # exactly what chance predicts.
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    assert A.cohens_kappa(pairs) == pytest.approx(0.0)


def test_systematic_disagreement_is_negative():
    pairs = [(True, False), (False, True), (True, False), (False, True)]
    assert A.cohens_kappa(pairs) < 0


def test_kappa_is_undefined_when_nobody_ever_says_true():
    # Full agreement, but on a single category: chance agreement is 1 and the
    # statistic has no denominator. Raw agreement, not kappa, is the number here.
    assert A.cohens_kappa([(False, False)] * 5) is None


def test_kappa_is_zero_when_one_rater_never_varies():
    # One rater says True throughout, so their labels carry no information and
    # every hit is chance -- distinct from the undefined case above.
    pairs = [(True, True), (True, False), (True, True), (True, False)]
    assert A.cohens_kappa(pairs) == pytest.approx(0.0)


def test_kappa_is_undefined_with_no_pairs():
    assert A.cohens_kappa([]) is None


# ---------------------------------------------------------------------------
# reading the export
# ---------------------------------------------------------------------------


def test_only_doubly_adjudicated_moments_are_kept(tmp_path):
    path = _write(
        tmp_path,
        [
            # no adjudicator at all
            _row("m1", [_adjudication("Tara", role="selector")]),
            # one adjudicator: nothing to compare
            _row("m2", [_adjudication("Tara")]),
            _row("m3", [_adjudication("Tara"), _adjudication("Erika")]),
            # a transcript with no key moments carries no "moment" key
            {"transcript_id": "t2", "no_key_moments_record": {}},
        ],
    )
    rows = A.doubly_adjudicated(path, {})
    assert [row["moment_id"] for row in rows] == ["m3"]


def test_selectors_and_reannotators_are_not_counted_as_adjudicators(tmp_path):
    path = _write(
        tmp_path,
        [
            _row(
                "m1",
                [
                    _adjudication("Paul", role="selector"),
                    _adjudication("Anita", role="reannotator"),
                    _adjudication("Tara"),
                ],
            )
        ],
    )
    assert A.doubly_adjudicated(path, {}) == []


def test_project_staff_are_excluded(tmp_path):
    path = _write(
        tmp_path,
        [_row("m1", [_adjudication("Tara"), _adjudication("Lucy")])],
    )
    assert A.doubly_adjudicated(path, {}) == []


def test_moments_with_three_adjudicators_are_skipped(tmp_path):
    path = _write(
        tmp_path,
        [
            _row(
                "m1",
                [
                    _adjudication("Tara"),
                    _adjudication("Erika"),
                    _adjudication("Amanda"),
                ],
            )
        ],
    )
    assert A.doubly_adjudicated(path, {}) == []


def test_raters_are_ordered_consistently_across_moments(tmp_path):
    # The export's order within a moment is arbitrary; kappa's chance term reads
    # each rater's marginal, so rater 1 has to be the same person every time.
    path = _write(
        tmp_path,
        [
            _row("m1", [_adjudication("Tara"), _adjudication("Erika")]),
            _row("m2", [_adjudication("Erika"), _adjudication("Tara")]),
        ],
    )
    rows = A.doubly_adjudicated(path, {"tara": "A05", "erika": "A18"})
    assert [row["raters"] for row in rows] == [("A05", "A18"), ("A05", "A18")]


def test_annotator_names_are_replaced_by_their_labels(tmp_path):
    path = _write(
        tmp_path, [_row("m1", [_adjudication("Tara"), _adjudication("Erika")])]
    )
    rows = A.doubly_adjudicated(path, {"tara": "A05", "erika": "A18"})
    assert rows[0]["raters"] == ("A05", "A18")


def test_labels_are_read_from_the_final_call(tmp_path):
    path = _write(
        tmp_path,
        [
            _row(
                "m1",
                [
                    _adjudication("Erika", rigor_present=True),
                    _adjudication("Tara", scaffolding_amount="over_scaffolding"),
                ],
            )
        ],
    )
    first, second = A.doubly_adjudicated(path, {"tara": "A05", "erika": "A18"})[0][
        "labels"
    ]
    assert first["rigor_present"] is False  # A05 (Tara) sorts first
    assert first["over_scaffolding_declared"] is True
    assert second["rigor_present"] is True
    assert second["over_scaffolding_declared"] is False


# ---------------------------------------------------------------------------
# thrown-out moments
# ---------------------------------------------------------------------------


def test_a_thrown_out_moment_has_no_labels_to_compare():
    # The trap: reading final=None as all-False would turn "this moment does not
    # belong in the benchmark" into five negative labels.
    labels = A.adjudicator_labels(_thrown_out())
    assert labels["throw_out"] is True
    assert all(labels[field] is None for field in A.FINAL_FIELDS)


def test_labelling_an_adjudication_is_not_throwing_it_out():
    assert A.adjudicator_labels(_adjudication())["throw_out"] is False


def test_a_thrown_out_moment_does_not_count_against_the_other_adjudicator(tmp_path):
    # One adjudicator threw the moment out, the other said the tutor scaffolded.
    # That is not a disagreement about scaffolding -- there is no second opinion
    # on scaffolding at all -- so the moment must not reach the label rows.
    path = _write(
        tmp_path,
        [
            _row(
                "m1",
                [_thrown_out("Tara"), _adjudication("Erika", scaffolding_present=True)],
            ),
            _row(
                "m2",
                [
                    _adjudication("Tara", scaffolding_present=True),
                    _adjudication("Erika", scaffolding_present=True),
                ],
            ),
        ],
    )
    results = A.score(A.doubly_adjudicated(path, {"tara": "A05", "erika": "A18"}))

    assert results["scaffolding_present"]["n"] == 1
    assert results["scaffolding_present"]["observed_agreement"] == pytest.approx(1.0)
    assert results["scaffolding_present"]["positives"] == [1, 1]


def test_throw_out_is_scored_on_every_moment(tmp_path):
    path = _write(
        tmp_path,
        [
            _row("m1", [_thrown_out("Tara"), _thrown_out("Erika")]),
            _row("m2", [_thrown_out("Tara"), _adjudication("Erika")]),
            _row("m3", [_adjudication("Tara"), _adjudication("Erika")]),
        ],
    )
    results = A.score(A.doubly_adjudicated(path, {"tara": "A05", "erika": "A18"}))

    assert results["throw_out"]["n"] == 3
    assert results["throw_out"]["positives"] == [2, 1]  # A05 twice, A18 once
    assert results["throw_out"]["observed_agreement"] == pytest.approx(2 / 3)
    # only m3 has two final calls behind it
    assert results["scaffolding_present"]["n"] == 1


def test_a_pair_who_threw_everything_out_has_no_label_rows(tmp_path):
    path = _write(tmp_path, [_row("m1", [_thrown_out("Tara"), _thrown_out("Erika")])])
    results = A.score(A.doubly_adjudicated(path, {"tara": "A05", "erika": "A18"}))

    assert results["scaffolding_present"]["n"] == 0
    assert results["scaffolding_present"]["kappa"] is None
    assert results["scaffolding_present"]["observed_agreement"] is None


def test_a_missing_amount_is_not_over_scaffolding():
    labels = A.adjudicator_labels(_adjudication(scaffolding_amount="appropriate"))
    assert labels["over_scaffolding_declared"] is False


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_score_reports_every_label_with_its_marginals(tmp_path):
    path = _write(
        tmp_path,
        [
            _row(
                "m1",
                [
                    _adjudication("Erika", scaffolding_present=True),
                    _adjudication("Tara", scaffolding_present=True),
                ],
            ),
            _row(
                "m2",
                [
                    _adjudication("Erika", scaffolding_present=True),
                    _adjudication("Tara"),
                ],
            ),
        ],
    )
    results = A.score(A.doubly_adjudicated(path, {"tara": "A05", "erika": "A18"}))
    assert set(results) == set(A.LABEL_FIELDS)

    scaffolding = results["scaffolding_present"]
    assert scaffolding["n"] == 2
    assert scaffolding["observed_agreement"] == pytest.approx(0.5)
    assert scaffolding["positives"] == [1, 2]  # A05 once, A18 twice
    # A18 called it True on both moments, so their labels carry no information
    # and the one hit is exactly what chance predicts.
    assert scaffolding["kappa"] == pytest.approx(0.0)

    assert results["rigor_present"]["positives"] == [0, 0]


def test_main_runs_and_writes_json(tmp_path, capsys):
    path = _write(
        tmp_path,
        [
            _row(
                f"m{i}",
                [
                    _adjudication("Erika", rigor_present=i % 2 == 0),
                    _adjudication("Tara", rigor_present=i % 2 == 0),
                ],
            )
            for i in range(4)
        ],
    )
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"tara": "A05", "erika": "A18"}), encoding="utf-8")
    out = tmp_path / "agreement.json"
    argv = [
        "--annotations",
        path,
        "--annotator-labels",
        str(labels),
        "--json-out",
        str(out),
    ]
    assert A.main(argv) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["n_moments"] == 4
    assert written["overall"]["rigor_present"]["kappa"] == pytest.approx(1.0)
    assert list(written["by_pair"]) == ["A05+A18"]
    assert "rigor_present" in capsys.readouterr().out


def test_main_reports_when_there_is_nothing_to_compare(tmp_path):
    path = _write(tmp_path, [_row("m1", [_adjudication("Tara")])])
    assert A.main(["--annotations", path, "--annotator-labels", "missing.json"]) == 1


# ---------------------------------------------------------------------------
# reading the pair below the adjudicator
# ---------------------------------------------------------------------------


def _adjudicated(moment_id, *, selector, reannotator, adjudicators):
    return _row(moment_id, [selector, reannotator, *adjudicators])


def test_latest_revision_of_a_role_wins():
    # The same person re-saving seconds later is one pass, not two annotators.
    annotations = [
        _annotation("Erin", revision=1, scaffolding_present=True),
        _annotation("Erin", revision=2, scaffolding_present=False),
    ]
    latest = A.latest_by_role(annotations, "selector")
    assert latest["revision"] == 2
    assert A.latest_by_role(annotations, "reannotator") is None


def test_every_adjudication_is_a_row_including_both_halves_of_a_pair(tmp_path):
    path = _write(
        tmp_path,
        [
            _adjudicated(
                "m1",
                selector=_annotation("Paul"),
                reannotator=_annotation("Anita", "reannotator"),
                adjudicators=[_adjudication("Tara"), _adjudication("Erika")],
            ),
            _adjudicated(
                "m2",
                selector=_annotation("Paul"),
                reannotator=_annotation("Anita", "reannotator"),
                adjudicators=[_adjudication("Tara")],
            ),
            _row("m3", [_annotation("Paul"), _annotation("Anita", "reannotator")]),
        ],
    )
    rows = A.adjudications(path, {"tara": "A05", "erika": "A18"})
    assert [(row["moment_id"], row["adjudicator"]) for row in rows] == [
        ("m1", "A05"),
        ("m1", "A18"),
        ("m2", "A05"),
    ]


def test_a_moment_with_no_pair_to_resolve_is_skipped(tmp_path):
    path = _write(
        tmp_path,
        [_row("m1", [_annotation("Paul"), _adjudication("Tara")])],
    )
    assert A.adjudications(path, {}) == []


def test_a_thrown_out_adjudication_is_a_row_with_no_final(tmp_path):
    path = _write(
        tmp_path,
        [
            _adjudicated(
                "m1",
                selector=_annotation("Paul"),
                reannotator=_annotation("Anita", "reannotator"),
                adjudicators=[_thrown_out("Tara")],
            )
        ],
    )
    assert A.adjudications(path, {})[0]["final"] is None


# ---------------------------------------------------------------------------
# how often the adjudicator backed the pair below them
# ---------------------------------------------------------------------------


def _resolution(tmp_path, *, selector, reannotator, adjudicator=None, **final):
    """Counts for one moment: the two annotators' passes and the call over them."""
    path = _write(
        tmp_path,
        [
            _adjudicated(
                "m1",
                selector=_annotation("Paul", **selector),
                reannotator=_annotation("Anita", "reannotator", **reannotator),
                adjudicators=[adjudicator or _adjudication("Tara", **final)],
            )
        ],
    )
    return A.resolution_counts(A.adjudications(path, {}))


def test_upholding_a_label_both_annotators_marked(tmp_path):
    counts = _resolution(
        tmp_path,
        selector={"rigor_present": True},
        reannotator={"rigor_present": True},
        rigor_present=True,
    )["rigor_present"]
    assert (counts["agreed"], counts["upheld"]) == (1, 1)
    assert (counts["agreed_yes"], counts["upheld_yes"]) == (1, 1)
    assert counts["agreed_no"] == 0
    assert counts["disagreed"] == 0


def test_upholding_a_label_neither_annotator_marked(tmp_path):
    counts = _resolution(
        tmp_path,
        selector={"rigor_present": False},
        reannotator={"rigor_present": False},
        rigor_present=False,
    )["rigor_present"]
    assert (counts["agreed"], counts["upheld"]) == (1, 1)
    assert (counts["agreed_no"], counts["upheld_no"]) == (1, 1)
    assert counts["agreed_yes"] == 0


def test_striking_a_label_both_annotators_marked(tmp_path):
    # An overturn is only visible in the yes half: pooling it with the no half
    # would hide which direction the adjudicator moves a lopsided label.
    counts = _resolution(
        tmp_path,
        selector={"scaffolding_appropriate": True},
        reannotator={"scaffolding_appropriate": True},
        scaffolding_appropriate=False,
    )["scaffolding_appropriate"]
    assert (counts["agreed"], counts["upheld"]) == (1, 0)
    assert (counts["agreed_yes"], counts["upheld_yes"]) == (1, 0)
    assert (counts["agreed_no"], counts["upheld_no"]) == (0, 0)


def test_adding_a_label_neither_annotator_marked(tmp_path):
    counts = _resolution(
        tmp_path,
        selector={"rigor_appropriate": False},
        reannotator={"rigor_appropriate": False},
        rigor_appropriate=True,
    )["rigor_appropriate"]
    assert (counts["agreed_no"], counts["upheld_no"]) == (1, 0)
    assert counts["upheld"] == 0


def test_picking_the_label_on_a_split(tmp_path):
    counts = _resolution(
        tmp_path,
        selector={"scaffolding_present": True},
        reannotator={"scaffolding_present": False},
        scaffolding_present=True,
    )["scaffolding_present"]
    assert counts["disagreed"] == 1
    assert counts["picked"] == 1
    assert counts["agreed"] == 0
    # One side of a split said True, so the union rule build_ground_truth uses
    # always lands on True: picked is agreement with the shipped label.
    assert counts["with_selector"] == 1
    assert counts["with_reannotator"] == 0


def test_leaving_the_label_off_on_a_split(tmp_path):
    counts = _resolution(
        tmp_path,
        selector={"rigor_appropriate": True},
        reannotator={"rigor_appropriate": False},
        rigor_appropriate=False,
    )["rigor_appropriate"]
    assert (counts["disagreed"], counts["picked"]) == (1, 0)
    assert (counts["with_selector"], counts["with_reannotator"]) == (0, 1)


def test_a_split_is_never_counted_as_agreement(tmp_path):
    counts = _resolution(
        tmp_path,
        selector={"scaffolding_present": True},
        reannotator={"scaffolding_present": False},
        scaffolding_present=True,
    )["scaffolding_present"]
    assert counts["agreed"] == counts["agreed_yes"] == counts["agreed_no"] == 0
    assert counts["upheld"] == 0


def test_over_scaffolding_is_read_from_the_declared_amount(tmp_path):
    counts = _resolution(
        tmp_path,
        selector={"scaffolding_amount": "over_scaffolding"},
        reannotator={"scaffolding_amount": "appropriate"},
        scaffolding_amount="over_scaffolding",
    )["over_scaffolding_declared"]
    assert counts["disagreed"] == 1
    assert counts["picked"] == 1


def test_a_thrown_out_adjudication_resolves_nothing(tmp_path):
    # They answered no label questions, so counting them would read every label
    # as silently upholding whatever the annotators said.
    results = _resolution(
        tmp_path,
        selector={"scaffolding_present": True},
        reannotator={"scaffolding_present": False},
        adjudicator=_thrown_out("Tara"),
    )
    assert results["scaffolding_present"]["n"] == 0
    assert results["scaffolding_present"]["disagreed"] == 0


def test_resolution_n_is_agreed_plus_disagreed(tmp_path):
    results = _resolution(
        tmp_path,
        selector={"scaffolding_present": True, "rigor_present": True},
        reannotator={"scaffolding_present": False, "rigor_present": True},
    )
    for field in A.FINAL_FIELDS:
        counts = results[field]
        assert counts["n"] == counts["agreed"] + counts["disagreed"] == 1
        assert counts["agreed"] == counts["agreed_yes"] + counts["agreed_no"]
        assert counts["upheld"] == counts["upheld_yes"] + counts["upheld_no"]


# ---------------------------------------------------------------------------
# free-text editing
# ---------------------------------------------------------------------------


def test_character_distance_counts_what_was_added_and_removed():
    stats = A.diff_stats("the tutor scaffolded", "the tutor scaffolded well")
    assert stats["net"] == 5
    assert stats["changed"] == 5
    assert stats["similarity"] > 0.8


def test_character_distance_of_a_replacement_counts_both_sides():
    stats = A.diff_stats("abcdef", "abcXYZ")
    assert stats["net"] == 0  # same length ...
    assert stats["changed"] == 6  # ... but half the text was replaced


def test_long_near_identical_prose_stays_similar():
    # difflib's autojunk heuristic treats characters appearing in over 1% of a
    # sequence longer than 200 as junk, which for prose is most of the alphabet.
    # Left on, it scores these two paragraphs as sharing almost nothing.
    base = "The tutor asked the student to explain their reasoning. " * 8
    stats = A.diff_stats(base, base + "It worked.")
    assert stats["similarity"] > 0.95


def _edit(final, selector="", reannotator=""):
    return A.classify_edit(final, {"selector": selector, "reannotator": reannotator})


def test_keeping_an_annotators_text_is_verbatim():
    edit = _edit("the student was stuck", reannotator="the student was stuck")
    assert edit["kind"] == "verbatim"
    assert edit["base"] == "reannotator"
    assert edit["changed"] == 0


def test_appending_to_an_annotators_text_is_an_edit():
    edit = _edit(
        "the student was stuck, so the tutor stepped in",
        selector="the student was stuck",
    )
    assert edit["kind"] == "edited"
    assert edit["base"] == "selector"
    assert edit["net"] == len(", so the tutor stepped in")


def test_a_text_sharing_almost_nothing_with_either_is_a_rewrite():
    edit = _edit("nobody scaffolded here", selector="the student was stuck")
    assert edit["kind"] == "rewritten"


def test_the_nearest_of_the_two_texts_is_the_base():
    edit = _edit(
        "the student was stuck and said so",
        selector="the student was stuck",
        reannotator="a completely different observation entirely",
    )
    assert edit["base"] == "selector"


def test_writing_where_neither_annotator_wrote_is_added():
    edit = _edit("worth a second look")
    assert edit["kind"] == "added"
    assert edit["base"] is None


def test_emptying_a_box_an_annotator_filled_is_cleared():
    edit = _edit("", selector="the student was stuck")
    assert edit["kind"] == "cleared"
    assert edit["base"] is None


def test_a_box_nobody_ever_filled_is_blank():
    assert _edit("")["kind"] == "blank"


def test_surrounding_whitespace_is_not_an_edit():
    # A text box collects trailing whitespace on its own; counting it would
    # report edits the adjudicator did not make.
    assert (
        _edit("the student was stuck ", selector="the student was stuck")["kind"]
        == "verbatim"
    )


def test_text_edits_counts_each_box_and_measures_only_sourced_ones(tmp_path):
    path = _write(
        tmp_path,
        [
            _adjudicated(
                "m1",
                selector=_annotation("Paul", why="the student was stuck"),
                reannotator=_annotation("Anita", "reannotator", why="stuck"),
                adjudicators=[
                    _adjudication(
                        "Tara",
                        why="the student was stuck and said so",
                        other_observations="worth a second look",
                    )
                ],
            )
        ],
    )
    results = A.text_edits(A.adjudications(path, {}))

    why = results["situation.why"]
    assert why["n"] == 1
    assert why["kinds"]["edited"] == 1
    assert why["base"] == {"selector": 1, "reannotator": 0}
    assert why["median_net_chars"] == len(" and said so")

    # other_observations was written from nothing: there is no distance from a
    # source text to report, so it must not enter the medians as a whole text's
    # worth of change.
    observations = results["other_observations"]
    assert observations["kinds"]["added"] == 1
    assert observations["base"] == {"selector": 0, "reannotator": 0}
    assert observations["median_net_chars"] is None

    assert results["result.explanation"]["kinds"]["blank"] == 1


def test_text_edits_skips_thrown_out_adjudications(tmp_path):
    path = _write(
        tmp_path,
        [
            _adjudicated(
                "m1",
                selector=_annotation("Paul", why="the student was stuck"),
                reannotator=_annotation("Anita", "reannotator"),
                adjudicators=[_thrown_out("Tara")],
            )
        ],
    )
    results = A.text_edits(A.adjudications(path, {}))
    assert results["situation.why"]["n"] == 0
    assert results["situation.why"]["median_similarity"] is None


# ---------------------------------------------------------------------------
# the whole run
# ---------------------------------------------------------------------------


def test_main_reports_resolution_and_editing(tmp_path, capsys):
    path = _write(
        tmp_path,
        [
            _adjudicated(
                f"m{i}",
                selector=_annotation(
                    "Paul", rigor_present=True, why="the student was stuck"
                ),
                reannotator=_annotation("Anita", "reannotator", why="stuck"),
                adjudicators=[
                    _adjudication(
                        "Tara",
                        rigor_present=True,
                        why="the student was stuck and said so",
                        rationale="keeping the selector's read",
                    )
                ],
            )
            for i in range(3)
        ],
    )
    out = tmp_path / "agreement.json"
    assert (
        A.main(
            [
                "--annotations",
                path,
                "--annotator-labels",
                "missing.json",
                "--json-out",
                str(out),
            ]
        )
        == 0
    )

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["n_adjudications"] == written["n_final_calls"] == 3
    assert written["resolution"]["rigor_present"]["disagreed"] == 3
    assert written["resolution"]["rigor_present"]["picked"] == 3
    assert written["text_edits"]["situation.why"]["kinds"]["edited"] == 3
    # no moment was adjudicated twice, so there is no kappa to report
    assert written["n_moments"] == 0

    printed = capsys.readouterr().out
    assert "How often the adjudicator backed the first and second pass" in printed
    assert "rationale note" in printed
