import pytest

from tutormoments.resources import resource_text


@pytest.mark.parametrize(
    "rel",
    [
        "annotate/scaffolding.md",
        "decompose/decompose_action.md",
        "decompose/decompose_result.md",
        "decompose/decompose_overscaffold.md",
        "structure/classify_action.md",
        "structure/classify_student_result.md",
    ],
)
def test_scorer_prompt_exists_and_nonempty(rel):
    text = resource_text(f"prompts/scorer/{rel}")
    assert text.strip(), f"empty prompts/scorer/{rel}"


# ---------------------------------------------------------------------------
# Task 2: Annotation dataclass + _build_synthetic_conversation
# ---------------------------------------------------------------------------

from tutormoments.conversation import Transcript
from tutormoments.moments import Moment as Scenario
from tutormoments.scoring import Annotation, _build_synthetic_conversation


@pytest.fixture
def fixture_scenario():
    return Scenario(
        id="testset:conv-abc__hum_5_12",
        context=[
            {"turn_number": 3, "role": "tutor", "text": "What do you think?"},
            {"turn_number": 4, "role": "student", "text": "I don't know."},
            {"turn_number": 5, "role": "tutor", "text": "Let's try breaking it down."},
        ],
        dimension="scaffolding",
        student={
            "mode": "oracle",
            "reference": "Turn 6. TUTOR: Good job.\nTurn 7. STUDENT: Thanks!",
            "context": "Grade 5, Math",
        },
        rubric={
            "gold": "scaffolding",
            "hint": "Student was stuck on long division.",
        },
        provenance={
            "conv_id": "conv-abc",
            "cut_turn": 5,
            "turn_start": 5,
            "turn_end": 12,
            "moment_id": "mom-001",
            "annotator_id": "ann-1",
            "chosen_cut_turn": 5,
            "cut_votes": {5: 2},
            "cluster_size": 2,
        },
    )


@pytest.fixture
def fixture_transcript(fixture_scenario):
    t = Transcript(scenario_id=fixture_scenario.id, tutor_model="test-model")
    t.generated_turns = [
        {"turn_number": 6, "role": "TUTOR", "text": "Can you try the first step?"},
        {"turn_number": 7, "role": "STUDENT", "text": "Is it 3?"},
        {"turn_number": 8, "role": "TUTOR", "text": "Almost, what comes next?"},
    ]
    return t


# --- Annotation round-trip ---


def test_annotation_to_dict_has_all_fields():
    ann = Annotation(
        scenario_id="testset:conv-abc__hum_5_12",
        annotation_type="scaffolding",
        turn_start=6,
        turn_end=8,
        situation="Student was stuck on long division.",
        action="Tutor asked a guiding question.",
        result="Student made partial progress.",
        action_decomposed=["asked guiding question"],
        overscaffold_decomposed=[],
        action_label="effective",
    )
    d = ann.to_dict()
    expected_fields = {
        "scenario_id",
        "annotation_type",
        "turn_start",
        "turn_end",
        "situation",
        "action",
        "result",
        "action_decomposed",
        "overscaffold_decomposed",
        "action_label",
    }
    assert expected_fields.issubset(set(d.keys()))
    assert d["scenario_id"] == "testset:conv-abc__hum_5_12"
    assert d["action_decomposed"] == ["asked guiding question"]
    assert d["overscaffold_decomposed"] == []
    # Student-outcome labelling was dropped from the runtime scorer
    assert "result_label" not in d
    assert "result_decomposed" not in d


def test_annotation_to_dict_round_trips_values():
    ann = Annotation(
        scenario_id="s1",
        annotation_type="rapport",
        turn_start=1,
        turn_end=3,
        situation="s",
        action="a",
        result="r",
        action_decomposed=["x", "y"],
        overscaffold_decomposed=["w"],
        action_label="ineffective",
    )
    d = ann.to_dict()
    assert d["annotation_type"] == "rapport"
    assert d["action_decomposed"] == ["x", "y"]
    assert d["overscaffold_decomposed"] == ["w"]
    assert d["action_label"] == "ineffective"


# --- _build_synthetic_conversation ---


def test_build_synthetic_conversation_keyed_by_scenario_id(
    fixture_scenario, fixture_transcript
):
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    # Both must be keyed by scenario.id
    assert fixture_scenario.id in conv_dict
    assert fixture_scenario.id in detections


def test_build_synthetic_conversation_turns_order(fixture_scenario, fixture_transcript):
    conv_dict, _ = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    conv = conv_dict[fixture_scenario.id]
    turns = conv["turns"]
    # Context has 3 turns (turn_numbers 3, 4, 5); generated has 3 (6, 7, 8)
    assert len(turns) == 6
    # First 3 are context turns in order
    assert turns[0]["turn_number"] == 3
    assert turns[1]["turn_number"] == 4
    assert turns[2]["turn_number"] == 5
    # Last 3 are generated turns
    assert turns[3]["turn_number"] == 6
    assert turns[4]["turn_number"] == 7
    assert turns[5]["turn_number"] == 8


def test_build_synthetic_conversation_roles_uppercased(
    fixture_scenario, fixture_transcript
):
    conv_dict, _ = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    turns = conv_dict[fixture_scenario.id]["turns"]
    # Context turns should have roles uppercased
    assert turns[0]["role"] == "TUTOR"
    assert turns[1]["role"] == "STUDENT"
    assert turns[2]["role"] == "TUTOR"
    # Generated turns already uppercase, preserved
    assert turns[3]["role"] == "TUTOR"
    assert turns[4]["role"] == "STUDENT"
    assert turns[5]["role"] == "TUTOR"


def test_build_synthetic_conversation_turn_shape(fixture_scenario, fixture_transcript):
    conv_dict, _ = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    turns = conv_dict[fixture_scenario.id]["turns"]
    for turn in turns:
        assert "turn_number" in turn
        assert "role" in turn
        assert "text" in turn
        assert turn.get("type") == "DIALOGUE"
        assert "timestamp" in turn


def test_build_synthetic_conversation_conversation_id(
    fixture_scenario, fixture_transcript
):
    conv_dict, _ = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    conv = conv_dict[fixture_scenario.id]
    assert conv["conversation_id"] == fixture_scenario.id


def test_build_synthetic_conversation_single_detection(
    fixture_scenario, fixture_transcript
):
    _, detections = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    det_entry = detections[fixture_scenario.id]
    detection_list = det_entry["detections"]
    assert len(detection_list) == 1


def test_build_synthetic_conversation_detection_turn_range(
    fixture_scenario, fixture_transcript
):
    _, detections = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    det = detections[fixture_scenario.id]["detections"][0]
    # turn_start = first generated turn number (6), turn_end = last (8)
    assert det["turn_start"] == 6
    assert det["turn_end"] == 8


def test_build_synthetic_conversation_detection_annotation_type(
    fixture_scenario, fixture_transcript
):
    _, detections = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    det = detections[fixture_scenario.id]["detections"][0]
    assert det["annotation_type"] == "scaffolding"


def test_build_synthetic_conversation_detection_situation_label_agg(
    fixture_scenario, fixture_transcript
):
    _, detections = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    det = detections[fixture_scenario.id]["detections"][0]
    # situation_label_agg = scenario.dimension (= rubric["gold"])
    assert det["situation_label_agg"] == fixture_scenario.dimension
    assert det["situation_label_agg"] == "scaffolding"


def test_build_synthetic_conversation_detection_situation_hint(
    fixture_scenario, fixture_transcript
):
    _, detections = _build_synthetic_conversation(fixture_scenario, fixture_transcript)
    det = detections[fixture_scenario.id]["detections"][0]
    # situation description includes the rubric hint
    assert fixture_scenario.rubric["hint"] in det["situation"]


# ---------------------------------------------------------------------------
# Task 3: Annotate pass — excerpt builder, suggestion text, build entries, parse
# ---------------------------------------------------------------------------

from tutormoments.scoring import (
    _build_annotate_entries,
    _format_excerpt,
    _parse_and_merge,
    _suggestion_text,
)

# --- _format_excerpt (context_window=0) ---


def test_format_excerpt_markers_present(fixture_scenario, fixture_transcript):
    """The excerpt must contain START and END markers."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    conv = conv_dict[fixture_scenario.id]
    det = detections[fixture_scenario.id]["detections"][0]
    excerpt = _format_excerpt(conv, det["turn_start"], det["turn_end"], 0, 0)
    assert f">>> DETECTED MOMENT START (Turn {det['turn_start']}) <<<" in excerpt
    assert f">>> DETECTED MOMENT END (Turn {det['turn_end']}) <<<" in excerpt


def test_format_excerpt_omit_header_emitted(fixture_scenario, fixture_transcript):
    """When context_window=0 and generated turns don't start at 1, omit header is emitted."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    conv = conv_dict[fixture_scenario.id]
    det = detections[fixture_scenario.id]["detections"][0]
    # turn_start=6, min_turn=3: excerpt_start=6 > min_turn=3 => header expected
    excerpt = _format_excerpt(conv, det["turn_start"], det["turn_end"], 0, 0)
    assert "[... turns 1-5 omitted ...]" in excerpt


def test_format_excerpt_only_detected_range(fixture_scenario, fixture_transcript):
    """With context_window=0, only turns [turn_start, turn_end] appear (plus markers)."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    conv = conv_dict[fixture_scenario.id]
    det = detections[fixture_scenario.id]["detections"][0]
    excerpt = _format_excerpt(conv, det["turn_start"], det["turn_end"], 0, 0)
    # Context turns (3,4,5) must not appear in excerpt
    assert "Turn 3." not in excerpt
    assert "Turn 4." not in excerpt
    assert "Turn 5." not in excerpt
    # Generated turns must appear
    assert "Turn 6." in excerpt
    assert "Turn 7." in excerpt
    assert "Turn 8." in excerpt


def test_format_excerpt_in_range_marker(fixture_scenario, fixture_transcript):
    """Each turn in [turn_start, turn_end] gets a ' <<<' suffix."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    conv = conv_dict[fixture_scenario.id]
    det = detections[fixture_scenario.id]["detections"][0]
    excerpt = _format_excerpt(conv, det["turn_start"], det["turn_end"], 0, 0)
    assert "Turn 6. TUTOR: Can you try the first step? <<<" in excerpt
    assert "Turn 7. STUDENT: Is it 3? <<<" in excerpt
    assert "Turn 8. TUTOR: Almost, what comes next? <<<" in excerpt


def test_format_excerpt_golden_string(fixture_scenario, fixture_transcript):
    """Golden: exact excerpt string for the fixture (context_window=0)."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    conv = conv_dict[fixture_scenario.id]
    det = detections[fixture_scenario.id]["detections"][0]
    excerpt = _format_excerpt(conv, det["turn_start"], det["turn_end"], 0, 0)
    expected = (
        "[... turns 1-5 omitted ...]\n"
        "\n"
        ">>> DETECTED MOMENT START (Turn 6) <<<\n"
        "Turn 6. TUTOR: Can you try the first step? <<<\n"
        "Turn 7. STUDENT: Is it 3? <<<\n"
        "Turn 8. TUTOR: Almost, what comes next? <<<\n"
        ">>> DETECTED MOMENT END (Turn 8) <<<"
    )
    assert excerpt == expected


# --- _suggestion_text ---


def test_suggestion_text_scaffolding():
    assert _suggestion_text("scaffolding") == (
        "A team of teachers believe that this moment is appropriate for scaffolding."
    )


def test_suggestion_text_rigor():
    assert _suggestion_text("rigor") == (
        "A team of teachers believe that this moment is appropriate for pushing for rigor."
    )


def test_suggestion_text_mixed():
    assert _suggestion_text("mixed") == (
        "A team of teachers believe that this moment is appropriate for either rigor or scaffolding."
    )


def test_suggestion_text_both():
    assert _suggestion_text("both") == (
        "A team of teachers believe that this moment is appropriate for either rigor or scaffolding."
    )


def test_suggestion_text_neither():
    assert _suggestion_text("neither") == (
        "A team of teachers believe that this moment is not appropriate for either rigor or scaffolding."
    )


def test_suggestion_text_unknown():
    assert _suggestion_text(None) == (
        "It's unclear to a team of teachers whether this moment is appropriate for rigor or scaffolding."
    )


def test_suggestion_text_unrecognized():
    assert _suggestion_text("some_unknown_label") == (
        "It's unclear to a team of teachers whether this moment is appropriate for rigor or scaffolding."
    )


# --- _build_annotate_entries: golden-prompt test ---


def test_build_annotate_entries_key_scheme(fixture_scenario, fixture_transcript):
    """Key must be '{scenario_id}__scaffolding__0' for the first detection."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    entries = _build_annotate_entries(conv_dict, detections)
    assert len(entries) == 1
    assert entries[0]["key"] == f"{fixture_scenario.id}__scaffolding__0"


def test_build_annotate_entries_prompt_contains_excerpt(
    fixture_scenario, fixture_transcript
):
    """The prompt must contain the exact markered excerpt."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    entries = _build_annotate_entries(conv_dict, detections)
    prompt = entries[0]["request"]["contents"][0]["parts"][0]["text"]
    assert ">>> DETECTED MOMENT START (Turn 6) <<<" in prompt
    assert ">>> DETECTED MOMENT END (Turn 8) <<<" in prompt
    assert "Turn 6. TUTOR: Can you try the first step? <<<" in prompt


def test_build_annotate_entries_prompt_contains_suggestion(
    fixture_scenario, fixture_transcript
):
    """The prompt must contain the suggestion sentence for 'scaffolding'."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    entries = _build_annotate_entries(conv_dict, detections)
    prompt = entries[0]["request"]["contents"][0]["parts"][0]["text"]
    assert (
        "A team of teachers believe that this moment is appropriate for scaffolding."
        in prompt
    )


def test_build_annotate_entries_annotator_style_absent(
    fixture_scenario, fixture_transcript
):
    """{annotator_style} must be replaced with '' (empty string, not present as literal)."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    entries = _build_annotate_entries(conv_dict, detections)
    prompt = entries[0]["request"]["contents"][0]["parts"][0]["text"]
    assert "{annotator_style}" not in prompt


def test_build_annotate_entries_json_mode_enabled(fixture_scenario, fixture_transcript):
    """The batch entry must have json_mode enabled (response_mime_type = application/json)."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    entries = _build_annotate_entries(conv_dict, detections)
    gen_config = entries[0]["request"]["generation_config"]
    assert gen_config.get("response_mime_type") == "application/json"


def test_build_annotate_entries_golden_prompt(fixture_scenario, fixture_transcript):
    """Golden: the substituted prompt is BYTE-EXACT equal to the expected prompt."""
    conv_dict, detections = _build_synthetic_conversation(
        fixture_scenario, fixture_transcript
    )
    entries = _build_annotate_entries(conv_dict, detections)
    assert entries[0]["key"] == f"{fixture_scenario.id}__scaffolding__0"

    actual_prompt = entries[0]["request"]["contents"][0]["parts"][0]["text"]

    # Build expected prompt by reading the actual template and applying the same 5 substitutions
    # the code applies, with the fixture's values.
    template = resource_text("prompts/scorer/annotate/scaffolding.md")

    suggestion = (
        "A team of teachers believe that this moment is appropriate for scaffolding."
    )
    expected_excerpt = (
        "[... turns 1-5 omitted ...]\n"
        "\n"
        ">>> DETECTED MOMENT START (Turn 6) <<<\n"
        "Turn 6. TUTOR: Can you try the first step? <<<\n"
        "Turn 7. STUDENT: Is it 3? <<<\n"
        "Turn 8. TUTOR: Almost, what comes next? <<<\n"
        ">>> DETECTED MOMENT END (Turn 8) <<<"
    )
    turn_start = 6
    turn_end = 8

    expected_prompt = template
    expected_prompt = expected_prompt.replace("{annotator_style}", "")
    expected_prompt = expected_prompt.replace("{suggestion}", suggestion)
    expected_prompt = expected_prompt.replace("{excerpt}", expected_excerpt)
    expected_prompt = expected_prompt.replace("{turn_start}", str(turn_start))
    expected_prompt = expected_prompt.replace("{turn_end}", str(turn_end))

    assert actual_prompt == expected_prompt


# --- _parse_and_merge ---


def test_parse_and_merge_extracts_fields():
    """Parse extracts situation/action/result from JSON response."""
    detections_by_conv = {
        "conv1": {
            "detections": [
                {"annotation_type": "scaffolding", "turn_start": 6, "turn_end": 8}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100},
        }
    }
    raw_entries = {
        "conv1__scaffolding__0": {
            "text": '{"situation": "Student stuck", "action": "Tutor guided", "result": "Effective"}',
            "usage": {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250},
        }
    }
    results = _parse_and_merge(raw_entries, detections_by_conv)
    assert "conv1" in results
    ann = results["conv1"]["annotations"][0]
    assert ann["situation"] == "Student stuck"
    assert ann["action"] == "Tutor guided"
    assert ann["result"] == "Effective"


def test_parse_and_merge_accumulates_usage():
    """Usage from p1 (detections) and p2 (parsed) are summed."""
    detections_by_conv = {
        "conv1": {
            "detections": [
                {"annotation_type": "scaffolding", "turn_start": 6, "turn_end": 8}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100},
        }
    }
    raw_entries = {
        "conv1__scaffolding__0": {
            "text": '{"situation": "s", "action": "a", "result": "r"}',
            "usage": {"input_tokens": 200, "output_tokens": 50, "total_tokens": 250},
        }
    }
    results = _parse_and_merge(raw_entries, detections_by_conv)
    usage = results["conv1"]["usage"]
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 350


def test_parse_and_merge_fallback_action_text():
    """When no parsed result for a key, fallback action text is used."""
    detections_by_conv = {
        "conv1": {
            "detections": [
                {"annotation_type": "scaffolding", "turn_start": 6, "turn_end": 8}
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    }
    raw_entries = {}  # no result for this key
    results = _parse_and_merge(raw_entries, detections_by_conv)
    ann = results["conv1"]["annotations"][0]
    assert ann["action"] == "[Analysis unavailable -- batch failed for this moment]"
    assert ann["situation"] == ""
    assert ann["result"] == ""


def test_parse_and_merge_list_response_takes_first():
    """If the JSON response is a list, [0] is used."""
    detections_by_conv = {
        "conv1": {
            "detections": [
                {"annotation_type": "scaffolding", "turn_start": 1, "turn_end": 2}
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    }
    raw_entries = {
        "conv1__scaffolding__0": {
            "text": '[{"situation": "s", "action": "a", "result": "r"}, {"situation": "x"}]',
            "usage": {},
        }
    }
    results = _parse_and_merge(raw_entries, detections_by_conv)
    ann = results["conv1"]["annotations"][0]
    assert ann["situation"] == "s"
    assert ann["action"] == "a"


def test_parse_and_merge_missing_json_keys_default_empty():
    """Missing situation/action/result keys default to ''."""
    detections_by_conv = {
        "conv1": {
            "detections": [
                {"annotation_type": "scaffolding", "turn_start": 1, "turn_end": 2}
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    }
    raw_entries = {
        "conv1__scaffolding__0": {
            "text": '{"situation": "s"}',  # action and result missing
            "usage": {},
        }
    }
    results = _parse_and_merge(raw_entries, detections_by_conv)
    ann = results["conv1"]["annotations"][0]
    assert ann["situation"] == "s"
    assert ann["action"] == ""
    assert ann["result"] == ""


def test_parse_and_merge_pass1_pass2_counts():
    """pass1_detections and pass2_analyzed counts are correct."""
    detections_by_conv = {
        "conv1": {
            "detections": [
                {"annotation_type": "scaffolding", "turn_start": 1, "turn_end": 2},
                {"annotation_type": "scaffolding", "turn_start": 3, "turn_end": 4},
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
    }
    raw_entries = {
        "conv1__scaffolding__0": {
            "text": '{"situation": "s", "action": "a", "result": "r"}',
            "usage": {},
        }
        # __1 missing: batch failed for 2nd detection
    }
    results = _parse_and_merge(raw_entries, detections_by_conv)
    assert results["conv1"]["pass1_detections"] == 2
    assert results["conv1"]["pass2_analyzed"] == 1


# ---------------------------------------------------------------------------
# Task 4: Decompose pass — _coerce_facets, _build_decompose_entries, junk skip
# ---------------------------------------------------------------------------

from tutormoments.scoring import _build_decompose_entries, _coerce_facets

JUNK_TEXTS = {"", "n/a", "test", "sdf", "this is a test annotation"}


# --- _coerce_facets ---


def test_coerce_facets_bare_array():
    """A bare JSON array should return the list of strings."""
    result = _coerce_facets(["a", "b", "c"])
    assert result == ["a", "b", "c"]


def test_coerce_facets_bare_array_coerces_to_str():
    """Non-string items in a bare array should be coerced to str."""
    result = _coerce_facets([1, 2, 3])
    assert result == ["1", "2", "3"]


def test_coerce_facets_object_with_list_value():
    """An object with a list-valued key should return the flattened list."""
    result = _coerce_facets({"facets": ["a", "b"]})
    assert result == ["a", "b"]


def test_coerce_facets_object_with_spans_key():
    """Object with 'spans' key (OpenAI overscaffold wrapper) should work."""
    result = _coerce_facets({"spans": ["x", "y"]})
    assert result == ["x", "y"]


def test_coerce_facets_object_with_empty_list_value():
    """Object with empty list value should return [] (not fall through to cram path)."""
    result = _coerce_facets({"spans": []})
    assert result == []


def test_coerce_facets_object_cram_path():
    """Object with no list values: interleave keys and string values."""
    result = _coerce_facets({"facet a": "facet b", "facet c": "facet d"})
    assert result == ["facet a", "facet b", "facet c", "facet d"]


def test_coerce_facets_returns_none_for_invalid():
    """Returns None when passed something that is not a list or dict."""
    assert _coerce_facets("not-a-list") is None
    assert _coerce_facets(42) is None
    assert _coerce_facets(None) is None


# --- _build_decompose_entries: action prompt ---


def test_build_decompose_entries_action_substitutes_action():
    """Action entry prompt must contain the action text via {action} substitution."""
    annotations = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "situation": "Student is stuck.",
                    "action": "Tutor asked a guiding question.",
                    "result": "Student got unstuck.",
                }
            ]
        }
    }
    entries, locations = _build_decompose_entries(annotations, "action")
    assert len(entries) == 1
    prompt = entries[0]["request"]["contents"][0]["parts"][0]["text"]
    assert "Tutor asked a guiding question." in prompt
    assert entries[0]["key"].startswith("action__conv1__0")


def test_build_decompose_entries_result_pass_removed():
    """The 'result' decompose pass was dropped with student-outcome labelling."""
    with pytest.raises(ValueError):
        _build_decompose_entries({}, "result")


def test_build_decompose_entries_overscaffold_substitutes_situation_action_result():
    """Overscaffold entry prompt must substitute {situation}, {action}, {result}."""
    annotations = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "situation": "Student confused.",
                    "action": "Tutor gave away the answer.",
                    "result": "Student just copied.",
                }
            ]
        }
    }
    entries, locations = _build_decompose_entries(annotations, "overscaffold")
    assert len(entries) == 1
    prompt = entries[0]["request"]["contents"][0]["parts"][0]["text"]
    assert "Student confused." in prompt
    assert "Tutor gave away the answer." in prompt
    assert "Student just copied." in prompt
    assert entries[0]["key"].startswith("overscaffold__conv1__0")


def test_build_decompose_entries_overscaffold_only_for_scaffolding_target():
    """Overscaffold entries are only built for annotation_type == 'scaffolding'."""
    annotations = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "rapport",
                    "situation": "Student seems upset.",
                    "action": "Tutor acknowledged feelings.",
                    "result": "Student engaged more.",
                }
            ]
        }
    }
    entries, locations = _build_decompose_entries(annotations, "overscaffold")
    # rapport annotations must not get overscaffold entries
    assert len(entries) == 0
    assert len(locations) == 0


def test_build_decompose_entries_junk_action_skipped():
    """Junk action text (e.g. 'n/a') must produce no entry and empty list returned."""
    for junk in list(JUNK_TEXTS):
        annotations = {
            "conv1": {
                "annotations": [
                    {
                        "annotation_type": "scaffolding",
                        "situation": "Student stuck.",
                        "action": junk,
                        "result": "Student improved.",
                    }
                ]
            }
        }
        entries, locations = _build_decompose_entries(annotations, "action")
        assert len(entries) == 0, f"Expected no entry for junk action: {junk!r}"
        assert len(locations) == 0


def test_build_decompose_entries_junk_case_insensitive():
    """Junk detection must be case-insensitive (strip + lower)."""
    annotations = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "situation": "S",
                    "action": "  N/A  ",  # uppercase, padded
                    "result": "R",
                }
            ]
        }
    }
    entries, locations = _build_decompose_entries(annotations, "action")
    assert len(entries) == 0


def test_build_decompose_entries_json_mode_enabled():
    """All decompose entries must have json_mode enabled."""
    annotations = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "situation": "S",
                    "action": "Tutor asked.",
                    "result": "Student answered.",
                }
            ]
        }
    }
    for pass_type in ("action", "overscaffold"):
        entries, _ = _build_decompose_entries(annotations, pass_type)
        assert len(entries) == 1
        gen_config = entries[0]["request"]["generation_config"]
        assert gen_config.get("response_mime_type") == "application/json", (
            f"{pass_type} entry missing json_mode"
        )


def test_build_decompose_entries_overscaffold_both_junk_skipped():
    """When both action and result are junk, overscaffold entry is also skipped."""
    annotations = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "situation": "S",
                    "action": "",
                    "result": "n/a",
                }
            ]
        }
    }
    entries, locations = _build_decompose_entries(annotations, "overscaffold")
    assert len(entries) == 0


# ---------------------------------------------------------------------------
# Task 5: Structure pass -- _parse_action_label, _parse_result_label,
#          _build_structure_entries (action+result in one batch, mixed json_mode)
# ---------------------------------------------------------------------------

from tutormoments.scoring import (
    _build_structure_entries,
    _parse_action_label,
    _parse_result_label,
)

# --- _parse_action_label ---


def test_parse_action_label_scaffolding_only():
    label, err = _parse_action_label('{"scaffolding":"yes","rigor":"no"}')
    assert label == "scaffolding"
    assert not err


def test_parse_action_label_both():
    label, err = _parse_action_label('{"scaffolding":"yes","rigor":"yes"}')
    assert label == "both"
    assert not err


def test_parse_action_label_rigor_only():
    label, err = _parse_action_label('{"scaffolding":"no","rigor":"yes"}')
    assert label == "rigor"
    assert not err


def test_parse_action_label_neither():
    label, err = _parse_action_label('{"scaffolding":"no","rigor":"no"}')
    assert label == "neither"
    assert not err


def test_parse_action_label_missing_rigor_returns_unclear():
    """Missing one dimension -> unclear."""
    label, err = _parse_action_label('{"scaffolding":"yes"}')
    assert label == "unclear"
    assert err


def test_parse_action_label_missing_scaffolding_returns_unclear():
    label, err = _parse_action_label('{"rigor":"yes"}')
    assert label == "unclear"
    assert err


def test_parse_action_label_empty_returns_unclear():
    label, err = _parse_action_label("{}")
    assert label == "unclear"
    assert err


def test_parse_action_label_list_wrapped():
    """A list-wrapped JSON response -- takes [0]."""
    label, err = _parse_action_label('[{"scaffolding":"yes","rigor":"no"}]')
    assert label == "scaffolding"
    assert not err


def test_parse_action_label_regex_fallback():
    """Regex fallback when JSON parse fails (extra text around a valid answer)."""
    text = 'Some preamble scaffolding: "yes", rigor: "no" end'
    label, err = _parse_action_label(text)
    assert label == "scaffolding"
    assert not err


def test_parse_action_label_regex_fallback_both():
    """Regex fallback for both dimensions present."""
    text = "scaffolding: yes, rigor: yes"
    label, err = _parse_action_label(text)
    assert label == "both"
    assert not err


def test_parse_action_label_invalid_value_unclear():
    """Non yes/no values -> unclear."""
    label, err = _parse_action_label('{"scaffolding":"maybe","rigor":"yes"}')
    assert label == "unclear"
    assert err


# --- _parse_result_label ---


def test_parse_result_label_A_returns_pos():
    label, err = _parse_result_label("A")
    assert label == "pos"
    assert not err


def test_parse_result_label_B_returns_neg():
    label, err = _parse_result_label("B")
    assert label == "neg"
    assert not err


def test_parse_result_label_lowercase_a():
    """Exact match is case-insensitive via strip+lower."""
    label, err = _parse_result_label("a")
    assert label == "pos"
    assert not err


def test_parse_result_label_lowercase_b():
    label, err = _parse_result_label("b")
    assert label == "neg"
    assert not err


def test_parse_result_label_trailing_period():
    """Trailing period is stripped before matching."""
    label, err = _parse_result_label("A.")
    assert label == "pos"
    assert not err


def test_parse_result_label_first_line_markdown_stripped():
    """Markdown emphasis on first line is stripped before matching."""
    label, err = _parse_result_label("**A**\n\nThe statements indicate understanding.")
    assert label == "pos"
    assert not err


def test_parse_result_label_first_word_regex_fallback():
    """First-word regex fallback when first line starts with a|b followed by word boundary."""
    label, err = _parse_result_label("a. The student shows understanding.")
    assert label == "pos"
    assert not err


def test_parse_result_label_unrecognized_returns_unclear():
    label, err = _parse_result_label("C")
    assert label == "unclear"
    assert err


def test_parse_result_label_empty_returns_unclear():
    label, err = _parse_result_label("")
    assert label == "unclear"
    assert err


def test_parse_result_label_whitespace_only_returns_unclear():
    label, err = _parse_result_label("   ")
    assert label == "unclear"
    assert err


def test_parse_result_label_letter_after_text_not_matched():
    """A letter mentioned mid-text should not be matched (first line/word only)."""
    label, err = _parse_result_label("The answer is A")
    # "the answer is a" -- first word is "the", not a/b -> unclear
    assert label == "unclear"
    assert err


# --- _build_structure_entries: action classification only ---


def _make_decomposed_results(action_facets, ann_type="scaffolding"):
    """Helper: build results dict with one annotation having given facets."""
    return {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": ann_type,
                    "action_decomposed": action_facets,
                }
            ]
        }
    }


def test_build_structure_entries_action_entry_json_mode_true():
    """Action entry must have json_mode=True (response_mime_type=application/json)."""
    results = _make_decomposed_results(["asked guiding question"])
    action_entries, skip_action = _build_structure_entries(results)
    assert len(action_entries) == 1
    gen_config = action_entries[0]["request"]["generation_config"]
    assert gen_config.get("response_mime_type") == "application/json"


def test_build_structure_entries_action_key_scheme():
    """Action entry key: 'action__{conv_id}__{idx}'."""
    results = _make_decomposed_results(["f1"])
    action_entries, _ = _build_structure_entries(results)
    assert action_entries[0]["key"] == "action__conv1__0"


def test_build_structure_entries_no_action_facets_skip():
    """No action facets -> skipped (not sent to model)."""
    results = _make_decomposed_results([])
    action_entries, skip_action = _build_structure_entries(results)
    assert len(action_entries) == 0
    assert skip_action == [("conv1", 0)]


def test_build_structure_entries_action_prompt_contains_facets():
    """Action prompt must contain the facet list via {action_list} substitution."""
    results = _make_decomposed_results(["asked guiding question", "broke it down"])
    action_entries, _ = _build_structure_entries(results)
    prompt = action_entries[0]["request"]["contents"][0]["parts"][0]["text"]
    assert "- asked guiding question" in prompt
    assert "- broke it down" in prompt


def test_build_structure_entries_target_filter():
    """Annotations with annotation_type != target are skipped."""
    results = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "action_decomposed": ["asked question"],
                },
                {
                    "annotation_type": "rapport",
                    "action_decomposed": ["built trust"],
                },
            ]
        }
    }
    # With target="scaffolding" (default), only the first annotation should be processed
    action_entries, _ = _build_structure_entries(results, target="scaffolding")
    assert len(action_entries) == 1
    assert action_entries[0]["key"] == "action__conv1__0"


def test_build_structure_entries_target_filter_rapport():
    """With target='rapport', only rapport annotations are processed."""
    results = {
        "conv1": {
            "annotations": [
                {
                    "annotation_type": "scaffolding",
                    "action_decomposed": ["asked question"],
                },
                {
                    "annotation_type": "rapport",
                    "action_decomposed": ["built trust"],
                },
            ]
        }
    }
    # With target="rapport", only the second annotation should be processed
    action_entries, _ = _build_structure_entries(results, target="rapport")
    assert len(action_entries) == 1
    assert action_entries[0]["key"] == "action__conv1__1"


# --- Default labels (no facets) ---


def test_no_action_facets_default_label_neither():
    """No action facets -> default action_label 'neither'."""
    # This tests the convention the caller must apply after _build_structure_entries.
    # The default is encoded in DEFAULT_ACTION_LABEL.
    from tutormoments.scoring import DEFAULT_ACTION_LABEL

    assert DEFAULT_ACTION_LABEL == "neither"


def test_default_result_label_kept_for_tutormoments_build():
    """DEFAULT_RESULT_LABEL is no longer used by the runtime scorer but is
    imported by tutormoments_build.groundtruth -- pin its value."""
    from tutormoments.scoring import DEFAULT_RESULT_LABEL

    assert DEFAULT_RESULT_LABEL == "no_evidence"


# ---------------------------------------------------------------------------
# Task 6: score(scenario, transcript) -> Annotation (3-pass end-to-end)
# ---------------------------------------------------------------------------

import json as _json
from unittest.mock import MagicMock, patch


def _make_raw_entries(mapping):
    """Build a run_batch-style {key: {text, usage}} dict from a plain dict."""
    return {
        k: {
            "text": v if isinstance(v, str) else _json.dumps(v),
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        for k, v in mapping.items()
    }


def _make_fake_score_client():
    """Build a fake ModelClient that does not need real API keys."""
    fake_client = MagicMock()
    fake_client.model = "claude-opus-4-6"
    fake_client.provider = "anthropic"
    return fake_client


def _patch_score(fake_run_batch, fake_client=None):
    """Return a context manager stack that patches run_batch (and optionally ModelClient)
    in the tutormoments.client module, which is where score() resolves them via local import.

    score() does:
        from .client import ModelClient, run_batch
    so the names to patch are tutormoments.client.run_batch and tutormoments.client.ModelClient.
    """
    if fake_client is None:
        fake_client = _make_fake_score_client()
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("tutormoments.client.run_batch", side_effect=fake_run_batch)
    )
    stack.enter_context(
        patch("tutormoments.client.ModelClient", return_value=fake_client)
    )
    return stack, fake_client


def test_score_returns_annotation(fixture_scenario, fixture_transcript):
    """score() with mocked run_batch returns a populated Annotation."""
    from tutormoments.scoring import score

    sid = fixture_scenario.id  # "testset:conv-abc__hum_5_12"

    annotate_key = f"{sid}__scaffolding__0"
    annotate_resp = {
        "situation": "Student stuck on division.",
        "action": "Tutor guided step-by-step.",
        "result": "Student made progress.",
    }
    decompose_action_key = f"action__{sid}__0"
    decompose_overscaffold_key = f"overscaffold__{sid}__0"
    structure_action_key = f"action__{sid}__0"

    def fake_run_batch(client, entries, **kwargs):
        keys = {e["key"] for e in entries}
        # Student-outcome (result__) entries were dropped from the runtime scorer
        assert not any(k.startswith("result__") for k in keys)
        if annotate_key in keys:
            return _make_raw_entries({annotate_key: annotate_resp})
        elif decompose_overscaffold_key in keys:
            # Decompose pass: action + overscaffold entries
            return _make_raw_entries(
                {
                    decompose_action_key: ["guided student", "broke down problem"],
                    decompose_overscaffold_key: [],
                }
            )
        else:
            # Structure pass: action (JSON)
            return _make_raw_entries(
                {
                    structure_action_key: {"scaffolding": "yes", "rigor": "no"},
                }
            )

    with _patch_score(fake_run_batch)[0]:
        result = score(fixture_scenario, fixture_transcript)

    assert isinstance(result, Annotation)
    assert result.scenario_id == fixture_scenario.id
    assert result.annotation_type == "scaffolding"
    assert result.turn_start == 6
    assert result.turn_end == 8
    assert result.situation == "Student stuck on division."
    assert result.action == "Tutor guided step-by-step."
    assert result.result == "Student made progress."
    assert result.action_decomposed == ["guided student", "broke down problem"]
    assert result.overscaffold_decomposed == []
    assert result.action_label == "scaffolding"
    assert not hasattr(result, "result_label")
    assert not hasattr(result, "result_decomposed")


def test_score_three_passes_in_order(fixture_scenario, fixture_transcript):
    """score() runs annotate -> decompose -> structure in that order."""
    from tutormoments.scoring import score

    sid = fixture_scenario.id
    annotate_key = f"{sid}__scaffolding__0"
    overscaffold_key = f"overscaffold__{sid}__0"

    # Distinguish passes:
    # - annotate pass:   contains {sid}__scaffolding__0
    # - decompose pass:  contains overscaffold__{sid}__0 (or action/overscaffold keys)
    # - structure pass:  does NOT contain overscaffold key and does NOT contain annotate key
    call_order = []

    def fake_run_batch(client, entries, **kwargs):
        keys = {e["key"] for e in entries}
        if annotate_key in keys:
            call_order.append("annotate")
            return _make_raw_entries(
                {
                    annotate_key: {
                        "situation": "s",
                        "action": "Tutor helped.",
                        "result": "r",
                    }
                }
            )
        elif overscaffold_key in keys:
            call_order.append("decompose")
            return _make_raw_entries(
                {
                    f"action__{sid}__0": ["facet1"],
                    overscaffold_key: [],
                }
            )
        else:
            call_order.append("structure")
            return _make_raw_entries(
                {
                    f"action__{sid}__0": {"scaffolding": "yes", "rigor": "no"},
                }
            )

    with _patch_score(fake_run_batch)[0]:
        score(fixture_scenario, fixture_transcript)

    assert call_order == ["annotate", "decompose", "structure"], (
        f"Expected annotate->decompose->structure, got {call_order}"
    )


def test_score_scorer_model_is_claude_opus_4_6(fixture_scenario, fixture_transcript):
    """score() must use the scorer model 'claude-opus-4-6' from config."""
    from tutormoments.scoring import score

    sid = fixture_scenario.id
    annotate_key = f"{sid}__scaffolding__0"

    def fake_run_batch(client, entries, **kwargs):
        keys = {e["key"] for e in entries}
        if annotate_key in keys:
            return _make_raw_entries(
                {
                    annotate_key: {
                        "situation": "s",
                        "action": "Tutor guided.",
                        "result": "r",
                    }
                }
            )
        elif f"overscaffold__{sid}__0" in keys:
            return _make_raw_entries(
                {
                    f"action__{sid}__0": ["facet1"],
                    f"overscaffold__{sid}__0": [],
                }
            )
        else:
            return _make_raw_entries(
                {
                    f"action__{sid}__0": {"scaffolding": "yes", "rigor": "no"},
                }
            )

    with (
        patch("tutormoments.client.run_batch", side_effect=fake_run_batch),
        patch("tutormoments.client.ModelClient") as mock_mc,
    ):
        # Make the mock instance have the right model attribute
        fake_client = MagicMock()
        fake_client.model = "claude-opus-4-6"
        fake_client.provider = "anthropic"
        mock_mc.return_value = fake_client

        score(fixture_scenario, fixture_transcript)

        # Check that ModelClient was instantiated with claude-opus-4-6
        assert mock_mc.called, "ModelClient was not instantiated"
        init_args = mock_mc.call_args
        model_arg = (
            init_args.args[0] if init_args.args else init_args.kwargs.get("model")
        )
        assert model_arg == "claude-opus-4-6", (
            f"Expected scorer model 'claude-opus-4-6', got {model_arg!r}"
        )


def test_score_accumulates_usage_across_passes(fixture_scenario, fixture_transcript):
    """score() must accumulate usage from all 3 passes into Annotation.usage."""
    from tutormoments.scoring import score

    sid = fixture_scenario.id
    annotate_key = f"{sid}__scaffolding__0"

    def make_resp(keys_vals, tokens_per_key):
        return {
            k: {
                "text": v if isinstance(v, str) else _json.dumps(v),
                "usage": {
                    "input_tokens": tokens_per_key,
                    "output_tokens": tokens_per_key,
                    "total_tokens": tokens_per_key * 2,
                },
            }
            for k, v in keys_vals.items()
        }

    def fake_run_batch(client, entries, **kwargs):
        keys = {e["key"] for e in entries}
        if annotate_key in keys:
            return make_resp(
                {
                    annotate_key: {
                        "situation": "s",
                        "action": "Tutor helped.",
                        "result": "r",
                    }
                },
                100,
            )
        elif f"overscaffold__{sid}__0" in keys:
            return make_resp(
                {
                    f"action__{sid}__0": ["f1"],
                    f"overscaffold__{sid}__0": [],
                },
                200,
            )
        else:
            return make_resp(
                {
                    f"action__{sid}__0": {"scaffolding": "yes", "rigor": "no"},
                },
                300,
            )

    with _patch_score(fake_run_batch)[0]:
        result = score(fixture_scenario, fixture_transcript)

    assert hasattr(result, "usage"), "Annotation must have a 'usage' attribute"
    usage = result.usage
    assert isinstance(usage, dict), "usage must be a dict"
    assert usage.get("total_tokens", 0) > 0, "total_tokens must be > 0"


# ---------------------------------------------------------------------------
# Ground-truth build shims: _parse_decomposed + _build_overscaffold_prompt
# ---------------------------------------------------------------------------

from tutormoments.scoring import _build_overscaffold_prompt, _parse_decomposed


def test_parse_decomposed_bare_array():
    facets, had_error = _parse_decomposed('["a", "b"]')
    assert facets == ["a", "b"]
    assert had_error is False


def test_parse_decomposed_object_wrapper():
    facets, had_error = _parse_decomposed('{"facets": ["x", "y"]}')
    assert facets == ["x", "y"]
    assert had_error is False


def test_parse_decomposed_regex_fallback_from_surrounding_text():
    facets, had_error = _parse_decomposed('Here you go: ["one", "two"]. Done.')
    assert facets == ["one", "two"]
    assert had_error is False


def test_parse_decomposed_unparseable_returns_empty_with_error():
    facets, had_error = _parse_decomposed("no array here at all")
    assert facets == []
    assert had_error is True


def test_build_overscaffold_prompt_skips_when_both_junk():
    template = "{situation}|{action}|{result}"
    assert _build_overscaffold_prompt("some situation", "", "n/a", template) is None


def test_build_overscaffold_prompt_substitutes_all_three():
    template = "S={situation}|A={action}|R={result}"
    out = _build_overscaffold_prompt("sit", "act", "res", template)
    assert out == "S=sit|A=act|R=res"


# ---------------------------------------------------------------------------
# score_batch: pooled 3-pass scoring across many scenarios
# ---------------------------------------------------------------------------


def _second_scenario():
    """A second, distinct scenario for pooling tests."""
    return Scenario(
        id="testset:conv-xyz__hum_2_4",
        context=[
            {"turn_number": 1, "role": "tutor", "text": "What is 2+2?"},
            {"turn_number": 2, "role": "student", "text": "5?"},
        ],
        dimension="rigor",
        student={"mode": "oracle", "reference": "", "context": "Grade 4, Math"},
        rubric={"gold": "rigor", "hint": "Student guessed."},
        provenance={
            "conv_id": "conv-xyz",
            "cut_turn": 2,
            "turn_start": 2,
            "turn_end": 4,
        },
    )


def _second_transcript(scenario):
    t = Transcript(scenario_id=scenario.id, tutor_model="test-model")
    t.generated_turns = [
        {
            "turn_number": 3,
            "role": "TUTOR",
            "text": "Are you sure? Walk me through it.",
        },
        {"turn_number": 4, "role": "STUDENT", "text": "Oh, it's 4."},
    ]
    return t


def _fake_pooled_run_batch(sids):
    """A fake run_batch answering annotate/decompose/structure for every sid.

    Also records each call's display_name so tests can assert pooling.
    """
    calls = []

    def fake_run_batch(client, entries, **kwargs):
        calls.append(
            {
                "display_name": kwargs.get("display_name"),
                "keys": [e["key"] for e in entries],
            }
        )
        keys = {e["key"] for e in entries}
        out = {}
        for sid in sids:
            if f"{sid}__scaffolding__0" in keys:
                out.update(
                    _make_raw_entries(
                        {
                            f"{sid}__scaffolding__0": {
                                "situation": f"situation for {sid}",
                                "action": f"action for {sid}",
                                "result": f"result for {sid}",
                            },
                        }
                    )
                )
            if f"overscaffold__{sid}__0" in keys:
                out.update(
                    _make_raw_entries(
                        {
                            f"action__{sid}__0": [f"facet-a {sid}"],
                            f"overscaffold__{sid}__0": [],
                        }
                    )
                )
            elif f"action__{sid}__0" in keys:
                # Structure pass (action JSON)
                out.update(
                    _make_raw_entries(
                        {
                            f"action__{sid}__0": {"scaffolding": "yes", "rigor": "no"},
                        }
                    )
                )
        return out

    return fake_run_batch, calls


def test_score_batch_pools_three_run_batch_calls(fixture_scenario, fixture_transcript):
    """N scenarios are scored with exactly 3 run_batch calls (one per pass)."""
    from tutormoments.scoring import score_batch

    s2 = _second_scenario()
    t2 = _second_transcript(s2)
    sids = [fixture_scenario.id, s2.id]

    fake_run_batch, calls = _fake_pooled_run_batch(sids)
    with _patch_score(fake_run_batch)[0]:
        annotations = score_batch([(fixture_scenario, fixture_transcript), (s2, t2)])

    assert len(calls) == 3, f"Expected 3 pooled run_batch calls, got {len(calls)}"
    assert [c["display_name"] for c in calls] == [
        "scorer_annotate",
        "scorer_decompose",
        "scorer_structure",
    ]
    # Every pass's entry set must cover BOTH scenarios (pooled, not per-moment)
    for call in calls:
        joined = " ".join(call["keys"])
        assert fixture_scenario.id in joined
        assert s2.id in joined
        # No student-outcome (result__) entries anywhere -- dropped from scorer
        assert not any(k.startswith("result__") for k in call["keys"])

    assert set(annotations) == set(sids)
    a1 = annotations[fixture_scenario.id]
    assert a1.scenario_id == fixture_scenario.id
    assert a1.situation == f"situation for {fixture_scenario.id}"
    assert a1.action_decomposed == [f"facet-a {fixture_scenario.id}"]
    assert a1.action_label == "scaffolding"
    a2 = annotations[s2.id]
    assert a2.situation == f"situation for {s2.id}"
    assert a2.action_decomposed == [f"facet-a {s2.id}"]


def test_score_batch_attributes_usage_per_scenario(
    fixture_scenario, fixture_transcript
):
    """Each Annotation.usage counts only its own scenario's entries."""
    from tutormoments.scoring import score_batch

    s2 = _second_scenario()
    t2 = _second_transcript(s2)
    fake_run_batch, _ = _fake_pooled_run_batch([fixture_scenario.id, s2.id])

    with _patch_score(fake_run_batch)[0]:
        annotations = score_batch([(fixture_scenario, fixture_transcript), (s2, t2)])

    # Per scenario: 1 annotate + 2 decompose (action, overscaffold) + 1 structure
    # = 4 entries x 15 tokens (result decompose + result classify were dropped)
    for sid, ann in annotations.items():
        assert ann.usage["total_tokens"] == 4 * 15, (
            f"{sid}: expected per-scenario usage, got {ann.usage}"
        )


def test_score_batch_empty_transcript_gets_minimal_annotation(
    fixture_scenario, fixture_transcript
):
    """A pair with no generated turns yields the minimal Annotation, no batch entries."""
    from tutormoments.scoring import score_batch

    s2 = _second_scenario()
    t2 = Transcript(scenario_id=s2.id, tutor_model="test-model")  # no generated turns

    fake_run_batch, calls = _fake_pooled_run_batch([fixture_scenario.id])
    with _patch_score(fake_run_batch)[0]:
        annotations = score_batch([(fixture_scenario, fixture_transcript), (s2, t2)])

    assert annotations[s2.id].action == "[No generated turns in transcript]"
    assert annotations[s2.id].usage["total_tokens"] == 0
    # The empty pair must not contribute entries to any pass
    for call in calls:
        assert not any(s2.id in k for k in call["keys"])
    # The non-empty pair still scores normally
    assert (
        annotations[fixture_scenario.id].situation
        == f"situation for {fixture_scenario.id}"
    )


def test_score_batch_empty_input():
    from tutormoments.scoring import score_batch

    def fail_run_batch(*a, **kw):
        raise AssertionError("run_batch must not be called for empty input")

    with _patch_score(fail_run_batch)[0]:
        assert score_batch([]) == {}


def test_score_delegates_to_pooled_path(fixture_scenario, fixture_transcript):
    """score() is the N=1 case of score_batch (same Annotation)."""
    from tutormoments.scoring import score, score_batch

    fake_run_batch, _ = _fake_pooled_run_batch([fixture_scenario.id])
    with _patch_score(fake_run_batch)[0]:
        single = score(fixture_scenario, fixture_transcript)
    fake_run_batch2, _ = _fake_pooled_run_batch([fixture_scenario.id])
    with _patch_score(fake_run_batch2)[0]:
        pooled = score_batch([(fixture_scenario, fixture_transcript)])[
            fixture_scenario.id
        ]

    assert single.to_dict() == pooled.to_dict()
