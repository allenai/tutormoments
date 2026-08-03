"""Tests for tutormoments_build.classify -- pure situation/effectiveness classification helpers.

These cover only the pure, LLM-free helpers used by the ground-truth build
pipeline.
"""

import pytest

from tutormoments_build.classify import (
    JUNK_TEXTS,
    VALID_LABELS,
    VALID_SITUATION_LABELS,
    _parse_situation_label,
    load_labeller_templates,
    load_situation_prompt,
    pick_template,
)

# --- _parse_situation_label -------------------------------------------------


def test_parse_situation_label_valid_json():
    label, had_error = _parse_situation_label('{"scaffolding": "yes", "rigor": "no"}')
    assert label == {"scaffolding": "yes", "rigor": "no"}
    assert had_error is False


def test_parse_situation_label_unwraps_list():
    label, had_error = _parse_situation_label('[{"scaffolding": "no", "rigor": "yes"}]')
    assert label == {"scaffolding": "no", "rigor": "yes"}
    assert had_error is False


def test_parse_situation_label_coerces_unknown_value_to_unclear():
    label, had_error = _parse_situation_label(
        '{"scaffolding": "maybe", "rigor": "yes"}'
    )
    assert label == {"scaffolding": "unclear", "rigor": "yes"}
    assert had_error is False


def test_parse_situation_label_regex_fallback_on_unquoted():
    label, had_error = _parse_situation_label("scaffolding: yes, rigor: no_mention")
    assert label == {"scaffolding": "yes", "rigor": "no_mention"}
    assert had_error is False


def test_parse_situation_label_unparseable_is_unclear_with_error():
    label, had_error = _parse_situation_label("complete gibberish with no labels")
    assert label == {"scaffolding": "unclear", "rigor": "unclear"}
    assert had_error is True


def test_valid_situation_labels_set():
    assert VALID_SITUATION_LABELS == {"yes", "no", "unclear", "no_mention"}


# --- load_labeller_templates / pick_template --------------------------------


def test_load_labeller_templates_dict_routes_per_type():
    templates = load_labeller_templates(
        {"scaffolding": "classify_scaffolding", "rapport": "classify_rapport"}
    )
    assert set(templates.keys()) == {"scaffolding", "rapport"}
    assert templates["scaffolding"]  # non-empty prompt text
    assert templates["rapport"]
    assert templates["scaffolding"] != templates["rapport"]


def test_load_labeller_templates_string_uses_none_fallback_key():
    templates = load_labeller_templates("classify_scaffolding")
    assert set(templates.keys()) == {None}
    assert templates[None]


def test_load_situation_prompt_returns_template():
    prompt = load_situation_prompt()
    assert isinstance(prompt, str)
    assert prompt.strip()


def test_pick_template_returns_type_specific():
    templates = {"scaffolding": "S", "rapport": "R"}
    assert pick_template(templates, "scaffolding") == "S"
    assert pick_template(templates, "rapport") == "R"


def test_pick_template_falls_back_to_none_key():
    templates = {None: "FALLBACK"}
    assert pick_template(templates, "scaffolding") == "FALLBACK"


def test_pick_template_raises_when_unmapped_and_no_fallback():
    with pytest.raises(KeyError):
        pick_template({"scaffolding": "S"}, "rapport")


# --- JUNK_TEXTS -------------------------------------------------------------


def test_junk_texts_contains_known_placeholders():
    assert "" in JUNK_TEXTS
    assert "n/a" in JUNK_TEXTS
    assert "test" in JUNK_TEXTS
    assert VALID_LABELS == {"effective", "partial", "ineffective"}
