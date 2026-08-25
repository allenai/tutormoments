"""Tests for the v2 excerpt classification round."""

import json
import os
from collections import Counter

import pytest

from tutormoments_build.v2 import classify_excerpts as C


def _excerpt(
    moment_id, *, scaffolding_present, scaffolding_appropriate=True, split="iteration"
):
    return {
        "moment_id": moment_id,
        "transcript_id": f"t-{moment_id}",
        "conversation_id": f"c-{moment_id}",
        "split": split,
        "post_cut_rows": 2,
        "post_cut_dialogue_rows": 1,
        # One rendering per width, as excerpts.py writes them. They differ so a
        # test can tell which rendering a prompt was actually filled from.
        "excerpts": {
            "full": {
                "excerpt": f"whole transcript\nTurn 1. TUTOR: hello ({moment_id})",
                "context_turns": None,
                "context_start_index": 0,
                "context_rows": 9,
            },
            "20": {
                "excerpt": f"wide lead-up\nTurn 1. TUTOR: hello ({moment_id})",
                "context_turns": 20,
                "context_start_index": 0,
                "context_rows": 4,
            },
            "5": {
                "excerpt": f"Turn 1. TUTOR: hello ({moment_id})",
                "context_turns": 5,
                "context_start_index": 3,
                "context_rows": 1,
            },
        },
        "labels": {
            "scaffolding_appropriate": scaffolding_appropriate,
            "rigor_appropriate": False,
            "scaffolding_present": scaffolding_present,
            "rigor_present": False,
            "over_scaffolding_present": False,
        },
    }


SCAFFOLDED = _excerpt("aaa", scaffolding_present=True)
UNSCAFFOLDED = _excerpt("bbb", scaffolding_present=False)
# Scaffolding delivered where none was called for: the ground truth labels this
# over-scaffolding by rule, so the prompt is not asked about it either.
UNCALLED_FOR = _excerpt("ccc", scaffolding_present=True, scaffolding_appropriate=False)


# ===========================================================================
# Prompt building
# ===========================================================================


def test_fill_substitutes_excerpt_and_keeps_literal_json_braces():
    """str.format would choke on the output-format block; str.replace must not."""
    template = PROMPTS.templates["action_direction"]
    filled = C.fill(template, "Turn 1. TUTOR: hi")

    assert "Turn 1. TUTOR: hi" in filled
    assert "{excerpt}" not in filled
    assert '"scaffolding": "yes or no"' in filled


def test_over_scaffolding_prompt_carries_the_excerpt():
    filled = C.fill(PROMPTS.templates["over_scaffolding"], "Turn 9. TUTOR: so")

    assert "Turn 9. TUTOR: so" in filled
    assert "{excerpt}" not in filled


# ===========================================================================
# Gating
# ===========================================================================


def test_over_scaffolding_needs_scaffolding_both_present_and_appropriate():
    assert C.wants_over_scaffolding(SCAFFOLDED) is True
    assert C.wants_over_scaffolding(UNSCAFFOLDED) is False
    assert C.wants_over_scaffolding(UNCALLED_FOR) is False


def test_missing_labels_do_not_trigger_the_over_scaffolding_pass():
    assert C.wants_over_scaffolding({"moment_id": "x"}) is False


def test_skip_reason_separates_the_two_halves_of_the_premise():
    assert C.skip_reason(UNSCAFFOLDED) == "no_scaffolding"
    assert C.skip_reason(UNCALLED_FOR) == "not_appropriate"


def test_build_entries_asks_action_of_all_and_over_scaffolding_of_gated_only():
    entries, counts = C.build_entries([SCAFFOLDED, UNSCAFFOLDED, UNCALLED_FOR], PROMPTS)
    keys = {entry["key"] for entry in entries}

    assert keys == {"action__aaa", "action__bbb", "action__ccc", "overscaffold__aaa"}
    assert counts["action_direction"] == 3
    assert counts["over_scaffolding"] == 1
    assert counts["over_scaffolding_not_asked"] == 2
    assert counts["over_scaffolding_skip_no_scaffolding"] == 1
    assert counts["over_scaffolding_skip_not_appropriate"] == 1


def test_entry_keys_split_back_on_the_first_separator():
    """The key scheme has to survive round-tripping to a moment_id."""
    entries, _ = C.build_entries([SCAFFOLDED], PROMPTS)
    prefix, moment_id = entries[0]["key"].split("__", 1)

    assert prefix == C.ACTION_PREFIX
    assert moment_id == "aaa"


# ===========================================================================
# Parsing: action direction
# ===========================================================================


@pytest.mark.parametrize(
    "scaffolding,rigor",
    [("yes", "no"), ("no", "yes"), ("yes", "yes"), ("no", "no")],
)
def test_parse_action_direction_reads_both_dimensions(scaffolding, rigor):
    text = json.dumps(
        {
            "description": "The tutor asks a guiding question.",
            "scaffolding": scaffolding,
            "rigor": rigor,
        }
    )
    fields, had_error = C.parse_action_direction(text)

    assert had_error is False
    assert fields["scaffolding"] is (scaffolding == "yes")
    assert fields["rigor"] is (rigor == "yes")
    assert fields["description"] == "The tutor asks a guiding question."


def test_parse_action_direction_unwraps_a_list_and_tolerates_prose():
    text = 'Here you go:\n[{"description": "d", "scaffolding": "yes", "rigor": "no"}]'
    fields, had_error = C.parse_action_direction(text)

    assert had_error is False
    assert (fields["scaffolding"], fields["rigor"]) == (True, False)
    assert fields["description"] == "d"


def test_parse_action_direction_reports_unparseable_output_as_none():
    fields, had_error = C.parse_action_direction("I could not decide.")

    assert had_error is True
    assert fields["scaffolding"] is None
    assert fields["rigor"] is None


def test_parse_action_direction_on_empty_text():
    fields, had_error = C.parse_action_direction("")

    assert had_error is True
    assert fields == {"scaffolding": None, "rigor": None, "description": ""}


# ===========================================================================
# Parsing: over-scaffolding
# ===========================================================================


@pytest.mark.parametrize(
    "key", ["over-scaffolding", "over_scaffolding", "overscaffolding"]
)
def test_parse_over_scaffolding_accepts_each_key_spelling(key):
    fields, had_error = C.parse_over_scaffolding(
        json.dumps({"description": "Gives away key steps.", key: "yes"})
    )

    assert had_error is False
    assert fields["over_scaffolding"] is True
    assert fields["description"] == "Gives away key steps."


def test_parse_over_scaffolding_reads_no():
    fields, had_error = C.parse_over_scaffolding(
        json.dumps({"description": "d", "over-scaffolding": "no"})
    )

    assert had_error is False
    assert fields["over_scaffolding"] is False


def test_parse_over_scaffolding_falls_back_to_regex():
    fields, had_error = C.parse_over_scaffolding("over-scaffolding: yes, because ...")

    assert had_error is False
    assert fields["over_scaffolding"] is True


def test_parse_over_scaffolding_reports_unparseable_output_as_none():
    fields, had_error = C.parse_over_scaffolding("maybe")

    assert had_error is True
    assert fields["over_scaffolding"] is None


# ===========================================================================
# Assembly
# ===========================================================================


CFG = {"model": "test-model", "thinking": "adaptive"}

# The real prompt set, loaded once. The tests assert on how the templates are
# filled and recorded, so a stub would only be asserting against itself.
PROMPTS = C.load_prompt_set(C.latest_prompt_version())
ACTION_PROMPT_PATH = PROMPTS.paths["action_direction"]
OVER_PROMPT_PATH = PROMPTS.paths["over_scaffolding"]

RAW = {
    "action__aaa": {
        "text": json.dumps({"description": "d", "scaffolding": "yes", "rigor": "no"}),
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    },
    "overscaffold__aaa": {
        "text": json.dumps({"description": "e", "over-scaffolding": "yes"}),
        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    },
    "action__bbb": {
        "text": json.dumps({"description": "f", "scaffolding": "no", "rigor": "yes"}),
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    },
}


def test_build_record_carries_both_passes_and_the_gold_labels():
    record = C.build_record(SCAFFOLDED, RAW, CFG, PROMPTS)

    assert record["moment_id"] == "aaa"
    assert record["model"] == "test-model"
    assert record["action_direction"]["scaffolding"] is True
    assert record["over_scaffolding"]["over_scaffolding"] is True
    assert record["over_scaffolding_asked"] is True
    assert record["labels"] == SCAFFOLDED["labels"]


def test_build_record_sums_usage_across_both_passes():
    record = C.build_record(SCAFFOLDED, RAW, CFG, PROMPTS)

    assert record["usage"] == {
        "input_tokens": 15,
        "output_tokens": 3,
        "total_tokens": 18,
    }


def test_ungated_moment_gets_null_over_scaffolding_not_false():
    """ "Not asked" has to stay distinguishable from "asked and told no"."""
    record = C.build_record(UNSCAFFOLDED, RAW, CFG, PROMPTS)

    assert record["over_scaffolding"] is None
    assert record["over_scaffolding_asked"] is False
    assert record["usage"]["total_tokens"] == 10


def test_scaffolding_where_none_was_called_for_is_not_asked_either():
    """The ground truth labels it over-scaffolding by rule, so the model is not
    scored on an answer that was fixed before it saw the text."""
    record = C.build_record(UNCALLED_FOR, RAW, CFG, PROMPTS)

    assert record["over_scaffolding"] is None
    assert record["over_scaffolding_asked"] is False


def test_build_record_keeps_the_raw_response():
    record = C.build_record(SCAFFOLDED, RAW, CFG, PROMPTS)

    assert record["action_direction"]["raw"] == RAW["action__aaa"]["text"]


def test_build_record_records_a_batch_entry_error():
    raw = {"action__aaa": {"text": "", "error": "overloaded"}}
    record = C.build_record(UNSCAFFOLDED, raw, CFG, PROMPTS)

    assert record["action_direction"]["error"] is None  # bbb has no entry in raw
    record = C.build_record(SCAFFOLDED, raw, CFG, PROMPTS)
    assert record["action_direction"]["error"] == "overloaded"
    assert record["action_direction"]["parse_error"] is True


def test_missing_batch_result_degrades_to_a_parse_error():
    record = C.build_record(SCAFFOLDED, {}, CFG, PROMPTS)

    assert record["action_direction"]["scaffolding"] is None
    assert record["action_direction"]["parse_error"] is True
    assert record["usage"] == C.ZERO_USAGE


# ===========================================================================
# Round-running (no API calls)
# ===========================================================================


def test_classify_splits_results_back_apart(monkeypatch):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)

    out, counts = C.classify(
        {"iteration": [SCAFFOLDED], "test": [UNSCAFFOLDED]}, CFG, PROMPTS
    )

    assert [r["moment_id"] for r in out["iteration"]] == ["aaa"]
    assert [r["moment_id"] for r in out["test"]] == ["bbb"]
    assert counts["action_direction"] == 2
    assert counts["action_direction_unparsed"] == 0


def test_classify_counts_unparsed_responses(monkeypatch):
    monkeypatch.setattr(
        C,
        "run_entries",
        lambda entries, cfg, batch_id=None: {"action__aaa": {"text": "nope"}},
    )

    _, counts = C.classify({"iteration": [SCAFFOLDED]}, CFG, PROMPTS)

    assert counts["action_direction_unparsed"] == 1
    assert counts["over_scaffolding_unparsed"] == 1


def test_run_entries_forwards_the_models_generation_knobs(monkeypatch):
    """thinking/effort/reasoning_effort have to reach the batch, or a comparison
    model runs on the wrong settings while the record claims otherwise."""
    import tutormoments.client as client

    seen = {}

    monkeypatch.setattr(
        client, "ModelClient", lambda model: seen.setdefault("model", model)
    )
    monkeypatch.setattr(
        client, "run_batch", lambda c, entries, **kwargs: seen.update(kwargs) or {}
    )

    C.run_entries(
        [],
        {
            "model": "gpt-5.5-2026-04-23",
            "thinking": True,
            "reasoning_effort": "high",
            "poll_interval": 30,
        },
    )

    assert seen["model"] == "gpt-5.5-2026-04-23"
    assert seen["thinking"] is True
    assert seen["reasoning_effort"] == "high"
    assert seen["effort"] == ""
    assert seen["poll_interval"] == 30


# ===========================================================================
# Config
# ===========================================================================


@pytest.mark.parametrize(
    "value,expected",
    [
        ("adaptive", True),
        ("enabled", True),
        (True, True),
        ("disabled", False),
        (False, False),
    ],
)
def test_use_thinking_normalises_the_config_value(value, expected):
    assert C.use_thinking({"thinking": value}) is expected


def test_phase_config_reads_the_v2_block():
    cfg = C.phase_config()

    assert cfg["model"] == "claude-opus-5"
    assert cfg["thinking"] == "adaptive"
    # Pinned, not left at the API default, so the baseline is effort-matched to
    # the xhigh comparison models and reproducible across rounds.
    assert cfg["effort"] == "xhigh"


def test_the_fallback_spec_matches_the_shipped_v2_block():
    """The fallback stands in for the v2 block when a stripped custom config has
    none. Drifting from the shipped block would run the baseline at a different
    reasoning depth than the default config does, silently."""
    shipped = C.phase_config()

    for key, value in C.FALLBACK_SPEC.items():
        assert shipped[key] == value


def test_model_override_takes_its_knobs_from_the_roster():
    """A comparison model is configured in the config file, not on the CLI."""
    cfg = C.phase_config(model="claude-sonnet-5")

    assert cfg["model"] == "claude-sonnet-5"
    assert cfg["thinking"] == "adaptive"
    assert cfg["effort"] == "xhigh"
    assert cfg["poll_interval"] == 60  # a round property, not a model one


def test_a_rostered_model_without_thinking_does_not_inherit_adaptive():
    """Otherwise a model configured thinking-off would silently run with the
    v2 block's adaptive thinking, and the record would name the wrong setting."""
    cfg = C.phase_config(model="deepseek-ai/DeepSeek-V4-Pro")

    assert cfg["thinking"] is False
    assert C.use_thinking(cfg) is False


def test_the_v2_model_needs_no_roster_entry():
    """It is fully specified by the v2 block; naming it explicitly must work
    even though it is not on the tutor roster."""
    assert C.phase_config(model="claude-opus-5")["model"] == "claude-opus-5"


def test_the_openai_comparison_model_resolves_and_can_be_batched():
    """gpt-5.6-sol is one of the cross-vendor comparison models for this round.
    Its `v2.models` entry has to resolve to an OpenAI reasoning config, and
    OpenAI has to be a provider run_batch supports -- either failure surfaces
    only after a whole round has been built and submitted."""
    from tutormoments.client import infer_provider

    cfg = C.phase_config(model="gpt-5.6-sol")

    assert cfg["model"] == "gpt-5.6-sol"
    assert C.use_thinking(cfg) is True
    assert cfg["reasoning_effort"] == "xhigh"
    # `effort` is the Anthropic knob; sending it to OpenAI would be rejected.
    assert cfg.get("effort", "") == ""
    assert infer_provider(cfg["model"]) == "openai"


def test_the_record_names_the_reasoning_depth_the_round_ran_at():
    """Without these, an effort-pinned round and one at the API default write
    identical metadata, and a prediction file cannot say how its labels were
    made."""
    record = C.build_record(
        SCAFFOLDED,
        RAW,
        {"model": "gpt-5.6-sol", "thinking": True, "reasoning_effort": "xhigh"},
        PROMPTS,
    )

    assert record["reasoning_effort"] == "xhigh"
    # The Anthropic knob was not in play, and must not read as if it were.
    assert record["effort"] is None
    assert record["thinking_budget"] is None


def test_the_report_names_the_effort_it_will_submit_at(capsys):
    cfg = {"model": "gpt-5.6-sol", "thinking": True, "reasoning_effort": "xhigh"}

    out = C.report({}, Counter(), dry_run=True, cfg=cfg)

    assert "model: gpt-5.6-sol (thinking: True, reasoning_effort: xhigh)" in out


def test_the_scoped_model_table_does_not_leak_into_the_spec():
    """`v2.models` is a lookup table, not a setting of the round. Left in, it
    would be recorded in every prediction record and passed to the batch."""
    assert "models" not in C.phase_config()
    assert "models" not in C.phase_config(model="gpt-5.6-sol")


def test_a_scoped_model_shadows_nothing_on_the_tutor_roster():
    """Both lookups have to keep working: the roster is where an id already
    configured for `tutormoments run` gets its knobs from."""
    cfg = C.phase_config(model="claude-sonnet-5")

    assert cfg["effort"] == "xhigh"
    assert cfg["thinking"] == "adaptive"


def test_model_knobs_rejects_an_unroutable_scoped_id():
    """A `v2.models` entry skips resolve_model, so provider inference is the only
    thing standing between a typo'd id and a failed batch submission."""
    with pytest.raises(ValueError, match="Cannot infer provider"):
        C.model_knobs("gtp-5.6-sol", {"gtp-5.6-sol": {"thinking": True}})


def test_an_unknown_model_error_names_both_config_places():
    """The id is missing from two tables; the error has to say so, or the fix
    looks like it belongs on the tutor roster."""
    with pytest.raises(ValueError, match="not in roster") as exc:
        C.phase_config(model="gpt-5.6-sol-typo")

    assert "v2.models" in str(exc.value)
    assert "gpt-5.6-sol" in str(exc.value)


def test_an_unrostered_model_is_refused_with_the_roster_listed():
    with pytest.raises(ValueError, match="not in roster"):
        C.phase_config(model="claude-opus-5-typo")


# ===========================================================================
# I/O
# ===========================================================================


def test_load_excerpts_skips_a_missing_split(tmp_path):
    path = tmp_path / "iteration.jsonl"
    path.write_text(json.dumps(SCAFFOLDED) + "\n", encoding="utf-8")

    out = C.load_excerpts(str(tmp_path))

    assert list(out) == ["iteration"]
    assert out["iteration"][0]["moment_id"] == "aaa"


def test_load_excerpts_raises_when_nothing_is_there(tmp_path):
    with pytest.raises(FileNotFoundError, match="v2.excerpts"):
        C.load_excerpts(str(tmp_path))


def test_prediction_dir_flattens_a_vendor_prefixed_id():
    """A vendor-prefixed id must not nest two directories."""
    assert C.prediction_dir("out", "deepseek-ai/DeepSeek-V4-Pro", "1") == os.path.join(
        "out", "deepseek-ai_DeepSeek-V4-Pro", "1"
    )


# ===========================================================================
# Prompt versions
# ===========================================================================


def test_load_prompt_set_reads_the_version_asked_for():
    prompts = C.load_prompt_set("1")

    assert prompts.version == "1"
    assert prompts.paths["action_direction"] == "prompts/v2/1/action_direction.md"
    assert set(prompts.templates) == {"action_direction", "over_scaffolding"}
    assert all(text.strip() for text in prompts.templates.values())


def test_an_unknown_prompt_version_names_the_ones_that_exist():
    """The error has to be actionable: a typo'd version should say what is there."""
    with pytest.raises(FileNotFoundError, match="versions available: 1"):
        C.load_prompt_set("nope")


def test_a_record_names_the_prompt_version_that_made_it():
    """A prediction file has to say which prompts wrote its labels."""
    record = C.build_record(SCAFFOLDED, RAW, CFG, PROMPTS)

    assert record["prompt_version"] == PROMPTS.version
    assert record["prompts"]["action_direction"] == PROMPTS.paths["action_direction"]


def test_the_default_prompt_version_is_the_newest_one_present(monkeypatch):
    """A bare run must classify with the newest prompts, not a pinned number.

    A default pinned to a version would leave an edit unexercised: the run would
    file predictions under the old version and the comparison the edit was made
    to settle would never be made.
    """
    monkeypatch.setattr(C, "available_prompt_versions", lambda: ["1", "2", "3"])

    assert C.latest_prompt_version() == "3"
    assert C.build_parser().parse_args([]).prompt_version == "3"


def test_the_newest_prompt_version_is_numeric_not_alphabetical(monkeypatch):
    """v10 is newer than v9, and a scratch directory is not a version at all."""
    monkeypatch.setattr(
        C, "available_prompt_versions", lambda: ["1", "9", "10", "scratch"]
    )

    assert C.latest_prompt_version() == "10"


def test_an_unreadable_prompt_root_falls_back_to_a_version(monkeypatch):
    """With nothing to read, load_prompt_set reports the failure, not this."""
    monkeypatch.setattr(C, "available_prompt_versions", list)

    assert C.latest_prompt_version() == C.FALLBACK_PROMPT_VERSION


def test_prediction_dir_files_a_prompt_version_apart_from_its_predecessor():
    """Two prompt versions of one model must not share a file."""
    assert C.prediction_dir("out", "claude-opus-5", "1") != C.prediction_dir(
        "out", "claude-opus-5", "2"
    )


def test_write_split_round_trips(tmp_path):
    record = C.build_record(SCAFFOLDED, RAW, CFG, PROMPTS)
    path = C.write_split(str(tmp_path), "iteration", [record])

    with open(path, encoding="utf-8") as fh:
        assert json.loads(fh.readline()) == record


# ===========================================================================
# CLI
# ===========================================================================


def _write_excerpts(tmp_path):
    (tmp_path / "iteration.jsonl").write_text(
        json.dumps(SCAFFOLDED) + "\n" + json.dumps(UNSCAFFOLDED) + "\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def test_the_test_split_is_held_out_unless_asked_for_by_name(tmp_path, monkeypatch):
    """The default run must not spend the held-out split on prompt iteration."""
    excerpt_dir = _write_excerpts(tmp_path)
    (tmp_path / "test.jsonl").write_text(
        json.dumps(dict(UNSCAFFOLDED, moment_id="ccc", split="test")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)
    out_dir = tmp_path / "out"

    assert C.main(["--excerpt-dir", excerpt_dir, "--out-dir", str(out_dir)]) == 0
    assert [p.name for p in sorted(out_dir.glob("*/*/*.jsonl"))] == ["iteration.jsonl"]

    assert (
        C.main(
            ["--excerpt-dir", excerpt_dir, "--out-dir", str(out_dir), "--splits", "test"]
        )
        == 0
    )
    assert [p.name for p in sorted(out_dir.glob("*/*/*.jsonl"))] == [
        "iteration.jsonl",
        "test.jsonl",
    ]


def test_dry_run_makes_no_api_calls_and_writes_nothing(tmp_path, monkeypatch, capsys):
    def _boom(*args, **kwargs):
        raise AssertionError("--dry-run must not call the API")

    monkeypatch.setattr(C, "run_entries", _boom)
    out_dir = tmp_path / "out"

    code = C.main(
        [
            "--excerpt-dir",
            _write_excerpts(tmp_path),
            "--out-dir",
            str(out_dir),
            "--splits",
            "iteration",
            "--dry-run",
        ]
    )

    assert code == 0
    assert not out_dir.exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_limit_truncates_each_split(tmp_path, monkeypatch):
    seen = {}

    def _capture(entries, cfg, batch_id=None):
        seen["keys"] = [entry["key"] for entry in entries]
        return RAW

    monkeypatch.setattr(C, "run_entries", _capture)

    C.main(
        [
            "--excerpt-dir",
            _write_excerpts(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--splits",
            "iteration",
            "--limit",
            "1",
        ]
    )

    assert seen["keys"] == ["action__aaa", "overscaffold__aaa"]


def test_predictions_are_filed_under_the_model_that_made_them(tmp_path, monkeypatch):
    """A second model must not overwrite the first round's predictions."""
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)
    out_dir = tmp_path / "out"

    code = C.main(
        [
            "--excerpt-dir",
            _write_excerpts(tmp_path),
            "--out-dir",
            str(out_dir),
            "--splits",
            "iteration",
            "--model",
            "claude-sonnet-5",
            # Pinned: this test is about the model directory, not the version.
            "--prompt-version",
            "1",
        ]
    )

    assert code == 0
    path = out_dir / "claude-sonnet-5" / "1" / "iteration.jsonl"
    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["model"] == "claude-sonnet-5"


def test_the_default_model_gets_its_own_directory_too(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)
    out_dir = tmp_path / "out"

    C.main(
        [
            "--excerpt-dir",
            _write_excerpts(tmp_path),
            "--out-dir",
            str(out_dir),
            "--splits",
            "iteration",
            "--prompt-version",
            "1",
        ]
    )

    assert (out_dir / "claude-opus-5" / "1" / "iteration.jsonl").exists()


def test_the_chosen_model_reaches_the_batch(tmp_path, monkeypatch):
    seen = {}

    def _capture(entries, cfg, batch_id=None):
        seen["cfg"] = cfg
        return RAW

    monkeypatch.setattr(C, "run_entries", _capture)

    C.main(
        [
            "--excerpt-dir",
            _write_excerpts(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--splits",
            "iteration",
            "--model",
            "claude-sonnet-5",
        ]
    )

    assert seen["cfg"]["model"] == "claude-sonnet-5"
    assert seen["cfg"]["effort"] == "xhigh"


def test_an_unknown_model_fails_on_the_dry_run(tmp_path, monkeypatch):
    """The typo should surface before a real round is ever submitted."""

    def _boom(*args, **kwargs):
        raise AssertionError("--dry-run must not call the API")

    monkeypatch.setattr(C, "run_entries", _boom)

    with pytest.raises(ValueError, match="not in roster"):
        C.main(
            [
                "--excerpt-dir",
                _write_excerpts(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--dry-run",
                "--model",
                "claude-sonnet-5-typo",
            ]
        )


def test_the_report_names_the_model_that_ran(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)

    C.main(
        [
            "--excerpt-dir",
            _write_excerpts(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--splits",
            "iteration",
            "--model",
            "claude-sonnet-5",
        ]
    )

    assert "model: claude-sonnet-5" in capsys.readouterr().out


def test_print_shows_both_prompts_for_a_scaffolded_moment(tmp_path, capsys):
    code = C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print", "aaa"])
    out = capsys.readouterr().out

    assert code == 0
    assert ACTION_PROMPT_PATH in out
    assert "Turn 1. TUTOR: hello (aaa)" in out
    assert "not asked" not in out


def test_print_says_why_the_over_scaffolding_prompt_is_absent(tmp_path, capsys):
    code = C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print", "bbb"])

    assert code == 0
    assert "not asked" in capsys.readouterr().out


def test_print_reports_an_unknown_moment(tmp_path, capsys):
    code = C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print", "zzz"])

    assert code == 1
    assert "no moment matching" in capsys.readouterr().err


def test_bare_print_picks_a_random_moment(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C.random, "choice", lambda pool: pool[-1])

    code = C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print"])
    out = capsys.readouterr().out

    assert code == 0
    assert "moment bbb (iteration)" in out
    assert ACTION_PROMPT_PATH in out


def test_random_pick_draws_from_every_loaded_split():
    seen = set()
    excerpts = {"iteration": [SCAFFOLDED], "test": [UNSCAFFOLDED]}
    for _ in range(50):
        seen.add(C.pick_moment(excerpts, C.RANDOM)["moment_id"])

    assert seen == {"aaa", "bbb"}


def test_random_pick_respects_a_narrowed_pool():
    picked = C.pick_moment({"iteration": [SCAFFOLDED]}, C.RANDOM)

    assert picked["moment_id"] == "aaa"


def test_print_names_the_moment_so_a_random_pick_is_reproducible(tmp_path, capsys):
    C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print", "aaa"])

    assert "===== moment aaa (iteration) =====" in capsys.readouterr().out


def test_pick_moment_returns_none_on_an_empty_pool():
    assert C.pick_moment({"iteration": []}, C.RANDOM) is None


def test_bare_print_reports_an_empty_pool(tmp_path, capsys):
    (tmp_path / "iteration.jsonl").write_text("", encoding="utf-8")

    code = C.main(["--excerpt-dir", str(tmp_path), "--print"])

    assert code == 1
    assert "no moments loaded" in capsys.readouterr().err


# ===========================================================================
# Post-cut gate
# ===========================================================================

EMPTY_POST_CUT = dict(
    SCAFFOLDED, moment_id="ccc", post_cut_rows=0, post_cut_dialogue_rows=0
)


def test_moment_with_nothing_after_the_cut_is_not_classified():
    assert C.has_post_cut_content(EMPTY_POST_CUT) is False
    assert C.has_post_cut_content(SCAFFOLDED) is True


def test_enrichment_only_post_cut_content_still_counts():
    """Screen activity after the cut is a pedagogical move, not emptiness."""
    screen_only = dict(SCAFFOLDED, post_cut_rows=1, post_cut_dialogue_rows=0)

    assert C.has_post_cut_content(screen_only) is True


def test_an_excerpt_predating_the_field_is_classified():
    assert C.has_post_cut_content({"moment_id": "old"}) is True


def test_no_entries_are_built_for_an_empty_post_cut_moment():
    entries, counts = C.build_entries([SCAFFOLDED, EMPTY_POST_CUT], PROMPTS)

    assert {e["key"] for e in entries} == {"action__aaa", "overscaffold__aaa"}
    assert counts["no_post_cut_content"] == 1


def test_skipped_moment_still_gets_a_record_naming_the_reason():
    record = C.build_record(EMPTY_POST_CUT, RAW, CFG, PROMPTS)

    assert record["classified"] is False
    assert record["skipped"] == "no_post_cut_content"
    assert record["action_direction"] is None
    assert record["over_scaffolding"] is None
    assert record["usage"] == C.ZERO_USAGE
    assert record["labels"] == EMPTY_POST_CUT["labels"]


def test_skipped_moments_are_not_counted_as_parse_failures(monkeypatch):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: {})

    out, counts = C.classify({"iteration": [EMPTY_POST_CUT]}, CFG, PROMPTS)

    assert counts["action_direction_unparsed"] == 0
    assert len(out["iteration"]) == 1


def test_report_names_the_skipped_moments(monkeypatch, capsys):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)

    out, counts = C.classify({"iteration": [SCAFFOLDED, EMPTY_POST_CUT]}, CFG, PROMPTS)

    assert "1 moment(s) not classified" in C.report(out, counts, dry_run=False)


def test_report_breaks_the_over_scaffolding_skips_down_by_reason(monkeypatch, capsys):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)

    out, counts = C.classify(
        {"iteration": [SCAFFOLDED, UNSCAFFOLDED, UNCALLED_FOR]}, CFG, PROMPTS
    )
    text = C.report(out, counts, dry_run=False)

    assert "2 not asked" in text
    assert "1 gold says no scaffolding here" in text
    assert "1 scaffolding not called for" in text


def test_missing_field_warns_with_the_rebuild_command(caplog):
    with caplog.at_level("WARNING"):
        C.build_entries(
            [
                {
                    "moment_id": "old",
                    "labels": {},
                    "excerpts": {C.EXCERPT_WIDTH: {"excerpt": "x"}},
                }
            ],
            PROMPTS,
        )

    assert "v2.excerpts" in caplog.text


# ===========================================================================
# Excerpt rendering the prompts read
# ===========================================================================


def test_both_prompts_are_filled_from_the_whole_transcript():
    """Neither pass reads a lead-up window; both get the full rendering."""
    entries, _ = C.build_entries([SCAFFOLDED], PROMPTS)
    by_key = {
        e["key"]: e["request"]["contents"][0]["parts"][0]["text"] for e in entries
    }

    assert "whole transcript" in by_key["action__aaa"]
    assert "whole transcript" in by_key["overscaffold__aaa"]
    assert "wide lead-up" not in by_key["overscaffold__aaa"]


def test_excerpt_at_defaults_to_the_full_rendering():
    assert C.EXCERPT_WIDTH == "full"
    assert C.excerpt_at(SCAFFOLDED).startswith("whole transcript")
    assert C.excerpt_at(SCAFFOLDED, "20").startswith("wide lead-up")


def test_a_missing_width_names_the_rebuild_command():
    """An excerpt file built before `full` existed must not silently truncate."""
    windowed_only = dict(SCAFFOLDED, excerpts={"5": {"excerpt": "x"}})

    with pytest.raises(KeyError, match="v2.excerpts --context-turns full"):
        C.excerpt_at(windowed_only)


def test_the_widths_each_pass_used_are_recorded():
    record = C.build_record(SCAFFOLDED, RAW, CFG, PROMPTS)

    assert record["context_turns"] == {
        "action_direction": "full",
        "over_scaffolding": "full",
    }


def test_print_labels_each_prompt_with_the_rendering_it_read(tmp_path, capsys):
    C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print", "aaa"])
    out = capsys.readouterr().out

    assert out.count(f"(@{C.EXCERPT_WIDTH} excerpt)") == 2


# ===========================================================================
# Overwrite protection
# ===========================================================================


def test_a_second_round_refuses_to_replace_the_first(tmp_path, monkeypatch):
    """Predictions cost money to make; a re-run must not silently discard them."""
    excerpt_dir = _write_excerpts(tmp_path)
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)
    out_dir = tmp_path / "out"
    # Version pinned: what is under test is the refusal, not which prompts ran.
    argv = ["--excerpt-dir", excerpt_dir, "--out-dir", str(out_dir), "--prompt-version", "1"]

    assert C.main(argv) == 0
    written = (out_dir / "claude-opus-5" / "1" / "iteration.jsonl").read_text(
        encoding="utf-8"
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        C.main(argv)

    assert (out_dir / "claude-opus-5" / "1" / "iteration.jsonl").read_text(
        encoding="utf-8"
    ) == written


def test_the_refusal_comes_before_the_batch_is_submitted(tmp_path, monkeypatch):
    """Refusing after the round would have already spent the money it protects."""
    excerpt_dir = _write_excerpts(tmp_path)
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)
    out_dir = tmp_path / "out"
    argv = ["--excerpt-dir", excerpt_dir, "--out-dir", str(out_dir)]
    assert C.main(argv) == 0

    def _boom(*args, **kwargs):
        raise AssertionError("submitted a batch it was going to refuse to write")

    monkeypatch.setattr(C, "run_entries", _boom)
    with pytest.raises(FileExistsError):
        C.main(argv)


def test_overwrite_replaces_the_earlier_round(tmp_path, monkeypatch):
    excerpt_dir = _write_excerpts(tmp_path)
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)
    out_dir = tmp_path / "out"
    argv = ["--excerpt-dir", excerpt_dir, "--out-dir", str(out_dir), "--prompt-version", "1"]

    assert C.main(argv) == 0
    assert C.main(argv + ["--overwrite"]) == 0
    assert (out_dir / "claude-opus-5" / "1" / "iteration.jsonl").exists()


def test_a_new_prompt_version_is_not_an_overwrite(tmp_path, monkeypatch):
    """The version directory is what keeps a revision from clobbering the last."""
    excerpt_dir = _write_excerpts(tmp_path)
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)
    monkeypatch.setattr(
        C, "load_prompt_set", lambda version: C.PromptSet(
            version=version, paths=PROMPTS.paths, templates=PROMPTS.templates
        )
    )
    out_dir = tmp_path / "out"
    argv = ["--excerpt-dir", excerpt_dir, "--out-dir", str(out_dir)]

    assert C.main(argv + ["--prompt-version", "1"]) == 0
    assert C.main(argv + ["--prompt-version", "2"]) == 0

    assert (out_dir / "claude-opus-5" / "1" / "iteration.jsonl").exists()
    assert (out_dir / "claude-opus-5" / "2" / "iteration.jsonl").exists()
