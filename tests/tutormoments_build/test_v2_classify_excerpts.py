"""Tests for the v2 excerpt classification round."""

import json

import pytest

from tutormoments_build.v2 import classify_excerpts as C


def _excerpt(moment_id, *, scaffolding_present, split="iteration"):
    return {
        "moment_id": moment_id,
        "transcript_id": f"t-{moment_id}",
        "conversation_id": f"c-{moment_id}",
        "split": split,
        "post_cut_rows": 2,
        "post_cut_dialogue_rows": 1,
        # One rendering per width, as excerpts.py writes them. The two differ so
        # a test can tell which window a prompt was actually filled from.
        "excerpts": {
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
            "scaffolding_appropriate": True,
            "rigor_appropriate": False,
            "scaffolding_present": scaffolding_present,
            "rigor_present": False,
            "over_scaffolding_present": False,
        },
    }


SCAFFOLDED = _excerpt("aaa", scaffolding_present=True)
UNSCAFFOLDED = _excerpt("bbb", scaffolding_present=False)


# ===========================================================================
# Prompt building
# ===========================================================================


def test_fill_substitutes_excerpt_and_keeps_literal_json_braces():
    """str.format would choke on the output-format block; str.replace must not."""
    template = C.resource_text(C.ACTION_DIRECTION_PROMPT)
    filled = C.fill(template, "Turn 1. TUTOR: hi")

    assert "Turn 1. TUTOR: hi" in filled
    assert "{excerpt}" not in filled
    assert '"scaffolding": "yes or no"' in filled


def test_over_scaffolding_prompt_carries_the_excerpt():
    filled = C.fill(C.resource_text(C.OVER_SCAFFOLDING_PROMPT), "Turn 9. TUTOR: so")

    assert "Turn 9. TUTOR: so" in filled
    assert "{excerpt}" not in filled


# ===========================================================================
# Gating
# ===========================================================================


def test_over_scaffolding_gated_on_the_gold_scaffolding_label():
    assert C.wants_over_scaffolding(SCAFFOLDED) is True
    assert C.wants_over_scaffolding(UNSCAFFOLDED) is False


def test_missing_labels_do_not_trigger_the_over_scaffolding_pass():
    assert C.wants_over_scaffolding({"moment_id": "x"}) is False


def test_build_entries_asks_action_of_all_and_over_scaffolding_of_scaffolded_only():
    entries, counts = C.build_entries([SCAFFOLDED, UNSCAFFOLDED])
    keys = {entry["key"] for entry in entries}

    assert keys == {"action__aaa", "action__bbb", "overscaffold__aaa"}
    assert counts["action_direction"] == 2
    assert counts["over_scaffolding"] == 1
    assert counts["over_scaffolding_not_asked"] == 1


def test_entry_keys_split_back_on_the_first_separator():
    """The key scheme has to survive round-tripping to a moment_id."""
    entries, _ = C.build_entries([SCAFFOLDED])
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
    record = C.build_record(SCAFFOLDED, RAW, CFG)

    assert record["moment_id"] == "aaa"
    assert record["model"] == "test-model"
    assert record["action_direction"]["scaffolding"] is True
    assert record["over_scaffolding"]["over_scaffolding"] is True
    assert record["over_scaffolding_asked"] is True
    assert record["labels"] == SCAFFOLDED["labels"]


def test_build_record_sums_usage_across_both_passes():
    record = C.build_record(SCAFFOLDED, RAW, CFG)

    assert record["usage"] == {
        "input_tokens": 15,
        "output_tokens": 3,
        "total_tokens": 18,
    }


def test_ungated_moment_gets_null_over_scaffolding_not_false():
    """ "Not asked" has to stay distinguishable from "asked and told no"."""
    record = C.build_record(UNSCAFFOLDED, RAW, CFG)

    assert record["over_scaffolding"] is None
    assert record["over_scaffolding_asked"] is False
    assert record["usage"]["total_tokens"] == 10


def test_build_record_keeps_the_raw_response():
    record = C.build_record(SCAFFOLDED, RAW, CFG)

    assert record["action_direction"]["raw"] == RAW["action__aaa"]["text"]


def test_build_record_records_a_batch_entry_error():
    raw = {"action__aaa": {"text": "", "error": "overloaded"}}
    record = C.build_record(UNSCAFFOLDED, raw, CFG)

    assert record["action_direction"]["error"] is None  # bbb has no entry in raw
    record = C.build_record(SCAFFOLDED, raw, CFG)
    assert record["action_direction"]["error"] == "overloaded"
    assert record["action_direction"]["parse_error"] is True


def test_missing_batch_result_degrades_to_a_parse_error():
    record = C.build_record(SCAFFOLDED, {}, CFG)

    assert record["action_direction"]["scaffolding"] is None
    assert record["action_direction"]["parse_error"] is True
    assert record["usage"] == C.ZERO_USAGE


# ===========================================================================
# Round-running (no API calls)
# ===========================================================================


def test_classify_splits_results_back_apart(monkeypatch):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)

    out, counts = C.classify({"iteration": [SCAFFOLDED], "test": [UNSCAFFOLDED]}, CFG)

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

    _, counts = C.classify({"iteration": [SCAFFOLDED]}, CFG)

    assert counts["action_direction_unparsed"] == 1
    assert counts["over_scaffolding_unparsed"] == 1


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


def test_write_split_round_trips(tmp_path):
    record = C.build_record(SCAFFOLDED, RAW, CFG)
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


def test_print_shows_both_prompts_for_a_scaffolded_moment(tmp_path, capsys):
    code = C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print", "aaa"])
    out = capsys.readouterr().out

    assert code == 0
    assert C.ACTION_DIRECTION_PROMPT in out
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
    assert C.ACTION_DIRECTION_PROMPT in out


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
    entries, counts = C.build_entries([SCAFFOLDED, EMPTY_POST_CUT])

    assert {e["key"] for e in entries} == {"action__aaa", "overscaffold__aaa"}
    assert counts["no_post_cut_content"] == 1


def test_skipped_moment_still_gets_a_record_naming_the_reason():
    record = C.build_record(EMPTY_POST_CUT, RAW, CFG)

    assert record["classified"] is False
    assert record["skipped"] == "no_post_cut_content"
    assert record["action_direction"] is None
    assert record["over_scaffolding"] is None
    assert record["usage"] == C.ZERO_USAGE
    assert record["labels"] == EMPTY_POST_CUT["labels"]


def test_skipped_moments_are_not_counted_as_parse_failures(monkeypatch):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: {})

    out, counts = C.classify({"iteration": [EMPTY_POST_CUT]}, CFG)

    assert counts["action_direction_unparsed"] == 0
    assert len(out["iteration"]) == 1


def test_report_names_the_skipped_moments(monkeypatch, capsys):
    monkeypatch.setattr(C, "run_entries", lambda entries, cfg, batch_id=None: RAW)

    out, counts = C.classify({"iteration": [SCAFFOLDED, EMPTY_POST_CUT]}, CFG)

    assert "1 moment(s) not classified" in C.report(out, counts, dry_run=False)


def test_missing_field_warns_with_the_rebuild_command(caplog):
    with caplog.at_level("WARNING"):
        C.build_entries(
            [
                {
                    "moment_id": "old",
                    "labels": {},
                    "excerpts": {
                        str(w): {"excerpt": "x"}
                        for w in (
                            C.ACTION_CONTEXT_TURNS,
                            C.OVER_SCAFFOLDING_CONTEXT_TURNS,
                        )
                    },
                }
            ]
        )

    assert "v2.excerpts" in caplog.text


# ===========================================================================
# Per-prompt context width
# ===========================================================================


def test_each_prompt_is_filled_from_its_own_window():
    """Action direction reads the 5-turn window, over-scaffolding the 20-turn one."""
    entries, _ = C.build_entries([SCAFFOLDED])
    by_key = {
        e["key"]: e["request"]["contents"][0]["parts"][0]["text"] for e in entries
    }

    assert "wide lead-up" not in by_key["action__aaa"]
    assert "wide lead-up" in by_key["overscaffold__aaa"]


def test_excerpt_at_picks_the_requested_width():
    assert C.excerpt_at(SCAFFOLDED, 5).startswith("Turn 1.")
    assert C.excerpt_at(SCAFFOLDED, 20).startswith("wide lead-up")


def test_a_missing_width_names_the_rebuild_command():
    narrow_only = dict(SCAFFOLDED, excerpts={"5": {"excerpt": "x"}})

    with pytest.raises(KeyError, match="v2.excerpts --context-turns"):
        C.excerpt_at(narrow_only, C.OVER_SCAFFOLDING_CONTEXT_TURNS)


def test_the_widths_each_pass_used_are_recorded():
    record = C.build_record(SCAFFOLDED, RAW, CFG)

    assert record["context_turns"] == {
        "action_direction": C.ACTION_CONTEXT_TURNS,
        "over_scaffolding": C.OVER_SCAFFOLDING_CONTEXT_TURNS,
    }


def test_print_labels_each_prompt_with_its_window(tmp_path, capsys):
    C.main(["--excerpt-dir", _write_excerpts(tmp_path), "--print", "aaa"])
    out = capsys.readouterr().out

    assert f"({C.ACTION_CONTEXT_TURNS}-turn window)" in out
    assert f"({C.OVER_SCAFFOLDING_CONTEXT_TURNS}-turn window)" in out
