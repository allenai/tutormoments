"""Tests for persona generation: what the generator reads, token usage, and resuming."""

import json
import threading
from types import SimpleNamespace

import pytest

from tutormoments_build.traits import ALL_DIMENSIONS
from tutormoments_build.v2 import replay_data as RD
from tutormoments_build.v2 import traits as T
from tutormoments_build.v2.replay_config import RoleSpec

SPEC = RoleSpec(
    role="traits",
    model="gemini-4-argon",
    provider="gemini",
    thinking={"include_thoughts": True, "thinking_level": "high"},
)


def _row(index, turn, role, text):
    return {"turn_index": index, "turn_number": turn, "role": role, "text": text}


ROWS = [
    _row(0, None, "", "[SCREEN UPDATE] board opens"),
    _row(1, 1, "Tutor", "[Tutor] hi there"),
    _row(2, None, "", "[PAUSE] 12 seconds"),
    _row(3, 2, "Student", "hello"),  # cut row
    _row(4, 3, "Tutor", "post-cut tutor"),
    _row(5, 4, "Student", "post-cut student"),
]


def _moment(moment_id="m1", rows=ROWS, cut_index=3):
    return RD.ReplayMoment(
        moment={"moment_id": moment_id, "transcript_id": "t1", "cut_turn": 2},
        conversation_id="c1",
        rows=rows,
        cut_index=cut_index,
        end_index=len(rows) - 1,
        problems=[],
    )


class FakeClient:
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        with self._lock:
            self.calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(
            text="reasoning DESCRIPTION: a trait",
            usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        )


def test_prefix_is_dialogue_through_the_cut_turn():
    assert RD.dialogue_prefix(_moment()) == (
        "Turn 1. Tutor: hi there\nTurn 2. Student: hello"
    )


def test_ids_must_be_unique_across_splits():
    gt = {"iteration": [{"moment_id": "a"}], "test": [{"moment_id": "a"}]}
    with pytest.raises(ValueError, match="appears in both 'iteration' and 'test'"):
        RD.check_unique_ids(gt)
    RD.check_unique_ids(
        {"iteration": [{"moment_id": "a"}], "test": [{"moment_id": "b"}]}
    )


def test_adapter_is_uncapped_and_sums_usage():
    client = FakeClient()
    persona, usage, calls, raw = T.generate_persona(
        "Turn 1. Tutor: hi", client, SPEC.thinking
    )

    # joined-3 mode makes one call per dimension.
    assert calls == len(ALL_DIMENSIONS) == len(client.calls)
    assert all(c["max_tokens"] == 0 for c in client.calls)
    assert all(c["thinking"] == SPEC.thinking for c in client.calls)
    assert all("Turn 1. Tutor: hi" in c["prompt"] for c in client.calls)
    assert usage["total_tokens"] == 13 * calls
    assert persona == "\n\n".join(["a trait"] * calls)
    assert set(raw) == set(ALL_DIMENSIONS)
    assert all(text == "reasoning DESCRIPTION: a trait" for text in raw.values())


def test_generate_writes_records_and_resumes(tmp_path):
    out = str(tmp_path / "traits.jsonl")
    client = FakeClient()

    first = T.generate([_moment("m1")], out, client=client, spec=SPEC)
    assert [r["moment_id"] for r in first] == ["m1"]
    record = T.load_traits(out)["m1"]
    assert record["generator_model"] == "gemini-4-argon"
    assert record["thinking"] == SPEC.thinking
    assert record["trait_mode"] == "joined-3"
    assert record["usage"]["total_tokens"] > 0

    calls_before = len(client.calls)
    second = T.generate([_moment("m1"), _moment("m2")], out, client=client, spec=SPEC)
    assert [r["moment_id"] for r in second] == ["m2"]
    assert len(client.calls) - calls_before == len(ALL_DIMENSIONS)
    assert set(T.load_traits(out)) == {"m1", "m2"}


def test_overwrite_regenerates_only_the_given_moments(tmp_path):
    out = tmp_path / "traits.jsonl"
    out.write_text(
        json.dumps({"moment_id": "m1", "persona": "old"})
        + "\n"
        + json.dumps({"moment_id": "keep", "persona": "kept"})
        + "\n"
    )

    T.generate(
        [_moment("m1")], str(out), client=FakeClient(), spec=SPEC, overwrite=True
    )

    traits = T.load_traits(str(out))
    assert traits["m1"]["persona"] != "old"
    assert traits["keep"]["persona"] == "kept"
    assert not (tmp_path / "traits.jsonl.partial").exists()


def test_a_missing_persona_raises():
    with pytest.raises(KeyError, match="no frozen persona for moment m9"):
        T.persona_for({"m1": {"persona": "x"}}, "m9")
    assert T.persona_for({"m1": {"persona": "x"}}, "m1") == "x"


class ScriptedClient(FakeClient):
    """Returns a fixed text for every call."""

    def __init__(self, text):
        super().__init__()
        self.text = text

    def generate(self, prompt, **kwargs):
        response = super().generate(prompt, **kwargs)
        response.text = self.text
        return response


def test_a_quoted_marker_in_the_plan_is_not_kept_as_the_persona():
    """v1 kept all text after the first marker, including a plan quoting the format."""
    leaked = (
        "Plan: output `DESCRIPTION:\\n[description]`.\n1. analyse...\n"
        "DESCRIPTION:\nThe student stays focused."
    )
    persona, _, calls, raw = T.generate_persona(
        "Turn 1. Tutor: hi", ScriptedClient(leaked), SPEC.thinking
    )

    assert persona == "\n\n".join(["The student stays focused."] * calls)
    assert all(text == leaked for text in raw.values())


def test_an_empty_description_is_not_written(tmp_path):
    """A failed moment isn't written, so the next run retries it."""
    out = str(tmp_path / "traits.jsonl")
    written = T.generate(
        [_moment("m1")], out, client=ScriptedClient("DESCRIPTION:   "), spec=SPEC
    )

    assert written == []
    assert T.load_traits(out) == {}


def test_a_failure_under_overwrite_keeps_the_old_file(tmp_path):
    out = tmp_path / "traits.jsonl"
    out.write_text(json.dumps({"moment_id": "m1", "persona": "old"}) + "\n")

    with pytest.raises(RuntimeError, match="left unchanged"):
        T.generate(
            [_moment("m1")],
            str(out),
            client=ScriptedClient("DESCRIPTION:"),
            spec=SPEC,
            overwrite=True,
        )
    assert T.load_traits(str(out))["m1"]["persona"] == "old"
