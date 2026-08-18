"""Streaming latency instrumentation: TTFC / TTFT / TTLT extraction.

The three anchors carry different meanings and each has a way of going
silently wrong, so every provider path is asserted against a fake stream with
an injected clock:

- ttft must skip reasoning (thinking blocks on Anthropic, ``part.thought`` on
  Gemini, inline ``<think>`` on Together). Counting a reasoning token as the
  first visible token would make thinking models look far faster to first
  token than a student ever experiences.
- ttlt must anchor on the last *content* delta, not on iterator exhaustion --
  the trailing usage-only chunk and connection teardown are not part of the
  student's wait.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tutormoments.client import (
    ModelClient,
    _InlineThinkGate,
    _StreamTimer,
)

# ---------------------------------------------------------------------------
# Fake clock
# ---------------------------------------------------------------------------


def _clock_from(ticks):
    """Return a clock callable yielding `ticks` in order, then repeating the last."""
    seq = list(ticks)
    state = {"i": 0}

    def _clock():
        i = state["i"]
        state["i"] = min(i + 1, len(seq) - 1)
        return seq[i]

    return _clock


# ---------------------------------------------------------------------------
# _StreamTimer
# ---------------------------------------------------------------------------


def test_timer_anchors_are_relative_to_t0():
    timer = _StreamTimer(clock=_clock_from([100.0, 100.5, 101.0, 102.5]))
    timer.chunk()  # 100.5 -> ttfc 0.5
    timer.visible()  # 101.0 -> ttft 1.0, ttlt 1.0
    timer.visible()  # 102.5 -> ttlt 2.5
    out = timer.as_dict()
    assert out["ttfc_seconds"] == pytest.approx(0.5)
    assert out["ttft_seconds"] == pytest.approx(1.0)
    assert out["ttlt_seconds"] == pytest.approx(2.5)


def test_timer_visible_backfills_ttfc():
    """A stream whose very first event is visible text still gets a ttfc."""
    timer = _StreamTimer(clock=_clock_from([10.0, 11.0]))
    timer.visible()
    out = timer.as_dict()
    assert out["ttfc_seconds"] == pytest.approx(1.0)
    assert out["ttft_seconds"] == pytest.approx(1.0)


def test_timer_output_tps_spans_the_whole_generation_window():
    """tokens/sec runs ttfc -> ttlt, not ttft -> ttlt.

    output_tokens counts thinking tokens too, so a text-only window would
    divide thinking-inclusive tokens by text-only time and overstate
    throughput several-fold on thinking models.
    """
    timer = _StreamTimer(clock=_clock_from([0.0, 1.0, 3.0, 5.0]))
    timer.chunk()  # ttfc = 1.0 (first thinking token)
    timer.visible()  # ttft = 3.0
    timer.visible()  # ttlt = 5.0
    out = timer.as_dict(output_tokens=100)
    assert out["output_tps"] == pytest.approx(25.0)  # 100 / (5.0 - 1.0)


def test_timer_output_tps_excludes_prefill():
    """The window still starts at first token, not at request send."""
    timer = _StreamTimer(clock=_clock_from([0.0, 10.0, 12.0]))
    timer.visible()  # 10s of prefill before the first token
    timer.visible()
    out = timer.as_dict(output_tokens=100)
    assert out["output_tps"] == pytest.approx(50.0)  # 100 / (12.0 - 10.0)


def test_timer_output_tps_none_without_token_count():
    """Together may not return usage; omit tps rather than guessing."""
    timer = _StreamTimer(clock=_clock_from([0.0, 1.0, 2.0]))
    timer.visible()
    timer.visible()
    assert timer.as_dict(output_tokens=None)["output_tps"] is None


def test_timer_no_visible_delta_leaves_ttft_none():
    timer = _StreamTimer(clock=_clock_from([0.0, 0.5]))
    timer.chunk()
    out = timer.as_dict()
    assert out["ttfc_seconds"] == pytest.approx(0.5)
    assert out["ttft_seconds"] is None
    assert out["ttlt_seconds"] is None


# ---------------------------------------------------------------------------
# _InlineThinkGate  (Together open-weight reasoners)
# ---------------------------------------------------------------------------


def test_think_gate_defers_until_after_closing_tag():
    gate = _InlineThinkGate()
    assert gate.visible("<think>") is False
    assert gate.visible("reasoning about the student") is False
    assert gate.visible("</think>") is False  # nothing visible yet
    assert gate.visible("Hi there!") is True


def test_think_gate_passes_through_when_no_think_block():
    gate = _InlineThinkGate()
    assert gate.visible("Hello") is True


def test_think_gate_waits_for_enough_chars_to_decide():
    """`<t` is still consistent with `<think>`; don't guess either way yet."""
    gate = _InlineThinkGate()
    assert gate.visible("<t") is False
    assert gate.visible("able>rows") is True  # turned out to be markup, not think


def test_think_gate_stays_open_once_visible():
    """A later literal '<think>' in answer text must not re-close the gate."""
    gate = _InlineThinkGate()
    assert gate.visible("Answer") is True
    assert gate.visible("<think>") is True


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------


def _evt(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _anthropic_stream_events():
    """thinking block, then a text block -- the shape a thinking tutor produces."""
    return [
        _evt("message_start"),
        _evt(
            "content_block_start",
            index=0,
            content_block=SimpleNamespace(type="thinking"),
        ),
        _evt("content_block_delta", index=0),
        _evt("content_block_delta", index=0),
        _evt("content_block_stop", index=0),
        _evt(
            "content_block_start", index=1, content_block=SimpleNamespace(type="text")
        ),
        _evt("content_block_delta", index=1),
        _evt("content_block_delta", index=1),
        _evt("content_block_stop", index=1),
        _evt("message_delta"),
        _evt("message_stop"),
    ]


def _final_message(text="Nice work!", in_tok=900, out_tok=40, cache_read=800):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        content=[block],
        usage=SimpleNamespace(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cache_read,
        ),
    )


def _anthropic_client_with_stream(events, final):
    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = stream_ctx
    stream_ctx.__exit__.return_value = False
    stream_ctx.__iter__.return_value = iter(events)
    stream_ctx.get_final_message.return_value = final
    client_obj = MagicMock()
    client_obj.messages.stream.return_value = stream_ctx
    return client_obj


def test_anthropic_stream_ttft_skips_thinking_deltas(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # generate() reads the clock once for call_t0, _StreamTimer.__init__ once
    # for t0, then exactly once per stream event.
    ticks = [0.0] + [0.1 * (i + 1) for i in range(12)]
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.monotonic", side_effect=_clock_from(ticks)),
    ):
        MockAnthropic.return_value = _anthropic_client_with_stream(
            _anthropic_stream_events(), _final_message()
        )
        c = ModelClient("claude-opus-4-8")
        resp = c.generate("Q", json_mode=False, stream=True)

    t = resp.timing
    assert t["ttfc_seconds"] < t["ttft_seconds"], "thinking must not set ttft"
    assert t["ttft_seconds"] <= t["ttlt_seconds"]
    assert resp.text == "Nice work!"


def test_anthropic_stream_ttlt_is_last_content_delta_not_stream_end(monkeypatch):
    """message_delta / message_stop arrive after generation; they must not move ttlt."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    events = [
        _evt("message_start"),
        _evt(
            "content_block_start", index=0, content_block=SimpleNamespace(type="text")
        ),
        _evt("content_block_delta", index=0),
        _evt("content_block_stop", index=0),
        _evt("message_delta"),
        _evt("message_stop"),
    ]
    # Only three reads happen: call_t0, t0, and message_start (which sets
    # ttfc=1.0). chunk() is free once ttfc is set, so the next read is the
    # text delta at 2.0. Every later event would read 90.0 -- so if the
    # implementation stamped ttlt on stream end rather than on the last
    # content delta, this assertion would fail.
    ticks = [0.0, 0.0, 1.0, 2.0, 90.0]
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.monotonic", side_effect=_clock_from(ticks)),
    ):
        MockAnthropic.return_value = _anthropic_client_with_stream(
            events, _final_message()
        )
        c = ModelClient("claude-opus-4-8")
        resp = c.generate("Q", json_mode=False, stream=True)

    assert resp.timing["ttlt_seconds"] == pytest.approx(2.0)


def test_anthropic_stream_records_cache_read_tokens(monkeypatch):
    """Cache state is read off the API, never inferred from turn position."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = _anthropic_client_with_stream(
            _anthropic_stream_events(), _final_message(cache_read=1234)
        )
        c = ModelClient("claude-opus-4-8")
        resp = c.generate("Q", json_mode=False, stream=True)

    assert resp.timing["cache_read_input_tokens"] == 1234
    assert resp.usage["cache_read_input_tokens"] == 1234


def test_streaming_does_not_redefine_latency_seconds(monkeypatch):
    """latency_seconds keeps its pre-streaming meaning: end-to-end wall clock.

    It predates this work and feeds the paper's latency figure, so streaming
    must not silently redefine it. TTLT is available separately on `timing`;
    the two are close but not the same number, and conflating them would
    break comparability across runs with nothing to signal it.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # call_t0, t0, ttfc, then the two text deltas (ttft=2.0, ttlt=3.0). The
    # final tick is the wall-clock read taken after the stream ends, far in
    # the future, so the two measurements are unmistakably distinct.
    ticks = [0.0, 0.0, 1.0, 2.0, 3.0, 99.0]
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.monotonic", side_effect=_clock_from(ticks)),
    ):
        MockAnthropic.return_value = _anthropic_client_with_stream(
            _anthropic_stream_events(), _final_message()
        )
        resp = ModelClient("claude-opus-4-8").generate(
            "Q", json_mode=False, stream=True
        )

    assert resp.timing["ttlt_seconds"] is not None
    assert resp.latency_seconds != resp.timing["ttlt_seconds"]
    assert resp.latency_seconds >= resp.timing["ttlt_seconds"]


def test_non_streamed_call_has_no_timing(monkeypatch):
    """The scorer/taxonomy paths stay non-streamed and unchanged."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        client_obj = MagicMock()
        client_obj.messages.create.return_value = _final_message()
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-8")
        resp = c.generate("Q", json_mode=False)

    assert resp.timing is None
    assert resp.latency_seconds >= 0
    client_obj.messages.stream.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAI-compatible streaming (OpenAI + Together)
# ---------------------------------------------------------------------------


def _oai_chunk(content=None, usage=None):
    if usage is not None:
        return SimpleNamespace(choices=[], usage=usage)
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def _oai_usage(prompt=500, completion=20, cached=400):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _openai_client_with_chunks(chunks):
    client_obj = MagicMock()
    client_obj.chat.completions.create.return_value = iter(chunks)
    return client_obj


def test_openai_stream_first_content_delta_is_ttft(monkeypatch):
    """chat.completions never streams reasoning, so ttft tracks ttfc closely.

    The real API opens with a role-only chunk carrying empty content, so the
    two anchors are near-identical rather than exactly equal -- verified
    against the live API at ttfc=3.9316 / ttft=3.9320. The leading empty
    chunk is reproduced here so the fixture matches the wire.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chunks = [
        _oai_chunk(""),  # role-only opener, no content
        _oai_chunk("Hel"),
        _oai_chunk("lo"),
        _oai_chunk(usage=_oai_usage()),
    ]
    # call_t0, t0, role-only chunk (ttfc), first content delta (ttft).
    ticks = [0.0, 0.0, 1.0, 1.001]
    with (
        patch("openai.OpenAI") as MockOpenAI,
        patch("tutormoments.client.time.monotonic", side_effect=_clock_from(ticks)),
    ):
        MockOpenAI.return_value = _openai_client_with_chunks(chunks)
        c = ModelClient("gpt-5.5-2026-04-23")
        resp = c.generate("Q", json_mode=False, stream=True)

    assert resp.text == "Hello"
    assert resp.timing["ttft_seconds"] - resp.timing["ttfc_seconds"] < 0.01
    assert resp.usage["output_tokens"] == 20


def test_openai_stream_requests_usage(monkeypatch):
    """Without include_usage there is no token count and no tokens/sec."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("openai.OpenAI") as MockOpenAI:
        client_obj = _openai_client_with_chunks([_oai_chunk("x")])
        MockOpenAI.return_value = client_obj
        ModelClient("gpt-5.5-2026-04-23").generate("Q", json_mode=False, stream=True)

    kwargs = client_obj.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


def test_openai_stream_usage_only_chunk_does_not_move_ttlt(monkeypatch):
    """The final choices==[] usage chunk is bookkeeping, not the student's wait."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    chunks = [_oai_chunk("hi"), _oai_chunk(usage=_oai_usage())]
    # call_t0, t0, content delta. The usage chunk reads no clock (ttfc is
    # already set), so 60.0 is only reachable if ttlt moved past generation.
    ticks = [0.0, 0.0, 1.0, 60.0]
    with (
        patch("openai.OpenAI") as MockOpenAI,
        patch("tutormoments.client.time.monotonic", side_effect=_clock_from(ticks)),
    ):
        MockOpenAI.return_value = _openai_client_with_chunks(chunks)
        resp = ModelClient("gpt-5.5-2026-04-23").generate(
            "Q", json_mode=False, stream=True
        )

    assert resp.timing["ttlt_seconds"] == pytest.approx(1.0)


def test_together_stream_defers_ttft_past_inline_think(monkeypatch):
    """DeepSeek emits <think> in the content stream; ttft must skip it."""
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
    chunks = [
        _oai_chunk("<think>"),
        _oai_chunk("the student is stuck on step 2"),
        _oai_chunk("</think>"),
        _oai_chunk("What did you try first?"),
        _oai_chunk(usage=_oai_usage(cached=0)),
    ]
    # call_t0, t0, first <think> delta (0.5). The two deltas inside the think
    # block read no clock (ttfc already set), so the next read is the answer
    # delta at 4.0 -- ttft must land there, not at 0.5.
    ticks = [0.0, 0.0, 0.5, 4.0]
    with (
        patch("openai.OpenAI") as MockOpenAI,
        patch("tutormoments.client.time.monotonic", side_effect=_clock_from(ticks)),
    ):
        MockOpenAI.return_value = _openai_client_with_chunks(chunks)
        resp = ModelClient("deepseek-ai/DeepSeek-V4-Pro").generate(
            "Q", json_mode=False, stream=True
        )

    assert resp.timing["ttfc_seconds"] == pytest.approx(0.5)
    assert resp.timing["ttft_seconds"] == pytest.approx(4.0)
    assert "<think>" in resp.text, "reasoning stays in text; only timing skips it"


def test_together_stream_missing_usage_records_no_tps(monkeypatch):
    """include_usage support is inconsistent on Together -- degrade, don't guess."""
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = _openai_client_with_chunks([_oai_chunk("hello")])
        resp = ModelClient("deepseek-ai/DeepSeek-V4-Pro").generate(
            "Q", json_mode=False, stream=True
        )

    assert resp.usage["output_tokens"] == 0
    assert resp.timing["output_tps"] is None
    assert resp.timing["ttft_seconds"] is not None


# ---------------------------------------------------------------------------
# Gemini streaming
# ---------------------------------------------------------------------------


def _gem_chunk(parts, usage=None):
    content = SimpleNamespace(parts=parts)
    cand = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[cand], usage_metadata=usage)


def _gem_part(text, thought=False):
    return SimpleNamespace(text=text, thought=thought)


def _gem_usage(prompt=300, candidates=25):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=prompt + candidates,
    )


def test_gemini_stream_skips_thought_parts_for_ttft(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    chunks = [
        _gem_chunk([_gem_part("planning...", thought=True)]),
        _gem_chunk([_gem_part("Try ")], usage=_gem_usage()),
        _gem_chunk([_gem_part("again.")], usage=_gem_usage()),
    ]
    # call_t0, t0, thought-only chunk (0.5), then the two visible parts.
    ticks = [0.0, 0.0, 0.5, 2.0, 3.0]
    with (
        patch("google.genai.Client") as MockGenai,
        patch("tutormoments.client.time.monotonic", side_effect=_clock_from(ticks)),
    ):
        client_obj = MagicMock()
        client_obj.models.generate_content_stream.return_value = iter(chunks)
        MockGenai.return_value = client_obj
        resp = ModelClient("gemini-3.5-flash").generate(
            "Q", json_mode=False, stream=True
        )

    assert resp.text == "Try again."
    assert resp.timing["ttfc_seconds"] == pytest.approx(0.5)
    assert resp.timing["ttft_seconds"] == pytest.approx(2.0)
    assert resp.usage["output_tokens"] == 25


def test_gemini_stream_tolerates_usage_only_chunk(monkeypatch):
    """A trailing chunk can carry usage with no candidates -- must not crash."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    chunks = [
        _gem_chunk([_gem_part("hi")]),
        SimpleNamespace(candidates=[], usage_metadata=_gem_usage()),
    ]
    with patch("google.genai.Client") as MockGenai:
        client_obj = MagicMock()
        client_obj.models.generate_content_stream.return_value = iter(chunks)
        MockGenai.return_value = client_obj
        resp = ModelClient("gemini-3.5-flash").generate(
            "Q", json_mode=False, stream=True
        )

    assert resp.text == "hi"
    assert resp.usage["output_tokens"] == 25
