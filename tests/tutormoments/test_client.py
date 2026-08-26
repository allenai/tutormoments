import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tutormoments.client import (
    ModelClient,
    ModelResponse,
    _base64_bytes,
    _build_image_blocks_anthropic,
    _build_image_blocks_gemini,
    _build_image_blocks_openai,
    _mime_from_path,
    _presigned_url,
    _should_use_presigned_url,
    _strip_json_fences,
    build_batch_entry,
    cancel_batch,
    infer_provider,
    normalize_usage,
    run_batch,
    run_sync_entries,
    write_jsonl,
)
from tutormoments.models import ThinkingConfigError


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-6", "anthropic"),
        ("claude-haiku-4-5-20251001", "anthropic"),
        ("gpt-5.5-2026-04-23", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini", "openai"),
        ("o4-mini", "openai"),
        ("gemini-3.1-pro-preview", "gemini"),
        ("deepseek-ai/DeepSeek-V3", "together"),
        ("meta-llama/Llama-3.3-70B", "together"),
    ],
)
def test_infer_provider_routes_by_prefix(model, expected):
    assert infer_provider(model) == expected


def test_infer_provider_is_case_insensitive():
    assert infer_provider("Claude-Opus-4-6") == "anthropic"


def test_infer_provider_unknown_raises():
    with pytest.raises(ValueError):
        infer_provider("totally-unknown-model")


class TestGenerateThinkingWire:
    """The thinking ladder resolves to the exact wire bytes the raw knobs sent.

    The full (family, rung) -> wire matrix is pinned in test_models.py; these
    tests prove the client actually places each resolved fragment on the
    request. Each expected payload is byte-identical to what the pre-ladder
    client produced for the equivalent raw-knob inputs (thinking=True,
    thinking_budget, effort, reasoning_effort) -- the wire-level
    reproducibility proof for the refactor.
    """

    def _anthropic(self, monkeypatch, model, **gen_kwargs):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("anthropic.Anthropic") as MockAnthropic:
            client_obj = MagicMock()
            client_obj.messages.create.return_value = _fake_anthropic_message("ok")
            MockAnthropic.return_value = client_obj
            ModelClient(model).generate("Q", json_mode=False, **gen_kwargs)
        return client_obj.messages.create.call_args.kwargs

    def test_anthropic_dynamic_sends_adaptive(self, monkeypatch):
        # Old thinking=True on adaptive-tier models.
        kwargs = self._anthropic(monkeypatch, "claude-opus-4-6", thinking="dynamic")
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert "extra_body" not in kwargs  # no effort alongside plain adaptive

    def test_anthropic_xhigh_sends_adaptive_plus_effort(self, monkeypatch):
        # Old thinking=True + effort="xhigh".
        kwargs = self._anthropic(monkeypatch, "claude-opus-4-8", thinking="xhigh")
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["extra_body"]["output_config"] == {"effort": "xhigh"}

    def test_anthropic_legacy_high_sends_enabled_budget_and_bumps_max_tokens(
        self, monkeypatch
    ):
        # Old haiku-4-5 thinking=True (legacy enabled + default budget 16384).
        # Enabled mode requires max_tokens >= budget + headroom.
        kwargs = self._anthropic(
            monkeypatch, "claude-haiku-4-5", thinking="high", max_tokens=100
        )
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16384}
        assert kwargs["max_tokens"] == 16384 + 64

    def test_anthropic_haiku_output_cap_comes_from_registry(self, monkeypatch):
        # max_tokens=0 resolves to the provider maximum (128000), which the
        # registry's per-model cap clamps to haiku's 64000. The dated id
        # resolves to the claude-haiku-4-5 entry by longest prefix.
        kwargs = self._anthropic(monkeypatch, "claude-haiku-4-5-20251001")
        assert kwargs["max_tokens"] == 64000

    def test_anthropic_none_level_sends_no_thinking_param(self, monkeypatch):
        kwargs = self._anthropic(monkeypatch, "claude-opus-4-6", thinking="none")
        assert "thinking" not in kwargs
        assert "extra_body" not in kwargs

    def test_no_stated_condition_sends_nothing_even_unregistered(self, monkeypatch):
        # thinking=None is unvalidated and unsent -- works for model ids the
        # registry has never heard of (dev/direct client use).
        kwargs = self._anthropic(
            monkeypatch, "claude-experimental-dev-model", thinking=None
        )
        assert "thinking" not in kwargs

    def _gemini(self, monkeypatch, model, **gen_kwargs):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        with patch("google.genai.Client") as MockGenai:
            client_obj = MagicMock()
            client_obj.models.generate_content.return_value = SimpleNamespace(
                text="ok", usage_metadata=None
            )
            MockGenai.return_value = client_obj
            ModelClient(model).generate("Q", json_mode=False, **gen_kwargs)
        return client_obj.models.generate_content.call_args.kwargs

    def test_gemini_dynamic_sends_budget_minus_one(self, monkeypatch):
        # Old thinking=True + thinking_budget=-1.
        kwargs = self._gemini(monkeypatch, "gemini-2.5-pro", thinking="dynamic")
        assert kwargs["config"]["thinking_config"] == {
            "include_thoughts": True,
            "thinking_budget": -1,
        }

    def test_gemini_flash_none_sends_zero_budget(self, monkeypatch):
        # Old thinking=False; only valid where the API accepts budget 0.
        kwargs = self._gemini(monkeypatch, "gemini-2.5-flash", thinking="none")
        assert kwargs["config"]["thinking_config"] == {
            "include_thoughts": False,
            "thinking_budget": 0,
        }

    def test_gemini_no_condition_omits_thinking_config(self, monkeypatch):
        kwargs = self._gemini(monkeypatch, "gemini-2.5-pro")
        assert "thinking_config" not in kwargs["config"]

    def _openai(self, monkeypatch, model, **gen_kwargs):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("openai.OpenAI") as MockOpenAI:
            client_obj = MagicMock()
            client_obj.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )
            MockOpenAI.return_value = client_obj
            ModelClient(model).generate("Q", json_mode=False, **gen_kwargs)
        return client_obj.chat.completions.create.call_args.kwargs

    def test_openai_high_sends_reasoning_effort(self, monkeypatch):
        # Old reasoning_effort="high".
        kwargs = self._openai(monkeypatch, "gpt-5.5", thinking="high")
        assert kwargs["reasoning_effort"] == "high"

    def test_openai_no_condition_omits_reasoning_effort(self, monkeypatch):
        kwargs = self._openai(monkeypatch, "gpt-5.5")
        assert "reasoning_effort" not in kwargs

    def test_unsatisfiable_level_fails_before_sdk_call_without_retries(
        self, monkeypatch
    ):
        # gemini-2.5-pro is always-thinking: `none` must die at resolve time,
        # before any request is made and without burning the retry budget.
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        with (
            patch("google.genai.Client") as MockGenai,
            patch("tutormoments.client.time.sleep") as mock_sleep,
        ):
            client_obj = MagicMock()
            MockGenai.return_value = client_obj
            c = ModelClient("gemini-2.5-pro")
            with pytest.raises(ThinkingConfigError):
                c.generate("Q", json_mode=False, thinking="none")
        client_obj.models.generate_content.assert_not_called()
        mock_sleep.assert_not_called()

    def test_unregistered_model_with_explicit_level_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("anthropic.Anthropic") as MockAnthropic:
            client_obj = MagicMock()
            MockAnthropic.return_value = client_obj
            c = ModelClient("claude-experimental-dev-model")
            with pytest.raises(ThinkingConfigError):
                c.generate("Q", json_mode=False, thinking="high")
        client_obj.messages.create.assert_not_called()

    def test_retired_raw_knob_kwargs_are_gone(self, monkeypatch):
        # The raw knobs were deleted, not deprecated: passing one is a
        # TypeError, so no caller can silently keep steering the old way.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("anthropic.Anthropic"):
            c = ModelClient("claude-opus-4-6")
        with pytest.raises(TypeError):
            c.generate("Q", thinking_budget=16384)
        with pytest.raises(TypeError):
            c.generate("Q", effort="xhigh")
        with pytest.raises(TypeError):
            c.generate("Q", reasoning_effort="high")


def test_strip_json_fences_removes_markdown_fence():
    assert _strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_json_fences_plain_passthrough():
    assert _strip_json_fences('{"a": 1}') == '{"a": 1}'


def test_mime_from_path_known_exts():
    assert _mime_from_path("x.png") == "image/png"
    assert _mime_from_path("x.jpg") == "image/jpeg"
    assert _mime_from_path("x.jpeg") == "image/jpeg"
    assert _mime_from_path("x.webp") == "image/webp"


def test_build_batch_entry_json_mode_shape():
    e = build_batch_entry("k1", "hello", json_mode=True, max_tokens=100)
    assert e["key"] == "k1"
    gc = e["request"]["generation_config"]
    assert gc["max_output_tokens"] == 100
    assert gc["response_mime_type"] == "application/json"
    assert e["request"]["contents"][0]["parts"][0]["text"] == "hello"
    assert e["request"]["contents"][0]["role"] == "user"
    assert "images" not in e and "cacheable_prefix" not in e


def test_build_batch_entry_optional_fields():
    e = build_batch_entry(
        "k", "p", images=["a.png"], json_mode=False, cacheable_prefix="PRE"
    )
    assert "response_mime_type" not in e["request"]["generation_config"]
    assert e["request"]["images"] == ["a.png"]
    assert e["cacheable_prefix"] == "PRE"


def test_write_jsonl_roundtrip(tmp_path):
    p = tmp_path / "b.jsonl"
    n = write_jsonl([{"key": "k", "request": {}}], str(p))
    assert n == 1
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["key"] == "k"


def test_modelclient_infers_provider_and_inits(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = MagicMock()
        c = ModelClient("claude-opus-4-6")
        assert c.provider == "anthropic"
        MockAnthropic.assert_called_once()
        assert c._client is MockAnthropic.return_value


def test_modelclient_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ModelClient("claude-opus-4-6")


# ===================================================================
# Task 6: generate() + provider builders + retry + usage/latency
# ===================================================================


def _fake_anthropic_message(text="hi", in_tok=10, out_tok=5):
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg.content = [block]
    msg.usage = MagicMock(
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return msg


def test_generate_anthropic_returns_text_and_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        client_obj = MagicMock()
        client_obj.messages.create.return_value = _fake_anthropic_message(
            "answer", 12, 3
        )
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        resp = c.generate("Q", json_mode=False, max_tokens=64)
        assert resp.text == "answer"
        assert resp.usage["input_tokens"] == 12
        assert resp.usage["output_tokens"] == 3
        assert resp.usage["total_tokens"] == 15
        assert resp.latency_seconds >= 0


def test_generate_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        client_obj.messages.create.side_effect = [
            RuntimeError("boom"),
            _fake_anthropic_message("ok"),
        ]
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        resp = c.generate("Q", json_mode=False)
        assert resp.text == "ok"
        assert client_obj.messages.create.call_count == 2


def test_generate_exhausts_retries_raises(monkeypatch):
    """After max_retries failures, generate() raises RuntimeError."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        client_obj.messages.create.side_effect = RuntimeError("always fails")
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        with pytest.raises(RuntimeError, match="API call failed"):
            c.generate("Q", json_mode=False)


def test_generate_latency_is_non_negative(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        client_obj = MagicMock()
        client_obj.messages.create.return_value = _fake_anthropic_message("x", 5, 2)
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        resp = c.generate("Q", json_mode=False)
        assert resp.latency_seconds is not None
        assert resp.latency_seconds >= 0


def test_generate_anthropic_json_mode_adds_system_message(monkeypatch):
    """json_mode=True must inject the JSON-only system prompt."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        client_obj = MagicMock()
        client_obj.messages.create.return_value = _fake_anthropic_message(
            '{"a":1}', 5, 3
        )
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        c.generate("Q", json_mode=True)
        call_kwargs = client_obj.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert "JSON" in call_kwargs["system"]


def test_generate_anthropic_output_schema_sets_format(monkeypatch):
    """output_schema must reach the request as output_config.format json_schema.

    This is the reproducibility guard for the taxonomy classifier rewire: the
    enum-constrained structured output must be preserved through ModelClient.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        client_obj = MagicMock()
        client_obj.messages.create.return_value = _fake_anthropic_message(
            '{"x":1}', 5, 2
        )
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-8")
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        c.generate("Q", json_mode=False, output_schema=schema)
        call_kwargs = client_obj.messages.create.call_args[1]
        oc = call_kwargs["extra_body"]["output_config"]
        assert oc["format"]["type"] == "json_schema"
        assert oc["format"]["schema"] == schema


def test_generate_output_schema_rejected_for_non_anthropic(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("openai.OpenAI"):
        c = ModelClient("gpt-5.5-2026-04-23")
        with pytest.raises(ValueError):
            c.generate("Q", output_schema={"type": "object"})


def test_get_retry_config_values():
    from tutormoments.config import get_batch_timeout, get_retry_config

    cfg = get_retry_config()
    assert cfg["max_retries"] == 5
    assert cfg["base_delay"] == 5
    assert get_batch_timeout() == 86400


# ===================================================================
# Task 7: image-block builders + cacheable-prefix
# ===================================================================


class TestShouldUsePresignedUrl:
    def test_returns_false_by_default(self):
        # Phase 1: local storage only -> always False.
        assert _should_use_presigned_url() is False


class TestBase64Bytes:
    def test_returns_base64_string_from_file(self, tmp_path):
        import base64

        p = tmp_path / "test.png"
        raw = b"\x89PNG fake"
        p.write_bytes(raw)
        # Patch _get_backend to return a LocalBackend-like object.
        with patch("tutormoments.client._get_backend") as mock_be:
            mock_be.return_value.read_bytes.return_value = raw
            result = _base64_bytes(str(p))
        assert result == base64.b64encode(raw).decode("ascii")


class TestPresignedUrl:
    def test_delegates_to_backend(self):
        with patch("tutormoments.client._get_backend") as mock_be:
            mock_be.return_value.get_presigned_url.return_value = (
                "https://example.com/img.png"
            )
            result = _presigned_url("images/img.png", expires_seconds=3600)
        assert result == "https://example.com/img.png"
        mock_be.return_value.get_presigned_url.assert_called_once_with(
            "images/img.png", expires_seconds=3600
        )


class TestBuildImageBlocksAnthropic:
    def _fake_b64(self, path):
        return "ZmFrZQ=="  # base64("fake")

    def test_base64_block_shape(self):
        with patch("tutormoments.client._base64_bytes", return_value="ZmFrZQ=="):
            blocks = _build_image_blocks_anthropic(
                ["photo.png"], use_url=False, enable_cache=False
            )
        assert len(blocks) == 1
        b = blocks[0]
        assert b["type"] == "image"
        assert b["source"]["type"] == "base64"
        assert b["source"]["media_type"] == "image/png"
        assert b["source"]["data"] == "ZmFrZQ=="
        assert "cache_control" not in b

    def test_cache_control_added_when_enable_cache(self):
        with patch("tutormoments.client._base64_bytes", return_value="ZmFrZQ=="):
            blocks = _build_image_blocks_anthropic(
                ["photo.jpg"], use_url=False, enable_cache=True
            )
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_url_block_shape(self):
        with patch(
            "tutormoments.client._presigned_url",
            return_value="https://s3.example.com/img.png",
        ):
            blocks = _build_image_blocks_anthropic(
                ["photo.png"], use_url=True, enable_cache=False
            )
        b = blocks[0]
        assert b["source"]["type"] == "url"
        assert b["source"]["url"] == "https://s3.example.com/img.png"

    def test_multiple_images(self):
        with patch("tutormoments.client._base64_bytes", side_effect=["b64a", "b64b"]):
            blocks = _build_image_blocks_anthropic(
                ["a.png", "b.webp"], use_url=False, enable_cache=False
            )
        assert len(blocks) == 2
        assert blocks[0]["source"]["media_type"] == "image/png"
        assert blocks[1]["source"]["media_type"] == "image/webp"


class TestBuildImageBlocksOpenAI:
    def test_base64_data_url_shape(self):
        with patch("tutormoments.client._base64_bytes", return_value="ZmFrZQ=="):
            blocks = _build_image_blocks_openai(["photo.png"], use_url=False)
        assert len(blocks) == 1
        b = blocks[0]
        assert b["type"] == "image_url"
        assert b["image_url"]["url"].startswith("data:image/png;base64,")
        assert "ZmFrZQ==" in b["image_url"]["url"]

    def test_presigned_url_shape(self):
        with patch(
            "tutormoments.client._presigned_url",
            return_value="https://cdn.example.com/img.jpg",
        ):
            blocks = _build_image_blocks_openai(["photo.jpg"], use_url=True)
        b = blocks[0]
        assert b["image_url"]["url"] == "https://cdn.example.com/img.jpg"

    def test_multiple_images(self):
        with patch("tutormoments.client._base64_bytes", side_effect=["b64x", "b64y"]):
            blocks = _build_image_blocks_openai(["x.png", "y.webp"], use_url=False)
        assert len(blocks) == 2
        assert "image/png" in blocks[0]["image_url"]["url"]
        assert "image/webp" in blocks[1]["image_url"]["url"]


class TestBuildImageBlocksGemini:
    def test_inline_data_block_shape(self):
        with patch("tutormoments.client._base64_bytes", return_value="ZmFrZQ=="):
            blocks = _build_image_blocks_gemini(["photo.png"])
        assert len(blocks) == 1
        b = blocks[0]
        assert "inline_data" in b
        assert b["inline_data"]["mime_type"] == "image/png"
        assert b["inline_data"]["data"] == "ZmFrZQ=="

    def test_multiple_images(self):
        with patch("tutormoments.client._base64_bytes", side_effect=["a", "b"]):
            blocks = _build_image_blocks_gemini(["a.jpg", "b.png"])
        assert len(blocks) == 2
        assert blocks[0]["inline_data"]["mime_type"] == "image/jpeg"
        assert blocks[1]["inline_data"]["mime_type"] == "image/png"


def test_anthropic_cacheable_prefix_is_separate_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as anth:
        client_obj = MagicMock()
        captured = {}

        def _create(**kwargs):
            captured.update(kwargs)
            m = MagicMock()
            b = MagicMock()
            b.type = "text"
            b.text = "ok"
            m.content = [b]
            m.usage = MagicMock(
                input_tokens=1,
                output_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
            return m

        client_obj.messages.create.side_effect = _create
        anth.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        c.generate("BODY", json_mode=False, cacheable_prefix="PREFIX")
        content = captured["messages"][0]["content"]
        assert isinstance(content, list)
        prefix_block = content[0]
        assert prefix_block["type"] == "text"
        assert prefix_block["text"] == "PREFIX"
        assert prefix_block["cache_control"] == {"type": "ephemeral"}


# ===================================================================
# Task 8: run_sync_entries + run_batch (anthropic/openai/gemini)
# ===================================================================


def test_run_sync_entries_collects_by_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = MagicMock()
        c = ModelClient("claude-opus-4-6")
        with patch.object(
            c,
            "generate",
            return_value=ModelResponse(
                "R", {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, 0.0
            ),
        ):
            out = run_sync_entries(
                c, [build_batch_entry("k1", "p1"), build_batch_entry("k2", "p2")]
            )
    assert set(out) == {"k1", "k2"}
    assert out["k1"]["text"] == "R"


def test_run_sync_entries_records_error_per_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = MagicMock()
        c = ModelClient("claude-opus-4-6")
        with patch.object(c, "generate", side_effect=RuntimeError("boom")):
            out = run_sync_entries(c, [build_batch_entry("k1", "p1")])
    assert out["k1"]["text"] == ""
    assert "boom" in out["k1"]["error"]
    # Error rows keep zeros but carry the canonical keys + provenance.
    usage = out["k1"]["usage"]
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_uncached",
        "cache_read",
        "cache_write",
        "output",
        "reasoning",
        "total",
    ):
        assert usage[key] == 0
    assert usage["provider"] == "anthropic"
    assert usage["model"] == "claude-opus-4-6"
    assert usage["endpoint"] == "sync"


def test_run_batch_anthropic_remaps_custom_ids(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        batch = MagicMock()
        batch.id = "b1"
        batch.processing_status = "ended"
        client_obj.messages.batches.create.return_value = batch
        client_obj.messages.batches.retrieve.return_value = batch

        def _result(i, text):
            r = MagicMock()
            r.custom_id = f"r{i}"
            r.result.type = "succeeded"
            msg = MagicMock()
            b = MagicMock()
            b.type = "text"
            b.text = text
            msg.content = [b]
            msg.usage = MagicMock(
                input_tokens=1,
                output_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
            r.result.message = msg
            return r

        client_obj.messages.batches.results.return_value = [
            _result(0, "A"),
            _result(1, "B"),
        ]
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        out = run_batch(
            c,
            [build_batch_entry("kA", "pA"), build_batch_entry("kB", "pB")],
            poll_interval=0,
        )
    assert out["kA"]["text"] == "A"
    assert out["kB"]["text"] == "B"


def test_run_batch_anthropic_sends_thinking_and_effort(monkeypatch):
    """Batch requests must carry the same wire thinking/effort fragment as sync
    calls (benchmark fidelity: batch mode may not silently diverge from
    run_conversation). thinking="xhigh" resolves to adaptive thinking plus
    output_config.effort -- the exact bytes the retired thinking=True +
    effort="xhigh" knobs produced."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        batch = MagicMock()
        batch.id = "b1"
        batch.processing_status = "ended"
        client_obj.messages.batches.create.return_value = batch
        client_obj.messages.batches.retrieve.return_value = batch
        client_obj.messages.batches.results.return_value = []
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-8")
        run_batch(
            c,
            [build_batch_entry("kA", "pA")],
            poll_interval=0,
            thinking="xhigh",
        )
        (submitted,) = client_obj.messages.batches.create.call_args.kwargs["requests"]
    assert submitted["params"]["thinking"] == {"type": "adaptive"}
    assert submitted["params"]["output_config"] == {"effort": "xhigh"}


def test_run_batch_anthropic_legacy_budget_and_registry_cap(monkeypatch):
    """Legacy (enabled+budget) models get budget_tokens on batch requests, no
    output_config, and the registry's per-model output cap clamps the entry's
    max_tokens (build_batch_entry's 65536 default > haiku's 64000 cap)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        batch = MagicMock()
        batch.id = "b1"
        batch.processing_status = "ended"
        client_obj.messages.batches.create.return_value = batch
        client_obj.messages.batches.retrieve.return_value = batch
        client_obj.messages.batches.results.return_value = []
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-haiku-4-5")
        run_batch(
            c,
            [build_batch_entry("kA", "pA")],
            poll_interval=0,
            thinking="high",
        )
        (submitted,) = client_obj.messages.batches.create.call_args.kwargs["requests"]
    assert submitted["params"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 16384,
    }
    assert submitted["params"]["max_tokens"] == 64000
    assert "output_config" not in submitted["params"]


def test_run_batch_anthropic_resume_rebuilds_id_map(monkeypatch):
    """Resume via existing_batch_id must rebuild the deterministic r{i} map."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        batch = MagicMock()
        batch.id = "b1"
        batch.processing_status = "ended"
        client_obj.messages.batches.retrieve.return_value = batch

        def _result(i, text):
            r = MagicMock()
            r.custom_id = f"r{i}"
            r.result.type = "succeeded"
            msg = MagicMock()
            b = MagicMock()
            b.type = "text"
            b.text = text
            msg.content = [b]
            msg.usage = MagicMock(
                input_tokens=1,
                output_tokens=1,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            )
            r.result.message = msg
            return r

        client_obj.messages.batches.results.return_value = [
            _result(0, "A"),
            _result(1, "B"),
        ]
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        created_ids = []
        out = run_batch(
            c,
            [build_batch_entry("kA", "pA"), build_batch_entry("kB", "pB")],
            poll_interval=0,
            existing_batch_id="b1",
            on_batch_created=created_ids.append,
        )
    # No fresh submission on resume, so the created-callback never fires.
    client_obj.messages.batches.create.assert_not_called()
    client_obj.messages.batches.retrieve.assert_called_with("b1")
    assert created_ids == []
    assert out["kA"]["text"] == "A"
    assert out["kB"]["text"] == "B"


def test_run_batch_calls_on_batch_created_after_fresh_submit(monkeypatch):
    """A fresh submission reports its batch id via on_batch_created (ctrl-C
    recovery hook) before the poll loop starts."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        batch = MagicMock()
        batch.id = "b-fresh"
        batch.processing_status = "ended"
        client_obj.messages.batches.create.return_value = batch
        client_obj.messages.batches.retrieve.return_value = batch
        client_obj.messages.batches.results.return_value = []
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        created_ids = []
        run_batch(
            c,
            [build_batch_entry("kA", "pA")],
            poll_interval=0,
            on_batch_created=created_ids.append,
        )
    assert created_ids == ["b-fresh"]


def test_run_batch_openai_parses_results(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("openai.OpenAI") as MockOpenAI, patch("tutormoments.client.time.sleep"):
        client_obj = MagicMock()
        uploaded = MagicMock()
        uploaded.id = "file-1"
        client_obj.files.create.return_value = uploaded
        batch = MagicMock()
        batch.id = "ob1"
        batch.status = "completed"
        batch.output_file_id = "out-1"
        client_obj.batches.create.return_value = batch
        client_obj.batches.retrieve.return_value = batch

        lines = "\n".join(
            json.dumps(
                {
                    "custom_id": cid,
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": text}}],
                            "usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": 3,
                                "total_tokens": 5,
                            },
                        }
                    },
                }
            )
            for cid, text in [("kA", "A"), ("kB", "B")]
        )
        content_obj = MagicMock()
        content_obj.content = lines.encode("utf-8")
        client_obj.files.content.return_value = content_obj
        MockOpenAI.return_value = client_obj
        c = ModelClient("gpt-5.4")
        out = run_batch(
            c,
            [build_batch_entry("kA", "pA"), build_batch_entry("kB", "pB")],
            poll_interval=0,
        )
    assert out["kA"]["text"] == "A"
    assert out["kB"]["text"] == "B"
    assert out["kA"]["usage"]["input_tokens"] == 2
    assert out["kA"]["usage"]["output_tokens"] == 3


def test_run_batch_gemini_parses_results(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-test")
    with (
        patch("google.genai.Client") as MockGenai,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        uploaded = MagicMock()
        uploaded.name = "files/up1"
        client_obj.files.upload.return_value = uploaded
        batch = MagicMock()
        batch.name = "batches/gb1"
        batch.state.name = "JOB_STATE_SUCCEEDED"
        batch.dest.file_name = "files/out1"
        client_obj.batches.create.return_value = batch
        client_obj.batches.get.return_value = batch

        lines = "\n".join(
            json.dumps(
                {
                    "key": key,
                    "response": {
                        "candidates": [{"content": {"parts": [{"text": text}]}}],
                        "usageMetadata": {
                            "promptTokenCount": 4,
                            "candidatesTokenCount": 6,
                            "totalTokenCount": 10,
                        },
                    },
                }
            )
            for key, text in [("kA", "A"), ("kB", "B")]
        )
        client_obj.files.download.return_value = lines.encode("utf-8")
        MockGenai.return_value = client_obj
        c = ModelClient("gemini-3.1-pro-preview")
        out = run_batch(
            c,
            [build_batch_entry("kA", "pA"), build_batch_entry("kB", "pB")],
            poll_interval=0,
        )
    assert out["kA"]["text"] == "A"
    assert out["kB"]["text"] == "B"
    assert out["kA"]["usage"]["input_tokens"] == 4
    assert out["kA"]["usage"]["output_tokens"] == 6


def test_run_batch_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "key-test")
    with patch("openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = MagicMock()
        c = ModelClient("deepseek-ai/DeepSeek-V3")
        with pytest.raises(ValueError, match="Batch API not supported"):
            run_batch(c, [build_batch_entry("k", "p")], poll_interval=0)


# ===================================================================
# Canonical usage vector (cost tracking, Phase 1)
# ===================================================================


def test_normalize_usage_total_is_derived_and_provenance_rides_along():
    u = normalize_usage(
        "gemini",
        "gemini-3.1-pro-preview",
        "sync",
        input_uncached=200,
        cache_read=100,
        cache_write=25,
        output=30,
        reasoning=50,
    )
    assert u["total"] == 405  # one definition: sum of the five buckets
    assert u["provider"] == "gemini"
    assert u["model"] == "gemini-3.1-pro-preview"
    assert u["endpoint"] == "sync"


def test_model_response_default_usage_has_canonical_zero_vector():
    usage = ModelResponse("x").usage
    for key in ("input_uncached", "cache_read", "cache_write", "output", "reasoning"):
        assert usage[key] == 0
    assert usage["total"] == 0


def _gemini_usage_meta(prompt=300, candidates=25, cached=100, thoughts=50):
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        total_token_count=prompt + candidates + thoughts,
        cached_content_token_count=cached,
        thoughts_token_count=thoughts,
    )


def _assert_gemini_vector(usage, endpoint):
    assert usage["input_uncached"] == 200  # prompt 300 minus 100 cached
    assert usage["cache_read"] == 100
    assert usage["cache_write"] == 0
    assert usage["output"] == 25
    assert usage["reasoning"] == 50
    # Canonical total agrees with the provider's thinking-inclusive total,
    # unlike legacy input+output recomputation.
    assert usage["total"] == 375
    assert usage["total_tokens"] == 375
    assert usage["provider"] == "gemini"
    assert usage["endpoint"] == endpoint


def test_gemini_sync_captures_cache_read_and_reasoning(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with patch("google.genai.Client") as MockGenai:
        client_obj = MagicMock()
        client_obj.models.generate_content.return_value = SimpleNamespace(
            text="hi", usage_metadata=_gemini_usage_meta()
        )
        MockGenai.return_value = client_obj
        resp = ModelClient("gemini-3.1-pro-preview").generate("Q", json_mode=False)
    _assert_gemini_vector(resp.usage, "sync")


def test_gemini_stream_captures_cache_read_and_reasoning(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    chunk = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="hi", thought=False)]
                )
            )
        ],
        usage_metadata=_gemini_usage_meta(),
    )
    with patch("google.genai.Client") as MockGenai:
        client_obj = MagicMock()
        client_obj.models.generate_content_stream.return_value = iter([chunk])
        MockGenai.return_value = client_obj
        resp = ModelClient("gemini-3.1-pro-preview").generate(
            "Q", json_mode=False, stream=True
        )
    _assert_gemini_vector(resp.usage, "stream")


def test_openai_sync_splits_cached_tokens_out_of_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(
            prompt_tokens=500,
            completion_tokens=20,
            total_tokens=520,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
        ),
    )
    with patch("openai.OpenAI") as MockOpenAI:
        client_obj = MagicMock()
        client_obj.chat.completions.create.return_value = response
        MockOpenAI.return_value = client_obj
        resp = ModelClient("gpt-5.5-2026-04-23").generate("Q", json_mode=False)
    usage = resp.usage
    assert usage["input_uncached"] == 100
    assert usage["cache_read"] == 400
    assert usage["reasoning"] == 0
    assert usage["output"] == 20
    assert usage["total"] == 520
    assert usage["input_tokens"] == 500  # legacy keys preserved
    assert usage["provider"] == "openai"
    assert usage["endpoint"] == "sync"


def test_openai_sync_captures_cache_write_tokens(monkeypatch):
    """GPT-5.6-generation models meter cache writes (billed at 1.25x input)
    under prompt_tokens_details.cache_write_tokens, a subset of prompt_tokens
    disjoint from cached_tokens. Mirrors live gpt-5.6-luna evidence
    (2026-08-24): prompt 1120 = 1117 written + 3 uncached, plus a mixed case.
    Older models' usage objects lack the attribute entirely -> 0 (covered by
    test_openai_sync_splits_cached_tokens_out_of_prompt's fixture)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
        usage=SimpleNamespace(
            prompt_tokens=1120,
            completion_tokens=5,
            total_tokens=1125,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=100, cache_write_tokens=1017
            ),
        ),
    )
    with patch("openai.OpenAI") as MockOpenAI:
        client_obj = MagicMock()
        client_obj.chat.completions.create.return_value = response
        MockOpenAI.return_value = client_obj
        resp = ModelClient("gpt-5.5-2026-04-23").generate("Q", json_mode=False)
    usage = resp.usage
    assert usage["cache_write"] == 1017
    assert usage["cache_read"] == 100
    assert usage["input_uncached"] == 3  # prompt - cached - written
    assert usage["output"] == 5
    assert usage["total"] == 1125
    assert usage["input_tokens"] == 1120  # legacy key untouched


def test_together_sync_and_stream_report_identical_vectors(monkeypatch):
    """Regression for the sync/stream asymmetry: the sync Together path used
    to drop cached_tokens that the streamed path captured."""
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
    usage_obj = SimpleNamespace(
        prompt_tokens=50,
        completion_tokens=10,
        total_tokens=60,
        prompt_tokens_details=SimpleNamespace(cached_tokens=7),
    )

    with patch("openai.OpenAI") as MockOpenAI:
        client_obj = MagicMock()
        client_obj.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
            usage=usage_obj,
        )
        MockOpenAI.return_value = client_obj
        sync_usage = (
            ModelClient("deepseek-ai/DeepSeek-V4-Pro")
            .generate("Q", json_mode=False)
            .usage
        )

    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))], usage=None
        ),
        SimpleNamespace(choices=[], usage=usage_obj),
    ]
    with patch("openai.OpenAI") as MockOpenAI:
        client_obj = MagicMock()
        client_obj.chat.completions.create.return_value = iter(chunks)
        MockOpenAI.return_value = client_obj
        stream_usage = (
            ModelClient("deepseek-ai/DeepSeek-V4-Pro")
            .generate("Q", json_mode=False, stream=True)
            .usage
        )

    assert sync_usage["cached_tokens"] == 7
    assert sync_usage["cache_read"] == 7
    assert sync_usage["input_uncached"] == 43
    assert sync_usage["provider"] == "together"
    assert sync_usage["endpoint"] == "sync"
    assert stream_usage["endpoint"] == "stream"
    for key, value in sync_usage.items():
        if key != "endpoint":
            assert stream_usage[key] == value, key


def test_anthropic_sync_cache_buckets_stay_disjoint_from_input(monkeypatch):
    """input_tokens already excludes the cache buckets, so input_uncached maps
    to it directly -- re-adding cache tokens at the base rate is the
    double-count trap the costing layer guards against."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    msg = _fake_anthropic_message("hi", in_tok=10, out_tok=5)
    msg.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=800,
    )
    with patch("anthropic.Anthropic") as MockAnthropic:
        client_obj = MagicMock()
        client_obj.messages.create.return_value = msg
        MockAnthropic.return_value = client_obj
        resp = ModelClient("claude-opus-4-6").generate("Q", json_mode=False)
    usage = resp.usage
    assert usage["input_uncached"] == 10
    assert usage["cache_read"] == 800
    assert usage["cache_write"] == 100
    assert usage["output"] == 5
    assert usage["total"] == 915
    assert usage["total_tokens"] == 15  # legacy definition unchanged
    assert usage["endpoint"] == "sync"


def test_run_batch_anthropic_cache_fields_survive(monkeypatch):
    """Regression for the open-coded batch usage dict that silently dropped
    the cache buckets: the batch path must route through _anthropic_usage."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with (
        patch("anthropic.Anthropic") as MockAnthropic,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        batch = MagicMock()
        batch.id = "b1"
        batch.processing_status = "ended"
        client_obj.messages.batches.create.return_value = batch
        client_obj.messages.batches.retrieve.return_value = batch

        r = MagicMock()
        r.custom_id = "r0"
        r.result.type = "succeeded"
        msg = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = "A"
        msg.content = [block]
        msg.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=55,
            cache_read_input_tokens=1234,
        )
        r.result.message = msg
        client_obj.messages.batches.results.return_value = [r]
        MockAnthropic.return_value = client_obj
        c = ModelClient("claude-opus-4-6")
        out = run_batch(c, [build_batch_entry("kA", "pA")], poll_interval=0)
    usage = out["kA"]["usage"]
    assert usage["cache_read"] == 1234
    assert usage["cache_write"] == 55
    assert usage["cache_read_input_tokens"] == 1234  # legacy key too
    assert usage["input_uncached"] == 10
    assert usage["total"] == 1304
    assert usage["endpoint"] == "batch"
    assert usage["provider"] == "anthropic"


def test_run_batch_openai_forces_cache_read_zero(monkeypatch):
    """Prompt caching does not apply on the OpenAI Batch API, so cache_read is
    0 by construction even if the response body carries a cached_tokens field."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("openai.OpenAI") as MockOpenAI, patch("tutormoments.client.time.sleep"):
        client_obj = MagicMock()
        uploaded = MagicMock()
        uploaded.id = "file-1"
        client_obj.files.create.return_value = uploaded
        batch = MagicMock()
        batch.id = "ob1"
        batch.status = "completed"
        batch.output_file_id = "out-1"
        client_obj.batches.create.return_value = batch
        client_obj.batches.retrieve.return_value = batch
        line = json.dumps(
            {
                "custom_id": "kA",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": "A"}}],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 3,
                            "total_tokens": 15,
                            "prompt_tokens_details": {"cached_tokens": 9},
                        },
                    }
                },
            }
        )
        content_obj = MagicMock()
        content_obj.content = line.encode("utf-8")
        client_obj.files.content.return_value = content_obj
        MockOpenAI.return_value = client_obj
        c = ModelClient("gpt-5.4")
        out = run_batch(c, [build_batch_entry("kA", "pA")], poll_interval=0)
    usage = out["kA"]["usage"]
    assert usage["cache_read"] == 0
    assert usage["input_uncached"] == 12
    assert usage["output"] == 3
    assert usage["total"] == 15
    assert usage["endpoint"] == "batch"
    assert usage["provider"] == "openai"


def test_run_batch_gemini_captures_cache_read_and_reasoning(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-test")
    with (
        patch("google.genai.Client") as MockGenai,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj = MagicMock()
        uploaded = MagicMock()
        uploaded.name = "files/up1"
        client_obj.files.upload.return_value = uploaded
        batch = MagicMock()
        batch.name = "batches/gb1"
        batch.state.name = "JOB_STATE_SUCCEEDED"
        batch.dest.file_name = "files/out1"
        client_obj.batches.create.return_value = batch
        client_obj.batches.get.return_value = batch
        line = json.dumps(
            {
                "key": "kA",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "A"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 300,
                        "candidatesTokenCount": 25,
                        "totalTokenCount": 375,
                        "cachedContentTokenCount": 100,
                        "thoughtsTokenCount": 50,
                    },
                },
            }
        )
        client_obj.files.download.return_value = line.encode("utf-8")
        MockGenai.return_value = client_obj
        c = ModelClient("gemini-3.1-pro-preview")
        out = run_batch(c, [build_batch_entry("kA", "pA")], poll_interval=0)
    _assert_gemini_vector(out["kA"]["usage"], "batch")


# ===================================================================
# Thinking ladder on the batch paths + cancel_batch + sync forwarding
# ===================================================================


def _mock_gemini_batch_client(result_text: str):
    """Gemini batch double that captures the uploaded JSONL before the submit
    path unlinks the temp file."""
    client_obj = MagicMock()
    captured = {}

    def _upload(file, config=None):
        with open(file, encoding="utf-8") as fh:
            captured["jsonl"] = fh.read()
        uploaded = MagicMock()
        uploaded.name = "files/up1"
        return uploaded

    client_obj.files.upload.side_effect = _upload
    batch = MagicMock()
    batch.name = "batches/gb1"
    batch.state.name = "JOB_STATE_SUCCEEDED"
    batch.dest.file_name = "files/out1"
    client_obj.batches.create.return_value = batch
    client_obj.batches.get.return_value = batch
    client_obj.files.download.return_value = result_text.encode("utf-8")
    return client_obj, captured


def test_run_batch_gemini_injects_thinking_config_without_mutating_entries(
    monkeypatch,
):
    """thinking="dynamic" lands in every uploaded JSONL line's
    generation_config as the include_thoughts/budget -1 wire shape, and the
    caller's entry dicts stay untouched (resume flows re-submit them)."""
    monkeypatch.setenv("GEMINI_API_KEY", "key-test")
    result_text = "\n".join(
        json.dumps(
            {
                "key": key,
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "A"}]}}],
                    "usageMetadata": {},
                },
            }
        )
        for key in ("kA", "kB")
    )
    entries = [build_batch_entry("kA", "pA"), build_batch_entry("kB", "pB")]
    entries_before = copy.deepcopy(entries)
    with (
        patch("google.genai.Client") as MockGenai,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj, captured = _mock_gemini_batch_client(result_text)
        MockGenai.return_value = client_obj
        c = ModelClient("gemini-2.5-pro")
        out = run_batch(c, entries, poll_interval=0, thinking="dynamic")
    lines = [json.loads(ln) for ln in captured["jsonl"].strip().splitlines()]
    assert len(lines) == 2
    for line in lines:
        assert line["request"]["generation_config"]["thinking_config"] == {
            "include_thoughts": True,
            "thinking_budget": -1,
        }
        # The rest of the generation_config is carried over unchanged.
        assert line["request"]["generation_config"]["max_output_tokens"] == 65536
    assert entries == entries_before
    assert set(out) == {"kA", "kB"}


def test_run_batch_gemini_skips_thought_parts_and_joins_text(monkeypatch):
    """Result parsing drops parts flagged thought: true (reasoning summaries,
    not the answer) and joins the remaining text parts, keeping batch text
    identical to sync response.text."""
    monkeypatch.setenv("GEMINI_API_KEY", "key-test")
    result_text = json.dumps(
        {
            "key": "kA",
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "reasoning...", "thought": True},
                                {"text": '{"a":'},
                                {"text": " 1}"},
                            ]
                        }
                    }
                ],
                "usageMetadata": {},
            },
        }
    )
    with (
        patch("google.genai.Client") as MockGenai,
        patch("tutormoments.client.time.sleep"),
    ):
        client_obj, _ = _mock_gemini_batch_client(result_text)
        MockGenai.return_value = client_obj
        c = ModelClient("gemini-2.5-pro")
        out = run_batch(
            c, [build_batch_entry("kA", "pA")], poll_interval=0, thinking="dynamic"
        )
    assert out["kA"]["text"] == '{"a": 1}'


def test_run_batch_unsatisfiable_level_fails_before_upload(monkeypatch):
    """A level the model cannot honor dies at resolve time -- nothing is
    uploaded and no batch job is created."""
    monkeypatch.setenv("GEMINI_API_KEY", "key-test")
    with patch("google.genai.Client") as MockGenai:
        client_obj = MagicMock()
        MockGenai.return_value = client_obj
        c = ModelClient("gemini-2.5-pro")
        with pytest.raises(ThinkingConfigError):
            run_batch(
                c, [build_batch_entry("k", "p")], poll_interval=0, thinking="none"
            )
    client_obj.files.upload.assert_not_called()
    client_obj.batches.create.assert_not_called()


def test_run_sync_entries_forwards_thinking_to_generate(monkeypatch):
    """run_sync_entries takes the same ladder level run_batch does and passes
    it straight through to generate(), so switching between the sync and batch
    paths keeps the same benchmark condition."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("anthropic.Anthropic"):
        c = ModelClient("claude-opus-4-6")
    with patch.object(c, "generate", return_value=ModelResponse("R")) as mock_gen:
        out = run_sync_entries(c, [build_batch_entry("k1", "p1")], thinking="dynamic")
    assert mock_gen.call_args.kwargs["thinking"] == "dynamic"
    assert out["k1"]["text"] == "R"


class TestCancelBatch:
    def test_gemini_routes_to_batches_cancel_by_name(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        with patch("google.genai.Client") as MockGenai:
            client_obj = MagicMock()
            MockGenai.return_value = client_obj
            cancel_batch(ModelClient("gemini-2.5-pro"), "batches/gb1")
        client_obj.batches.cancel.assert_called_once_with(name="batches/gb1")

    def test_openai_routes_to_batches_cancel(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch("openai.OpenAI") as MockOpenAI:
            client_obj = MagicMock()
            MockOpenAI.return_value = client_obj
            cancel_batch(ModelClient("gpt-5.5"), "ob1")
        client_obj.batches.cancel.assert_called_once_with("ob1")

    def test_anthropic_routes_to_messages_batches_cancel(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        with patch("anthropic.Anthropic") as MockAnthropic:
            client_obj = MagicMock()
            MockAnthropic.return_value = client_obj
            cancel_batch(ModelClient("claude-opus-4-6"), "mb1")
        client_obj.messages.batches.cancel.assert_called_once_with("mb1")

    def test_unsupported_provider_raises(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "key-test")
        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()
            c = ModelClient("deepseek-ai/DeepSeek-V4-Pro")
        with pytest.raises(ValueError, match="Batch API not supported"):
            cancel_batch(c, "b1")


def test_openai_usage_captures_reasoning_tokens():
    """The informational reasoning_tokens key rides alongside the usage vector.

    It is a subset of completion_tokens (not a separate cost bucket), so the
    canonical `reasoning` bucket stays 0 -- moving it there would double-count
    against `output`. The smoke layer reads it as thinking evidence.
    """
    from tutormoments.client import _openai_batch_usage, _openai_style_usage

    usage_obj = SimpleNamespace(
        prompt_tokens=53,
        completion_tokens=109,
        total_tokens=162,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=64),
    )
    usage = _openai_style_usage(
        usage_obj, provider="openai", model="gpt-5.5", endpoint="sync"
    )
    assert usage["reasoning_tokens"] == 64
    assert usage["reasoning"] == 0
    assert usage["output"] == 109
    assert usage["total"] == 53 + 109

    batch = _openai_batch_usage(
        {
            "prompt_tokens": 53,
            "completion_tokens": 109,
            "total_tokens": 162,
            "completion_tokens_details": {"reasoning_tokens": 64},
        },
        model="gpt-5.5",
    )
    assert batch["reasoning_tokens"] == 64
    assert batch["reasoning"] == 0

    # Absent details: key present, zero -- "reported dimension, zero observed".
    bare = _openai_style_usage(
        SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        provider="together",
        model="deepseek-ai/DeepSeek-V4-Pro",
        endpoint="sync",
    )
    assert bare["reasoning_tokens"] == 0
