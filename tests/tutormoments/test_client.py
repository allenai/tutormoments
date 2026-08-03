import json
from unittest.mock import MagicMock, patch

import pytest

from tutormoments.client import (
    ModelClient,
    ModelResponse,
    _anthropic_thinking_param,
    _mime_from_path,
    _strip_json_fences,
    build_batch_entry,
    infer_provider,
    run_batch,
    run_sync_entries,
    write_jsonl,
)


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


class TestAnthropicThinkingParam:
    def test_adaptive_models_get_adaptive_shape(self):
        for m in (
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
        ):
            assert _anthropic_thinking_param(m, 0) == {"type": "adaptive"}

    def test_haiku_4_5_gets_legacy_enabled_shape(self):
        # Haiku 4.5 is the one current model without adaptive-thinking support
        # (extended thinking only, per the Anthropic models overview) --
        # sending {"type": "adaptive"} there would 400.
        assert _anthropic_thinking_param("claude-haiku-4-5", 4096) == {
            "type": "enabled",
            "budget_tokens": 4096,
        }
        assert _anthropic_thinking_param("claude-haiku-4-5-20251001", 0) == {
            "type": "enabled",
            "budget_tokens": 16384,
        }

    def test_unknown_or_future_model_defaults_to_adaptive(self):
        # A brand-new Anthropic model must work with no code change (README
        # "Running new tutor models"). Anything not in the frozen legacy set
        # gets adaptive.
        for m in ("claude-opus-5", "claude-sonnet-6", "totally-new-claude"):
            assert _anthropic_thinking_param(m, 0) == {"type": "adaptive"}

    def test_legacy_model_gets_enabled_with_budget(self):
        assert _anthropic_thinking_param("claude-sonnet-4-5", 8192) == {
            "type": "enabled",
            "budget_tokens": 8192,
        }

    def test_legacy_model_zero_budget_defaults_16384(self):
        assert _anthropic_thinking_param("claude-sonnet-4-5", 0) == {
            "type": "enabled",
            "budget_tokens": 16384,
        }


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
# Task 7: vision image-block builders + cacheable-prefix
# ===================================================================

from tutormoments.client import (
    VISION_CAPABLE_PREFIXES,
    _base64_bytes,
    _build_image_blocks_anthropic,
    _build_image_blocks_gemini,
    _build_image_blocks_openai,
    _presigned_url,
    _should_use_presigned_url,
    validate_vision_support,
)


class TestVisionSupport:
    def test_vision_capable_prefixes_covers_current_anthropic_tutors(self):
        # Every Anthropic model in the tutor roster must pass the vision gate.
        validate_vision_support("claude-sonnet-5")
        validate_vision_support("claude-fable-5")

    def test_vision_capable_prefixes_contains_known_models(self):
        # Spot-check a few entries from the source list.
        assert "claude-opus-4" in VISION_CAPABLE_PREFIXES
        assert "gpt-4o" in VISION_CAPABLE_PREFIXES
        assert "gemini-2" in VISION_CAPABLE_PREFIXES

    def test_validate_vision_support_passes_for_capable_model(self):
        # Should not raise.
        validate_vision_support("claude-opus-4-6")
        validate_vision_support("gpt-4o-mini")
        validate_vision_support("gemini-2.0-flash")

    def test_validate_vision_support_raises_for_unknown_model(self):
        with pytest.raises(ValueError, match="not in the vision-capable list"):
            validate_vision_support("claude-2")


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
    assert out["k1"]["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


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
            msg.usage = MagicMock(input_tokens=1, output_tokens=1)
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
    """Batch requests must carry the same thinking/effort params as sync calls
    (benchmark fidelity: batch mode may not silently diverge from run_conversation)."""
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
            thinking=True,
            effort="xhigh",
        )
        (submitted,) = client_obj.messages.batches.create.call_args.kwargs["requests"]
    assert submitted["params"]["thinking"] == {"type": "adaptive"}
    assert submitted["params"]["output_config"] == {"effort": "xhigh"}


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
            msg.usage = MagicMock(input_tokens=1, output_tokens=1)
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
            existing_batch_id="b1",
        )
    # No fresh submission on resume.
    client_obj.messages.batches.create.assert_not_called()
    client_obj.messages.batches.retrieve.assert_called_with("b1")
    assert out["kA"]["text"] == "A"
    assert out["kB"]["text"] == "B"


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
