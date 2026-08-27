"""Model provider routing and calling (sync + batch), shared by tutor/student/scorer."""

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google.genai import types

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Order matters (first match wins).
PROVIDER_PREFIXES = [
    ("gemini", "gemini"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    ("deepseek-ai/", "together"),
    ("moonshotai/", "together"),
    ("minimaxai/", "together"),
    ("google/gemma", "together"),
    ("meta-llama/", "together"),
    ("qwen/", "together"),
]


VISION_CAPABLE_PREFIXES = (
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-sonnet-5",
    "claude-fable-5",
    "gemini-2",
    "gemini-3",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o4",
)


def validate_vision_support(model: str) -> None:
    """Raise ValueError if the model is not known to support vision input."""
    m = model.lower()
    if not any(m.startswith(p) for p in VISION_CAPABLE_PREFIXES):
        raise ValueError(
            f"Model '{model}' is not in the vision-capable list. "
            f"Vision-capable prefixes: {', '.join(VISION_CAPABLE_PREFIXES)}."
        )


def infer_provider(model: str) -> str:
    """Infer provider from model name string.

    Examples:
        'gemini-3.1-pro-preview' -> 'gemini'
        'gpt-4o'               -> 'openai'
        'o3-mini'              -> 'openai'
        'claude-sonnet-4-6'    -> 'anthropic'
    """
    model_lower = model.lower()
    for prefix, provider in PROVIDER_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Expected prefix: {', '.join(p for p, _ in PROVIDER_PREFIXES)}"
    )


@dataclass
class ModelResponse:
    """Unified response from any provider."""

    text: str
    # Zero legacy counters plus the canonical cost vector (see
    # normalize_usage). No provenance here: this default is only used where
    # no real API call happened (e.g. registered/callable tutors), so there
    # is no provider/model/endpoint to attribute the zeros to.
    usage: dict = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_uncached": 0,
            "cache_read": 0,
            "cache_write": 0,
            "output": 0,
            "reasoning": 0,
            "total": 0,
        }
    )
    # Wall-clock seconds from the start of the successful generate() attempt
    # to the response landing. Stamped by ModelClient.generate(). Retries
    # are not included (they're bookkeeping, not model reasoning time).
    # Unaffected by streaming, so runs before and after the streaming work
    # remain comparable -- the streaming metrics are additive, on `timing`.
    latency_seconds: float | None = None
    # Populated only on streamed calls (generate(stream=True)). Shape:
    #   {ttfc_seconds, ttft_seconds, ttlt_seconds, output_tokens,
    #    cache_read_input_tokens, output_tps}
    # See _StreamTimer for what each anchor means.
    timing: dict | None = None


TOGETHER_BASE_URL = "https://api.together.xyz/v1"


# ===================================================================
# Streaming latency instrumentation
# ===================================================================


class _StreamTimer:
    """Stamps TTFC / TTFT / TTLT as stream deltas arrive.

    The three anchors answer different questions:

    - ``ttfc`` -- first chunk of *any* kind, reasoning included. Diagnostic:
      the gap between ttfc and ttft is how long the model spent thinking.
    - ``ttft`` -- first *visible answer* token. This is what a student
      actually sees appear, and the reason we don't use the Artificial
      Analysis definition (first token of any kind): on the OpenAI path
      reasoning is never streamed, so first-token-of-any-kind is not
      computable uniformly across our providers.
    - ``ttlt`` -- last *visible* delta, i.e. when the student can reply. It is
      deliberately not iterator exhaustion: the trailing usage-only chunk and
      connection teardown are not part of anyone's wait.

    Call exactly one of chunk()/visible() per stream event: each reads the
    clock, so calling both would attribute two timestamps to one arrival.

    The clock is injectable so tests can assert exact values. It is resolved
    at construction rather than bound as a default argument so that patching
    ``tutormoments.client.time.monotonic`` reaches it.
    """

    def __init__(self, clock=None):
        self._clock = clock or time.monotonic
        self.t0 = self._clock()
        self._ttfc: float | None = None
        self._ttft: float | None = None
        self._ttlt: float | None = None

    def chunk(self) -> None:
        """Record that some chunk arrived (thinking, usage, or content)."""
        if self._ttfc is None:
            self._ttfc = self._clock() - self.t0

    def visible(self) -> None:
        """Record that a visible answer delta arrived."""
        elapsed = self._clock() - self.t0
        if self._ttfc is None:
            self._ttfc = elapsed
        if self._ttft is None:
            self._ttft = elapsed
        self._ttlt = elapsed

    def as_dict(
        self,
        *,
        output_tokens: int | None = None,
        cache_read_input_tokens: int | None = None,
    ) -> dict:
        """Build the timing block recorded on ModelResponse.timing.

        output_tps spans ttfc -> ttlt, not ttft -> ttlt. `output_tokens`
        counts everything the model generated, thinking included, so the
        window has to start when generation started rather than when visible
        text did. Measuring thinking-inclusive tokens over a text-only window
        overstates throughput several-fold on thinking models (a live Haiku
        call reported 870 tok/s that way). For non-thinking models the two
        anchors nearly coincide, so this changes little.
        """
        tps = None
        if (
            output_tokens
            and self._ttfc is not None
            and self._ttlt is not None
            and self._ttlt > self._ttfc
        ):
            tps = output_tokens / (self._ttlt - self._ttfc)
        return {
            "ttfc_seconds": self._ttfc,
            "ttft_seconds": self._ttft,
            "ttlt_seconds": self._ttlt,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "output_tps": tps,
        }


class _InlineThinkGate:
    """Defer TTFT past an inline ``<think>...</think>`` block.

    Together-hosted open-weight reasoners (DeepSeek-V4-Pro, Kimi) emit their
    chain of thought inside the ordinary content stream rather than in a
    separate block the way Anthropic and Gemini do. Without this gate the
    first content delta would be counted as the first visible token, making
    those models look far faster to first token than a student would ever
    experience.

    Feed every content delta to :meth:`visible`; it returns True from the
    first delta that is genuinely visible answer text.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._buf = ""
        self._in_think: bool | None = None  # None until the stream reveals it
        self._done = False

    def visible(self, piece: str) -> bool:
        if self._done:
            return True
        self._buf += piece

        if self._in_think is None:
            stripped = self._buf.lstrip()
            if not stripped:
                return False
            head = stripped[: len(self._OPEN)]
            if self._OPEN.startswith(head):
                # Still consistent with an opening tag -- wait for enough
                # characters to decide rather than guessing either way.
                if len(head) < len(self._OPEN):
                    return False
                self._in_think = True
            else:
                self._in_think = False

        if not self._in_think:
            self._done = True
            return True

        idx = self._buf.find(self._CLOSE)
        if idx == -1:
            return False
        # Closing tag seen; the first non-blank text after it is visible.
        if self._buf[idx + len(self._CLOSE) :].strip():
            self._done = True
            return True
        return False


# Adaptive thinking ({"type": "adaptive"}) is the modern default and the shape
# every current Opus/Sonnet-tier Anthropic model uses. The legacy
# enabled+budget_tokens shape is needed only by a *closed, frozen set* of
# pre-adaptive models -- no new model is ever added here. So we default to
# adaptive and enumerate only the legacy models; a brand-new Anthropic model
# works with no code change (see README "Running new tutor models"). Legacy
# enabled+budget is rejected on Opus 4.7+/Sonnet 5/Fable 5. Haiku 4.5 is the
# one current model still on the legacy shape (extended thinking only, no
# adaptive support per the Anthropic models overview).
_ANTHROPIC_LEGACY_THINKING_MODELS = (
    "claude-3-7-sonnet",
    "claude-haiku-4-5",
    "claude-opus-4-0",
    "claude-opus-4-20250514",
    "claude-opus-4-1",
    "claude-opus-4-5",
    "claude-sonnet-4-0",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5",
)


def _anthropic_thinking_param(model: str, thinking_budget: int) -> dict:
    """Return the right `thinking` kwarg shape for the given model.

    Defaults to adaptive ({"type": "adaptive"}), which every current Anthropic
    model requires. Only the frozen pre-adaptive models in
    _ANTHROPIC_LEGACY_THINKING_MODELS get the enabled+budget_tokens form.
    """
    if model and any(
        model.startswith(prefix) for prefix in _ANTHROPIC_LEGACY_THINKING_MODELS
    ):
        budget = thinking_budget if thinking_budget > 0 else 16384
        return {"type": "enabled", "budget_tokens": budget}
    return {"type": "adaptive"}


class ModelClient:
    """Provider-agnostic synchronous model client.

    Instantiates the appropriate SDK client based on the model name,
    and provides a unified `generate()` method with retry logic.
    """

    def __init__(self, model: str):
        self.model = model
        self.provider = infer_provider(model)
        self._client = self._init_client()

    def _init_client(self):
        """Initialize the SDK client for the inferred provider."""
        if self.provider == "gemini":
            from google import genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not found in environment")
            return genai.Client(api_key=api_key)

        elif self.provider == "openai":
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found in environment")
            return OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not found in environment")
            return anthropic.Anthropic(api_key=api_key)

        elif self.provider == "together":
            # Together is OpenAI-compatible -- same SDK, different base_url + key.
            from openai import OpenAI

            api_key = os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise RuntimeError("TOGETHER_API_KEY not found in environment")
            return OpenAI(api_key=api_key, base_url=TOGETHER_BASE_URL)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        json_mode: bool = True,
        max_tokens: int = 0,
        timeout: int = 120,
        thinking: bool = False,
        thinking_budget: int = 0,
        reasoning_effort: str = "",
        effort: str = "",
        enable_cache: bool = False,
        *,
        output_schema: dict | None = None,
        cacheable_prefix: str | None = None,
        stream: bool = False,
    ) -> "ModelResponse":
        """Generate a response from the model with retry logic.

        output_schema: optional JSON Schema for structured output. When set,
        the Anthropic path constrains the response via output_config.format
        (json_schema). Only supported on the anthropic provider today.

        stream: when True, consume the response as a token stream and record
        per-call TTFT/TTLT on ModelResponse.timing. Text and usage are
        identical to the non-streamed path -- streaming changes how the
        response is transported, not how it is sampled. Only the conversation
        path (tutor/student turns) streams; the scorer, taxonomy and
        ground-truth paths stay non-streamed so their behaviour is unchanged.
        """
        from tutormoments.config import get_retry_config

        if output_schema is not None and self.provider != "anthropic":
            raise ValueError(
                "output_schema is only supported on the anthropic provider"
            )

        if max_tokens <= 0:
            max_tokens = MAX_OUTPUT_TOKENS.get(self.provider, 8192)

        retry_cfg = get_retry_config()
        max_retries = retry_cfg.get("max_retries", 5)
        base_delay = retry_cfg.get("base_delay", 5)

        last_error = None
        call_t0 = time.monotonic()
        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    resp = self._generate_gemini(
                        prompt,
                        json_mode,
                        max_tokens,
                        timeout,
                        thinking,
                        thinking_budget,
                        images,
                        cacheable_prefix=cacheable_prefix,
                        stream=stream,
                    )
                elif self.provider == "openai":
                    resp = self._generate_openai(
                        prompt,
                        json_mode,
                        max_tokens,
                        timeout,
                        thinking,
                        thinking_budget,
                        reasoning_effort=reasoning_effort,
                        images=images,
                        cacheable_prefix=cacheable_prefix,
                        stream=stream,
                    )
                elif self.provider == "anthropic":
                    resp = self._generate_anthropic(
                        prompt,
                        json_mode,
                        max_tokens,
                        timeout,
                        thinking,
                        thinking_budget,
                        reasoning_effort=reasoning_effort,
                        effort=effort,
                        images=images,
                        enable_cache=enable_cache,
                        output_schema=output_schema,
                        cacheable_prefix=cacheable_prefix,
                        stream=stream,
                    )
                elif self.provider == "together":
                    resp = self._generate_together(
                        prompt,
                        json_mode,
                        max_tokens,
                        timeout,
                        cacheable_prefix=cacheable_prefix,
                        stream=stream,
                    )
                else:
                    raise RuntimeError(f"unknown provider {self.provider}")
                # Stamp wall-clock latency for the successful attempt only
                # (retries are bookkeeping, not the model's reasoning time).
                # Deliberately unchanged by streaming: this field predates the
                # streaming work and feeds the paper's latency figure, so it
                # keeps meaning exactly what it always did -- end-to-end
                # seconds for the call. The streaming metrics are additive and
                # live on `timing`; overriding this one with TTLT would
                # silently redefine a published field.
                resp.latency_seconds = time.monotonic() - call_t0
                return resp
            except Exception as e:
                last_error = e
                delay = base_delay * (2**attempt)
                if attempt < max_retries - 1:
                    logger.warning(
                        "API error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("API failed after %d attempts: %s", max_retries, e)

        raise RuntimeError(
            f"API call failed after {max_retries} attempts: {last_error}"
        )

    def _generate_gemini(
        self,
        prompt,
        json_mode,
        max_tokens,
        timeout,
        thinking=False,
        thinking_budget=0,
        images=None,
        cacheable_prefix: str | None = None,
        stream: bool = False,
    ):
        """Gemini API call via google-genai SDK."""
        config = {
            "max_output_tokens": max_tokens,
            "http_options": {"timeout": timeout * 1000},
        }
        if json_mode:
            config["response_mime_type"] = "application/json"
        if thinking:
            # thinking_budget = -1 means "dynamic" (model self-paces).
            # 0 = no thinking. Positive = fixed budget. None/unset = default 16384.
            if thinking_budget is None or thinking_budget == 0:
                budget = 16384
            else:
                budget = thinking_budget  # may be -1 (dynamic) or positive
            config["thinking_config"] = {
                "include_thoughts": True,
                "thinking_budget": budget,
            }

        # Gemini has no server-side prompt cache wired here; the cacheable
        # head is concatenated into the prompt, which is semantically
        # equivalent (just without the cache-hit cost savings).
        effective_prompt = (cacheable_prefix or "") + prompt
        if images:
            image_blocks = _build_image_blocks_gemini(images)
            parts = _interleave_text_and_images(
                effective_prompt,
                image_blocks,
                lambda s: {"text": s},
            )
            contents = [{"role": "user", "parts": parts}]
        else:
            contents = effective_prompt

        if stream:
            return self._stream_gemini(contents, config)

        response = self._client.models.generate_content(
            model=f"models/{self.model}",
            contents=contents,
            config=config,
        )

        text = response.text or ""
        usage = _gemini_usage(
            response.usage_metadata, model=self.model, endpoint="sync"
        )
        return ModelResponse(text=text, usage=usage)

    def _stream_gemini(self, contents, config):
        """Stream a Gemini response, timing first-visible and last deltas.

        Gemini marks reasoning parts with ``part.thought`` -- those advance
        ttfc but never ttft, so TTFT reflects the first token a student sees.
        """
        timer = _StreamTimer()
        pieces: list[str] = []
        usage_meta = None

        for chunk in self._client.models.generate_content_stream(
            model=f"models/{self.model}",
            contents=contents,
            config=config,
        ):
            if getattr(chunk, "usage_metadata", None) is not None:
                usage_meta = chunk.usage_metadata
            saw_visible = False
            for part in _gemini_stream_parts(chunk):
                if getattr(part, "thought", False):
                    continue
                text = getattr(part, "text", None)
                if text:
                    timer.visible()
                    pieces.append(text)
                    saw_visible = True
            if not saw_visible:
                timer.chunk()

        usage = _gemini_usage(usage_meta, model=self.model, endpoint="stream")
        return ModelResponse(
            text="".join(pieces),
            usage=usage,
            # Gemini has no explicit cache wired here (the prefix is
            # concatenated into the prompt), so cache state is unknowable.
            timing=timer.as_dict(output_tokens=usage["output_tokens"]),
        )

    def _generate_openai(
        self,
        prompt,
        json_mode,
        max_tokens,
        timeout,
        thinking=False,
        thinking_budget=0,
        reasoning_effort: str = "",
        images=None,
        cacheable_prefix: str | None = None,
        stream: bool = False,
    ):
        """OpenAI API call via openai SDK."""
        if images:
            image_blocks = _build_image_blocks_openai(
                images,
                use_url=_should_use_presigned_url(),
            )
            # Prepend cacheable head so auto-cache sees the same prefix on repeats.
            head_text = cacheable_prefix or ""
            content = _interleave_text_and_images(
                head_text + prompt,
                image_blocks,
                lambda s: {"type": "text", "text": s},
            )
        else:
            content = (cacheable_prefix or "") + prompt

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": max_tokens,
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if stream:
            return self._stream_openai_compatible(kwargs, inline_think=False)

        response = self._client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        usage = _openai_style_usage(
            response.usage, provider=self.provider, model=self.model, endpoint="sync"
        )
        return ModelResponse(text=text, usage=usage)

    def _generate_together(
        self,
        prompt,
        json_mode,
        max_tokens,
        timeout,
        cacheable_prefix: str | None = None,
        stream: bool = False,
    ):
        """Together (open-weight) call via OpenAI-compatible chat completions.

        Together uses `max_tokens` (not `max_completion_tokens`) and does not
        accept `reasoning_effort`. Open-weight reasoners (DeepSeek-V4, Kimi)
        produce their own chain-of-thought internally; there's no depth knob
        to pass. There is no cache_control-style API, so the cacheable head is
        just concatenated into the prompt (same as the Gemini path); Together
        still reports server-side cache hits via cached_tokens (observed live
        on DeepSeek-V4-Pro), whether or not a billing discount applies.
        """
        content = (cacheable_prefix or "") + prompt
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": min(max_tokens, MAX_OUTPUT_TOKENS["together"]),
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if stream:
            resp = self._stream_openai_compatible(kwargs, inline_think=True)
            if json_mode:
                resp.text = _strip_json_fences(resp.text)
            return resp

        response = self._client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        if json_mode:
            text = _strip_json_fences(text)
        usage = _openai_style_usage(
            response.usage, provider=self.provider, model=self.model, endpoint="sync"
        )
        return ModelResponse(text=text, usage=usage)

    def _stream_openai_compatible(self, kwargs: dict, *, inline_think: bool):
        """Stream an OpenAI-compatible chat completion, timing the deltas.

        Shared by the OpenAI and Together paths -- same wire protocol, one
        difference that matters for TTFT:

        - OpenAI (chat.completions) never streams reasoning tokens, so the
          first content delta is both ttfc and ttft.
        - Together's open-weight reasoners emit ``<think>...</think>`` inline
          in the content stream, so `inline_think=True` gates ttft past the
          closing tag (see _InlineThinkGate).

        `include_usage` support is inconsistent on Together; when the final
        usage chunk never arrives we record zeros rather than guessing, and
        output_tps is simply omitted.
        """
        kwargs = {
            **kwargs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        timer = _StreamTimer()
        gate = _InlineThinkGate() if inline_think else None
        pieces: list[str] = []
        usage_obj = None

        for chunk in self._client.chat.completions.create(**kwargs):
            if getattr(chunk, "usage", None) is not None:
                usage_obj = chunk.usage
            # The final usage-only chunk carries no choices.
            choices = getattr(chunk, "choices", None)
            delta = getattr(choices[0], "delta", None) if choices else None
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece and (gate is None or gate.visible(piece)):
                timer.visible()
            else:
                timer.chunk()
            if piece:
                pieces.append(piece)

        usage = _openai_style_usage(
            usage_obj, provider=self.provider, model=self.model, endpoint="stream"
        )
        return ModelResponse(
            text="".join(pieces),
            usage=usage,
            timing=timer.as_dict(
                output_tokens=usage["output_tokens"] or None,
                cache_read_input_tokens=usage["cached_tokens"],
            ),
        )

    def _generate_anthropic(
        self,
        prompt,
        json_mode,
        max_tokens,
        timeout,
        thinking=False,
        thinking_budget=0,
        reasoning_effort="",
        effort="",
        images=None,
        enable_cache=False,
        output_schema: dict | None = None,
        cacheable_prefix: str | None = None,
        stream: bool = False,
    ):
        """Anthropic API call via anthropic SDK."""
        system_parts = []
        if json_mode:
            system_parts.append(
                "You must respond with valid JSON only. "
                "Do not include markdown code fences, explanations, or any text "
                "outside the JSON object."
            )

        if images:
            image_blocks = _build_image_blocks_anthropic(
                images,
                use_url=_should_use_presigned_url(),
                enable_cache=enable_cache,
            )
            content = _interleave_text_and_images(
                prompt,
                image_blocks,
                lambda s: {"type": "text", "text": s},
            )
            if cacheable_prefix is not None:
                # Prepend the cacheable head as its own text block.
                content = [
                    {
                        "type": "text",
                        "text": cacheable_prefix,
                        "cache_control": {"type": "ephemeral"},
                    },
                ] + content
        elif cacheable_prefix is not None:
            content = [
                {
                    "type": "text",
                    "text": cacheable_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        # Clamp max_tokens to the model's per-model output cap (Haiku 4.5 and
        # Sonnet 4.6 max out at 64K -- requests above the cap are rejected).
        capped_max = max_tokens
        for prefix, cap in _ANTHROPIC_MAX_OUTPUT_CAP.items():
            if self.model and self.model.startswith(prefix):
                capped_max = min(capped_max, cap)
                break
        kwargs = {
            "model": self.model,
            "max_tokens": capped_max,
            "messages": [{"role": "user", "content": content}],
            "timeout": timeout,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)
        if thinking:
            kwargs["thinking"] = _anthropic_thinking_param(self.model, thinking_budget)
            if kwargs["thinking"].get("type") == "enabled":
                # Legacy enabled mode needs max_tokens >= budget + headroom.
                budget = kwargs["thinking"]["budget_tokens"]
                if kwargs["max_tokens"] < budget + 64:
                    kwargs["max_tokens"] = budget + 64

        # effort and structured-output format both live under output_config.
        # effort is only valid on adaptive-thinking models that support it
        # (Haiku 4.5 400s if effort is sent -- skip there). output_schema
        # constrains the response via output_config.format (json_schema).
        # SDK <= 0.71 doesn't expose output_config as a top-level kwarg, so
        # we forward it via extra_body. The server accepts it either way.
        output_config: dict = {}
        if effort and self.model and not self.model.startswith("claude-haiku-4-5"):
            output_config["effort"] = effort
        if output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}
        if output_config:
            kwargs["extra_body"] = {"output_config": output_config}

        if stream:
            return self._stream_anthropic(kwargs, json_mode=json_mode)

        response = self._client.messages.create(**kwargs)

        # Extract text from content blocks (skip thinking blocks)
        text = _extract_anthropic_text(response.content)
        # Strip markdown fences if present
        if json_mode:
            text = _strip_json_fences(text)

        usage = _anthropic_usage(response.usage, model=self.model, endpoint="sync")
        return ModelResponse(text=text, usage=usage)

    def _stream_anthropic(self, kwargs: dict, *, json_mode: bool):
        """Stream an Anthropic message, timing first-visible and last deltas.

        Anthropic streams thinking and text as separate content blocks, so we
        track each block's type from its ``content_block_start`` and count a
        delta as visible only inside a ``text`` block. Thinking deltas advance
        ttfc but never ttft -- the student sees nothing while the model thinks.

        Text and usage come from ``get_final_message()``, which accumulates
        exactly what the non-streamed path would have returned.
        """
        timer = _StreamTimer()
        block_types: dict[int, str] = {}

        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block_types[event.index] = getattr(event.content_block, "type", "")
                    timer.chunk()
                elif (
                    etype == "content_block_delta"
                    and block_types.get(event.index) == "text"
                ):
                    timer.visible()
                else:
                    timer.chunk()
            message = stream.get_final_message()

        text = _extract_anthropic_text(message.content)
        if json_mode:
            text = _strip_json_fences(text)

        usage = _anthropic_usage(message.usage, model=self.model, endpoint="stream")
        return ModelResponse(
            text=text,
            usage=usage,
            timing=timer.as_dict(
                output_tokens=usage["output_tokens"] or None,
                cache_read_input_tokens=usage["cache_read_input_tokens"],
            ),
        )

    def __repr__(self):
        return f"ModelClient(model='{self.model}', provider='{self.provider}')"


# ===================================================================
# Shared client cache (conversation path)
# ===================================================================

_CLIENT_CACHE: dict[str, "ModelClient"] = {}


def get_client(model: str) -> "ModelClient":
    """Return a shared ModelClient for `model`, constructing it once.

    Conversations previously built a fresh client per moment, so each moment's
    first request paid a fresh TLS handshake. That cost lands squarely on the
    conversation's *first* turn -- the cold-cache turn -- systematically
    inflating exactly the latency figure the probe reports, by an amount that
    has nothing to do with the model. Reusing one client per model keeps the
    SDK's HTTP connection pool warm across moments.

    Results are unaffected: a ModelClient holds no per-call state, and the
    underlying SDK clients are thread-safe, so the replay pool can share one.
    Scoring and taxonomy still construct their own -- they are one-shot per
    run and not latency-measured.
    """
    client = _CLIENT_CACHE.get(model)
    if client is None:
        client = ModelClient(model)
        _CLIENT_CACHE[model] = client
    return client


def _reset_client_cache() -> None:
    """Clear the shared client cache (for testing)."""
    _CLIENT_CACHE.clear()


def resolve_max_tokens(client: "ModelClient", max_tokens: int) -> int:
    """Resolve ``max_tokens <= 0`` to the model's maximum output.

    Zero means "no limit we impose" -- the benchmark must not cap tutor
    output. A cap that a thinking model can exhaust before emitting any
    visible text turns that turn into an empty response, which
    run_conversation then records as "..." and the scorer grades as if the
    tutor said it. That penalises thinking models for a harness setting.

    Takes the client rather than a model string so it reuses the provider the
    client already resolved at construction, instead of re-parsing the name.
    """
    if max_tokens <= 0:
        max_tokens = MAX_OUTPUT_TOKENS.get(client.provider, 8192)
    model = getattr(client, "model", "") or ""
    for prefix, cap in _ANTHROPIC_MAX_OUTPUT_CAP.items():
        if model.startswith(prefix):
            return min(max_tokens, cap)
    return max_tokens


# Provider-specific max output token limits
MAX_OUTPUT_TOKENS = {
    "gemini": 65536,
    "openai": 128000,
    "anthropic": 128000,
    "together": 16384,  # open-weight reasoners (DeepSeek/Kimi) need room to think
}


# Per-model max output token caps. Requests above the cap are rejected by the
# API, so this clamps rather than lets the call fail. Keys match by startswith().
# Verified against the Models API (`client.models.retrieve(id).max_tokens`) on
# 2026-08-17; re-check there rather than trusting this table if a model 400s on
# max_tokens. Sonnet 4.6 was previously listed at 64000 and is now 128000, i.e.
# the provider table's default -- no entry needed.
_ANTHROPIC_MAX_OUTPUT_CAP = {
    "claude-haiku-4-5": 64000,
}


# MIME type mapping by file extension
_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _mime_from_path(path: str) -> str:
    """Resolve MIME type from file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MIME_BY_EXT:
        raise ValueError(f"unknown image extension: {ext} (path: {path})")
    return _MIME_BY_EXT[ext]


def _strip_json_fences(text: str) -> str:
    """Strip markdown JSON code fences from text.

    Handles: ```json ... ``` and ``` ... ```
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


# ===================================================================
# Usage normalization (canonical cost vector)
# ===================================================================
#
# Every LLM call records the same five-bucket vector, built from the exact
# cached/uncached split the provider reported -- never estimated. `total` is
# derived from the buckets so there is exactly one definition of "total".
# Provenance (provider/model/endpoint) rides alongside as strings; only the
# integer-valued keys are meaningful to sum. The legacy per-provider keys
# (input_tokens/output_tokens/total_tokens, cached_tokens, cache_*_input_tokens)
# stay alongside so existing readers and old transcripts keep working.


def normalize_usage(
    provider: str,
    model: str,
    endpoint: str,
    *,
    input_uncached: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    output: int = 0,
    reasoning: int = 0,
) -> dict:
    """Build the canonical usage vector + provenance for one LLM call.

    input_uncached: prompt tokens billed at the full input rate.
    cache_read: tokens served from the provider's prompt cache.
    cache_write: tokens written to cache (Anthropic; 0 elsewhere today).
    output: completion tokens, excluding reasoning where separable.
    reasoning: thinking tokens billed at the output rate (Gemini reports
        them separately; other providers fold them into output).
    endpoint: "sync" | "stream" | "batch" -- batch-tagged usage is what the
        batch discount applies to in cost computation.
    """
    return {
        "input_uncached": input_uncached,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "output": output,
        "reasoning": reasoning,
        "total": input_uncached + cache_read + cache_write + output + reasoning,
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
    }


def _gemini_usage(usage_meta, *, model: str, endpoint: str) -> dict:
    """Normalize a Gemini usage_metadata object (sync and stream paths).

    prompt_token_count includes the cached share, so the uncached part is the
    difference. thoughts_token_count is disjoint from candidates_token_count
    (their sum plus the prompt is total_token_count), which is why the legacy
    provider total disagrees with input+output on thinking models.
    """
    prompt = getattr(usage_meta, "prompt_token_count", 0) or 0
    candidates = getattr(usage_meta, "candidates_token_count", 0) or 0
    total = getattr(usage_meta, "total_token_count", 0) or 0
    cached = getattr(usage_meta, "cached_content_token_count", 0) or 0
    thoughts = getattr(usage_meta, "thoughts_token_count", 0) or 0
    return {
        "input_tokens": prompt,
        "output_tokens": candidates,
        "total_tokens": total,
        **normalize_usage(
            "gemini",
            model,
            endpoint,
            input_uncached=prompt - cached,
            cache_read=cached,
            output=candidates,
            reasoning=thoughts,
        ),
    }


def _gemini_batch_usage(usage_meta: dict, *, model: str) -> dict:
    """Normalize the camelCase usageMetadata dict from a Gemini batch result."""
    prompt = usage_meta.get("promptTokenCount", 0) or 0
    candidates = usage_meta.get("candidatesTokenCount", 0) or 0
    total = usage_meta.get("totalTokenCount", 0) or 0
    cached = usage_meta.get("cachedContentTokenCount", 0) or 0
    thoughts = usage_meta.get("thoughtsTokenCount", 0) or 0
    return {
        "input_tokens": prompt,
        "output_tokens": candidates,
        "total_tokens": total,
        **normalize_usage(
            "gemini",
            model,
            "batch",
            input_uncached=prompt - cached,
            cache_read=cached,
            output=candidates,
            reasoning=thoughts,
        ),
    }


def _openai_style_usage(usage_obj, *, provider: str, model: str, endpoint: str) -> dict:
    """Normalize an OpenAI-shaped usage object (OpenAI + Together, sync/stream).

    prompt_tokens includes the cached share, reported under
    prompt_tokens_details.cached_tokens. Together populates it too (observed
    live on DeepSeek-V4-Pro); whether Together discounts those tokens is a
    pricing-table question, not a capture question. GPT-5.6-generation models
    additionally meter cache writes (billed at 1.25x input) under
    prompt_tokens_details.cache_write_tokens -- also a subset of
    prompt_tokens, disjoint from cached_tokens (verified live on
    gpt-5.6-luna: prompt 1120 = 1117 written + 3 uncached). usage_obj may be
    None (Together streams sometimes never deliver the usage chunk): every
    bucket records 0 rather than a guess.
    """
    prompt = getattr(usage_obj, "prompt_tokens", 0) or 0
    completion = getattr(usage_obj, "completion_tokens", 0) or 0
    total = getattr(usage_obj, "total_tokens", 0) or 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    written = (
        (getattr(details, "cache_write_tokens", 0) or 0) if details is not None else 0
    )
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached,
        **normalize_usage(
            provider,
            model,
            endpoint,
            input_uncached=prompt - cached - written,
            cache_read=cached,
            cache_write=written,
            output=completion,
        ),
    }


def _openai_batch_usage(usage: dict, *, model: str) -> dict:
    """Normalize the usage dict from an OpenAI batch result line.

    Prompt caching does not apply on the OpenAI Batch API, so the whole prompt
    is uncached by construction -- cache_read is forced to 0 even if a
    cached_tokens field ever appears in a batch response body.
    """
    prompt = usage.get("prompt_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    total = usage.get("total_tokens", 0) or 0
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        **normalize_usage(
            "openai",
            model,
            "batch",
            input_uncached=prompt,
            output=completion,
        ),
    }


def _anthropic_usage(usage, *, model: str, endpoint: str) -> dict:
    """Normalize an Anthropic usage object into the shared usage dict.

    Shared by the sync, streamed, and batch paths so all report identically.
    Anthropic's input_tokens already excludes the cache buckets (the three are
    disjoint), so it maps straight to input_uncached -- the cache buckets must
    never be re-added at the base input rate.
    """
    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        **normalize_usage(
            "anthropic",
            model,
            endpoint,
            input_uncached=input_tokens,
            cache_read=cache_read,
            cache_write=cache_write,
            output=output_tokens,
        ),
    }


def _zero_usage(provider: str, model: str, endpoint: str) -> dict:
    """Zeroed usage row for failed calls -- keeps the canonical keys present."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        **normalize_usage(provider, model, endpoint),
    }


def _gemini_stream_parts(chunk):
    """Yield content parts from a Gemini stream chunk.

    Chunks late in a stream can carry usage metadata with no candidates, so
    every level is guarded rather than indexed.
    """
    candidates = getattr(chunk, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    if content is None:
        return []
    return getattr(content, "parts", None) or []


def _extract_anthropic_text(content) -> str:
    """Concatenate all text blocks from an Anthropic response/message.

    Anthropic returns a list of content blocks; non-text blocks (e.g. thinking)
    are skipped. There can be more than one text block -- when thinking is
    enabled, or when output is interleaved -- so all text blocks must be joined.
    Keeping only the first silently truncates the response (and can produce
    invalid JSON from a response split mid-string across two blocks).
    """
    return "".join(block.text for block in content if block.type == "text")


# Marker emitted by format_transcript / format_excerpt at each anchored screenshot.
# Permissive on the content between SCREEN and `image N]` so future enrichments
# (e.g. timestamp) don't break interleaving.
_SCREEN_MARKER_RE = re.compile(
    r"^[ \t]*\[SCREEN[^\]]*?image (\d+)\][ \t]*$",
    re.MULTILINE,
)


def _interleave_text_and_images(
    prompt: str,
    image_blocks: list[dict],
    text_block,
) -> list[dict]:
    """Split prompt at screenshot markers and insert image blocks at their referenced positions."""
    parts: list[dict] = []
    cursor = 0
    used: set[int] = set()

    for m in _SCREEN_MARKER_RE.finditer(prompt):
        chunk = prompt[cursor : m.end()]
        if chunk:
            parts.append(text_block(chunk))
        cursor = m.end()

        idx = int(m.group(1)) - 1  # 1-based markers, 0-based list
        if 0 <= idx < len(image_blocks):
            parts.append(image_blocks[idx])
            used.add(idx)

    if cursor < len(prompt):
        tail = prompt[cursor:]
        if tail:
            parts.append(text_block(tail))

    for i, block in enumerate(image_blocks):
        if i not in used:
            parts.append(block)

    return parts if parts else [text_block(prompt)]


# ===================================================================
# Storage backend stub (Phase 1: local-only)
# ===================================================================


class _LocalBackend:
    """Minimal local-file storage backend for Phase 1.

    NOTE(Phase 2): swap out for a real S3-backed backend when cloud
    storage is wired. Phase 2 must also update _get_backend() to detect
    the configured storage type and return the appropriate backend.
    """

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def get_presigned_url(self, path: str, expires_seconds: int = 172800) -> str:
        raise NotImplementedError(
            "get_presigned_url is only available with the S3 backend (Phase 2). "
            "Local backend has no pre-signed URL capability."
        )


_backend_instance: "_LocalBackend | None" = None


def _get_backend() -> "_LocalBackend":
    """Return the active storage backend.

    Phase 1: always returns the LocalBackend singleton.
    Phase 2: detect configured backend (S3 vs local) and return accordingly.
    """
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = _LocalBackend()
    return _backend_instance


def _should_use_presigned_url() -> bool:
    """True when the storage backend is S3 (pre-signed URLs available).

    Phase 1: always returns False (LocalBackend is always local).
    Phase 2: update once S3 backend is wired.
    """
    return not isinstance(_get_backend(), _LocalBackend)


def _base64_bytes(rel_path: str) -> str:
    """Read a file via the storage backend and return its base64-encoded bytes."""
    raw = _get_backend().read_bytes(rel_path)
    return base64.b64encode(raw).decode("ascii")


def _presigned_url(rel_path: str, expires_seconds: int = 172800) -> str:
    """Get a pre-signed URL for a file from the storage backend.

    Phase 1: raises NotImplementedError (local backend has no presigned URLs).
    Phase 2: returns a real S3 pre-signed URL once the backend is wired.
    """
    return _get_backend().get_presigned_url(rel_path, expires_seconds=expires_seconds)


def _build_image_blocks_anthropic(
    image_paths: list[str],
    use_url: bool,
    enable_cache: bool,
) -> list[dict]:
    """Build Anthropic image content blocks."""
    blocks = []
    for path in image_paths:
        media_type = _mime_from_path(path)
        if use_url:
            source = {"type": "url", "url": _presigned_url(path)}
        else:
            source = {
                "type": "base64",
                "media_type": media_type,
                "data": _base64_bytes(path),
            }
        block = {"type": "image", "source": source}
        if enable_cache:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def _build_image_blocks_openai(
    image_paths: list[str],
    use_url: bool,
) -> list[dict]:
    """Build OpenAI image content blocks."""
    blocks = []
    for path in image_paths:
        if use_url:
            url = _presigned_url(path)
        else:
            b64 = _base64_bytes(path)
            url = f"data:{_mime_from_path(path)};base64,{b64}"
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


def _build_image_blocks_gemini(image_paths: list[str]) -> list[dict]:
    """Build Gemini image content blocks.

    Gemini does not accept S3 URIs; always inline.
    """
    blocks = []
    for path in image_paths:
        blocks.append(
            {
                "inline_data": {
                    "mime_type": _mime_from_path(path),
                    "data": _base64_bytes(path),
                }
            }
        )
    return blocks


def _extract_entry(entry: dict) -> tuple[str, str, bool, int, list[str]]:
    """Extract key, prompt, json_mode, max_tokens, images from a batch entry."""
    key = entry["key"]
    parts = entry["request"]["contents"][0]["parts"]
    prompt_text = parts[0]["text"]
    gen_config = entry["request"].get("generation_config", {})
    json_mode = "application/json" in gen_config.get("response_mime_type", "")
    max_tokens = gen_config.get("max_output_tokens", 0)
    images = entry["request"].get("images", [])
    return key, prompt_text, json_mode, max_tokens, images


def build_batch_entry(
    key: str,
    prompt_text: str,
    images: list[str] | None = None,
    json_mode: bool = True,
    max_tokens: int = 65536,
    cacheable_prefix: str | None = None,
) -> dict:
    """Build a single batch entry from a key and prompt text.

    Uses a provider-neutral internal format. run_batch() and run_sync_entries()
    both consume these entries.

    cacheable_prefix: when set, the Anthropic batch path will emit the
    two-block structured content (prefix with cache_control + prompt text).
    For Gemini and OpenAI batch, the prefix is concatenated into the prompt
    text (auto-cache handles it; Gemini has no explicit batch cache API yet).
    """
    gen_config = {"max_output_tokens": max_tokens}
    if json_mode:
        gen_config["response_mime_type"] = "application/json"
    request = {
        "contents": [{"parts": [{"text": prompt_text}], "role": "user"}],
        "generation_config": gen_config,
    }
    if images:
        request["images"] = list(images)
    entry = {"key": key, "request": request}
    if cacheable_prefix is not None:
        entry["cacheable_prefix"] = cacheable_prefix
    return entry


def write_jsonl(entries: list[dict], jsonl_path: str) -> int:
    """Write a list of batch entry dicts to a JSONL file.

    Returns the number of entries written.
    """
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return len(entries)


def run_sync_entries(
    client: "ModelClient",
    entries: list[dict],
    json_mode: bool = True,
    max_tokens: int = 0,
) -> dict:
    """Run entries synchronously one at a time.

    Returns {key: {text, usage}} dict (same shape as run_batch).
    """
    raw_entries = {}
    total = len(entries)
    for i, entry in enumerate(entries):
        key, prompt_text, entry_json_mode, entry_max_tokens, images = _extract_entry(
            entry
        )
        if not entry_max_tokens:
            entry_max_tokens = max_tokens

        logger.debug("[%d/%d] %s...", i + 1, total, key[:60])
        try:
            response = client.generate(
                prompt_text,
                images=images or None,
                json_mode=entry_json_mode if json_mode else False,
                max_tokens=entry_max_tokens,
            )
            raw_entries[key] = {
                "text": response.text,
                "usage": response.usage,
            }
        except Exception as e:
            logger.error("ERROR on %s: %s", key, e)
            raw_entries[key] = {
                "text": "",
                "error": str(e),
                "usage": _zero_usage(client.provider, client.model, "sync"),
            }
    return raw_entries


# ===================================================================
# Unified batch API
# ===================================================================


def run_batch(
    client: "ModelClient",
    entries: list[dict],
    json_mode: bool = True,
    display_name: str = "batch",
    poll_interval: int = 60,
    thinking: bool = False,
    thinking_budget: int = 0,
    reasoning_effort: str = "",
    effort: str = "",
    enable_cache: bool = False,
    existing_batch_id: str | None = None,
    on_batch_created=None,
) -> dict:
    """Run entries as a batch job via the provider's batch API.

    Resume support: if `existing_batch_id` is set, skip submission and resume
    polling on that batch (the entries list still drives result parsing, so
    the caller must pass the same entries that were submitted with the batch).
    `on_batch_created` is called with the provider's batch id immediately after
    a fresh submission succeeds, before the poll loop starts -- callers use
    this to persist the id for ctrl-C recovery.
    """
    provider = client.provider
    if existing_batch_id:
        logger.info(
            "Resuming in-flight %s batch %s (%d entries)",
            provider,
            existing_batch_id,
            len(entries),
        )
    else:
        logger.info(
            "Running batch (%s): %d entries, display_name=%s",
            provider,
            len(entries),
            display_name,
        )

    if provider == "gemini":
        return _run_batch_gemini(
            client,
            entries,
            json_mode,
            display_name,
            poll_interval,
            thinking,
            thinking_budget,
            existing_batch_id=existing_batch_id,
            on_batch_created=on_batch_created,
        )
    elif provider == "openai":
        return _run_batch_openai(
            client,
            entries,
            json_mode,
            display_name,
            poll_interval,
            thinking,
            thinking_budget,
            reasoning_effort,
            existing_batch_id=existing_batch_id,
            on_batch_created=on_batch_created,
        )
    elif provider == "anthropic":
        return _run_batch_anthropic(
            client,
            entries,
            json_mode,
            display_name,
            poll_interval,
            thinking,
            thinking_budget,
            reasoning_effort,
            effort=effort,
            enable_cache=enable_cache,
            existing_batch_id=existing_batch_id,
            on_batch_created=on_batch_created,
        )
    else:
        raise ValueError(f"Batch API not supported for provider: {provider}")


# ===================================================================
# Gemini Batch API
# ===================================================================


def _run_batch_gemini(
    client,
    entries,
    json_mode,
    display_name,
    poll_interval,
    thinking=False,
    thinking_budget=0,
    existing_batch_id=None,
    on_batch_created=None,
):
    """Gemini batch: upload JSONL, submit, poll, download.

    If existing_batch_id is set, skip upload+submit and retrieve that job.
    """
    import tempfile

    from tutormoments.config import get_batch_timeout

    gemini_client = client._client
    jsonl_path = None

    if existing_batch_id:
        batch_job = gemini_client.batches.get(name=existing_batch_id)
    else:
        # Write Gemini-format JSONL to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for entry in entries:
                key, prompt_text, entry_json_mode, entry_max_tokens, images = (
                    _extract_entry(entry)
                )
                # Gemini has no explicit batch cache API; concatenate prefix (auto-cache).
                cacheable_prefix = entry.get("cacheable_prefix")
                effective_prompt = (cacheable_prefix or "") + prompt_text
                if images:
                    image_blocks = _build_image_blocks_gemini(images)
                    parts = _interleave_text_and_images(
                        effective_prompt,
                        image_blocks,
                        lambda s: {"text": s},
                    )
                else:
                    parts = [{"text": effective_prompt}]
                gem_entry = {
                    "key": key,
                    "request": {
                        "contents": [{"parts": parts, "role": "user"}],
                        "generation_config": entry["request"].get(
                            "generation_config", {}
                        ),
                    },
                }
                f.write(json.dumps(gem_entry, ensure_ascii=False) + "\n")
            jsonl_path = f.name

        logger.info("Uploading batch request file...")
        uploaded_file = gemini_client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )
        logger.info("Uploaded file: %s", uploaded_file.name)

        logger.info("Submitting batch job...")
        batch_job = gemini_client.batches.create(
            model=f"models/{client.model}",
            src=uploaded_file.name,
            config={"display_name": display_name},
        )
        logger.info("Batch job created: %s", batch_job.name)
        if on_batch_created:
            on_batch_created(batch_job.name)

    try:
        poll_start = time.monotonic()
        batch_timeout = get_batch_timeout()
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
        while batch_job.state.name not in completed_states:
            if time.monotonic() - poll_start > batch_timeout:
                raise RuntimeError(
                    f"Gemini batch timed out after {batch_timeout}s "
                    f"(state: {batch_job.state.name})"
                )
            logger.info(
                "Batch in progress: %s (%dm elapsed, next poll in %ds)",
                batch_job.state.name,
                int(time.monotonic() - poll_start) // 60,
                poll_interval,
            )
            time.sleep(poll_interval)
            batch_job = gemini_client.batches.get(name=batch_job.name)

        logger.info("Batch job finished: %s", batch_job.state.name)
        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Gemini batch failed: {batch_job.state.name}")

        if not batch_job.dest or not batch_job.dest.file_name:
            raise RuntimeError("No output file in batch job result")

        logger.info("Downloading results from: %s", batch_job.dest.file_name)
        result_bytes = gemini_client.files.download(file=batch_job.dest.file_name)
        result_text = result_bytes.decode("utf-8")

        raw_entries = {}
        for line in result_text.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            key = result.get("key")
            response = result.get("response")

            if response:
                candidates = response.get("candidates", [])
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                usage_meta = response.get("usageMetadata", {})
                raw_entries[key] = {
                    "text": text,
                    "usage": _gemini_batch_usage(usage_meta, model=client.model),
                }
            else:
                error = result.get("error")
                raw_entries[key] = {
                    "text": "",
                    "error": str(error) if error else "No response",
                    "usage": _zero_usage("gemini", client.model, "batch"),
                }
        return raw_entries
    finally:
        if jsonl_path:
            os.unlink(jsonl_path)


# ===================================================================
# OpenAI Batch API
# ===================================================================


def _run_batch_openai(
    client,
    entries,
    json_mode,
    display_name,
    poll_interval,
    thinking=False,
    thinking_budget=0,
    reasoning_effort="",
    existing_batch_id=None,
    on_batch_created=None,
):
    """OpenAI batch: upload JSONL, create batch, poll, download results.

    If existing_batch_id is set, skip upload+create and retrieve that batch.
    """
    import tempfile

    from tutormoments.config import get_batch_timeout

    openai_client = client._client
    max_tokens = MAX_OUTPUT_TOKENS["openai"]
    jsonl_path = None

    if existing_batch_id:
        batch_job = openai_client.batches.retrieve(existing_batch_id)
    else:
        # Write OpenAI-format JSONL
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for entry in entries:
                key, prompt_text, entry_json_mode, entry_max_tokens, images = (
                    _extract_entry(entry)
                )
                if not entry_max_tokens or entry_max_tokens > max_tokens:
                    entry_max_tokens = max_tokens

                # OpenAI uses automatic prefix caching; concatenate the prefix so the
                # same static head is always at the front of the message.
                cacheable_prefix = entry.get("cacheable_prefix")
                effective_prompt = (cacheable_prefix or "") + prompt_text

                if images:
                    image_blocks = _build_image_blocks_openai(
                        images,
                        use_url=_should_use_presigned_url(),
                    )
                    content = _interleave_text_and_images(
                        effective_prompt,
                        image_blocks,
                        lambda s: {"type": "text", "text": s},
                    )
                else:
                    content = effective_prompt

                body = {
                    "model": client.model,
                    "messages": [{"role": "user", "content": content}],
                    "max_completion_tokens": entry_max_tokens,
                }
                if json_mode and entry_json_mode:
                    body["response_format"] = {"type": "json_object"}
                if reasoning_effort:
                    body["reasoning_effort"] = reasoning_effort
                line = {
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            jsonl_path = f.name

        logger.info("Uploading batch request file...")
        with open(jsonl_path, "rb") as f:
            uploaded_file = openai_client.files.create(file=f, purpose="batch")
        logger.info("Uploaded file: %s", uploaded_file.id)

        logger.info("Submitting batch job...")
        batch_job = openai_client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": display_name},
        )
        logger.info("Batch job created: %s", batch_job.id)
        if on_batch_created:
            on_batch_created(batch_job.id)

    try:
        poll_start = time.monotonic()
        batch_timeout = get_batch_timeout()
        terminal_states = {"completed", "failed", "expired", "cancelled"}
        while batch_job.status not in terminal_states:
            if time.monotonic() - poll_start > batch_timeout:
                raise RuntimeError(
                    f"OpenAI batch timed out after {batch_timeout}s "
                    f"(state: {batch_job.status})"
                )
            logger.info(
                "Batch in progress: %s (%dm elapsed, next poll in %ds)",
                batch_job.status,
                int(time.monotonic() - poll_start) // 60,
                poll_interval,
            )
            time.sleep(poll_interval)
            batch_job = openai_client.batches.retrieve(batch_job.id)

        logger.info("Batch job finished: %s", batch_job.status)
        if batch_job.status != "completed":
            raise RuntimeError(f"OpenAI batch failed: {batch_job.status}")

        if not batch_job.output_file_id:
            raise RuntimeError("No output file in batch job result")

        logger.info("Downloading results from: %s", batch_job.output_file_id)
        result_bytes = openai_client.files.content(batch_job.output_file_id).content
        result_text = result_bytes.decode("utf-8")

        raw_entries = {}
        for line in result_text.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            key = result.get("custom_id")
            error = result.get("error")

            if error:
                raw_entries[key] = {
                    "text": "",
                    "error": str(error),
                    "usage": _zero_usage("openai", client.model, "batch"),
                }
                continue

            response_body = result.get("response", {}).get("body", {})
            choices = response_body.get("choices", [])
            text = ""
            if choices:
                text = choices[0].get("message", {}).get("content", "")

            usage = response_body.get("usage", {})
            raw_entries[key] = {
                "text": text,
                "usage": _openai_batch_usage(usage, model=client.model),
            }
        return raw_entries
    finally:
        if jsonl_path:
            os.unlink(jsonl_path)


# ===================================================================
# Anthropic Batch API
# ===================================================================


def _run_batch_anthropic(
    client,
    entries,
    json_mode,
    display_name,
    poll_interval,
    thinking=False,
    thinking_budget=0,
    reasoning_effort="",
    effort="",
    enable_cache=False,
    existing_batch_id=None,
    on_batch_created=None,
):
    """Anthropic batch: create message batch, poll, stream results.

    If existing_batch_id is set, skip submission and retrieve that batch.
    The id_to_key mapping is rebuilt deterministically from `entries` order
    (so callers must pass the same entries that were originally submitted).
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from tutormoments.config import get_batch_timeout, get_retry_config

    anthropic_client = client._client
    max_tokens = MAX_OUTPUT_TOKENS["anthropic"]
    # Apply per-model output cap (e.g. Haiku 4.5 / Sonnet 4.6 cap at 64K).
    for prefix, cap in _ANTHROPIC_MAX_OUTPUT_CAP.items():
        if client.model and client.model.startswith(prefix):
            max_tokens = min(max_tokens, cap)
            break

    # id_to_key mapping is deterministic in entries order, so it can be rebuilt
    # on resume without re-submitting. Anthropic custom_id has a 64-char limit,
    # so we use short indexed IDs.
    id_to_key = {f"r{i}": _extract_entry(e)[0] for i, e in enumerate(entries)}

    if existing_batch_id:
        message_batch = anthropic_client.messages.batches.retrieve(existing_batch_id)
    else:
        thinking_param = (
            _anthropic_thinking_param(client.model, thinking_budget)
            if thinking
            else None
        )
        thinking_min = 0
        if thinking_param is not None and thinking_param.get("type") == "enabled":
            # Legacy enabled mode needs max_tokens >= budget + headroom.
            thinking_min = thinking_param["budget_tokens"] + 64

        requests = []
        for i, entry in enumerate(entries):
            key, prompt_text, entry_json_mode, entry_max_tokens, images = (
                _extract_entry(entry)
            )
            if not entry_max_tokens or entry_max_tokens > max_tokens:
                entry_max_tokens = max_tokens
            if thinking_min and entry_max_tokens < thinking_min:
                entry_max_tokens = thinking_min

            cacheable_prefix = entry.get("cacheable_prefix")

            if images:
                image_blocks = _build_image_blocks_anthropic(
                    images,
                    use_url=_should_use_presigned_url(),
                    enable_cache=enable_cache,
                )
                content = _interleave_text_and_images(
                    prompt_text,
                    image_blocks,
                    lambda s: {"type": "text", "text": s},
                )
                if cacheable_prefix is not None:
                    content = [
                        {
                            "type": "text",
                            "text": cacheable_prefix,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ] + content
            elif cacheable_prefix is not None:
                content = [
                    {
                        "type": "text",
                        "text": cacheable_prefix,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": prompt_text},
                ]
            else:
                content = prompt_text

            params = {
                "model": client.model,
                "max_tokens": entry_max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
            if json_mode and entry_json_mode:
                params["system"] = (
                    "You must respond with valid JSON only. "
                    "Do not include markdown code fences, explanations, or any text "
                    "outside the JSON object."
                )
            if thinking_param is not None:
                params["thinking"] = thinking_param
            # effort goes inside output_config, mirroring the sync path
            # (_generate_anthropic). Haiku 4.5 rejects effort -- skip there.
            if (
                effort
                and client.model
                and not client.model.startswith("claude-haiku-4-5")
            ):
                params["output_config"] = {"effort": effort}

            requests.append(
                Request(
                    custom_id=f"r{i}",
                    params=MessageCreateParamsNonStreaming(**params),
                )
            )

        logger.info("Submitting batch (%d requests)...", len(requests))
        retry_cfg = get_retry_config()
        max_retries = retry_cfg.get("max_retries", 5)
        base_delay = retry_cfg.get("base_delay", 5)
        for attempt in range(max_retries):
            try:
                message_batch = anthropic_client.messages.batches.create(
                    requests=requests
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Batch submit error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    raise
        logger.info("Batch created: %s", message_batch.id)
        if on_batch_created:
            on_batch_created(message_batch.id)

    poll_start = time.monotonic()
    batch_timeout = get_batch_timeout()
    while message_batch.processing_status != "ended":
        if time.monotonic() - poll_start > batch_timeout:
            raise RuntimeError(
                f"Anthropic batch timed out after {batch_timeout}s "
                f"(state: {message_batch.processing_status})"
            )
        logger.info(
            "Batch in progress: %s, %s (%dm elapsed, next poll in %ds)",
            message_batch.processing_status,
            message_batch.request_counts,
            int(time.monotonic() - poll_start) // 60,
            poll_interval,
        )
        time.sleep(poll_interval)
        message_batch = anthropic_client.messages.batches.retrieve(message_batch.id)

    logger.info("Batch finished: %s", message_batch.processing_status)
    logger.info("  Counts: %s", message_batch.request_counts)

    # Parse results -- map short IDs back to original keys
    raw_entries = {}
    for result in anthropic_client.messages.batches.results(message_batch.id):
        key = id_to_key.get(result.custom_id, result.custom_id)

        if result.result.type == "succeeded":
            message = result.result.message
            # Skip thinking blocks, extract (and concatenate) text blocks
            text = _extract_anthropic_text(message.content)
            if json_mode:
                text = _strip_json_fences(text)
            usage = _anthropic_usage(
                message.usage, model=client.model, endpoint="batch"
            )
            raw_entries[key] = {"text": text, "usage": usage}
        else:
            error_msg = f"{result.result.type}"
            if hasattr(result.result, "error") and result.result.error:
                error_msg = f"{result.result.type}: {result.result.error}"
            raw_entries[key] = {
                "text": "",
                "error": error_msg,
                "usage": _zero_usage("anthropic", client.model, "batch"),
            }

    return raw_entries
