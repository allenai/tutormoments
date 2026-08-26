"""Offline tests for the live smoke layer (tutormoments smoke).

The smoke command's own logic must be testable without any network: the
client factory and batch submit/cancel callables are injectable, so canned
responses drive every verdict path.
"""

import json
from types import SimpleNamespace

import pytest

from tutormoments.smoke import (
    FAIL,
    PASS,
    WARN,
    BatchCheck,
    SmokePlan,
    SyncCheck,
    build_smoke_plan,
    format_smoke_report,
    run_smoke,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(provider="anthropic", total=100, reasoning=0, **extra):
    usage = {
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": total,
        "total": total,
        "reasoning": reasoning,
        "provider": provider,
    }
    usage.update(extra)
    return usage


class _FakeClient:
    def __init__(self, model, response=None, error=None):
        self.model = model
        from tutormoments.models import infer_provider

        self.provider = infer_provider(model)
        self._response = response
        self._error = error
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self._error is not None:
            raise self._error
        return self._response


def _factory(clients):
    def get(model):
        return clients[model]

    return get


def _sync_check(model="claude-opus-4-8", thinking="xhigh", label="arm:test"):
    from tutormoments.models import ThinkingLevel, infer_provider

    return SyncCheck(
        label=label,
        model=model,
        provider=infer_provider(model),
        thinking=ThinkingLevel.coerce(thinking),
        json_mode=False,
    )


def _plan(sync=(), batch=(), skipped=()):
    return SmokePlan(
        sync_checks=list(sync), batch_checks=list(batch), skipped=list(skipped)
    )


# ---------------------------------------------------------------------------
# Plan building (pure, from a config file)
# ---------------------------------------------------------------------------


@pytest.fixture()
def smoke_config(tmp_path, monkeypatch):
    from tutormoments.config import _reset_config_cache

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
providers:
  anthropic: { env: ANTHROPIC_API_KEY }
  gemini:    { env: GEMINI_API_KEY }
models:
  sonnet-high: { model: claude-sonnet-4-6, thinking: high }
  gemini-dyn:  { model: gemini-2.5-pro, thinking: dynamic }
student: { model: claude-opus-4-6, mode: oracle, thinking: none }
scorer:  { model: claude-opus-4-6, thinking: dynamic }
defaults: { trials: 1, max_turns: 5 }
retry:    { max_retries: 5, base_delay: 5 }
batch:    { timeout: 86400 }
""",
        encoding="utf-8",
    )
    _reset_config_cache()
    yield str(cfg)
    _reset_config_cache()


def test_build_plan_enumerates_arms_and_roles(smoke_config):
    plan = build_smoke_plan(config_path=smoke_config)
    labels = [c.label for c in plan.sync_checks]
    assert labels == ["arm:sonnet-high", "arm:gemini-dyn", "student", "scorer"]
    arm = plan.sync_checks[0]
    assert arm.model == "claude-sonnet-4-6"
    assert arm.provider == "anthropic"
    assert arm.thinking == "high"
    # One batch representative per batch-capable provider; the scorer's model
    # is preferred for its provider.
    by_provider = {b.provider: b for b in plan.batch_checks}
    assert set(by_provider) == {"anthropic", "gemini"}
    assert by_provider["anthropic"].model == "claude-opus-4-6"


def test_build_plan_filters(smoke_config):
    plan = build_smoke_plan(
        config_path=smoke_config, arms=["gemini-dyn"], roles=["tutor"]
    )
    assert [c.label for c in plan.sync_checks] == ["arm:gemini-dyn"]

    plan = build_smoke_plan(config_path=smoke_config, providers=["gemini"])
    assert {c.provider for c in plan.sync_checks} == {"gemini"}

    plan = build_smoke_plan(config_path=smoke_config, include_batch=False)
    assert plan.batch_checks == []


def test_build_plan_unknown_arm_raises(smoke_config):
    with pytest.raises(ValueError, match="not in roster"):
        build_smoke_plan(config_path=smoke_config, arms=["typo-arm"])


# ---------------------------------------------------------------------------
# Sync check verdicts
# ---------------------------------------------------------------------------


def test_sync_pass_with_thinking_evidence():
    check = _sync_check("claude-haiku-4-5", "high")  # legacy enabled: required
    client = _FakeClient(
        "claude-haiku-4-5",
        response=SimpleNamespace(
            text="answer", usage=_usage(reasoning=0, thinking_blocks=2)
        ),
    )
    report = run_smoke(
        _plan(sync=[check]), client_factory=_factory({"claude-haiku-4-5": client})
    )
    (row,) = report.results
    assert row.status == PASS
    assert row.thinking_evidence == "2"
    # The check must send the arm's real condition.
    assert client.calls[0]["thinking"] == "high"


def test_sync_fail_on_empty_text():
    check = _sync_check()
    client = _FakeClient(
        "claude-opus-4-8", response=SimpleNamespace(text="  ", usage=_usage())
    )
    report = run_smoke(
        _plan(sync=[check]), client_factory=_factory({"claude-opus-4-8": client})
    )
    assert report.results[0].status == FAIL
    assert "empty response" in report.results[0].detail


def test_sync_fail_on_missing_usage():
    check = _sync_check()
    client = _FakeClient(
        "claude-opus-4-8",
        response=SimpleNamespace(text="ok", usage={"total_tokens": 0, "total": 0}),
    )
    report = run_smoke(
        _plan(sync=[check]), client_factory=_factory({"claude-opus-4-8": client})
    )
    assert report.results[0].status == FAIL
    assert "usage" in report.results[0].detail


def test_required_thinking_without_evidence_fails():
    # Gemini fixed budget: thinking is required; zero reasoning tokens = FAIL.
    check = _sync_check("gemini-2.5-pro", "high")
    client = _FakeClient(
        "gemini-2.5-pro",
        response=SimpleNamespace(
            text="answer", usage=_usage(provider="gemini", reasoning=0)
        ),
    )
    report = run_smoke(
        _plan(sync=[check]), client_factory=_factory({"gemini-2.5-pro": client})
    )
    assert report.results[0].status == FAIL
    assert "no thinking observed" in report.results[0].detail


def test_thinking_off_with_evidence_fails():
    check = _sync_check("gemini-2.5-flash", "none")
    client = _FakeClient(
        "gemini-2.5-flash",
        response=SimpleNamespace(
            text="answer", usage=_usage(provider="gemini", reasoning=57)
        ),
    )
    report = run_smoke(
        _plan(sync=[check]), client_factory=_factory({"gemini-2.5-flash": client})
    )
    assert report.results[0].status == FAIL
    assert "thinking-off" in report.results[0].detail


def test_dynamic_without_evidence_warns_then_fails_strict():
    check = _sync_check("gemini-2.5-pro", "dynamic")
    response = SimpleNamespace(
        text="answer", usage=_usage(provider="gemini", reasoning=0)
    )
    client = _FakeClient("gemini-2.5-pro", response=response)
    factory = _factory({"gemini-2.5-pro": client})

    report = run_smoke(_plan(sync=[check]), client_factory=factory)
    assert report.results[0].status == WARN

    report = run_smoke(
        _plan(sync=[check]), client_factory=factory, strict_thinking=True
    )
    assert report.results[0].status == FAIL


def test_openai_reasoning_tokens_are_evidence():
    # OpenAI reports thinking as usage.completion_tokens_details.reasoning_tokens,
    # surfaced by the client as the informational reasoning_tokens key (the
    # canonical `reasoning` bucket stays 0 there -- it is a subset of output,
    # not a separate cost). The smoke must read it as evidence.
    check = _sync_check("gpt-5.5", "high")
    client = _FakeClient(
        "gpt-5.5",
        response=SimpleNamespace(
            text="answer",
            usage=_usage(provider="openai", reasoning=0, reasoning_tokens=1857),
        ),
    )
    report = run_smoke(_plan(sync=[check]), client_factory=_factory({"gpt-5.5": client}))
    assert report.results[0].status == PASS
    assert report.results[0].thinking_evidence == "1857"

    client = _FakeClient(
        "gpt-5.5",
        response=SimpleNamespace(
            text="answer",
            usage=_usage(provider="openai", reasoning=0, reasoning_tokens=0),
        ),
    )
    report = run_smoke(_plan(sync=[check]), client_factory=_factory({"gpt-5.5": client}))
    assert report.results[0].status == FAIL


def test_together_evidence_not_asserted():
    check = _sync_check("deepseek-ai/DeepSeek-V4-Pro", "dynamic")
    client = _FakeClient(
        "deepseek-ai/DeepSeek-V4-Pro",
        response=SimpleNamespace(
            text="answer", usage=_usage(provider="together", reasoning=0)
        ),
    )
    report = run_smoke(
        _plan(sync=[check]),
        client_factory=_factory({"deepseek-ai/DeepSeek-V4-Pro": client}),
    )
    assert report.results[0].status == PASS
    assert report.results[0].thinking_evidence == "n/a"


def test_one_check_exception_does_not_abort_others():
    bad = _sync_check("claude-opus-4-8", "xhigh", label="arm:bad")
    good = _sync_check("claude-sonnet-4-6", "high", label="arm:good")
    clients = {
        "claude-opus-4-8": _FakeClient("claude-opus-4-8", error=RuntimeError("boom")),
        "claude-sonnet-4-6": _FakeClient(
            "claude-sonnet-4-6",
            response=SimpleNamespace(
                text="ok", usage=_usage(reasoning=0, thinking_blocks=1)
            ),
        ),
    }
    report = run_smoke(_plan(sync=[bad, good]), client_factory=_factory(clients))
    by_label = {r.label: r for r in report.results}
    assert by_label["arm:bad"].status == FAIL
    assert "boom" in by_label["arm:bad"].detail
    assert by_label["arm:good"].status == PASS
    assert report.failed


# ---------------------------------------------------------------------------
# Batch checks
# ---------------------------------------------------------------------------


def _batch_check(model="claude-opus-4-6", thinking="dynamic"):
    from tutormoments.models import ThinkingLevel, infer_provider

    return BatchCheck(
        provider=infer_provider(model),
        model=model,
        thinking=ThinkingLevel.coerce(thinking),
    )


def test_batch_submit_and_cancel_pass():
    submitted = {}
    cancelled = {}

    def submit(client, entries, json_mode, display_name, thinking):
        submitted["entries"] = entries
        submitted["thinking"] = thinking
        return "batch-123"

    def cancel(client, batch_id):
        cancelled["id"] = batch_id

    client = _FakeClient("claude-opus-4-6")
    report = run_smoke(
        _plan(batch=[_batch_check()]),
        client_factory=_factory({"claude-opus-4-6": client}),
        submit_batch_fn=submit,
        cancel_batch_fn=cancel,
    )
    (row,) = report.results
    assert row.status == PASS
    assert row.batch_id == "batch-123"
    assert cancelled["id"] == "batch-123"
    assert len(submitted["entries"]) == 2
    assert submitted["thinking"] == "dynamic"


def test_batch_cancel_failure_warns_with_id():
    def submit(client, entries, json_mode, display_name, thinking):
        return "batch-456"

    def cancel(client, batch_id):
        raise RuntimeError("cancel unsupported")

    client = _FakeClient("claude-opus-4-6")
    report = run_smoke(
        _plan(batch=[_batch_check()]),
        client_factory=_factory({"claude-opus-4-6": client}),
        submit_batch_fn=submit,
        cancel_batch_fn=cancel,
    )
    (row,) = report.results
    assert row.status == WARN
    assert "batch-456" == row.batch_id
    assert "24h" in row.detail
    assert not report.failed  # WARN alone is not a failure


def test_batch_submit_failure_fails():
    def submit(client, entries, json_mode, display_name, thinking):
        raise RuntimeError("400 bad thinking_config")

    report = run_smoke(
        _plan(batch=[_batch_check()]),
        client_factory=_factory({"claude-opus-4-6": _FakeClient("claude-opus-4-6")}),
        submit_batch_fn=submit,
        cancel_batch_fn=lambda c, b: None,
    )
    assert report.results[0].status == FAIL
    assert "bad thinking_config" in report.results[0].detail


# ---------------------------------------------------------------------------
# Rendering + JSON report
# ---------------------------------------------------------------------------


def test_report_renders_ascii_and_json():
    check = _sync_check("claude-opus-4-8", "xhigh")
    client = _FakeClient(
        "claude-opus-4-8",
        response=SimpleNamespace(
            text="ok", usage=_usage(reasoning=0, thinking_blocks=1)
        ),
    )
    report = run_smoke(
        _plan(sync=[check], skipped=[("batch:together", "provider has no batch API")]),
        client_factory=_factory({"claude-opus-4-8": client}),
        config_source="tutormoments:default_config.yaml",
    )
    text = format_smoke_report(report)
    assert text.isascii()
    assert "arm:test" in text
    assert "effort=xhigh" in text
    assert "SKIP" in text
    assert "1 passed" in text

    payload = json.loads(report.to_json())
    assert payload["config_source"] == "tutormoments:default_config.yaml"
    assert payload["results"][0]["status"] == PASS


def test_cli_smoke_exit_codes(monkeypatch, capsys):
    import tutormoments.cli as cli
    from tutormoments.smoke import CheckResult, SmokeReport

    def fake_run_smoke(plan, **kwargs):
        report = SmokeReport()
        report.results.append(
            CheckResult(
                label="arm:x", model="m", provider="anthropic", wire="", status=FAIL
            )
        )
        return report

    monkeypatch.setattr("tutormoments.smoke.run_smoke", fake_run_smoke)
    monkeypatch.setattr("tutormoments.smoke.build_smoke_plan", lambda **kw: _plan())
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["smoke"])
    assert exc_info.value.code == 1

    monkeypatch.setattr(
        "tutormoments.smoke.build_smoke_plan",
        lambda **kw: (_ for _ in ()).throw(ValueError("Arm 'x' not in roster")),
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["smoke", "--arms", "x"])
    assert exc_info.value.code == 2
