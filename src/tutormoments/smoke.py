"""Live smoke checks: verify the active config's real wire formats.

The offline test suite mocks every provider SDK, so it can prove the code
agrees with itself but never that a provider accepts what we send. This
module is the live half: one tiny real call per configured arm/role (and one
submit-then-cancel batch per provider) with the arm's EXACT configured
thinking condition. Run it via `tutormoments smoke` before merging changes to
core API logic (see AGENTS.md).

It deliberately calls only models already present in the active config --
choosing models is the benchmark owner's job, never this tool's.

Three layers, so the logic itself stays testable offline:
  build_smoke_plan()  -- pure: config -> checks
  run_smoke()         -- executes checks via an injectable client factory
  format_smoke_report() / SmokeReport.to_json()
"""

import datetime
import json
import logging
from dataclasses import dataclass, field

from tutormoments.models import ThinkingLevel, resolve_thinking
from tutormoments.resources import resource_text

logger = logging.getLogger(__name__)

_BATCH_PROVIDERS = ("gemini", "openai", "anthropic")

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass(frozen=True)
class SyncCheck:
    label: str  # e.g. "arm:claude-opus-4-8" or "student"
    model: str
    provider: str
    thinking: "ThinkingLevel | None"
    json_mode: bool


@dataclass(frozen=True)
class BatchCheck:
    provider: str
    model: str
    thinking: "ThinkingLevel | None"


@dataclass
class SmokePlan:
    sync_checks: list
    batch_checks: list
    skipped: list  # (label, reason) pairs surfaced in the report


@dataclass
class CheckResult:
    label: str
    model: str
    provider: str
    wire: str
    status: str
    detail: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_evidence: str = ""
    batch_id: str = ""


@dataclass
class SmokeReport:
    results: list = field(default_factory=list)
    started_at: str = ""
    config_source: str = ""

    @property
    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)

    def to_json(self) -> str:
        return json.dumps(
            {
                "started_at": self.started_at,
                "config_source": self.config_source,
                "results": [vars(r) for r in self.results],
            },
            indent=2,
            default=str,
        )


def build_smoke_plan(
    config_path=None,
    arms: "list[str] | None" = None,
    roles: "list[str] | None" = None,
    providers: "list[str] | None" = None,
    include_sync: bool = True,
    include_batch: bool = True,
) -> SmokePlan:
    """Enumerate the checks for the active config (pure; no network).

    arms/roles/providers narrow the selection. Roles are from
    {tutor, student, scorer, taxonomy, groundtruth}. An unknown arm name
    raises ValueError (exit code 2 territory -- a typo must not smoke-pass).
    """
    from tutormoments.config import (
        get_groundtruth_phase_config,
        get_registered_student,
        load_config,
        resolve_arm,
        scorer_spec,
        student_spec,
        taxonomy_spec,
    )

    cfg = load_config(config_path)
    roles = (
        list(roles)
        if roles
        else ["tutor", "student", "scorer", "taxonomy", "groundtruth"]
    )
    sync_checks: list[SyncCheck] = []
    skipped: list[tuple[str, str]] = []

    def _selected(provider: str) -> bool:
        return providers is None or provider in providers

    if "tutor" in roles:
        arm_names = arms if arms is not None else list(cfg.get("models") or {})
        for name in arm_names:
            arm = resolve_arm(name, config_path)  # raises on unknown arm
            if _selected(arm.provider):
                sync_checks.append(
                    SyncCheck(
                        label=f"arm:{name}",
                        model=arm.model,
                        provider=arm.provider,
                        thinking=arm.thinking,
                        json_mode=False,  # tutor turns run json_mode=False
                    )
                )
    elif arms:
        raise ValueError("--arms requires the tutor role in --roles")

    def _role_check(role: str, model: str, thinking, json_mode: bool):
        from tutormoments.models import infer_provider

        provider = infer_provider(model)
        if _selected(provider):
            sync_checks.append(
                SyncCheck(
                    label=role,
                    model=model,
                    provider=provider,
                    thinking=thinking,
                    json_mode=json_mode,
                )
            )

    if "student" in roles and cfg.get("student"):
        spec = student_spec(config_path)
        if get_registered_student(spec.model) is not None:
            skipped.append(("student", f"registered student '{spec.model}' (no API)"))
        else:
            _role_check("student", spec.model, spec.thinking, json_mode=False)
    if "scorer" in roles and cfg.get("scorer"):
        spec = scorer_spec(config_path)
        _role_check("scorer", spec.model, spec.thinking, json_mode=True)
    if "taxonomy" in roles and cfg.get("taxonomy"):
        spec = taxonomy_spec(config_path)
        _role_check("taxonomy", spec.model, spec.thinking, json_mode=True)
    if "groundtruth" in roles and cfg.get("groundtruth"):
        spec = get_groundtruth_phase_config(config_path)
        _role_check("groundtruth", spec.model, spec.thinking, json_mode=True)

    batch_checks: list[BatchCheck] = []
    if include_batch:
        # One representative per provider present in the sync selection: the
        # real batch consumers are the scorer/groundtruth paths, so prefer
        # those models for their providers; else the provider's first check.
        preferred: dict[str, SyncCheck] = {}
        for check in sync_checks:
            if (
                check.label in ("scorer", "groundtruth")
                or check.provider not in preferred
            ):
                if check.provider not in preferred or check.label in (
                    "scorer",
                    "groundtruth",
                ):
                    preferred[check.provider] = check
        for provider, check in preferred.items():
            if provider in _BATCH_PROVIDERS:
                batch_checks.append(
                    BatchCheck(
                        provider=provider, model=check.model, thinking=check.thinking
                    )
                )
            else:
                skipped.append((f"batch:{provider}", "provider has no batch API"))

    return SmokePlan(
        sync_checks=sync_checks if include_sync else [],
        batch_checks=batch_checks,
        skipped=skipped,
    )


def _thinking_expectation(wire, level) -> str:
    """What the wire fragment promises: required / optional / off / na."""
    if level is None:
        return "na"
    if wire.gemini_thinking_config is not None:
        cfgd = wire.gemini_thinking_config
        budget = cfgd.get("thinking_budget")
        if budget == 0:
            return "off"
        if budget == -1:
            return "optional"
        return "required"  # positive budget or thinking_level string
    if wire.anthropic_thinking is not None:
        kind = wire.anthropic_thinking.get("type")
        if kind == "disabled":
            return "off"
        if kind == "enabled":
            return "required"
        return "optional"  # adaptive: the model decides
    if wire.openai_reasoning_effort is not None:
        return "off" if wire.openai_reasoning_effort == "none" else "required"
    if level == ThinkingLevel.NONE:
        return "off"  # omit-style off (anthropic 4.x tiers)
    return "na"  # noop mechanism (together internal reasoners)


def _thinking_evidence(usage: dict) -> "int | None":
    """Wire-level evidence that thinking happened, or None when unobservable."""
    reasoning = usage.get("reasoning", 0) or 0
    if reasoning:
        return int(reasoning)
    # Anthropic: thinking blocks; OpenAI: the informational reasoning_tokens
    # subset of completion_tokens. Either key's presence means the provider
    # reported the dimension, so zero IS the observation.
    for key in ("thinking_blocks", "reasoning_tokens"):
        if key in usage:
            return int(usage.get(key) or 0)
    # Together never reports a reasoning dimension at all.
    if usage.get("provider") == "together":
        return None
    return int(reasoning)


def _judge_thinking(expectation: str, evidence, strict: bool) -> tuple[str, str]:
    """(status, note) from the expectation vs the observed evidence."""
    if expectation == "na" or evidence is None:
        return PASS, ""
    if expectation == "required" and evidence == 0:
        return FAIL, "thinking knob sent but no thinking observed"
    if expectation == "off" and evidence > 0:
        return FAIL, "thinking observed under a thinking-off condition"
    if expectation == "optional" and evidence == 0:
        status = FAIL if strict else WARN
        return status, "model-paced thinking produced no observed thinking"
    return PASS, ""


def run_smoke(
    plan: SmokePlan,
    *,
    client_factory=None,
    submit_batch_fn=None,
    cancel_batch_fn=None,
    strict_thinking: bool = False,
    max_tokens: int = 0,
    config_source: str = "",
) -> SmokeReport:
    """Execute the plan. One check's failure never aborts the others."""
    from tutormoments import client as client_mod

    if client_factory is None:
        client_factory = client_mod.get_client
    if submit_batch_fn is None:
        submit_batch_fn = client_mod.submit_batch
    if cancel_batch_fn is None:
        cancel_batch_fn = client_mod.cancel_batch

    report = SmokeReport(
        started_at=datetime.datetime.now().isoformat(timespec="seconds"),
        config_source=config_source,
    )

    ping = resource_text("prompts/smoke/ping.md").strip()
    ping_json = resource_text("prompts/smoke/ping_json.md").strip()

    for check in plan.sync_checks:
        wire = resolve_thinking(check.model, check.thinking)
        row = CheckResult(
            label=check.label,
            model=check.model,
            provider=check.provider,
            wire=wire.describe(),
            status=PASS,
        )
        try:
            client = client_factory(check.model)
            resp = client.generate(
                ping_json if check.json_mode else ping,
                json_mode=check.json_mode,
                max_tokens=max_tokens,
                thinking=check.thinking,
            )
            usage = resp.usage or {}
            row.input_tokens = int(usage.get("input_tokens", 0) or 0)
            row.output_tokens = int(usage.get("output_tokens", 0) or 0)
            if not (resp.text or "").strip():
                row.status = FAIL
                row.detail = "empty response text"
            elif not usage.get("total_tokens") and not usage.get("total"):
                row.status = FAIL
                row.detail = "no token usage recorded"
            else:
                expectation = _thinking_expectation(wire, check.thinking)
                evidence = _thinking_evidence(usage)
                row.thinking_evidence = "n/a" if evidence is None else str(evidence)
                status, note = _judge_thinking(expectation, evidence, strict_thinking)
                row.status = status
                row.detail = note
        except Exception as e:  # noqa: BLE001 -- isolate per check
            row.status = FAIL
            row.detail = f"{type(e).__name__}: {e}"
        report.results.append(row)

    for check in plan.batch_checks:
        wire = resolve_thinking(check.model, check.thinking)
        row = CheckResult(
            label=f"batch:{check.provider}",
            model=check.model,
            provider=check.provider,
            wire=wire.describe(),
            status=PASS,
        )
        try:
            client = client_factory(check.model)
            entries = [
                client_mod.build_batch_entry(f"smoke-{i}", ping_json, json_mode=True)
                for i in range(2)
            ]
            batch_id = submit_batch_fn(
                client,
                entries,
                json_mode=True,
                display_name="tutormoments-smoke",
                thinking=check.thinking,
            )
            row.batch_id = str(batch_id)
            if not batch_id:
                row.status = FAIL
                row.detail = "submission returned no batch id"
            else:
                try:
                    cancel_batch_fn(client, batch_id)
                    row.detail = "submitted and cancelled"
                except Exception as e:  # noqa: BLE001 -- submission already proven
                    row.status = WARN
                    row.detail = (
                        f"submitted ok; cancel failed ({type(e).__name__}: {e}). "
                        "Cancel it manually; provider batches expire in 24h."
                    )
        except Exception as e:  # noqa: BLE001 -- isolate per check
            row.status = FAIL
            row.detail = f"{type(e).__name__}: {e}"
        report.results.append(row)

    for label, reason in plan.skipped:
        report.results.append(
            CheckResult(
                label=label,
                model="",
                provider="",
                wire="",
                status="SKIP",
                detail=reason,
            )
        )

    return report


def format_smoke_report(report: SmokeReport) -> str:
    """Render the report as an ASCII table (repo convention: no Unicode)."""
    headers = (
        "check",
        "model",
        "provider",
        "wire knobs sent",
        "in/out tok",
        "thinking",
        "result",
    )
    rows = []
    for r in report.results:
        tok = (
            f"{r.input_tokens}/{r.output_tokens}"
            if r.input_tokens or r.output_tokens
            else ""
        )
        rows.append(
            (
                r.label,
                r.model,
                r.provider,
                r.wire,
                tok,
                r.thinking_evidence,
                r.status + (f"  {r.detail}" if r.detail else ""),
            )
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows
        else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        f"config: {report.config_source}" if report.config_source else "",
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(
            "  ".join(row[i].ljust(widths[i]) for i in range(len(row))).rstrip()
        )
    n_fail = sum(1 for r in report.results if r.status == FAIL)
    n_warn = sum(1 for r in report.results if r.status == WARN)
    n_pass = sum(1 for r in report.results if r.status == PASS)
    lines.append("")
    lines.append(f"smoke: {n_pass} passed, {n_warn} warnings, {n_fail} failed")
    return "\n".join(line for line in lines if line is not None)
