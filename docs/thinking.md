# Benchmark thinking configuration

How TutorMoments states, validates, and transmits model reasoning settings.
Every configured condition is benchmark-defining: changing a provider knob
changes the experiment. Verify changed settings against the live API with
`tutormoments smoke` before relying on them.

## Every condition is explicit, in provider parlance

Providers expose reasoning depth through incompatible knobs
(`thinking_budget`, `thinking_level`, `reasoning`, `effort`, provider-specific
`thinking` blocks). Reviewers need to see the exact settings used for each
benchmarked model in that provider's parlance, so config states them
directly — there is no intermediate "thinking level" vocabulary to decode.
The tutor roster is `benchmark_models:`:

```yaml
benchmark_models:
  gemini-2.5-low: { model: gemini-2.5-pro, thinking_budget: 4096, condition: low }
  gpt-5.5-low:    { model: gpt-5.5-2026-04-23, reasoning: low, condition: low }
  opus-4.8-xhigh: { model: claude-opus-4-8, thinking: { type: adaptive }, effort: xhigh, condition: xhigh }
```

The arm key is the selectable tutor id and the result join key. `model` is the
provider model id. Provider-native keys are validated at config load and
forwarded by the client. `condition` is the grouping label written to
`config.json`, `summary.json`, latency probe output, and leaderboard rows.

The student/scorer/taxonomy/groundtruth role blocks state their thinking
parameters the same way (minus `condition` — they are benchmark apparatus,
not reported arms):

```yaml
student: { model: claude-opus-4-6, mode: oracle, thinking: { type: disabled } }
scorer:  { model: claude-opus-4-6, thinking: { type: adaptive } }
```

The maintainer-only `v2:` block (the annotation-round classifier,
`python -m tutormoments_build.v2.classify_excerpts`) is a role block with a
comparison table hung off it. Its own model states the round's default
condition; `v2.models` is a lookup table keyed by model id, consulted when
`--model` names a scoring model that is not on the tutor roster:

```yaml
v2:
  model: claude-opus-5
  thinking: { type: adaptive }
  effort: xhigh
  models:
    gpt-5.6-sol: { reasoning: xhigh }
```

`--model` resolves against `v2.models` first, then `benchmark_models` (by arm
name or model id), so a model already on the roster scores at exactly the
parameters the benchmark replays it with and cannot drift from its arm. An
override takes its entry whole and inherits nothing from the block above it:
merging would carry one vendor's knob onto another's model. Unlike the
runtime's own blocks the `v2:` block is not validated by the config loader —
`classify_excerpts.phase_config` validates it through the same
`resolve_thinking`, before a batch is built.

## The per-provider keys

The allowed keys mirror exactly the knobs the client knows how to send; an
unknown or provider-mismatched key is a config error at load time, before any
tokens are spent. The authoritative validation is
`tutormoments.models.resolve_thinking`; the contract tests in
`tests/tutormoments/test_models.py` pin every wire fragment.

| provider | keys | wire form |
|---|---|---|
| anthropic | `thinking` (required; a thinking block or `null` = send no thinking param), `effort` (adaptive or disabled) | `thinking={...}` plus `output_config.effort` (extra_body on sync, params on batch) |
| gemini | exactly one of `thinking_budget` / `thinking_level`, optional `include_thoughts` | `generation_config.thinking_config` |
| openai | `reasoning` (required) | `reasoning_effort` |
| together | `reasoning_effort` for hybrid reasoners; none for open-weight internal reasoners | `reasoning_effort`; nothing sent when none |
| meta | `reasoning_effort` (required) | `reasoning_effort` |

Validation is shape-level: key ownership, mutual exclusion (Gemini's budget
vs. level; the API 400s if both are sent), and structural rules (Anthropic
`budget_tokens` only with `type: enabled` and positive; `effort` only with
`type: adaptive` or `disabled`, since effort also sets answer length). Value vocabularies (which effort tiers or reasoning levels
a given model accepts) are the provider's to extend, so they are proven live
with `tutormoments smoke`, not enumerated offline.

Provider notes:
- Anthropic: what an omitted thinking param (`thinking: null`) means depends
  on the model generation — thinking off on the 4.x tiers, adaptive on
  Sonnet 5 and later. Prefer the explicit `thinking: {type: disabled}` for
  "off": the condition then survives a model swap, and smoke can assert it
  (an omitted param has no verifiable expectation and judges as "na").
- Anthropic Opus 5.5, Fable 5.1 and Sonnet 5.5 have no off switch:
  `{type: disabled}` is a 400, so their arms state `adaptive` and vary
  `effort` (`low`..`max`; `low` is the floor). Opus 5.5 defaults to
  `medium`, not `high`, so its effort must be stated. Sonnet 5.5 also
  accepts `{type: between_tools}` (thinking only between tool calls); the
  benchmark makes no tool calls, and the config does not accept that type.
- Anthropic Haiku 5.5 is adaptive by default (default effort `medium`), but
  unlike the three above it does accept `{type: disabled}` (checked live
  2026-10-07). Its arm (`claude-haiku-5-5-none`) states `disabled` with
  effort `low`.
- Together hybrid reasoners (DeepSeek-V4-Pro-0813) think by default and
  return reasoning in a separate field, reported as `reasoning_tokens`, so
  smoke asserts a stated `reasoning_effort` against it. Together documents
  `reasoning_effort` as accepting `high` and `max` (other values
  auto-mapped); the arm uses `max`, the documented top tier. Older
  open-weight reasoners (DeepSeek-V4-Pro, Kimi) expose no knob, so they take
  `{}` and judge as "na".
- Observed on DeepSeek-V4-Pro-0813 (2026-09-28), contrary to Together's
  reasoning guide: `reasoning: {enabled: false}` does not turn thinking off
  (reasoning tokens are still produced), which is why the config does not
  accept that key; only the undocumented `chat_template_kwargs: {thinking:
  false}` did. In a small probe (one prompt, 3 calls each) `max` produced
  fewer reasoning tokens than `high` (606-1419 vs 2306-2746), comparable
  to sending no effort (430-2014), so the `max` label is Together's tier
  name, not a measured "most thinking". Recheck before comparing effort
  tiers across runs.
- Meta (Muse Wonder, `muse-wonder-1.0`) always reasons: `reasoning_effort:
  none` is a 400, and an omitted effort still reasons at a model-chosen
  level, so config requires the key rather than letting omission stand for a
  condition. Meta documents `minimal`/`low`/`medium`/`high`/`xhigh`/`max`
  for Muse Wonder; the arm uses `high`. Reasoning is private (never streamed
  or returned as text) and reported as `reasoning_tokens`, so smoke asserts
  a stated effort against it; `minimal` is treated like Gemini's (zero
  thinking is a WARN). See https://dev.meta.ai/docs/reasoning.
- Anthropic `enabled` mode keeps the `max_tokens >= budget + 64` headroom
  rule; the client enforces it.
- Gemini `thinking_budget: -1` is model-paced; `0` is off (rejected by
  always-thinking models such as 2.5 Pro). `include_thoughts` defaults to
  true except with budget 0. The 3.x line replaced `thinking_budget` with
  `thinking_level`; the 3.5 Flash arm keeps the budget `-1` shape it was
  published with, and later 3.x arms state `thinking_level`.
- Gemini `thinking_level` vocabularies differ by model and there is no off
  level: 3.6 Flash accepts `minimal`/`low`/`medium`/`high`; 3.7 and 3.8
  Flash accept only `low`/`medium`/`high` (`minimal` is a 400). Google
  describes `minimal` as using as few thinking tokens as possible, not as
  off: on 3.6 Flash (2026-09-30) a smoke ping produced zero thinking tokens,
  so smoke treats `minimal` like budget `-1` (zero thinking is a WARN, not a
  FAIL). Any other level is expected to produce thinking. Gemini 4 Argon
  accepts `low`/`medium`/`high` (default `medium`); its arm uses `high`.
- The OpenAI o-series exposes only a depth knob (`reasoning`), never an off
  switch.

## Choosing budget operating points

The default arms use model-paced settings or provider effort tiers. When an
experiment needs fixed numeric budgets, the numbers this project has used are

    low = 4096   high = 16384

anchored in provider guidance and the literature: 16k+ is Anthropic's
guidance for complex tasks, roughly where published budget-vs-accuracy sweeps
find diminishing returns, and this codebase's historical default budget;
4096 sits a quarter of the way there — a genuinely shallow condition that is
still above Anthropic's documented 1,024 minimum. State the number directly
in the arm (`thinking_budget: 4096` / `budget_tokens: 4096`).

Sources: Anthropic extended-thinking docs
(https://docs.claude.com/en/docs/build-with-claude/extended-thinking), Gemini
thinking docs (https://ai.google.dev/gemini-api/docs/thinking), LiteLLM's
reasoning_effort-to-budget mapping
(https://docs.litellm.ai/docs/providers/anthropic), and budget-vs-accuracy
findings (e.g. https://arxiv.org/pdf/2507.04023).

## Adding a model

`src/tutormoments/models.yaml` holds stable per-model facts only: `provider`,
`max_output_cap` (if the API caps output), and `pricing` (per-MTok rates for
cost tracking; empty means "not priced yet", never "free"). Add one entry
there so the run is priced and capped correctly — an unregistered id still
routes by name prefix, but runs unpriced. Model ids match exact-first, then
longest prefix, case-insensitively, so a dated point release resolves to its
base entry. Then verify the arm with `tutormoments smoke` before
benchmarking.

## Verifying a new or changed condition

1. Add the arm to a scratch config (or use `--arms`).
2. Run `tutormoments smoke --arms <arm>` and confirm: the call succeeds AND
   the thinking-evidence column matches the stated condition (reasoning
   tokens present when the knobs demand thinking, absent when they turn it
   off).
3. Include the smoke output in the PR.
