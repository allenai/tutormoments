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

## The per-provider keys

The allowed keys mirror exactly the knobs the client knows how to send; an
unknown or provider-mismatched key is a config error at load time, before any
tokens are spent. The authoritative validation is
`tutormoments.models.resolve_thinking`; the contract tests in
`tests/tutormoments/test_models.py` pin every wire fragment.

| provider | keys | wire form |
|---|---|---|
| anthropic | `thinking` (required; a thinking block or `null` = send no thinking param), `effort` (adaptive only) | `thinking={...}` plus `output_config.effort` (extra_body on sync, params on batch) |
| gemini | exactly one of `thinking_budget` / `thinking_level`, optional `include_thoughts` | `generation_config.thinking_config` |
| openai | `reasoning` (required) | `reasoning_effort` |
| together | none (open-weight internal reasoners) | nothing sent |

Validation is shape-level: key ownership, mutual exclusion (Gemini's budget
vs. level; the API 400s if both are sent), and structural rules (Anthropic
`budget_tokens` only with `type: enabled` and positive; `effort` only with
`type: adaptive`). Value vocabularies (which effort tiers or reasoning levels
a given model accepts) are the provider's to extend, so they are proven live
with `tutormoments smoke`, not enumerated offline.

Provider notes:
- Anthropic: what an omitted thinking param (`thinking: null`) means depends
  on the model generation — thinking off on the 4.x tiers, adaptive on
  Sonnet 5 and later. Prefer the explicit `thinking: {type: disabled}` for
  "off": the condition then survives a model swap, and smoke can assert it
  (an omitted param has no verifiable expectation and judges as "na").
- Anthropic `enabled` mode keeps the `max_tokens >= budget + 64` headroom
  rule; the client enforces it.
- Gemini `thinking_budget: -1` is model-paced; `0` is off (rejected by
  always-thinking models such as 2.5 Pro). `include_thoughts` defaults to
  true except with budget 0. The 3.x line replaced `thinking_budget` with
  `thinking_level`; every published 3.x run so far has used the budget `-1`
  shape.
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
