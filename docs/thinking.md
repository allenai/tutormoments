# Benchmark thinking configuration

How TutorMoments states, validates, and transmits model reasoning settings.
Tutor arms are benchmark-defining: changing a configured provider knob changes
the experiment. Verify changed arm settings against the live API with
`tutormoments smoke` before relying on them.

## Tutor arms are explicit

Providers expose reasoning depth through incompatible knobs
(`thinking_budget`, `thinking_level`, `reasoning`, `effort`, provider-specific
`thinking` blocks). Reviewers need to see the exact settings used for each
benchmarked model in that provider's parlance, so the tutor roster uses
`benchmark_models:`:

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

## Infrastructure roles use the ladder

The student/scorer/taxonomy/groundtruth blocks use a smaller canonical ladder:

```yaml
student: { model: claude-opus-4-6, mode: oracle, thinking: none }
scorer:  { model: claude-opus-4-6, thinking: dynamic }
```

Those roles are benchmark apparatus rather than reported tutor arms. Keeping
their settings on the ladder keeps one reviewed default for the fixed
evaluation machinery while tutor-arm configs stay fully explicit.

Rung meanings: `none` = thinking verifiably off; `low`/`high`/`xhigh` =
explicit depth rungs; `dynamic` = the model decides (Gemini budget -1,
Anthropic adaptive, open-weight internal reasoning).

## Legacy ladder mapping

The authoritative copy is `src/tutormoments/models.yaml`; the contract tests
in `tests/tutormoments/test_models.py` pin every cell. The mapping is used by
infrastructure role blocks and by older `models:` configs still accepted for
compatibility. Summary:

| family | none | low | high | xhigh | dynamic |
|---|---|---|---|---|---|
| anthropic 4.6 tier (opus/sonnet-4-6) | omit `thinking` | adaptive + effort low | high | – (no xhigh on 4.6) | `{type: adaptive}`, no effort |
| anthropic 4.7+ (opus-4-7/4-8) | omit | adaptive + effort low | high | xhigh | `{type: adaptive}` |
| anthropic sonnet-5 | `{type: disabled}` (omission runs adaptive on Sonnet 5) | low | high | xhigh | `{type: adaptive}` |
| anthropic legacy (frozen pre-adaptive set) | omit | enabled+4096 | enabled+16384 | – | – |
| gemini-2.5-pro | – (API rejects budget 0) | budget 4096 | 16384 | 32768 (2x high; the Pro budget cap — Pro only, Flash caps at 24576) | budget −1 |
| gemini-2.5-flash | budget 0, include_thoughts false | 4096 | 16384 | – | budget −1 |
| gemini-3.x | – (thinking_level floor is not off) | thinking_level low ⚠ | high ⚠ | – | budget −1 (the proven wire shape) |
| openai gpt-5 line | reasoning_effort none ⚠ | low | high | xhigh ⚠ | – |
| openai o-series | – (no off switch) | low | high | – | – |
| together open-weight | – (always-thinking) | – | – | – | emit nothing |

"–" = unsatisfiable: config load raises. "⚠" = documented but not yet
verified live from this codebase: the registry lists these rungs under
`unverified`, and they raise a "not yet verified" error until proven.

Notes:
- Anthropic effort rides in `output_config.effort` (extra_body on sync,
  params on batch), exactly as before the ladder.
- Anthropic legacy `enabled` mode keeps the `max_tokens >= budget + 64`
  headroom rule.
- Gemini 3.x: `thinking_level` and `thinking_budget` are mutually exclusive
  on the wire (the API 400s if both are sent), which is why the family is
  split from 2.5.

## Shared budget rungs

Legacy budget-based ladder families use:

    low = 4096   high = 16384

Anchors, from provider guidance and the literature:
- 16k+ is Anthropic's guidance for complex tasks, roughly where published
  budget-vs-accuracy sweeps find diminishing returns, and this codebase's
  historical default budget.
- 4096 sits a quarter of the way there — a genuinely shallow condition that
  is still above Anthropic's documented 1,024 minimum on every family.

Tutor arms that need a different numeric operating point should state it
directly in `benchmark_models:` instead of adding a ladder rung.

Sources: Anthropic extended-thinking docs
(https://docs.claude.com/en/docs/build-with-claude/extended-thinking), Gemini
thinking docs (https://ai.google.dev/gemini-api/docs/thinking), LiteLLM's
reasoning_effort-to-budget mapping
(https://docs.litellm.ai/docs/providers/anthropic), and budget-vs-accuracy
findings (e.g. https://arxiv.org/pdf/2507.04023).

## Adding a model

One entry in `src/tutormoments/models.yaml` under `models:` — name the
`family` (add a `families:` entry only for a genuinely new capability shape)
and fill the per-model facts (`max_output_cap` if the API caps output,
`pricing` when known). Model ids match exact-first, then longest prefix,
case-insensitively, so a dated point release resolves to its base entry.
Then verify with `tutormoments smoke` before benchmarking.

## Verifying an `unverified` rung

1. Add the model/rung to a scratch config (or use `--arms`).
2. Temporarily remove the rung from the family's `unverified` list.
3. Run `tutormoments smoke --arms <arm>` and confirm: the call succeeds AND
   the thinking-evidence column matches the stated condition (reasoning
   tokens present for on-rungs, absent for `none`).
4. Commit the `unverified` removal with the smoke output in the PR.
