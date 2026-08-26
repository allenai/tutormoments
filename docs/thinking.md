# The thinking ladder and the model registry

How TutorMoments states, validates, and transmits each arm's reasoning
condition. This is benchmark-defining configuration: changing a mapping below
changes the experiment. Never change it without consulting the benchmark
owner, and verify any new or changed rung against the live API with
`tutormoments smoke` before relying on it.

## Why a ladder

Providers expose reasoning depth through four incompatible knobs
(`thinking_budget`, `thinking_level`, `reasoning_effort`, `effort`), and
before this design each config consumer re-interpreted the raw values its own
way — which produced validation/runtime divergence and made "the same model
at two thinking levels" inexpressible. Now:

- Config states ONE canonical value per arm/role:
  `thinking: none | minimal | low | medium | high | xhigh | max | dynamic`
  (required — every benchmarked condition is explicit).
- The model registry (`src/tutormoments/models.yaml`) translates it to the
  provider's wire knob, exactly once, at config load
  (`tutormoments.models.resolve_thinking`).
- The translation is fail-closed: a rung a model cannot honor (e.g. `none`
  on an always-thinking model), an unregistered model, or a retired raw knob
  in config is rejected before any tokens are spent.

Rung meanings: `none` = thinking verifiably off; `minimal`–`max` = explicit
depth rungs; `dynamic` = the model decides (Gemini budget −1, Anthropic
adaptive, open-weight internal reasoning).

## The ladder -> wire mapping

The authoritative copy is `src/tutormoments/models.yaml`; the contract tests
in `tests/tutormoments/test_models.py` pin every cell. Summary:

| family | none | minimal | low | medium | high | xhigh | max | dynamic |
|---|---|---|---|---|---|---|---|---|
| anthropic 4.6 tier (opus/sonnet-4-6) | omit `thinking` | – | adaptive + effort low | medium | high | – (no xhigh on 4.6) | max | `{type: adaptive}`, no effort |
| anthropic 4.7+ (opus-4-7/4-8) | omit | – | adaptive + effort low | medium | high | xhigh | max | `{type: adaptive}` |
| anthropic sonnet-5 | `{type: disabled}` (omission runs adaptive on Sonnet 5) | – | low | medium | high | xhigh | max | `{type: adaptive}` |
| anthropic legacy (frozen pre-adaptive set) | omit | enabled+1024 | 4096 | 8192 | 16384 | – | 32768 | – |
| gemini-2.5-pro | – (API rejects budget 0) | 1024 | 4096 | 8192 | 16384 | – | 32768 | budget −1 |
| gemini-2.5-flash | budget 0, include_thoughts false | 1024 | 4096 | 8192 | 16384 | – | 24576 | budget −1 |
| gemini-3.x | – (thinking_level floor is not off) | thinking_level minimal ⚠ | low ⚠ | medium ⚠ | high ⚠ | – | – | budget −1 (the proven wire shape) |
| openai gpt-5 line | reasoning_effort none ⚠ | minimal ⚠ | low | medium | high | xhigh ⚠ | – | – |
| openai o-series | – (no off switch) | – | low | medium | high | – | – | – |
| together open-weight | – (always-thinking) | – | – | – | – | – | – | emit nothing |

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

## The shared budget ladder (why these numbers)

All budget-based families (Gemini 2.5, Anthropic legacy) use one ladder so
rung names stay comparable across providers:

    minimal = 1024   low = 4096   medium = 8192   high = 16384
    max = provider cap (32768 Gemini Pro / 24576 Gemini Flash / 32768 Anthropic legacy)

Anchors, from provider guidance and the literature:
- 1,024 is Anthropic's documented minimum thinking budget, and the low-effort
  anchor cross-provider translation layers use.
- 16k+ is Anthropic's guidance for complex tasks, roughly where published
  budget-vs-accuracy sweeps find diminishing returns, and this codebase's
  historical default budget.
- Anthropic recommends batch processing above 32k, and 32768/24576 are the
  Gemini 2.5 Pro/Flash caps.

Sources: Anthropic extended-thinking docs
(https://docs.claude.com/en/docs/build-with-claude/extended-thinking), Gemini
thinking docs (https://ai.google.dev/gemini-api/docs/thinking), LiteLLM's
reasoning_effort-to-budget mapping
(https://docs.litellm.ai/docs/providers/anthropic), and budget-vs-accuracy
findings (e.g. https://arxiv.org/pdf/2507.04023).

An exact numeric budget (say 3000 tokens) is deliberately inexpressible from
config: if an experiment needs a new operating point, add a rung value in the
registry (a reviewed, benchmark-defining change), not a per-config knob.

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
