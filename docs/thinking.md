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
  `thinking: none | low | high | xhigh | dynamic`
  (required — every benchmarked condition is explicit). The ladder is
  deliberately small: a rung exists only when an experiment needs it, and
  adding one (e.g. `medium` for OpenAI's default effort tier, or `minimal`/
  `max`) is a reviewed registry line plus a `tutormoments smoke`
  verification.
- The model registry (`src/tutormoments/models.yaml`) translates it to the
  provider's wire knob, exactly once, at config load
  (`tutormoments.models.resolve_thinking`).
- The translation is fail-closed: a rung a model cannot honor (e.g. `none`
  on an always-thinking model), an unregistered model, or a retired raw knob
  in config is rejected before any tokens are spent.

Rung meanings: `none` = thinking verifiably off; `low`/`high`/`xhigh` =
explicit depth rungs; `dynamic` = the model decides (Gemini budget −1,
Anthropic adaptive, open-weight internal reasoning).

## The ladder -> wire mapping

The authoritative copy is `src/tutormoments/models.yaml`; the contract tests
in `tests/tutormoments/test_models.py` pin every cell. Summary:

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

## The shared budget ladder (why these numbers)

All budget-based families (Gemini 2.5, Anthropic legacy) use one ladder so
rung names stay comparable across providers:

    low = 4096   high = 16384

Anchors, from provider guidance and the literature:
- 16k+ is Anthropic's guidance for complex tasks, roughly where published
  budget-vs-accuracy sweeps find diminishing returns, and this codebase's
  historical default budget.
- 4096 sits a quarter of the way there — a genuinely shallow condition that
  is still above Anthropic's documented 1,024 minimum on every family.

The ladder deliberately omits rungs no experiment uses today (`minimal`,
`medium` — OpenAI's default effort tier, `max` — provider caps, which would
mean a different depth per family). Re-adding one is a reviewed registry
line plus a smoke verification.

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
