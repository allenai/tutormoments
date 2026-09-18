# Cost

Two numbers matter, and they answer different questions.

**Tutor cost per conversation** is the deployment question: what would it cost to run this
model as the tutor? Tutor-model tokens only, priced at list rates with the *actual* cache
mix the run produced (cache reads at the cache-read price), and **never batch-discounted** —
batching is a harness throughput choice, not a model property, and conversations are
structurally synchronous anyway (the tutor and the simulated student alternate; there is
nothing to batch). This is the leaderboard figure, reported per conversation
(`tutor_list_cost / n_conversations`) so it does not scale with how many moments a run
happened to replay.

**Cost to run the benchmark** is the replication question: what does one full run cost,
invoices included? Every role — tutor, student, scorer, taxonomy classifier — **with** the
batch discount applied to batch-tagged usage (the scorer's three pooled passes are the only
batch consumer in a runtime run). This is the number that should reconcile against the
provider console for the API keys the run billed to.

Keeping them separate is the point. Folding the student and scorer into a "cost" column
would rank tutors partly by how much the *harness* spent judging them; quoting the tutor
figure as the replication cost would understate a run by the scorer's entire bill.

## Exact arithmetic, no estimation

Every provider reports the cached/uncached split on every response
(`prompt_tokens_details.cached_tokens`, `cached_content_token_count`,
`cache_read_input_tokens`), so cost is exact arithmetic over recorded usage. No cache-hit-rate
assumption appears anywhere in the pipeline — measured hit behavior is too erratic to model
(OpenAI's automatic cache returned 0 cached tokens on an immediate retry and 5,058/5,061 on a
probe minutes later; per-call recording handles both, any rate assumption handles neither).

Every LLM call is normalized at the client boundary
([`normalize_usage`](../src/tutormoments/client.py)) into one canonical vector:

| bucket | billed at |
|---|---|
| `input_uncached` | full input rate |
| `cache_read` | cache-read rate (~0.1× input) |
| `cache_write` | cache-write rate (Anthropic 1.25× input; 0 where no write premium is metered) |
| `output` | output rate |
| `reasoning` | output rate (tracked separately so no information is lost; no provider prices it apart, so it folds into output for pricing) |
| `total` | derived: the sum of the five — the single consistent definition of "total tokens" |

plus provenance strings that ride inside the dict and survive aggregation: `provider`,
`model`, and `endpoint` (`sync` | `stream` | `batch`). Aggregation
([`usage.py`](../src/tutormoments/usage.py)) sums every integer key rather than an
allowlist, keeps provenance when uniform, and `+`-merges it when mixed (`"stream+sync"`) —
so the one fact costing needs from an aggregate, whether `batch` contributed, is never
destroyed.

Provider normalization, in brief: OpenAI's `cached_tokens` (and, on GPT-5.6-gen models,
`cache_write_tokens`) are split out of `prompt_tokens`; Gemini's
`cached_content_token_count` and `thoughts_token_count` are captured directly (its legacy
provider total already includes thoughts); Anthropic's `input_tokens` already *excludes* its
cache buckets, so they add on top; Together reports `cached_tokens` OpenAI-style. OpenAI's
Batch API applies no caching, so batch usage forces `cache_read = 0` there.

## Pricing table

Rates live in the model registry ([`models.yaml`](../src/tutormoments/models.yaml)) —
one file of stable per-model facts, resolved by exact id then longest prefix, so a dated
snapshot like `gpt-5.5-2026-04-23` resolves to its base entry. A populated `pricing:` entry
carries exactly seven keys, shape-validated at registry load:

| key | meaning |
|---|---|
| `input` | $/MTok, uncached prompt tokens |
| `output` | $/MTok, completion + reasoning tokens |
| `cache_read` | $/MTok, tokens served from cache |
| `cache_write` | $/MTok, tokens written to cache; 0 where the provider meters no write premium |
| `batch_multiplier` | the batch API discount, in (0, 1]; 1.0 where none applies |
| `as_of` | the date (YYYY-MM-DD) the rates were read from the provider's docs |
| `source` | the documentation URL they were read from |

**To update the table:** read the provider's pricing page, set all seven keys (`as_of` =
the day you looked, `source` = the page), and bump the top-level `pricing_version`. Rates
change over time, so a dollar figure is only meaningful alongside its lookup date — that is
why `as_of` is mandatory and why every run summary records the `pricing_version` (and a
snapshot of the resolved rates) it was costed with. An empty `pricing: {}` means "not priced
yet", never "free": costing an unpriced model yields `null` plus a logged warning, never a
silent 0.

Two rate subtleties the table encodes rather than the code:

- **Cache-write TTL.** Where a provider offers multiple cache-write rates by TTL
  (Anthropic: 1.25× input for the 5-minute cache, 2× for the 1-hour), the entry states the
  rate for the TTL the runtime actually requests. The client only ever sends bare
  `cache_control: {type: ephemeral}` — the 5-minute default — so Anthropic entries carry
  1.25×. Usage reports do not say which TTL a write used, so opting into 1-hour caching
  would force a schema revisit, not just a repricing.
- **Gemini cache storage.** Gemini's cache *storage* price ($/MTok/hour) applies only to
  explicit `CachedContent` objects, which the runtime never creates (the client prepends
  the cacheable prefix as plain content — Gemini has no `cache_control`-style API). So no
  storage term appears anywhere — and none could flow through the cost arithmetic
  regardless, since storage bills in token-hours, a time dimension no per-response usage
  field reports. Adopting explicit caching would be a schema revisit, like Anthropic 1h.

Long-prompt tiers (gemini-2.5-pro above 200k input, gpt-5.5 above 272k) are not modelled:
benchmark prompts sit orders of magnitude below both thresholds, so entries carry the base
tier, with a YAML comment marking the assumption.

## The computation

[`costing.py`](../src/tutormoments/costing.py) is pure functions over the run summary's
`tokens` block:

- `cost_usd(usage, rates, batch=)` — one vector × one rate grid. Each bucket is priced
  exactly once at its own rate; the cache buckets are disjoint from `input_uncached` by
  construction at the capture boundary (in particular, Anthropic's `input_tokens` already
  excludes them — re-adding cache reads at the full input rate is the classic double-count,
  and a regression test pins it with a real smoke-run vector).
- `role_cost_usd(usage)` — prices one role aggregate from its own embedded provenance:
  model → registry rates, endpoint → batch flag. Null (with a logged warning) whenever the
  aggregate cannot be priced *exactly*: no or mixed model provenance, an unpriced model, a
  batch+sync endpoint mix (the discount cannot be apportioned), or pre-vector contamination
  (below).
- `list_cost(tokens)` — figure (1): the tutor aggregate only, never discounted. A
  batch-tagged tutor aggregate raises: conversations have been structurally sync since the
  dead conversation-batch path was removed, so that tag is an invariant violation, not a
  case to price.
- `billed_cost_estimate(tokens)` — figure (2): every role summed, batch discount on
  batch-tagged usage. Any unpriceable role nulls the whole estimate — a partial sum would
  quietly understate replication cost, which is worse than no number.
- `summary_cost_block(tokens, n_conversations)` — packages both figures for
  `summary.json`.

## Where the figures surface

Every run's `summary.json` carries a `cost` block:

```json
"cost": {
  "tutor_list_cost_usd": 2.7086,
  "tutor_cost_per_conversation_usd": 0.0026,
  "n_conversations": 1040,
  "run_billed_cost_estimate_usd": 9.4312,
  "pricing_version": "2026-09-16",
  "rates": {"claude-opus-4-8": {"input": 5.0, "...": "..."}}
}
```

`n_conversations` is the recorded denominator — the conversations (all trials pooled) whose
tutor usage the token block aggregated. `pricing_version` plus the `rates` snapshot keep a
published cost interpretable and recomputable after the registry's rates move on.

`tutormoments report` adds a `tutor_cost_per_conversation` leaderboard column (markdown,
CSV, and the HTML viewer), formatted at 4 decimals because per-conversation costs are
cents. The end-of-run terminal summary prints both figures. The public website does not
publish cost for now — that is a separate decision, deliberately out of scope.

## Old runs cannot be back-costed

Runs from before usage-vector capture recorded only legacy per-provider counters; the
cached/uncached split was discarded at the time and no arithmetic can recover it. Costing
detects this rather than guessing: at the capture boundary the canonical `total` is ≥ the
legacy `total_tokens` on every provider (equal on OpenAI/Gemini/Together; Anthropic exceeds
it by the cache buckets its legacy input count excludes), so an aggregate where
`total_tokens` wins must contain pre-vector contributions — for example a resume over an
old run's transcripts — and is reported as uncosted (`null`) rather than priced from the
fraction of its usage that carried a vector. Pre-cost summaries simply have no `cost`
block; the leaderboard shows `-` for both cases, never `$0`.

Known residual gaps, in the same spirit of null-over-wrong: registered/callable tutors
synthesize empty usage, so their cost is legitimately null; failed retry attempts bill
tokens that are not captured; and `tutormoments_build/` records no usage at all yet, so
figure (2) covers a benchmark *run*, not ground-truth reconstruction.

## Reconciliation

`run_billed_cost_estimate_usd` is designed to be checked against the provider console for
the API keys the run billed to — the capture layer's buckets have been verified
bucket-for-bucket against the OpenAI and Anthropic dashboards. The standing per-run
procedure (and the remaining empirical questions: Gemini implicit caching inside its batch
endpoint, Together's cached-token discount on an actual invoice) is tracked in the cost
plan and lands with the reconciliation workstream.
