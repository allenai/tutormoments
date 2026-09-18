# Cost

## Why we track cost

TutorMoments exists to help people choose a tutoring model, and price is part of that
choice. A model that scores a few points higher on Appropriate Scaffolding but costs ten
times as much per conversation is a different product decision than one that costs the
same, and a leaderboard that hides that difference is answering only half of the question
an ed-tech team is actually asking. Ed-tech is a famously resource-constrained industry;
cost vs performance matters a lot in ed-tech.

There is a second audience: anyone deciding whether to replicate a run. Reproducibility is
this project's first priority, and "what will this cost me?" is a real barrier to
reproduction. Publishing a benchmark result without its price makes replication a leap of
faith; publishing the price makes it a budget line.

Finally, tracking cost keeps us honest with ourselves. Every LLM call in the pipeline
records its token usage, and the computed run cost is designed to be checked against the
provider's own invoice. If the two disagree, either our usage capture or our understanding
of the provider's billing is wrong — and both are things we want to find out.

## Reported figures

The summary reports the cost per conversation (relevant to ed techs) and the cost to run
the benchmark (relevant to reproducers).

**Tutor cost per conversation** answers the deployment question: what would it cost to run
this model as the tutor? It counts tutor-model tokens only, priced at the provider's list
rates with the cache mix the run actually produced (tokens served from cache are priced at
the cheaper cache-read rate). It is never batch-discounted because conversations cannot be batched efficiently — the tutor and the simulated student alternate
turns, so each call depends on the last. The leaderboard reports this figure divided by the
number of conversations in the run, so it does not scale with how many moments a particular
run happened to replay.

**Cost to run the benchmark** answers the replication question: what does one full run
cost, invoices included? It sums every role — tutor, student, scorer, and the taxonomy
classifier — and it *does* apply the batch discount to usage that actually went through a
batch API (in a runtime run, that is only the scorer's three pooled passes). This is the
number that should match the provider console for the API keys the run billed to.

Keeping the two separate is deliberate. If the student's and scorer's spending were folded
into the leaderboard column, tutors would be ranked partly by how much the *harness* spent
judging them. If the tutor figure were quoted as the cost of replication, it would
understate a run by the scorer's entire bill.

## How usage is recorded

Cost here is exact arithmetic over recorded usage, with no estimation anywhere. That is
possible because every provider reports the cached/uncached breakdown on each response
(`prompt_tokens_details.cached_tokens` on OpenAI, `cached_content_token_count` on Gemini,
`cache_read_input_tokens` on Anthropic). We record those splits per call rather than
assuming a cache hit rate, because measured hit behavior is too erratic to model: in one
live test, OpenAI's automatic cache returned zero cached tokens on an immediate retry and
then 5,058 of 5,061 on a probe a few minutes later. Per-call recording handles both cases
correctly; any assumed rate would handle neither.

Each provider reports usage in its own vocabulary, so every LLM call is normalized at the
client boundary ([`normalize_usage`](../src/tutormoments/client.py)) into one canonical
vector:

| bucket | billed at |
|---|---|
| `input_uncached` | full input rate |
| `cache_read` | cache-read rate (roughly 0.1× input) |
| `cache_write` | cache-write rate (Anthropic meters writes at 1.25× input; 0 where no write premium exists) |
| `output` | output rate |
| `reasoning` | output rate — tracked separately so the information is kept, but no provider prices reasoning tokens differently, so they fold into output when priced |
| `total` | derived: the sum of the five buckets, giving one consistent definition of "total tokens" |

Alongside the numbers, three provenance strings ride inside every usage dict: `provider`,
`model`, and `endpoint` (`sync`, `stream`, or `batch`). Aggregation
([`usage.py`](../src/tutormoments/usage.py)) sums every integer field rather than a fixed
list, keeps a provenance string when all contributions agree, and joins disagreements with
`+` (for example `"stream+sync"`). The one fact costing later needs from an aggregate —
whether any batch usage contributed — therefore survives summation.

The per-provider normalization, briefly: OpenAI's `cached_tokens` (and, on GPT-5.6-generation
models, `cache_write_tokens`) are subtracted out of `prompt_tokens`; Gemini's
`cached_content_token_count` and `thoughts_token_count` are captured directly; Anthropic's
`input_tokens` already *excludes* its cache buckets, so they add on top of it; Together
reports `cached_tokens` in the OpenAI style. One special case: OpenAI's Batch API does not
apply caching, so batch usage on that path records `cache_read = 0`.

## The pricing table

Token counts become dollars through the model registry
([`models.yaml`](../src/tutormoments/models.yaml)), which holds stable per-model facts:
provider routing, output caps, and pricing. Model ids resolve by exact match first and then
by longest prefix, so a dated snapshot such as `gpt-5.5-2026-04-23` finds its base entry. A
populated `pricing:` entry carries exactly seven keys, and the registry refuses to load an
entry that is malformed:

| key | meaning |
|---|---|
| `input` | $/MTok for uncached prompt tokens |
| `output` | $/MTok for completion and reasoning tokens |
| `cache_read` | $/MTok for tokens served from cache |
| `cache_write` | $/MTok for tokens written to cache; 0 where the provider meters no write premium |
| `batch_multiplier` | the batch API discount, in (0, 1]; 1.0 where none applies |
| `as_of` | the date (YYYY-MM-DD) the rates were read from the provider's documentation |
| `source` | the documentation URL they were read from |

Provider prices change over time, so a dollar figure is only meaningful next to the date
its rates were looked up. That is why `as_of` is mandatory, and why every run summary
records the `pricing_version` (and a snapshot of the exact rates) it was costed with.

**To update the table:** read the provider's pricing page, fill in all seven keys — `as_of`
is the day you looked, `source` is the page you looked at — and bump the top-level
`pricing_version`. An empty `pricing: {}` entry means "not priced yet", never "free":
costing a model without rates produces `null` and a logged warning, never a silent zero.

Two billing subtleties are encoded in the table's conventions rather than in code:

- **Cache-write TTL.** Some providers price cache writes by time-to-live (Anthropic:
  1.25× input for the 5-minute cache, 2× for the 1-hour cache). An entry states the rate
  for the TTL the runtime actually requests. Our client only ever sends bare
  `cache_control: {type: ephemeral}` — the 5-minute default — so Anthropic entries carry
  the 1.25× rate. Usage reports do not say which TTL a given write used, so if the runtime
  ever opts into 1-hour caching, this single-rate scheme has to be redesigned, not just
  repriced.
- **Gemini cache storage.** Gemini charges a storage price ($/MTok/hour) for explicit
  `CachedContent` objects. The runtime never creates one — the client prepends the
  cacheable prefix as plain content, because Gemini offers no `cache_control`-style API —
  so no storage term appears anywhere. It could not flow through the cost arithmetic even
  if it did: storage bills in token-*hours*, a time dimension that no per-response usage
  field reports. Adopting explicit caching would be a schema revisit, like the 1-hour TTL.

Long-prompt pricing tiers (gemini-2.5-pro bills higher above 200k input tokens, gpt-5.5
above 272k) are not modelled. Those thresholds apply per request, and the largest request
this benchmark makes stays well under them: the longest moment context in the release is
55,681 characters (see [docs/latency.md](latency.md)), which together with the largest
prompt template (~10k characters, the scorer's annotate pass) and a full conversation's
generated turns comes to roughly 20k tokens — about a tenth of the lower threshold — and
the median request is a few thousand tokens. The entries therefore carry the base-tier
rates, with a comment in the YAML marking the assumption. A future release with much
longer source transcripts would need this re-checked before its costs are trusted.

## From tokens to dollars

[`costing.py`](../src/tutormoments/costing.py) turns the run summary's `tokens` block into
the two figures. It is a small set of pure functions:

- `cost_usd(usage, rates, batch=)` prices one usage vector against one rate grid. Each
  bucket is priced exactly once, at its own rate. The cache buckets are disjoint from
  `input_uncached` by construction at the capture boundary — in particular, Anthropic's
  `input_tokens` already excludes them. Re-adding cache reads at the full input rate is
  the classic double-counting mistake, and a regression test pins it using a real vector
  from a live smoke run.
- `role_cost_usd(usage)` prices one role's aggregate using the provenance embedded in it:
  the model resolves to registry rates, the endpoint decides whether the batch discount
  applies. It returns `null` (with a logged warning) whenever the aggregate cannot be
  priced *exactly*: missing or mixed model provenance, an unpriced model, a mix of batch
  and non-batch usage the discount cannot be apportioned across, or pre-vector
  contamination (explained below).
- `list_cost(tokens)` computes the tutor figure — tutor aggregate only, never discounted.
  If the tutor aggregate is somehow batch-tagged, it raises an error rather than pricing
  it: conversations have been structurally synchronous since the dead conversation-batch
  path was removed, so that tag would mean an invariant was violated, not that a discount
  is due.
- `billed_cost_estimate(tokens)` computes the replication figure — every role summed, with
  the batch discount on batch-tagged usage. If any role cannot be priced, the whole
  estimate is `null`: a partial sum would quietly understate what replication costs, which
  is worse than reporting no number.
- `summary_cost_block(tokens, n_conversations)` packages both figures, the pricing
  version, and the rate snapshot for `summary.json`.

## Where the figures appear

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

`n_conversations` records the denominator explicitly — the conversations (across all
trials) whose tutor usage the token block aggregated. `pricing_version` and the `rates`
snapshot let anyone recompute a published cost even after the registry's rates have moved
on.

`tutormoments report` adds a `tutor_cost_per_conversation` column to the leaderboard
(markdown, CSV, and the HTML viewer), formatted to four decimal places because
per-conversation costs are cents. The end-of-run terminal summary prints both figures. The
public website does not publish cost for now; whether it should is a separate decision,
deliberately out of scope here.

## Old runs cannot be back-costed

Runs made before usage-vector capture existed recorded only each provider's legacy
counters. The cached/uncached split was discarded at the time, and no arithmetic can
recover it, so those runs cannot be costed accurately — and the code detects this rather
than guessing. The detection works because, when everything is captured, the canonical
`total` is always greater than or equal to the legacy `total_tokens` (they are equal on
OpenAI, Gemini, and Together; on Anthropic the canonical total is larger by the cache
buckets that the legacy input count excludes). An aggregate where `total_tokens` comes out
larger must therefore contain pre-vector contributions — a resume over an old run's
transcripts is the typical case — and it is reported as uncosted (`null`) rather than
priced from only the fraction of its usage that carried a vector. Summaries that predate
cost tracking entirely simply have no `cost` block. In both cases the leaderboard shows
`-`, never `$0`.

A few smaller gaps are handled in the same null-over-wrong spirit: registered and callable
tutors synthesize empty usage, so their cost is legitimately null; tokens spent on failed
retry attempts are billed by the provider but not captured; and the dataset-construction
code in `tutormoments_build/` records no usage at all yet, so the replication figure covers
a benchmark *run*, not reconstruction of the ground truth.

## Reconciliation against invoices

The replication figure is designed to be checked against the provider console for the API
keys a run billed to, and the capture layer's buckets have already been verified
bucket-for-bucket against the OpenAI and Anthropic dashboards. The standing per-run
reconciliation procedure, along with the remaining empirical questions (does Gemini's
implicit caching apply inside its batch endpoint? does Together's invoice actually discount
the cached tokens it reports?), is tracked in the cost plan and lands with the
reconciliation workstream.
