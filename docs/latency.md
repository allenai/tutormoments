# Latency

Two numbers matter for a tutoring product.

**Time to first token (TTFT)** drives how responsive the tutor feels. Ed-tech partners
name this as important to maintaining student engagement.

**Time to last token (TTLT)** is when the student can actually reply. It bounds the pace of
the whole exchange.

**TTFT is the headline; TTLT is reported beside it.** Both are real, but they are not
interchangeable, and TTLT should not be the number a model is ranked on:

- TTLT = TTFT + the time to stream the answer out. That window is 0–11% of TTLT across the
  roster, so the two rank models almost identically — headlining TTFT gives up very little.
- What the window *does* carry is generation length × throughput. Length is a content
  property this benchmark already evaluates: a model that over-explains is penalised by
  Avoids-Over-Scaffolding. Ranking on TTLT partly double-counts that.
- More importantly, **TTLT rewards saying less.** A curt, dismissive turn beats a
  well-scaffolded longer one on TTLT irrespective of teaching quality. TTFT cannot be gamed
  that way: it measures time until the student sees anything, and is indifferent to what
  the tutor then says.

Report TTLT because "when can the student reply" is a genuine question. Rank on TTFT
because it isolates responsiveness from verbosity.

Throughput (tokens per second) is measured by Artificial Analysis but is not reported here. Good tutor turns are short, so high throughout only matters to the extent that it reduces time to last token.

Tokens per second is recorded as `output_tps` but is never a headline.

## Definitions

| Metric | Anchored at | Notes |
|---|---|---|
| `ttft_seconds` | first **visible answer** token | what the student sees appear |
| `ttlt_seconds` | last visible delta | when the student can reply |
| `ttfc_seconds` | first chunk of any kind, reasoning included | diagnostic; `ttft − ttfc` is roughly thinking time |
| `output_tps` | `output_tokens / (ttlt − ttft)` | diagnostic only |
| `n_no_visible_output` | calls that emitted no visible token | sanity check; should be 0 |

A call that emits no visible token has no TTFT, so the percentiles are conditional on a
turn having produced output. `n_no_visible_output` makes that conditioning visible —
otherwise a model could look fast by not answering. It should be 0; a non-zero value means
a model exhausted even its maximum output budget on reasoning, and both the latency figures
and that run's scores need checking before they are trusted.

TTLT deliberately anchors on the last content delta rather than on stream exhaustion. The
trailing usage-only chunk and connection teardown arrive after generation finishes and are
not part of anyone's wait.

## Getting the number

```bash
tutormoments latency --tutor claude-sonnet-5 --mode scaffolding_rigor
```

Writes `results/<run_id>/latency.json` and prints a summary. This is the reportable figure.

### How many samples you get

A sample is one **tutor call**, not one moment. Each conversation makes several:

```
samples = moments × tutor calls per conversation
```

Tutor calls per conversation is `ceil(max_turns / 2)` — the tutor speaks on turns 1, 3, 5 —
so 3 at the config default `max_turns` of 5, and 2 at `--max-turns 3`. With the frozen
112-moment subsample:

| invocation | samples | cold | warm |
|---|---|---|---|
| defaults (`max_turns` 5) | 112 × 3 = **336** | 112 | 224 |
| `--max-turns 3` | 112 × 2 = **224** | 112 | 112 |

Exactly one sample per conversation is cold (turn 1); the rest are warm. So a *cold*
percentile is computed over 112 regardless of `max_turns` — raising `max_turns` buys warm
samples only.

**There is no repeat option, deliberately.** Measuring the same moment twice does not give
two independent samples: the second request is byte-identical, and any provider with
automatic prefix caching serves it from cache, so the repeated "cold" sample is not cold.
Sample size therefore comes from distinct moments, and the subsample is already every
conversation in the release. If 112 moments do not resolve what you need, the honest
options are a larger release or reporting tiers rather than neighbour-level rankings — not
re-measuring the same prompts.

To check whether the measurement environment is stable, run the probe twice and compare the
two `latency.json` files. Each carries its own `measured_at`, which a pooled repeat would
have averaged away — see [the student as drift control](#the-student-is-a-free-drift-control)
for a check that comes free with every run.

`tutormoments run` also records TTFT/TTLT on every transcript, but those are diagnostics —
see [Why a separate command](#why-a-separate-command).

## Analysis

### Two departures from Artificial Analysis

We follow [Artificial Analysis](https://artificialanalysis.ai/methodology/performance-benchmarking)
on percentiles (P5/P50/P95) but diverge on two points.

**We measure the first *visible* token, not the first token of any kind.** AA's headline
TTFT counts the first *reasoning* token for reasoning models, and its methodology assumes
2,000 reasoning tokens when actual counts aren't available. That makes AA's TTFT not
apples-to-apples between a provider that streams reasoning and one that hides it. It is
also not computable on our OpenAI path at all: `chat.completions` never streams reasoning
tokens. First-visible-token is the only definition computable uniformly across all four of
our providers — and it is the one a student actually experiences. `ttfc_seconds` still
records the raw first chunk wherever the provider distinguishes it.

Concretely, per provider:

- **Anthropic** — thinking and text arrive as separate content blocks; only deltas inside a
  `text` block count as visible.
- **Gemini** — reasoning parts carry `part.thought`; those are skipped.
- **Together** — open-weight reasoners (DeepSeek-V4-Pro, Kimi) emit `<think>…</think>`
  *inline in the content stream*, so TTFT is held until after the closing tag. Without this
  they would look dramatically faster to first token than any student would experience.
- **OpenAI** — reasoning is never streamed, so first content delta is both `ttfc` and `ttft`.

**We report warm and cold separately rather than a single number.** See below.

### Why TTLT is almost equal to TTFT

A streaming response arrives as a series of **deltas** — the term for one
incremental piece of content (`content_block_delta` on Anthropic, `choices[0].delta.content`
on OpenAI). A long response streams as hundreds of them, which is what produces the familiar
token-by-token typing effect. **A tutor turn does not: it is short enough to arrive in one or two deltas.**

A direct instrumented call counted exactly **2 text deltas** for both `claude-sonnet-4-6`
and `claude-opus-4-8`, carrying answers of 72 and 157 characters. Across the frozen
subsample, `claude-sonnet-4-6` delivered its whole visible answer inside a 5ms window on
44 of 80 calls. Median generation window (TTFT → TTLT): 0.00s for Sonnet 4.6, 0.05s for
Gemini 2.5 Pro, 1.16s for Opus 4.8.

The practical consequence: streaming buys a tutoring product almost no progressive
rendering. The student waits, then the message appears essentially at once. **TTFT is the
metric; TTLT is TTFT plus about a second; throughput is noise.** Streaming is still
required — it is the only way to observe TTFT at all — but not for the usual reason.

A corollary for reading `output_tps`: on thinking models most `output_tokens` are thinking
tokens (Opus 4.8: ~500 of 551 on one measured call), so tokens/sec largely describes
reasoning speed, not how fast text reaches the student.

### Caching does not systematically improve latency

A tutoring provider would cache. The prompt is input-heavy and output-light — system prompt
plus full transcript in, a few hundred tokens of tutor turn out — so input dominates cost,
and prompt caching cuts the dominant term by roughly 90% on a hit. Cache reads cost ~0.1×
base input against writes at 1.25× (5-minute TTL), so two requests break even; the write
adds no meaningful latency beyond the prefill you pay anyway. Caching is not a close call.

So the benchmark caches too ([`cacheable_prefix`](../src/tutormoments/conversation.py) —
system prompt plus pre-cut transcript), and the two cache states map onto real student
experiences:

- **miss** ≈ the first message of a session
- **hit** ≈ every later message in that session

They are reported separately. Pooling them would make the figure drift with how many turns
each conversation happened to run, since `[END]` truncates some conversations early.

**Cache state is read off `cache_read_input_tokens`, never inferred from turn position.**
Turn index is a bad proxy for two reasons. The minimum cacheable prefix is model-dependent
and *not monotonic* across generations — 512 tokens on Opus 5, 1024 on Opus 4.8 / Sonnet 5 /
Sonnet 4.6, but 4096 on Opus 4.6 and Haiku 4.5 — and a prefix below the minimum fails to
cache **silently**. So the same short-transcript moment can cache on one roster model and
not another. And providers that report no cache tokens are recorded as `unknown` rather than
guessed at.

### Caching fidelity — the biggest caveat

**Only the Anthropic path sends a real cache breakpoint.** Gemini
([client.py](../src/tutormoments/client.py)) and Together concatenate the cacheable prefix
into the prompt instead, so neither caches *this conversation's* transcript.

That does not mean they report no cache hits. Together reports `cached_tokens` from its own
automatic prefix caching, and measured on `deepseek-ai/DeepSeek-V4-Pro` it returns a **0.91
hit rate** — higher than Anthropic's structural 0.50. The hits are not what they look like:

| | hit/miss by turn | median tokens read back |
|---|---|---|
| Anthropic, explicit breakpoint | perfect `miss, hit` alternation | **7,400–10,000** |
| Together, automatic prefix cache | misses cluster at *run start*, unrelated to turn | **256** |

Together is caching one quantised block of the system prompt every moment shares — run
warmup, not session warmth. Reading `cached_tokens > 0` as "warm" would compare that against
Anthropic's genuine 8k-token transcript cache.

So a warm figure is published only when the median hit actually reads back a conversation
head: `cache_read_p50_on_hits >= MIN_SESSION_CACHE_READ_TOKENS`. The threshold sits above any
incidental block (256) and below the smallest real head observed (1,180).

The warm figure is **withheld** (rendered `-`) unless all three hold: the provider reports
cache tokens; there are at least `MIN_CACHE_HIT_SAMPLES` hits; and the median hit reads back
a real conversation head. The sample gate is a count rather than a hit *rate* deliberately —
the rate is fixed by `max_turns` (exactly 0.5 at `--max-turns 3`, 0.67 at 5), so a rate
threshold would let a run knob decide whether a model gets a published figure. In practice
one `claude-sonnet-4-6` run came in at 0.49 because a single turn missed; under a rate gate
its warm figure would have vanished over one stray call.

Every latency block publishes `cache_hit_rate` and `cache_read_p50_on_hits` alongside the
numbers. **Until real prefix caching is wired for Gemini and Together, the warm column is
only comparable within the Anthropic-hosted models.** That is tracked as follow-up work.

### What is not modelled

The Anthropic ephemeral TTL is 5 minutes, refreshed on each hit. A real session mixes hits
and misses depending on how long students pause — a student who reads a hint and replies in
90 seconds stays warm; one who works a problem for eight minutes does not. The probe runs
turns back-to-back, so it measures the hit and miss *endpoints*, not the blend a deployment
sees. We report both endpoints rather than inventing a hit-rate weighting.

### Concurrency impacts latency

`tutormoments run` replays moments through a thread pool (`--concurrency`, default 4), and
concurrency distorts latency by a **model-dependent** amount:

1. **Shared decode batches** — concurrent requests from one account land in the same decode
   batch, so per-stream throughput falls as concurrency rises. This is why Artificial
   Analysis publishes single-request and 10-parallel as two separate numbers.
2. **Edge queueing** — near ITPM/RPM ceilings providers queue before prefill, inflating
   TTFT. An outright 429 doesn't pollute the figure (only successful attempts are stamped),
   but soft queueing does, invisibly.
3. **Rate-limit tiers differ per model**, so the size of the distortion differs per model.
   That is what actually breaks cross-model comparison — more than the absolute error.

The probe runs strictly serially, so its numbers mean the same thing for every model. Run
figures are stamped `"source": "run"` with the concurrency they were gathered at, and the
leaderboard withholds the TTFT column for them.

Conversations also share one `ModelClient` per model
([`get_client`](../src/tutormoments/client.py)). Previously each moment built its own,
paying a fresh TLS handshake on its first request — which landed squarely on the cold-cache
turn, inflating exactly the figure we report.

Batch mode records no latency at all: the batch APIs expose none. The probe requires a sync run.

### The student is a free drift control

Student turns are streamed and timed like tutor turns, and `latency.json` carries a
`student` block. It is never surfaced in the summary or the leaderboard — no human waits on
a simulated student, so its latency is not a product metric.

It is useful for something else. **The student model is fixed across every run in a sweep**
(the config's `student.model`, not the tutor under test), so its latency is the same
measurement repeated under changing conditions — a control for whether the environment
drifted while the sweep ran.

Comparing tutors measured hours apart is only valid if conditions held. Check the student
block across runs before trusting a cross-model comparison: if its median moves comparably
to the tutor differences you are claiming, those differences may be the network or the hour
of day rather than the models.

Measured on the seven-model sweep of 2026-08-17 (14:50→16:20): student TTFT p50 ranged
2.29–2.50s, a spread of **0.21s**, against a within-run p5–p95 spread of ~1.5s. Drift
between runs was far smaller than noise inside them, so the tutor spread over that sweep
(3.4s to 14.9s) is not an artifact of the environment changing.

## The latency subsample

Measuring all 520 moments per model would be needlessly expensive, so the probe measures a
subsample — and that subsample is **frozen**, committed as
`src/tutormoments/latency_probe_ids.json`.

This is what makes latency capable of being a **time series**. Selecting at run time would silently re-pick
the sample whenever the dataset changed, so a later measurement would be over different
prompts than an earlier one and the numbers would drift for reasons unrelated to the model.
A future release that is a *superset* of the current one still resolves every frozen id, so
old and new measurements stay comparable.

The list is resolved in three tiers:

1. **`latency_probe_ids.json` in the release directory** (`subsample_source: frozen_release`)
   — a dataset's own statement about itself outranks anything shipped with the code.
   `write_release` copies the list into every release so a downloaded release directory is
   self-describing.
2. **The list packaged with the runtime** (`frozen_packaged`). This is the path most runs
   take: the default config loads moments from the published Hugging Face dataset, where
   there is no local release directory to read from.
3. **Derivation** (`derived`) — first *n* in released order. A last resort that spans no
   particular length distribution and **is not comparable to anything**.

It lives in the runtime package rather than `tutormoments_build/` because of what it is.
`balanced_520_ids.json` is an *input to the build* — it tells the builder which moments to
include, and the runtime never needs it. This list is the opposite: an output the runtime
reads at measurement time, which the build merely relays.

Selection rule (in [`select_latency_subsample`](../tutormoments_build/latency_subsample.py): sort moments by context length, then stride-sample, endpoints included. Prompt length is the dominant driver
of TTFT — in the current release, context spans 984 to 55,681 characters, a 56× range — so
striding across the sorted distribution guarantees short, middling, and long prompts are all
represented.

Every `latency.json` records:

| Field | Meaning |
|---|---|
| `subsample_id` | short hash of the id list; a different hash means the samples are not comparable |
| `subsample_source` | `frozen_release`, `frozen_packaged`, or `derived` (**not** comparable to frozen runs) |
| `subsample_complete` | false when the release has dropped a frozen id, breaking the series |
| `missing_ids` | which ids were dropped |

## Measurement environment

Artificial Analysis pins itself to GCP `us-central1-a`. We run wherever you are, so rather
than pretending to a fixed environment we record the one we had:

```json
"measurement_environment": {
  "measured_at": "...", "concurrency": 1,
  "location": "gcp-us-central1", "tutormoments_version": "..."
}
```

Set `TUTORMOMENTS_LATENCY_LOCATION` to label it. **Figures are only comparable within one
measurement environment** — network distance to the provider is a first-order term in TTFT.

Cross-day drift is not captured; AA samples 8× daily, and we do not. Run the probe again on
another day and compare the two files if you need that.

## Interpreting a result

**A p50 gap is not evidence that one model is faster.** TTFT spread is wide — Opus 4.8 runs
p5 4.62 / p50 9.45 / p95 28.34 — so two close models can differ at p50 while being tied.

Compare them **paired by (moment, turn)** instead, which cancels the moment-to-moment
variation that dominates the raw spread. Opus 4.8 and Sonnet 4.6 differ by 0.55s at p50, but
paired, Opus is faster on 39 of 80 calls — 48.7%, a coin flip.

Sample size sets what you can resolve. The 95% CI half-width on a paired win rate near 50%
is `0.98 / sqrt(n)`:

| samples | 95% CI half-width | where this comes from |
|---|---|---|
| 112 | ±9.3 pts | cold figure, any `max_turns` |
| 224 | ±6.5 pts | warm figure at default `max_turns` 5 |
| 336 | ±5.3 pts | all samples pooled, default run |

So a default run separates models differing by roughly 7 points or more on the warm figure,
and places everything else in a tier. It cannot rank close neighbours, and no run option
changes that — the sample is already every conversation in the release. An earlier
40-moment run put Opus 4.8 and Sonnet 4.6 0.55s apart at p50 while paired they split
39/80: 48.7%, CI [37.8%, 59.7%].

Reporting one model as faster than another means checking the paired win rate, not the p50
gap, and treating an interval that straddles 50% as a tie.

Sanity checks that should hold on any live run:

- `ttfc ≤ ttft ≤ ttlt` on every sample.
- `n_no_visible_output` is 0.
- `cache_hit_rate` ≈ 0.5 at `--max-turns 3` on providers with a real session cache. That is
  structural (one miss then one hit per moment), not a quality signal — a rate far from it
  means the cache is behaving differently than assumed, in either direction.
- `ttfc` well below `ttft` on thinking models, and near-equal on the OpenAI path, where
  reasoning is not streamed.

**Do not expect warm to be much faster than cold.** Measured across the roster, the warm
TTFT gain is ~1–3.6s on Anthropic and *negative* on `gpt-5.5` (6.57 warm vs 6.39 cold).
Prompt caching skips prefill, and prefill is not the bottleneck: `ttfc` moves by ~0.04s
between cold and warm on Opus 4.8. On thinking models TTFT is dominated by thinking time by
roughly 7×. Caching remains a ~90% cost lever; it is close to irrelevant for latency.

One note when comparing against the June 2026 Preview paper: its figures use
`tutor_latencies`, which is end-to-end wall-clock seconds per call and is **unchanged** by
the streaming work — the streaming metrics are additive, on a separate `timing` field. Those
figures stay comparable across runs. They have never captured time to first token; that is
what this probe is for.