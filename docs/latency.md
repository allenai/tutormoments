# Latency

Two numbers matter for a tutoring product.

**Time to first token (TTFT)** drives how responsive the tutor feels. Ed-tech partners
name this as important to maintaining student engagement.

**Time to last token (TTLT)** is when the student can actually reply. It bounds the pace of
the whole exchange.

**TTFT is the headline; TTLT is reported beside it.** Both are real, but they are not
interchangeable, and TTLT should not be the number a model is ranked on:

- TTLT = TTFT + the time to stream the answer out. That window is 0–12% of TTLT across the
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

Throughput (tokens per second) is measured by Artificial Analysis but is not reported here.
Good tutor turns are short, so high throughput only matters to the extent that it reduces
time to last token. It is recorded as `output_tps`, but never as a headline.

## Definitions

| Metric | Anchored at | Notes |
|---|---|---|
| `ttft_seconds` | first **visible answer** token | what the student sees appear |
| `ttlt_seconds` | last visible delta | when the student can reply |
| `ttfc_seconds` | first chunk of any kind, reasoning included | diagnostic; `ttft − ttfc` is roughly thinking time |
| `output_tps` | `output_tokens / (ttlt − ttfc)` | diagnostic only; window starts at `ttfc` because `output_tokens` counts thinking too |
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
see [Concurrency impacts latency](#concurrency-impacts-latency).

### Where the figures surface

Probe results live in their own run directory, which nothing else reads, so two consumers
join them in ([`probe_runs`](../src/tutormoments/latency.py)):

- **`tutormoments report`** — the leaderboard gains `ttft_p50`, `ttft_first_p50` and
  `ttft_later_p50`, joined onto each run summary by `(tutor_model, mode)`. A cell with no probe
  shows `-`; it never borrows the TTFT its benchmark run recorded under concurrency. The
  run-based `tutor_lat_p50` / `tutor_lat_p95` columns stay where they were, and are still
  end-to-end wall clock that compares a model against its own history rather than against
  another model.
- **The website's latency chart** — `website/scripts/refresh-data.py` reads the same figures
  into `static/data/latency.json` as `ttft_s` and `ttlt_s` plus the same first/later split
  (`ttft_first_s` / `ttft_later_s`), and plots TTFT on the x-axis with TTLT and the split in
  the tooltip. That script imports this module rather
  than restating its rules; an earlier version restated them and gated on cache hit *rate*,
  which is the one thing this code deliberately refuses to do.

Two rules keep the join from quietly comparing unlike things:

**Only frozen, complete subsamples are eligible.** A derived sample spans no particular
prompt-length distribution, and an incomplete one dropped ids, so neither is the same
measurement as the run before it. Among eligible probes for one cell the newest `measured_at`
wins, so re-measuring a model supersedes its earlier figure without anyone deleting the old
run directory — which is how the 40-moment pilot stopped being published the moment the
112-moment sweep landed beside it. Eligibility is per-probe, though, so two *different* frozen
samples would each pass it; both consumers therefore check that the selected probes agree on
`subsample_id` and warn when they do not.

**The published split keys on turn position, not cache state.** "First message" is the 112
turn-1 calls; "later" is the 224 on turns 3 and 5. The probe recorded every call's turn
itself, so this split is ground truth on all four providers and needs no publishability gate.
A cache-state split is the natural first idea and is wrong twice over: it does not exist on
Gemini (no cache tokens reported), and where automatic caching exists it mislabels — a "cold"
bucket holds every call whose cache missed, which on `gpt-5.5` included 27 later-turn calls
whose prefix silently failed to cache, diluting its "first message" figure from 7.53s to
6.91s, and on Together tracks run warmup rather than session position entirely. The
cache-state aggregates stay in `latency.json` (fidelity-gated, see
[caching fidelity](#caching-fidelity--the-biggest-caveat)) as the diagnostic for what
*caching* does; the turn split is what a student's first and later messages actually cost.

**The pooled figure is the ranking number.** One number per model, over an identical sample —
112 first + 224 later for every model in a full run. The mix is fixed by run structure, not by
provider behavior, so pooling is fair; the split is published beside it so nothing is hidden.

## Analysis

### The roster, measured

Seven models, `scaffolding_rigor`, the frozen 112-moment subsample
(`subsample_id` `589e8acf8ac761f2`) at the default `max_turns` of 5 — 336 tutor calls each,
2,352 in total. Measured 2026-08-18, 10:27–15:31 local, on one unlabelled machine
(`location` null): **these seconds are comparable to each other and to nothing else.**
Non-Anthropic lanes ran concurrently with each other; the two Anthropic tutors ran alone,
because every probe's simulated student is `claude-opus-4-6` on the same account.

| model | TTFT p50 | p5 / p95 | first message | later | TTLT p50 | thinking share of TTFT |
|---|---|---|---|---|---|---|
| `gpt-5.4-mini-2026-03-17` | **3.16** | 1.77 / 7.16 | 3.76 | 2.71 | 3.47 | not streamed |
| `gemini-3.5-flash` | **5.51** | 3.62 / 10.50 | 6.11 | 5.21 | 5.72 | 3.75s of 5.51 |
| `gpt-5.5-2026-04-23` | **6.38** | 3.40 / 9.65 | 7.53 | 5.89 | 7.28 | not streamed |
| `claude-sonnet-4-6` | **8.47** | 3.12 / 21.09 | 9.62 | 7.63 | 8.73 | 7.13s of 8.47 |
| `claude-opus-4-8` | **9.04** | 4.34 / 27.90 | 13.22 | 7.99 | 10.52 | 7.86s of 9.04 |
| `deepseek-ai/DeepSeek-V4-Pro` | **11.09** | 4.42 / 36.43 | 12.91 | 9.97 | 11.86 | 10.35s of 11.09 |
| `gemini-2.5-pro` | **14.08** | 9.13 / 24.34 | 14.74 | 13.68 | 14.15 | 11.59s of 14.08 |

TTFT p50 is pooled over all 336 calls, which is what the leaderboard and the website rank on.
"First message" / "later" split it **by turn position** — the 112 turn-1 calls against the 224
on turns 3 and 5. Turn position is the probe's own record, so the split exists for every
provider identically; it does not depend on what a provider reports about caching, which is
why Gemini and DeepSeek have figures here despite having no usable cache split (see
[caching fidelity](#caching-fidelity--the-biggest-caveat)). On Anthropic it coincides with
the cache split to a hundredth of a second (turn 1 is always the miss); on `gpt-5.5` it
corrects it — the cache-based "cold" was 6.91 because 27 later-turn calls silently failed to
cache and diluted the bucket, while the actual first-message p50 is 7.53. The thinking column
subtracts the `ttfc` p50 from the `ttft` p50; the
[decomposition below](#caching-does-not-systematically-improve-latency) takes the p50 of the
per-call difference instead, which is why its figures differ by a tenth of a second or so.

The split is also where the roster's other pattern shows: every model adapts its effort to
position. Even Gemini 2.5 Pro, with no session cache in this harness at all, is 1.06s faster
on later turns — pure thinking adaptation. DeepSeek's 2.94s and Opus 4.8's 5.22s gaps say the
first message of a session is where reasoning models spend their budget.

The order is not the score order: the two strongest tutors in the benchmark sit 8.5–9.0s from
first token, and the fastest model on the roster is the weakest. That trade-off is the finding
the website chart plots.

Sanity checks (see [interpreting a result](#interpreting-a-result)) all hold: `ttfc ≤ ttft ≤
ttlt` on every one of the 2,352 calls, no failed moments, and every frozen id resolved on
every model. One exception, on one call: `deepseek-ai/DeepSeek-V4-Pro` recorded
`n_no_visible_output` 1 of 336 (0.3%), so its percentiles are conditional on 335. That call
emitted 572 output tokens and never produced a visible one. The probe keeps timings rather
than text, so the mechanism is not directly observable here, but an unterminated
`<think>` block produces exactly this on the Together path: TTFT is held until the closing
tag, and a tag that never arrives leaves nothing ever counted as visible. In a benchmark run
that turn is recorded as `"..."` and scored as if the tutor said it, which is why this counter
exists — and why a non-zero value is worth chasing rather than rounding away.

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
and `claude-opus-4-8`, carrying answers of 72 and 157 characters. Median generation window
(TTFT → TTLT) across the roster:

| model | window p50 | share of TTLT |
|---|---|---|
| `claude-sonnet-4-6` | 0.00s | 0.0% |
| `gemini-2.5-pro` | 0.06s | 0.4% |
| `gemini-3.5-flash` | 0.15s | 2.6% |
| `gpt-5.4-mini-2026-03-17` | 0.30s | 8.6% |
| `deepseek-ai/DeepSeek-V4-Pro` | 0.69s | 5.8% |
| `gpt-5.5-2026-04-23` | 0.84s | 11.6% |
| `claude-opus-4-8` | 1.23s | 11.7% |

Sonnet 4.6's median window is **0.00s**: the whole visible answer lands inside one
millisecond-scale window, in one or two deltas.

The practical consequence: streaming buys a tutoring product almost no progressive
rendering. The student waits, then the message appears essentially at once. **TTFT is the
metric; TTLT is TTFT plus at most about a second; throughput is noise.** Streaming is still
required — it is the only way to observe TTFT at all — but not for the usual reason.

Two corollaries for reading `output_tps`. On models that stream reasoning, most
`output_tokens` are thinking tokens (Opus 4.8: ~500 of 551 on one measured call), so
tokens/sec largely describes reasoning speed rather than how fast text reaches the student.
On the OpenAI path it is not even that: reasoning is finished before `ttfc`, so the window
`ttlt − ttfc` covers only the visible text while `output_tokens` still counts the reasoning,
which is how `gpt-5.4-mini` posts a nominal 1,518 tokens/sec. Diagnostic only, as labelled.

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

They are reported separately, and the pooled figure is reported too — the pooled one is what
ranks the roster, because Gemini has no cache states to split on at all (see
[where the figures surface](#where-the-figures-surface)). The reason to keep the split beside
it is that pooling can drift with how many turns each conversation happened to run: `[END]`
truncates some conversations early, and a model that ends more of them shifts its own cold/warm
mix. Worth checking rather than assuming — on the 2026-08-18 sweep it did not happen at all.
Every model returned exactly 112 first turns and 224 later ones, so the pooled figures there
differ only in the models, not in their mix.

**Cache state is read off `cache_read_input_tokens`, never inferred from turn position.**
Turn index is a bad proxy for two reasons. The minimum cacheable prefix is model-dependent
and *not monotonic* across generations — 512 tokens on Opus 5, 1024 on Opus 4.8 / Sonnet 5 /
Sonnet 4.6, but 4096 on Opus 4.6 and Haiku 4.5 — and a prefix below the minimum fails to
cache **silently**. So the same short-transcript moment can cache on one roster model and
not another. And providers that report no cache tokens are recorded as `unknown` rather than
guessed at.

**Measured, the cold/warm gap is mostly not a caching effect.** Every model with a usable
split is faster warm — 1.0s on both OpenAI models, 2.0s on Sonnet 4.6, and 5.2s on Opus 4.8.
That last number looks like a strong case for caching until the gap is decomposed. Splitting
TTFT into prefill (`ttfc`, first chunk of any kind) and thinking (`ttft − ttfc`) at p50:

| model | | prefill | thinking | TTFT |
|---|---|---|---|---|
| `claude-opus-4-8` | first message | 1.18 | 11.75 | 13.22 |
| | later | 1.18 | 6.14 | 7.99 |
| `claude-sonnet-4-6` | first message | 1.76 | 7.27 | 9.62 |
| | later | 1.23 | 6.17 | 7.63 |

Caching can only touch prefill, and on Opus 4.8 prefill does not move *at all* — 1.18s cold,
1.18s warm, on a cache reading back a median 9,390 tokens. The entire 5.2s belongs to
thinking, and thinking falls monotonically with turn index (11.75s, then 6.96s, then 6.00s):
the model reasons hardest when it first meets the problem and less on each continuation.
Sonnet 4.6 does show a real prefill saving, and it is 0.53s.

So the split is honestly read as **"first message of a session" vs "a later message"** — a
position-in-conversation effect that caching contributes a fraction of a second to. That
reading is now literal: the split the leaderboard and website publish keys on turn position
directly, so it exists for every provider and measures the position effect wherever thinking
adapts — including Gemini 2.5 Pro, 1.06s faster on later turns with no session cache in this
harness at all. On the OpenAI path the decomposition is unavailable (reasoning is not
streamed, so `ttfc` ≈ `ttft`) and the whole ~1.0s gain lands in prefill by construction; that
is an upper bound on its caching effect, not a measurement of one.

None of this weakens the case for caching, which is a cost argument — a ~90% cut on the
dominant term. It just means **caching is not a latency lever**, and a product that wants a
faster first token has to spend elsewhere.

### Caching fidelity — the biggest caveat

**Only the Anthropic path sends a real cache breakpoint.** Gemini
([client.py](../src/tutormoments/client.py)) and Together concatenate the cacheable prefix
into the prompt instead, so neither caches *this conversation's* transcript.

That does not mean they report no cache hits. Together reports `cached_tokens` from its own
automatic prefix caching, and measured on `deepseek-ai/DeepSeek-V4-Pro` it returns a **0.786
hit rate** — *above* the 0.667 ceiling the run structure allows for a real session cache, which
is the tell. The hits are not what they look like:

| | hit/miss by turn | median tokens read back |
|---|---|---|
| Anthropic, explicit breakpoint | perfect `miss, hit, hit` per moment | **6,767–9,390** |
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
its warm figure would have vanished over one stray call. The roster bears this out from the
other side: `gpt-5.5-2026-04-23` lands at 0.586 against the 0.667 ceiling — 24 calls that
should have hit did not — while reading back 5,888 tokens a time, which is a real conversation
head by any measure.

Every latency block publishes `cache_hit_rate` and `cache_read_p50_on_hits` alongside the
numbers.

Note what the gate is and is not testing. It asks what the hits actually read back, not who
sent a breakpoint — so it is not a list of Anthropic models. Measured on the roster, the
OpenAI path passes it: `chat.completions` caches automatically and reads back a median of
5,888 tokens, which is a real conversation head rather than a shared block of system prompt.
Together fails it at 256 tokens, and Gemini reports nothing to judge. **So the warm column is
comparable within the Anthropic and OpenAI paths, and absent elsewhere.** Wiring an explicit
Gemini/Together cache remains follow-up work; the gate is what keeps its absence honest in the
meantime.

**What the gate guards — and what routed around it.** A withheld warm figure withholds the
cold one too: the gate's conditions establish that a provider's hit/miss labels mean session
warmth *at all*, and labels that cannot be trusted for the hits cannot be trusted for their
complement. Together is the concrete case: its misses are the calls its automatic prefix
cache happened not to serve, which cluster at run start rather than at session start. On the
roster that put its "cold" p50 at 12.86s and its "warm" at 10.86s — a 2.0s gap that reads
exactly like the session effect the Anthropic models show and is nothing of the kind. The 72
calls in that bucket are the ones Together's cache happened to miss, not the 112 first
messages.

This is why the *published* first/later split keys on turn position instead
(`probe_figures`): turn index is the probe's own record, needs no gate, and answers the
product question directly. The cache split this section describes remains in `latency.json`
and the probe's terminal summary as the caching diagnostic — the summary prints the cold
figure with a NOTE naming what was withheld and why, because there it is read by whoever just
ran the probe, not lifted into a table. Where both splits exist they nearly agree on
Anthropic (turn 1 ≡ miss, to a hundredth of a second) and diverge exactly where the cache
labels lie, which is itself a useful check.

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
figures are stamped `"source": "run"` with the concurrency they were gathered at, so a
reader can tell the two apart. Only probe figures are published — see
[where the figures surface](#where-the-figures-surface).

Conversations also share one `ModelClient` per model
([`get_client`](../src/tutormoments/client.py)). Previously each moment built its own,
paying a fresh TLS handshake on its first request — which landed squarely on the cold-cache
turn, inflating exactly the figure we report.

Batch mode records no latency at all: the batch APIs expose none. The probe requires a sync run.

### The student is a free drift control

Student turns are streamed and timed like tutor turns, and `latency.json` carries a
`student` block. Neither the probe's terminal summary nor the leaderboard ever shows it —
no human waits on a simulated student, so its latency is not a product metric. (A benchmark
run's `summary.json` does carry a `student_streamed` block; it is written, not displayed.)

It is useful for something else. **The student model is fixed across every run in a sweep**
(the config's `student.model`, not the tutor under test), so its latency is the same
measurement repeated under changing conditions — a control for whether the environment
drifted while the sweep ran.

Comparing tutors measured hours apart is only valid if conditions held. Check the student
block across runs before trusting a cross-model comparison: if its median moves comparably
to the tutor differences you are claiming, those differences may be the network or the hour
of day rather than the models.

Measured on the seven-model sweep of 2026-08-18 (10:27→15:31): student TTFT p50 ranged
2.44–2.76s, a spread of **0.32s**, against a within-run p5–p95 spread of ~1.5s and a tutor
spread of 3.2s to 14.1s. Drift was an order of magnitude smaller than the differences being
claimed, so the roster figures are not an artifact of the environment changing over five
hours.

The control also picked up the one thing it was pointed at. The three slowest student medians
(2.69–2.76s) are all runs from the first phase of the sweep, when three provider lanes ran
concurrently and each was making student calls against the same Anthropic account; the two
runs that had that account to themselves came in at 2.46–2.48s. That is contention showing up
exactly where the [concurrency](#concurrency-impacts-latency) argument says it should — and it
is the reason the two Anthropic *tutors* were measured alone rather than in a lane beside the
others.

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

Selection rule (in [`select_latency_subsample`](../tutormoments_build/latency_subsample.py)),
two constraints:

1. **One moment per source conversation.** Moments cut from the same conversation share a
   long transcript prefix, and a provider with automatic prefix caching serves the second
   one from cache — so it is not an independent measurement, and a sample labelled cold
   would not be cold. Measured on the 40-moment pilot: every `gpt-5.5` turn-1 cache hit
   came from a conversation contributing more than one moment, reading back 4.9k–9.0k
   tokens. This caps the sample at the release's 112 conversations.
2. **Match the release's context-length distribution.** Prompt length is the dominant
   driver of TTFT — context spans 984 to 55,681 characters, a 57× range — so each of 112
   quantiles of that distribution is assigned a distinct conversation, which contributes
   the moment nearest its quantile.

The first constraint fixes *how many* moments each conversation contributes, not *which*;
the second spends that freedom on coverage. The assignment is greedy from the most
constrained targets inward — an extreme target is reachable by only a few conversations, so
it is matched before a central target consumes the conversation holding the release's
longest moment — then refined by swapping any pair of assignments that lowers total error.
Ties break on (distance, length, id) throughout, so the result does not depend on the order
moments arrive in.

The committed sample spans the full 984–55,681 characters and tracks the population's
quantile curve to within 203 characters on average (worst 797, against a median prompt of
~15k).

> An earlier rule took each conversation's *median* moment. That satisfied constraint 1 but
> discarded every conversation's shortest and longest moment, so the sample covered only the
> 9th–98th percentile of prompt length — 5,223 to 37,695 characters — and never measured the
> tails of the axis TTFT depends on most. It was replaced before any figures were published
> against it; the two samples are not comparable, and `subsample_id` distinguishes them.

Every `latency.json` records:

| Field | Meaning |
|---|---|
| `subsample_id` | short hash of the id list; a different hash means the samples are not comparable |
| `subsample_source` | `frozen_release`, `frozen_packaged`, or `derived` (**not** comparable to frozen runs) |
| `subsample_complete` | false when the release has dropped a frozen id, breaking the series |
| `missing_ids` | which ids were dropped |
| `n_requested` | how many ids the list asked for, against `tutor.n_samples` actually measured |
| `failed_moments` | top-level, not in `subsample`: moments whose conversation raised and was skipped, so a partial run is visibly partial rather than quietly short |

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
p5 4.34 / p50 9.04 / p95 27.90 — so two close models can differ at p50 while being tied.

Compare them **paired by (moment, turn)** instead, which cancels the moment-to-moment
variation that dominates the raw spread.

Sample size sets what you can resolve. The 95% CI half-width on a paired win rate near 50%
is `0.98 / sqrt(n)`:

| samples | 95% CI half-width | where this comes from |
|---|---|---|
| 112 | ±9.3 pts | cold figure, any `max_turns` |
| 224 | ±6.5 pts | warm figure at default `max_turns` 5 |
| 336 | ±5.3 pts | all samples pooled, default run |

So a default run separates models differing by roughly 6 points or more, and places everything
else in a tier. It cannot rank arbitrarily close neighbours, and no run option changes that —
the sample is already every conversation in the release.

On the 2026-08-18 roster the full sample resolves every adjacent pair, including the two
closest:

| pair | p50 gap | paired win rate | verdict |
|---|---|---|---|
| `gemini-3.5-flash` vs `gpt-5.5` | 0.87s | flash faster on 57.1% ±5.3 | separated, barely |
| `claude-sonnet-4-6` vs `claude-opus-4-8` | 0.57s | Sonnet faster on 60.4% ±5.3 | separated |
| `gpt-5.4-mini` vs `gpt-5.5` | 3.22s | mini faster on 91.7% ±5.3 | separated |

The middle row is the one to keep in mind. A 0.57s p50 gap between two models whose p5–p95
spans are 18–24 seconds wide looks like noise, and on the 40-moment pilot it was: Opus came
out ahead on 48.7% of pairs, CI [37.8%, 59.7%], a coin flip. At 336 samples the same pairing
lands at 60.4% ±5.3 for Sonnet — the ranking was real, the pilot simply could not see it. The
first row is now the limiting case: 57.1% ±5.3 clears 50% with 1.8 points to spare, so a pair
this close is at the edge of what this benchmark can resolve, not comfortably inside it.

Reporting one model as faster than another means checking the paired win rate, not the p50
gap, and treating an interval that straddles 50% as a tie.

Sanity checks that should hold on any live run:

- `ttfc ≤ ttft ≤ ttlt` on every sample.
- `n_no_visible_output` is 0.
- `cache_hit_rate` at or below the structural ceiling — 0.5 at `--max-turns 3`, 0.67 at 5 —
  on providers with a real session cache. The ceiling is one miss then hits per moment, not
  a quality signal. Expect to land *under* it: a moment whose cacheable head falls below the
  model's minimum prefix fails to cache silently, and 32% of the subsample sits below the
  4,096-token minimum that `claude-opus-4-6` and `claude-haiku-4-5` impose (1% below the
  1,024-token one). So ~0.45 at `max_turns` 5 is normal on those two models and ~0.67 on the
  rest. A rate *above* the ceiling, or far below it on a model with a low minimum, means the
  cache is behaving differently than assumed. Observed on the roster: 0.667 and 0.661 on the
  two Anthropic models (the ceiling, as expected), 0.649 and 0.586 on OpenAI, and **0.786 on
  DeepSeek — above the ceiling**, which is the automatic-prefix-cache signature rather than a
  session cache.
- `ttfc` well below `ttft` on thinking models, and near-equal on the OpenAI path, where
  reasoning is not streamed. Reported in the same cache-state split as the headline
  metrics, so this reads straight off the block. Observed: thinking is 2.1× to 14.1× `ttfc` on
  the five models that stream it, and `ttft − ttfc` is 0.02s or less on both OpenAI models.

**Do not read the warm figure as the effect of caching.** Measured across the roster the warm
gain is 1.0–5.2s, but on Opus 4.8 — the largest of them — prefill (`ttfc`) is identical cold
and warm to within a millisecond, and the whole gap is the model thinking less on a
continuation than on a fresh problem. See
[caching does not systematically improve latency](#caching-does-not-systematically-improve-latency)
for the decomposition. Caching remains a ~90% cost lever; as a latency lever it is worth
fractions of a second.

One note when comparing against the June 2026 Preview paper: its figures use
`tutor_latencies`, which is end-to-end wall-clock seconds per call and is **unchanged** by
the streaming work — the streaming metrics are additive, on a separate `timing` field. Those
figures stay comparable across runs. They have never captured time to first token; that is
what this probe is for.