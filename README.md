# TutorMoments

TutorMoments is a benchmark for measuring how well language models tutor. It runs a
model-under-test against a simulated student on frozen tutoring **moments**
(expert teacher-annotated key moments cut from real tutoring transcripts), then scores
the generated continuation. 

The implemented metrics are:
- Appropriate Scaffolding: How often does the LM tutor introduce scaffolds that make content more accessible in moments when expert teachers think that scaffolding should occur, without over-scaffolding
- Appropriate Rigor: How often does the LM tutor push for rigor by increasing the cognitive demands of the task in moments when expert teachers think that pushes for rigor should occur (without over-scaffolding)
- Avoids Over-Scaffolding: How often does the LM avoid introducing more supports than needed given the student's state?
- Action Taxonomy: How do the tutor's move differ from human tutor moves?

## Installation

TutorMoments requires Python 3.11 or newer. Install from a checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the `tutormoments` command and the `tutormoments` Python package — the
benchmark runtime. Everything a run depends on ships inside the package:
`src/tutormoments/default_config.yaml` and the tutor/student/scorer prompts under
`src/tutormoments/prompts/`.

## Running

Provide an API key and run the benchmark; the dataset source defaults to the
released Hugging Face dataset (see `--dataset` below).

```bash
export ANTHROPIC_API_KEY=...   # or the provider's key

tutormoments run \
  --tutors claude-opus-4-8 \
  --modes plain scaffolding_rigor \
  --sample 10
```

Useful run options:

| Option | Purpose |
|---|---|
| `--tutors MODEL_ID ...` | Tutor model IDs or registered custom tutor names to evaluate. |
| `--modes MODE ...` | Tutor prompt modes. Defaults to `plain scaffolding_rigor`. |
| `--dataset HF_ID` | Hugging Face dataset id. Defaults to [`allenai/tutormoments-preview`](https://huggingface.co/datasets/allenai/tutormoments-preview) (`dataset.id` in config). |
| `--data_path DIR` | Run from a local data dir; wins over `--dataset`. |
| `--dataset-revision REV` | Pin a HF dataset revision for reproducibility. |
| `--sample N` | Use the first `N` moments from the dataset. |
| `--trials N` | Run each tutor/mode cell multiple times and summarize mean/spread. |
| `--max-turns N` | Maximum generated tutor/student turns per conversation. |
| `--log-level LEVEL` | Log verbosity: `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. |
| `--log-file FILE` | Optional combined log for the whole invocation (per-run logs are always written). |

Supported tutor prompt modes are:

| Mode | Description |
|---|---|
| `plain` | Baseline tutor prompt. |
| `scaffolding_rigor` | Tutor prompt that explicitly targets scaffolding and rigor. |
| `oracle` | Reference-aware tutor prompt that requires a moment's reference transcript. |

Each run writes under `results/`. Re-running skips moments whose transcript and
score are already present. Partial failures are recorded in `summary.json`; a
run where zero moments complete raises an error and does not produce a
leaderboard-ready empty summary.

Run output layout:

```text
results/<run_id>/
  config.json
  run.log
  transcripts/<scenario_id>.json
  scores/<scenario_id>.json
  summary.json
  taxonomy/                 action-taxonomy classification: classified.csv + sidecars
```

Each run's `config.json` records the dataset source (id, revision or local
path, config name, record count, and the record-level content hash), the
resolved config, package version, git commit, config hash, prompt hashes, and
the dataset manifest when running from a local release dir. A run is
reproducible from: pinned dataset revision + content hash and the resolved
config (up to LLM sampling nondeterminism).

## Latency

`run` records time-to-first-token and time-to-last-token on every transcript, but those
figures are gathered under `--concurrency` and are not comparable across models. For the
reportable number, run the serial probe:

```bash
tutormoments latency --tutor claude-sonnet-5 --mode scaffolding_rigor
```

This writes `results/<run_id>/latency.json`. `tutormoments report` then joins that onto the
leaderboard as `ttft_p50`, split into `ttft_first_p50` / `ttft_later_p50` (the first message
of a session against turns 3 and 5), matching on tutor model and prompt mode; a cell with no
probe run shows `-` rather than the concurrency-distorted figure from its benchmark run.

See [docs/latency.md](docs/latency.md) for what the metrics mean, why warm and cold cache
states are reported separately, and the caveats that apply before quoting a number.

## Reports And Viewers

Aggregate completed run summaries into leaderboard files:

```bash
tutormoments report --results-root results --out leaderboard
```

This writes `leaderboard.md` and `leaderboard.csv`. Columns are the paper's three metrics,
then TTFT from any serial [latency](#latency) probe in the same results root, then the run's
own end-to-end tutor latency and token totals.

Build a self-contained HTML viewer:

```bash
tutormoments view --results-root results --out viewer.html
```

The report and viewer commands read `summary.json` files from run directories.

## Running new tutor models

The `benchmark_models:` roster maps **arms** — benchmark conditions — not
bare models. Each entry names the provider model id, states the exact
provider-native reasoning parameters the run will send, and includes a
`condition:` label for grouping results. This keeps the benchmark config
auditable in provider parlance. The same model can run under several arms:

```yaml
benchmark_models:
  gemini-2.5-low: { model: gemini-2.5-pro, thinking_budget: 4096, condition: low }
  gemini-2.5-dyn: { model: gemini-2.5-pro, thinking_budget: -1, condition: dynamic }
  gpt-5.5-low:    { model: gpt-5.5-2026-04-23, reasoning: low, condition: low }
```

```bash
tutormoments run --tutors gemini-2.5-low gpt-5.5-low --data_path <release dir>
```

Running a model for the first time needs the provider API key exported and a
model id whose provider can be inferred. `src/tutormoments/models.yaml` still
holds stable per-model facts such as pricing and output caps; it no longer
hides tutor-arm reasoning parameters from the run config.

Tutors are selectable per-run via the CLI. The Student model is set in the config; changing the student model would change the shape of the evaluation. Modify your 
config to swap the student model with another API-callable model:

```yaml
student: { model: claude-opus-4-6, mode: oracle, thinking: none } # default
```

The student/scorer/taxonomy/groundtruth blocks still use `{model, thinking}`
with the canonical ladder because they are the fixed evaluation apparatus,
not the tutor conditions being reported. Keeping them inline means retuning a
tutor arm can never silently change the student or scoring regime.

## Custom Tutor / Student Models

You can register a different approach to the LM Tutor / Synthetic student,
for example a synthetic student without student-traits.

```python
from tutormoments import register_tutor, register_student

@register_tutor("my-tutor")
def my_tutor(conversation):
    return "next tutor turn"

@register_student("my-student")
def my_student(conversation):
    return "next student turn"
```

For CLI use, import the registration module before calling `tutormoments.cli.main()`
from a small wrapper script. There is not yet a plugin discovery mechanism.

Example wrapper:

```python
from tutormoments import register_tutor, register_student
from tutormoments.cli import main


@register_tutor("my-tutor")
def my_tutor(conversation):
    return "next tutor turn"


@register_student("my-student")
def my_student(conversation):
    return "next student turn"


if __name__ == "__main__":
    main()
```

Then run the wrapper with normal CLI arguments, for example:

```bash
python my_tutormoments_wrapper.py run --tutors my-tutor --sample 1 --data_path <release dir>
```

## The dataset

The released dataset carries two representations of the same underlying data:

- **Source configs** for data consumers — `transcripts` (full sessions),
  `annotations` (human SAR annotations), `ground_truth` (labeled key moments).
- **`moments`** for eval runners — the frozen, self-contained runnable set
  (520 moments: 260 scaffolding / 260 rigor). Each record carries the pre-cut
  transcript context, gold dimension, the real human continuation (for the
  oracle student), the frozen student trait (the exact persona the paper's
  simulated student ran with — generated from the pre-cut prefix only, so
  every benchmark run everywhere evaluates against the same students), and
  full provenance back to the source transcripts. It is built once by
  maintainers and published; the runtime never constructs or regenerates any
  of it.

Validate a local release directory:

```bash
tutormoments-build dataset validate --data_path data/balanced_520_release
```

The release ships `moments.jsonl` with `moments.manifest.json` and
`moments.schema.json` beside it (flat, like the other release files — so a
full release download is itself a valid `--data_path` dir). The manifest
records the set name, version, schema version, record count,
a record-level `content_hash` (comparable across the HF and local load
paths), a file-level `file_sha256`, creation date, and provenance.


## Scoring

TutorMoments scores generated continuations with model-backed annotation passes and
then aggregates per-moment scores into leaderboard metrics:

| Metric | summary.json source | Interpretation |
|---|---|---|
| Appropriate Scaffolding | `scaffold_calibrated.score` | Of scaffolding-appropriate moments, fraction where the tutor scaffolded without over-scaffolding. |
| Appropriate Rigor | `rigor_calibrated.score` | Of rigor-appropriate moments, fraction where the tutor pushed for rigor without over-scaffolding. |
| Avoids Over-Scaffolding | `1 - overscaffold.rate` | Fraction of all moments free of over-scaffolding. Computed at report time. |

These are the three metrics the paper reports; all are higher-is-better and
appear in the leaderboard as `appropriate_scaffolding`, `appropriate_rigor`,
and `avoids_overscaffold`.

## Action taxonomy

The action taxonomy classifies each decomposed tutor action into a 13-letter
scheme and produces the tables behind the paper's action-distribution figures.
The *data generation* lives in the runtime (`tutormoments.taxonomy` +
`tutormoments taxonomy`); the *figures* are rendered by the notebooks under
`analysis/working-paper-*`, which import `from tutormoments import taxonomy`.

**Every `tutormoments run` classifies its own LM-side actions** — alongside the
headline metrics — writing `results/<run_id>/taxonomy/classified.csv` and a
`taxonomy` block (raw category counts) into `summary.json`. This adds an LLM
classification pass per run (model set by the `taxonomy` config block, default
`claude-opus-4-8`); it is resume-safe and never fails the run.

The comparison figures plot each LM against a fixed **human reference**
distribution. That reference is the paper's frozen, published distribution —
[`analysis/working-paper-20260630/v1_action_taxonomy_distribution.csv`](analysis/working-paper-20260630/v1_action_taxonomy_distribution.csv)
— so you do **not** re-classify the ground truth; the figure notebooks load it
by default via `tutormoments.taxonomy.read_paper_distribution(...)`. Only the LM
side comes from your run. Rendering the figures needs pandas + matplotlib —
install the `analysis` extra:

```bash
pip install -e ".[analysis]"
```

Then point a figure notebook's `LM_CLASSIFIED` at your run's
`results/<run_id>/taxonomy/classified.csv` and run it — the human baseline is
already wired to the paper distribution.

Regenerating (optional): to rebuild the human reference from scratch, or to
produce the full headline tables from classified facets:

```bash
# Regenerate the human reference from the ground-truth bundle:
tutormoments taxonomy classify --kind key_moments --input key_moments.jsonl --output ./human

# Re-classify a completed run's LM side (normally emitted by the run itself):
tutormoments taxonomy classify --kind tutormoments --input results/<run_id>/ \
  --scenarios data/balanced_520_release/moments.jsonl --output ./lm

# Headline tables from classified facets (needs a classified human side):
tutormoments taxonomy headline --human ./human/classified.csv --lm ./lm/classified.csv --output ./headline
```

## Repository Layout

```text
.
├── pyproject.toml
├── configs/
│   └── local.example.yaml
├── src/tutormoments/            installable benchmark runtime
├── tutormoments_build/           maintainer-only dataset construction + release tooling
├── analysis/                paper notebooks, plots, taxonomy figures
├── docs/                    methodology docs (latency.md)
├── data/                    local datasets and release dirs, gitignored
├── results/                 run outputs, gitignored
└── tests/                   tutormoments/ (runtime), tutormoments_build/, analysis/
```

The core rule: code needed to run or score the benchmark lives in
`src/tutormoments/`; code that creates the dataset lives in `tutormoments_build/`;
code that explains or visualizes results lives in `analysis/`. Build and
analysis code may import the runtime; the runtime never imports them.

## Configuration

TutorMoments loads config in this order:

1. `tutormoments run --config path/to/config.yaml`
2. `TUTORMOMENTS_CONFIG`
3. `./config.yaml`
4. packaged `src/tutormoments/default_config.yaml`

Use `configs/local.example.yaml` as a starting point for local overrides.

Config files have these top-level sections:

| Key | Purpose |
|---|---|
| `dataset` | Released benchmark source: `id` (HF dataset), `revision` (pin), `config` (which config holds the runnable moments; default `moments`). |
| `providers` | Maps provider names to the environment variable that contains the API key. |
| `models` | Model roster for hosted tutor models. The provider is inferred from the model ID. |
| `student` | Hosted model and mode used for the simulated student. |
| `scorer` | Hosted model used for the scoring passes. |
| `defaults` | Default `trials` and `max_turns` values for `tutormoments run`. |
| `retry` | Retry behavior for direct model calls. |
| `batch` | Batch API polling timeout for scorer/provider paths that use batch APIs. |


## Logging

Both CLIs (`tutormoments` and `tutormoments-build`) log progress to the console at
INFO level, and every run automatically writes its own log file — nothing to
enable:

- `tutormoments run` writes `results/<run_id>/run.log` per cell, next to
  `config.json` and `summary.json`. The run directory is the complete record
  of the run: what ran, what was resumed, and which moments failed.
- `tutormoments-build dataset build` / `build-from-run` / `build-ground-truth`
  write a `build.log` into their output directory (skipped on `--dry-run`).

Log files append, so a resumed run continues the same log. In a multi-tutor
sweep, parallel lanes each write only to their own run's log, and every
console line is tagged with its cell (e.g. `[gpt-5-4/plain]`) so interleaved
lanes stay readable. Logs narrate the two phases of each run: Replay
(generating tutor/student continuations) and Classification (the 3-pass
batch scorer).

Two optional knobs: `--log-level DEBUG|INFO|WARNING|ERROR` controls
verbosity, and `--log-file FILE` additionally writes one combined log for the
whole invocation — useful for watching a sweep across lanes, or for the
subcommands without an output directory (`report`, `view`, `validate`).
`TUTORMOMENTS_LOG_LEVEL` / `TUTORMOMENTS_LOG_FILE` environment variables set defaults
for the flags.

## Development

Maintainer extras (not needed to run the benchmark):

```bash
pip install -e ".[dev]"                    # + pytest (runtime tests)
pip install -e ".[build,build-dev]"        # + dataset construction tooling
pip install -e ".[analysis]"               # + pandas/matplotlib for analysis figures
```

Run from a local directory

```
tutormoments run \
  --tutors claude-opus-4-8 \
  --modes plain scaffolding_rigor \
  --sample 10 \
  --data_path data/balanced_520_release
```

Run the runtime test suite without real API calls:

```bash
pytest tests/tutormoments -q          # runtime only (needs [dev])
pytest tests -q                   # full suite (needs [dev,build-dev,analysis]; missing extras skip)
```

### Live smoke checks

The test suite mocks every provider SDK, so it can never prove a provider
actually accepts our wire formats. `tutormoments smoke` is the live half: one
tiny real call per configured arm/role with its exact thinking condition
(costs cents), plus one submit-then-cancel batch per provider, with a
thinking-evidence column that catches silently-ignored knobs. Run it before
merging changes to core API logic (`client.py`, `models.py`, `models.yaml`,
`config.py`, `default_config.yaml`, `smoke.py`) — CI posts an advisory
reminder on such PRs but stays offline by design.

```bash
tutormoments smoke                          # everything in the active config
tutormoments smoke --arms gemini-2.5-pro    # one arm
tutormoments smoke --providers anthropic --no-batch
```

Exit codes: 0 = all passed, 1 = a check failed, 2 = config/environment error
(a missing API key for a selected provider is an error, not a silent skip).


### Dataset construction (maintainers)

Dataset construction lives in `tutormoments_build/` — outside the runtime package,
because changing it can change the benchmark itself. Install with the `build`
extra and use the `tutormoments-build` CLI:

```bash
# Raw human annotations -> per-conversation ground-truth JSON (LLM batch pipeline)
tutormoments-build dataset build-ground-truth --input <annotations.jsonl> --labeller hybrid

# Ground truth + transcripts + id list -> release dir (moments.jsonl + manifest)
tutormoments-build dataset build \
  --set balanced_520 \
  --ids tutormoments_build/balanced_520_ids.json \
  --ground-truth <ground-truth dir> \
  --transcripts <transcripts dir> \
  --tutoring-provider-a-jsonl <transcripts.jsonl> \
  --out data/balanced_520_release \
  --created 2026-07-01

# Rebuild the paper's exact 520 from the published reference run
# (the canonical record of the benchmark-time detections)
tutormoments-build dataset build-from-run \
  --set balanced_520 \
  --reference-run <benchmark_520_full_run.jsonl> \
  --tutoring-provider-a-jsonl <transcripts.jsonl> \
  --ids tutormoments_build/balanced_520_ids.json \
  --out data/balanced_520_release \
  --created 2026-07-01
```

`tutormoments_build/balanced_520_ids.json` is the canonical, frozen selection of
the paper's 520 moments (committed; deidentified UUID surrogates only). The
selection is not derivable — the sampler that chose it predates this repo —
so the id list plus the published reference run are the reproduction path.
