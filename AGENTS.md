# AGENTS.md

TutorMoments is a language-model benchmarking framework. It replays real
human-student tutoring transcripts from expert-annotated cut points using LM
tutors and simulated students, scores the generated continuations, and reports
the results. See README.md for project goals, usage, and the full repository
layout.

## Layout

- `src/tutormoments/` — the installable benchmark runtime (`tutormoments` CLI): runs
  and scores a released dataset. Everything a run needs ships in the package.
- `tutormoments_build/` — maintainer-only dataset construction (`tutormoments-build`
  CLI): ground-truth building and release writing.
- `analysis/` — paper notebooks and plots (incl. the taxonomy figures). The
  taxonomy *data generation* lives in the runtime (`tutormoments.taxonomy`); these
  notebooks render its tables.
- `docs/` — methodology docs for runtime features, kept out of the README to
  stop it growing without bound (`latency.md`).
- `tests/` — `tutormoments/` (runtime), `tutormoments_build/`, `analysis/`.

Import rule: build and analysis code may import `tutormoments`; the runtime never
imports them.

## Invariants

- This is research code: reproducibility comes first. Doing it right beats
  doing it fast.
- Benchmark-defining code lives in `tutormoments_build/` — `moments_build.py`,
  `moments.schema.json`, and the frozen `balanced_520_ids.json` used in the 
  June 2026 Preview paper. These determine what the benchmark *is*. 
  Changes there can change published results; treat them with care.
- `src/tutormoments/latency_probe_ids.json` is frozen the same way
  `balanced_520_ids.json` is: committed and never regenerated — re-picking it
  breaks comparability with every prior latency measurement. It lives in the
  runtime package (not `tutormoments_build/`) because it is an *output* the
  runtime reads at measurement time, whereas `balanced_520_ids.json` is an
  *input* telling the builder which moments to include. Neither file has (or
  should get) a regeneration command; both stay auditable via tests that
  recompute them against the released dataset. The *selection rule* stays
  build-side in `tutormoments_build/latency_subsample.py`.
- The `src/tutormoments` runtime consumes released datasets only. It never constructs, filters, or regenerates benchmark data (including student traits — those are frozen in the release).
- `data/` and `results/` are gitignored and must stay that way. Never commit
  datasets, transcripts, or run outputs. The datasets are de-identified, but still take care to never commit anything containing student data.
- All LLM prompts live under consolidated `prompts/{my prompt}.md` directories as standalone markdown files, never inline in Python source. Templates are loaded from disk and filled at call time.
- Every LLM call path records token usage (input/output/total tokens) — this
  is the project's cost-tracking mechanism.
- The benchmark imposes **no output token cap**: `run_conversation` passes
  `max_tokens=0`, which the client resolves to the model's maximum. A cap a
  thinking model can exhaust before emitting visible text yields an empty
  response, which is recorded as `"..."` and scored as if the tutor said it.
  Never reintroduce a cap to save tokens.
- Never make choices about which language model to use or language model configuration for any API call without consulting the user; this is a language model benchmark, the choice of model matters
- Tutor benchmark arms live in `benchmark_models:` and state exact
  provider-native reasoning parameters in the config YAML, plus a `condition`
  label for grouping results. Reviewers should not have to reverse-engineer
  hidden ladder mappings to know what was run. The student/scorer/taxonomy/
  groundtruth role blocks still use canonical `thinking:` ladder levels
  (none/low/high/xhigh/dynamic) because they are fixed evaluation apparatus.
  See docs/thinking.md.
- `src/tutormoments/models.yaml` (the model registry) remains benchmark-
  defining for stable model facts and the legacy/role thinking ladder. Never
  edit it without consulting the user, and verify new or changed ladder rungs
  live with `tutormoments smoke` before relying on them.
- The offline suite can only prove the code agrees with itself. The live
  verification layer is `tutormoments smoke` (one tiny real call per
  configured arm/role plus a submit-then-cancel batch per provider): run it
  before merging any PR that touches core API wire-format paths
  (`src/tutormoments/client.py`, `models.py`, `models.yaml`, `config.py`,
  `default_config.yaml`, `smoke.py` -- keep this list in sync with the
  smoke-reminder job in .github/workflows/ci.yml). Smoke output goes to
  stdout / gitignored `results/smoke/`, never committed. CI stays offline: no
  API secrets in GitHub, ever (public repo, fork PRs).
- This repo is public. PR commits are public. Before pushing anything to remote, think about whether it could expose unnecessary information.

## Tests

```bash
pytest tests/tutormoments -q   # runtime only (needs the [dev] extra)
pytest tests -q            # full suite (needs [dev,build-dev,analysis]; missing extras skip)
```

The suite runs without real API calls. New features and bug fixes need
accompanying tests; test business logic, not boilerplate.
