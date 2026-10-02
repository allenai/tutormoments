# TutorMoments project website

Project site for **TutorMoments-Preview: When Help is Unhelpful — Evaluating AI Tutors for
Productive Struggle**. Plain static site (no build step), served by nginx on
[Skiff2](https://skiff.allenai.org/getting-started) (Cloud Run) at
<https://tutormoments.allen.ai>.

## Local preview

Quick preview (no container):

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

Full-fidelity preview of the exact image Skiff2 deploys:

```sh
docker build -t tutormoments-web .
docker run --rm -p 8080:8080 tutormoments-web
# open http://localhost:8080
```

## Layout

- `index.html` — the whole page (Ai2-brand styling in `static/css/site.css`)
- `static/js/main.js` — renders the leaderboard table and the interactive charts from `static/data/*.json`
- `static/data/` — chart data, all checked in (`leaderboard.json`, `latency.json`, `action_distribution.json`, `cost.json`). If `action_distribution.json` or `cost.json` is absent, its section hides itself
- `static/paper/tutormoments-preview.pdf`, `static/animation/index.html` — published copies of the paper and pipeline animation
- `Dockerfile`, `nginx.conf` — the nginx image Skiff2 builds and runs (serves the static files on port 8080)

## Deployment

Deployed by [Skiff2](https://skiff.allenai.org/getting-started). The service is declared in the
repo-root `skiff2.json` (a single public nginx service, `cwd: ./website`). Two workflows drive it:

- `.github/workflows/plan.yml` — on a PR to `main` that touches `website/**`, `skiff2.json`,
  or the workflow, runs a terraform **plan** (no apply) as a dry run.
- `.github/workflows/deploy.yml` — on a push to `main` touching those same paths, builds the
  image, pushes it, and deploys to Cloud Run at <https://tutormoments.allen.ai>.

All asset, data, and animation references are relative, so the site works unchanged at the domain
root.

## Refreshing chart data

Results live in the benchmark's `results/` (gitignored) alongside the `analysis/` exports.
After running new models:

```sh
/path/to/tutormoments/.venv/bin/python scripts/refresh-data.py /path/to/tutormoments
```

Run it with an interpreter that can `import tutormoments` — the checkout's own venv is the
easy one. The TTFT figures come with publishability rules that live in
`tutormoments.latency`, and the script reads them out rather than restating them. Under a
bare `python3` it refreshes everything else and says that it skipped `ttft_s`.

This regenerates `static/data/{leaderboard,latency,action_distribution,cost}.json`. The
action-distribution figure reads the repo's
`analysis/working-paper-20260630/action_taxonomy_distribution.csv` export; pass
`--action-csv path/to/action_taxonomy_distribution.csv` to use a copy outside the checkout. Add
new models to the `MODELS` list in the script (plus `ACTION_CSV_MODELS`, and `MODEL_STYLE` in
`static/js/main.js`). Partial refreshes are fine — missing inputs just skip that JSON.

### The two latency figures

`static/data/latency.json` carries both, from different sources, and they are not
interchangeable:

- **`ttft_s`** — median time to first *visible* token, from `tutormoments latency`. That probe
  runs strictly serially, so this is the figure that is comparable across models, and it is
  what the chart's x-axis plots. `ttft_first_s` / `ttft_later_s` split it by turn position —
  the first message of a session against turns 3 and 5 — which the probe recorded itself, so
  every probed model gets the split regardless of what its provider reports about caching.
  A model with no probe run has no `ttft_s` key and is left off the chart rather than
  plotted at zero.
- **`ttlt_s`** — median time to last token from the same probe: when the student can actually
  reply. Shown in the tooltip as "Full turn, end to end".
- **`latency_s`** — end-to-end seconds per tutor turn from a benchmark run, which replays
  moments under `--concurrency`. Rate-limit tiers differ per model, so this compares a model
  against its own history but not against another model. It is kept in the JSON for
  correspondence with the paper's Figure 7 but is **not displayed** — the tooltip's
  end-to-end row is the probe's `ttlt_s`.

`ttft_s` values are read from probe runs under `<checkout>/results` (override with
`--probe-root`), taking the newest run per model that measured the frozen subsample in full.
The `ttft.subsample_id` recorded in the JSON is what makes the series auditable: a different
hash means different prompts were measured, and the script warns rather than charting two
samples together. See `docs/latency.md` in the main repo.

`latency_s` is still read off the paper's Figure 7 (±0.2s) for every model; a checkout with
`results/benchmark/` present replaces those with exact values. Since nothing on the page
renders it, that estimate no longer needs a footnote.

### The cost figure

`static/data/cost.json` plots the latency chart's models, at the same `score`, against
`uncached_cost_per_response_usd`: the tutor's list cost per response with every prompt token
priced at the full input rate (no prompt caching), output and reasoning at the output rate,
divided by the counted tutor calls. The chart shows it per 1,000 responses on a log axis,
with tokens per response and the list prices used in the tooltip. See `docs/cost.md` in the
main repo for why it is uncached.

Each row's `source` says where its tokens came from:

- **`"run"`** — the model's newest full scaffolding_rigor benchmark run under the results root
  (`summary.json`'s tutor tokens and call count, via `tutormoments.costing.costed_runs`). This
  is the normal case: any run made since usage-vector capture is costable, so a new model
  needs no extra spend to appear here.
- **`"probe"`** — a `tutormoments latency --mode scaffolding_rigor` probe, used only for models
  whose runs predate usage capture (via `tutormoments.latency.probe_cost_figures`).

Prices are the checkout's registry rates at refresh time, whichever the source, so re-running
this script after a price change updates the chart; `cost.pricing_version` and each row's
`rates.as_of` record which rates were used. A site model with neither source gets no row; it
is listed in `omitted` and counted in the footnote, never plotted at zero. The roster follows
`latency.json`, so a model must be in `MODELS` to appear. If new points' labels collide,
adjust the `labelLeft` / `labelBelow` maps in `renderCost`.
