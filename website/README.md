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
- `static/js/main.js` — renders the leaderboard table and the two interactive charts from `static/data/*.json`
- `static/data/` — chart data, all checked in (`leaderboard.json`, `latency.json`, `action_distribution.json`). If `action_distribution.json` is ever removed, its section hides itself
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
python3 scripts/refresh-data.py /path/to/tutormoments/checkout
```

This regenerates `static/data/{leaderboard,latency,action_distribution}.json`. The
action-distribution figure reads the repo's
`analysis/working-paper-20260630/action_taxonomy_distribution.csv` export; pass
`--action-csv path/to/action_taxonomy_distribution.csv` to use a copy outside the checkout. Add
new models to the `MODELS` list in the script (plus `ACTION_CSV_MODELS`, and `MODEL_STYLE` in
`static/js/main.js`). Partial refreshes are fine — missing inputs just skip that JSON.

## TODOs when things go live

- `static/data/latency.json` — latencies are currently read off the paper's Figure 7 (±0.2s); the refresh script replaces them with exact values from a checkout with results
- `latency_s` is end-to-end seconds per tutor turn, taken from benchmark runs. `ttft_s` (time to first visible token) is separate: it appears only for models that have a `tutormoments latency` probe run, because benchmark runs measure latency under `--concurrency` and are not comparable across models. Models without a probe run — and models on providers with no real prompt cache — simply have no `ttft_s` key. See `docs/latency.md` in the main repo.
