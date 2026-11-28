# Point the v2 research code at the release build

Status: phase 0 and phase 1 done (uncommitted, branch `feature/v2-research-on-release`), 2026-11-27. Follows the "Other work" item in `2026_11_11_v2_data_release.md`: make the first release, bring the research path into sync with it, and have every experiment record the build it used.

## Goal

The research scripts in `tutormoments_build/v2/` read the release build (`release-v2/builds/<build_id>/`) instead of loose files under `data/v2_annotations/source/` and `data/ground_truth/`, and every output they write records the build ID and the sha256 of the inputs it read. The first target is the classify path (`excerpts` → `classify_excerpts` → `evaluate_scorer`), so the classifier work can start on release data.

## Where things stand

- Release candidate: bucket build `20261124T2322Z-081a944`. The local build `20261124T2137Z-2b21ca7` (in `temp/release-local/localstore/`) is byte-identical to it, so nothing needs downloading. Its own manifest names a dirty tree, though, so outputs must cite the bucket build, not the local one (step 0).
- `origin/v2-replays` (about 9,200 lines, not yet merged) adds replays, personas, scoring and taxonomy on top of the same scripts. It reads data through `excerpts.DEFAULT_GROUND_TRUTH_DIR`, `excerpts.DEFAULT_TRANSCRIPTS`, `excerpts.load_ground_truth`, `classify_excerpts.SPLIT_STEMS` / `DEFAULT_SPLITS`, and imports from `build_ground_truth`. Renaming any of these would break that code without producing a git conflict, so phase 1 keeps these names and changes only their values.
- The release's `benchmark/{dev,test}.jsonl` rows equal the old `ground_truth_v2/{iteration,test}.jsonl` rows, except: `iteration` becomes `dev`; `annotators` + `annotator_roles` become one `annotators` array of `{annotator, role}`; `moment_status` and `moment_created_at` are dropped; `original_boundaries` is added. The classify path reads none of the changed fields.
- The latest scorer prompts (v8, v9) fill `{descriptions}` from `student_state.jsonl`, which is model-generated and not in the release.

## Phase 0: local data (no code)

1. Fetch only the bucket build's `manifest.json` into `temp/release/builds/20261124T2322Z-081a944/`, and hard-link the local build's files beside it (no copies, no 3 GB download). Check every file's sha256 against that manifest.
2. **Clean up superseded local data** (approved 2026-11-27). Delete the earlier local builds and snapshots, which the `20261124` ones supersede. Also delete the early provider A snapshot extract and the 2026-11-24 upstream comparison files (about 15 GB in all, under `temp/release-local/`). The per-provider review zips (about 2.7 GB) go only once their delivery is confirmed. Kept: the current local build and its snapshot, and the localstore inputs the next local release build needs. Afterwards, `release pull --prune` keeps the local release folder down to the current build.

## Staying on the current build

The risk with pinning to a build is that research quietly stays on an old one. Scripts therefore follow the build the code names, and moving to a new build is a one-line PR.

- **The pointer lives in the repo** (decided 2026-11-27), e.g. `tutormoments_build/v2/release.json`. It holds the build ID and its manifest's sha256. Moving to a new build is a PR that changes it, made once the build passes `verify`. Why the repo and not the bucket:
  - It needs no new command and no bucket writes.
  - The change is reviewed and recorded in git history.
  - A commit fixes its data. Code and the data shape it expects move together, and checking out an old commit gets that commit's build, the same way `splits.json` is pinned.

  Branches can point at different builds; merging `v2` brings the current one.
- **`tutormoments-build release pull`** reads the pointer and syncs that build into a gitignored local folder (`temp/release/builds/<build_id>/`). It checks the downloaded manifest against the pinned sha256.
  - Files whose sha256 matches one already held locally are hard-linked instead of downloaded. Builds mostly share files, so a new build costs only what changed.
  - `--prune` removes local builds other than the one pointed to, so old builds don't accumulate.
- **Research scripts read the build the pointer names.** If it isn't local, they fail and say to run `release pull`, so a script never silently uses a build other than the code's. That makes the network unnecessary at run time. After a `git pull` that moves the pointer, the next run says to pull the new build.
- The `TUTORMOMENTS_V2_RELEASE` environment variable overrides the pointer (no flag: every script's existing input paths already follow it), for trying a build before its PR. Outputs then record that build, and the run says it isn't the pinned one.
- Either way, every output records the build it actually read.

## Phase 1: the classify path on release data (PR into `v2` now)

Branch `feature/v2-research-on-release` off `v2`. The `v2-replays` PR will then conflict where both touched the same lines, and those conflicts get reviewed when it lands.

1. **One way to find the build.** Add a small module (e.g. `tutormoments_build/v2/release_data.py`):
   - Find the build folder: `TUTORMOMENTS_V2_RELEASE`, then the build named by the repo's pointer. Fail with a clear message pointing to `release pull` if it isn't local (see "Staying on the current build").
   - Load the manifest and work out each input's path (`benchmark/<split>.jsonl`, `transcripts/<source>.jsonl`).
   - Check the sha256 of each file a script reads.
   - Return a `release` provenance block: `build_id` plus `{path: sha256}` of the files read.
2. **Keep the names, change the values.** `excerpts.DEFAULT_GROUND_TRUTH_DIR`, `DEFAULT_TRANSCRIPTS` and `load_ground_truth` resolve against the build. `SPLIT_STEMS` becomes `("dev", "test")` and `DEFAULT_SPLITS` becomes `("dev",)`. The `v2-replays` code then follows without edits beyond the conflicts.
3. **`excerpts.py`** reads `benchmark/{dev,test}.jsonl` and `transcripts/<source>.jsonl` and stamps each excerpt record with the `release` block.
4. **`classify_excerpts.py`** carries the excerpt's `release` block onto every prediction. `student_state.jsonl` stays a research input with its own path. Its sha256 goes on the predictions, which already record where the facets came from.
5. **`evaluate_scorer.py`** reads gold labels from the build and reports the build ID. It fails if the predictions name a different build than the gold labels, unless `--allow-build-mismatch` is given.
6. **Old outputs.** `iteration` is retired (decided 2026-11-27): no alias in code. The README says to regenerate excerpts (free) and to rename old prediction files to `dev.jsonl`, or regenerate them.
7. **The pointer file and `release pull`** (see "Staying on the current build"). The pointer starts at `20261124T2322Z-081a944`. Tests run against `LocalStore`:
   - files with a sha256 already held locally are hard-linked;
   - a manifest that doesn't match the pinned sha256 is refused;
   - `--prune` keeps only the pointed build.
8. **README** (`tutormoments_build/v2/README.md`): replace the `aws s3 cp` instructions with `tutormoments-build release pull`, and explain `TUTORMOMENTS_V2_RELEASE`.

Tests, written first (red/green), in `tests/tutormoments_build/`:

- Resolving the build: env var beats pointer; a pointed build that isn't local fails with the `release pull` hint; a sha256 mismatch fails; an override is recorded as not the pinned build.
- An excerpt record carries the build ID and its input hashes.
- Splits are `dev`/`test`.
- A prediction carries its excerpt's provenance through.
- `evaluate_scorer` refuses mismatched builds.

Fixtures move to the release row shape.

Regression check, run locally before the PR:

- Build excerpts from the old inputs (snapshot `20261124T2137Z`: `source/` transcripts plus `synthetic/ground_truth_v2/`) and from the release build. Diff them: they should match apart from `split` and the new `release` block.
- Run `evaluate_scorer` on the same predictions against old and new gold. The scores should be identical.

### Phase 0 and 1 results (2026-11-27)

- **Bucket build in place.** Build `20261124T2322Z-081a944`'s manifest was fetched, and the local build's files were hard-linked beside it: all 11,263 match. Cleanup freed 16 GB.
- **`release pull` checked live** against the bucket: the manifest matched the pin and all 11,263 files were kept (7 s).
- **Regression check.** Excerpts rendered from the old inputs (snapshot GT + source transcripts) and from the release build are identical for all 1,271 moments, apart from `split` (`iteration` → `dev`) and the new `release` block. Gold labels are identical, so scores are unchanged.
- **Tests.** Full suite: 1,584 passed, 13 skipped.
- **Also changed.** `decompose_situation` now reads its moment list from the release build too, since it shares the split names.

## Phase 1b: student-state facets in the release (release PR, then a build)

Decided 2026-11-27: the facets ship, because the over-scaffolding prompts (v8, v9) use them for both human excerpts and replays. That makes them part of how the benchmark is scored, and the release must hold everything scoring needs. This follows the stage 4 rule: model-generated inputs go on the moments row with their model, prompt and settings.

Where things stand: `synthetic/ground_truth_v2/student_state.jsonl` is already a snapshot input (snapshot `20261124T2137Z`, sha256 checked). All 1,271 rows were made with claude-opus-5-5 (adaptive thinking, effort high) from `prompts/v2_helper/student_state.md`, with no errors or parse failures. The merged adjudicator account behind each row was recomputed from the snapshot's annotation exports (no LLM calls): it matches on all 1,271, so the facets are not stale.

1. **Schema first (benchmark-defining: needs your approval).** Add a field to `moments.schema.json` per row, roughly:
   - the facets;
   - the merged account they came from, and which adjudicators wrote it;
   - the merge method and similarity;
   - model, provider-native thinking parameters, and the prompt's text sha256 (its path has already moved once, on `v2-replays`).

   The research-only fields (`split`, `raw`, `usage`) stay out. Draft it with real rows in `temp/release-proposal/` for review.
2. **The prompt ships.** Its text goes in the release (e.g. `docs/prompts/student_state.md`), so the hash can be checked without the repo.
3. **Writer check.** The moments writer recomputes each merged account from the build's own annotations and fails on any mismatch, so a new annotation export can never ship with stale facets. A moment whose account changed needs its facets regenerated, which is an LLM step and stays outside the build.
4. **Local build, then a bucket build** (when asked), then a PR moving the pointer to it. Then `classify_excerpts` reads the facets from `moments.jsonl` instead of `student_state.jsonl`.

## Models used by the research steps

Checked 2026-11-27 from `default_config.yaml` on `v2` and `v2-replays`, the code defaults, and the frozen files. Every step resolves its model's parameters from config; none is hard-coded except `decompose_situation`'s default model id, whose parameters come from its benchmark arm.

| Step | Where set | `v2` | `v2-replays` |
|---|---|---|---|
| Student-state facets (`decompose_situation`) | `--model` default, parameters from the arm | claude-opus-5-5, adaptive, effort high; recorded on all 1,271 frozen rows | same |
| Action classifier, human excerpts (`classify_excerpts`) | `v2` block | claude-opus-5, adaptive, effort xhigh | claude-opus-5-5, adaptive, effort high (claude-opus-5 xhigh kept as a comparison entry) |
| Comparison classifiers | `v2.models`, else the tutor arm | gpt-5.6-sol (reasoning xhigh); gemini-3.5-flash via its arm | same; README examples use gemini-4-argon |
| Replay classifier (`classify_replays`) | `v2_replay.scorer` | — | gemini-4-argon, thinking_level high |
| Simulated student (`replays`) | `v2_replay.student` | — | gemini-4-argon, thinking_level low |
| Student personas (`traits`) | `v2_replay.traits` | — | gemini-4-argon, thinking_level high |
| Provider C `problem_text` (upstream, in the release) | provider pipeline | gemini-3-flash-preview (stated in the schema) | same |

Notes, kept as a record. This plan changes no model choices.

- **Replays and human excerpts are classified by different models by default.** On `v2-replays`, replays use gemini-4-argon (high) and human excerpts use claude-opus-5-5 (high). Which models the v2 scorer, the simulated student and the personas use is being settled by separate research (out of scope here).
- **What existing predictions were actually made with** can't be checked from this checkout: there are no `outputs/` here. Each prediction record stores its model and settings, so this gets checked when those files are regenerated or collected.

## Phase 2: rename `build_ground_truth` and read the released annotations (after `v2-replays` merges)

1. **Rename** `build_ground_truth.py` to `benchmark_labels.py`, after the release file `benchmark_labels.schema.json` it produces. Its output is `benchmark/dev.jsonl` and `test.jsonl` in the release shape: no more `data/ground_truth/`, no `iteration`.
2. **Read `annotations/<source>.jsonl`** from the build, not the raw exports. The released annotations already carry annotator labels, keep only the last save, and drop retracted moments. That makes `annotator_labels.json`, `excluded_annotators.json` and the "latest save" logic unnecessary in research code.
3. **Byte-identical acceptance test:** the script run on build `20261124T2322Z-081a944`'s annotations reproduces that build's `benchmark/dev.jsonl` and `test.jsonl`. This also meets the release plan's "update `build_ground_truth.py` before publishing" item.
4. **Remove the duplicate.** Have `tutormoments_build/release/benchmark.py` call `benchmark_labels` instead of keeping its own copy of the release shape. This is benchmark-defining code: the release build must stay byte-identical, so check it with a local build.
5. **Move the other annotation readers onto release annotations:** `splits.py`, `adjudicator_agreement.py`, `decompose_situation.py`. Regression check: the same reports and the same loaded descriptions as before. No LLM calls are rerun.
6. Update imports in the merged `v2-replays` modules (`evaluate_taxonomy.py` imports from `build_ground_truth`).

## Later (not in this plan)

- Replays starting from `benchmark/moments.jsonl` instead of rebuilding moments from labels plus transcripts. Replay strategies and simulated-student configurations are still being explored. Phase 1 only makes sure `replay_data` reads its transcripts and labels from the build, so switching later stays easy.
- Provider C slide text: `replay_data` (on `v2-replays`) reads `tutoring_provider_c_lesson_slides.jsonl`, which the release replaces with each session's `problems` (`problem_text`). Check that the two hold the same text before replays move onto the release.
- The release workflow decision (options A/B/C in the status doc). Pointing research at builds is a step towards option C. The decision is still needed before the second release.

## Open questions

1. **The facets field's name and shape** in `moments.jsonl` (phase 1b, step 1), for review with real rows.
