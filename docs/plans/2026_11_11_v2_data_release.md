# v2 data release: implementation status

Status as of 2026-11-24. The release layout and principles live in the internal
release plan doc; this file tracks implementation. Code: `tutormoments_build/release/`,
CLI `tutormoments-build release snapshot|build|verify`.

## How we work

- Schema first: each release file's schema is drafted with real rows converted to
  it, reviewed and approved before its writer is built. Drafts for review go in the
  gitignored `temp/release-proposal/` (schemas, diffs vs the bucket, README).
- Everything runs in the de-identified bucket, never the PII bucket: frozen input
  snapshots in `release-v2/snapshots/<id>/`, builds in `release-v2/builds/<build_id>/`
  (`<UTC minute>-<commit>`). The bucket is passed via `--root` /
  `TUTORMOMENTS_RELEASE_ROOT` and never written in code or docs.
- Publish target: a new Hugging Face dataset repo (name TBD), screenshots included.
- Bucket builds run only when asked: one build per real change. Writers are checked first with a local build (`LocalStore`) on the snapshot's inputs, sha256-checked against the snapshot.
- PRs: stages 0–4 were a stack (#97 ← #98 ← #99 ← #100), merged into `v2` 2026-11-12/13. Later stages branch from `v2` and PR into it (stage 5: #102).

## Stages

| # | Stage | Status | Open |
|---|---|---|---|
| 0 | Plumbing: snapshot, build, checks, manifest, verify | Done; merged into `v2` (#97) | `verify` needs retries on long S3 reads if it times out again |
| 1 | `transcripts/<provider>.jsonl` | Schemas approved 2026-11-04; writer done; first real build 2026-11-11 (below) | A's `screenshots` field (2026-11-19 delivery) added to the release schema (#109); in build `20261123T2330Z-4091066` |
| 2 | `annotations/<provider>.jsonl` | Schema, writer and guide merged into `v2` (#98); first real build 2026-11-11 (below) | Review of the build by the annotation team. The `transcript_selection` change (2026-11-11) is not in that build |
| 3 | `benchmark/dev.jsonl`, `test.jsonl` | Schema approved 2026-11-11; schema and writer merged into `v2` (#99) | None |
| 4 | `benchmark/moments.jsonl` | Schema approved 2026-11-11; schema and writer merged into `v2` (#100) | Open research questions (persona, continuation length) are runtime choices; the runtime needs a v2 loader and a wait-time tool |
| 5 | `assessments/<provider>.jsonl`, `reference/eedi_*.jsonl` | Schemas approved 2026-11-12; schemas and writer merged into `v2` (#102) | Asked Eedi: feedback text on unpublished questions, and the reference tables' licence. Duplicated provider A answers fixed upstream 2026-11-13; the writer now derives `batch` (first delivery) from `batches` (#109) |
| 6 | `screenshots/tutoring_provider_a/` | De-identified set delivered 2026-11-19 (10,118 frames); snapshot input, writer and checks merged into `v2` (#109); in build `20261123T2330Z-4091066` | None |
| 7 | `manifest.json` schema, README dataset card | Not started (builds already write a manifest) | Needs the Hugging Face repo name and the dataset's licence (which depends on Eedi's answer if the reference tables ship). Card carries A's two dropped fields, the surrogate-name caveat, and stage 5's notes (Eedi attribution, what the quiz data can't be joined to) |
| — | Publish | Not started | HF repo, full build, verify, dry-run push |

## Decisions so far

- **Transcripts** ship as the bucket has them, validated against each provider's
  reviewed schema plus the shared `transcript_core.schema.json`. Release schemas
  differ from the bucket's only in public-facing text and these declared changes,
  which every build re-checks (`transcripts.py`):
  - A drops `session.prior_session_count` (one batch only) and
    `eedi_content_link[].type` (always `quiz_session`; the build fails on any other value).
  - C pins `batch` to `"2026-07-06"` (single delivery, no more coming).
- **Annotators** are released as their `A01`–`A21` labels: one per person, across
  providers. The label map (`v2_annotations/annotator_labels.json` in the bucket,
  a snapshot input) covers every annotator in the exports. Staff are now excluded
  upstream, so no exclusion list is needed.
- **C programs**: assessments now carry `program_id` (fixed upstream 2026-11-04);
  pre/post-program assessments join on it, not `student_id`. Verified: 7,032 of
  7,110 programs have sessions; the other 78 are programs whose sessions were all
  excluded.
- **Annotations** ship as one schema for all providers (`release/schemas/annotations.schema.json`), one row per moment or per "no key moments" record, with every annotator's entry on it. Approved 2026-11-11:
  - `purpose` is `rigor_scaffolding` and `batch` is `v2`; later kinds of annotation or passes become new values.
  - Annotator names are stripped and IDs replaced by labels; the writer fails on any unmapped ID.
  - Each entry's answers ship exactly as exported, so the ground-truth build can be rerun from the release.
  - Dropped: retracted moments (A 3, C 1); earlier saves of the same role by the same annotator (the last save is kept, as `build_ground_truth.py` does) and `revision`; the transcript fingerprints (`selection_source_sha256`, `corpus_revision`), which match no released transcript file; the tool-computed turn counts (`dialogue_turns` and variants), wrong on 17 early A moments; A's flat `enrichment_numbers` (always the events listed in `transcript_selection`; dropped 2026-11-11).
  - Kept: thrown-out moments with their status, and the v1 rigor hint fields, so researchers can account for annotators being shown hints.
  - A's `transcript_selection` ships (reversed 2026-11-11): it is the only record of what each annotator saw. The tool showed some screen events as part of the turn before them instead of on their own row, varying by tool version (none before 2.10) and annotator, so an event's `before_turn` does not say which side of the cut the annotator saw it on. The tool's one-row `is_cut` flag becomes `in_situation` on every row (true up to and including the cut row). Found while drafting stage 4: in the moments' after-cut screen events, 85 of 333 were shown before the cut.
  - No summary fields (outcome, agreement): resolved labels belong in `benchmark/`.
  - Field descriptions follow the annotators' guiding sheet.
- **Annotation guide** ships as `docs/annotation_guide.md` (`release/docs/annotation_guide.md`): the annotators' guiding sheet with its comments, feedback tab, images (crossed out in the source), internal notes and author line removed; the guidance text is otherwise word for word.
- **Benchmark labels** (`benchmark/dev.jsonl`, `test.jsonl`) ship as one schema for both files (`release/schemas/benchmark_labels.schema.json`): the `build_ground_truth.py` records, one row per moment. Approved 2026-11-11:
  - `iteration` is renamed `dev` (file name and `split` value). Hugging Face reads `dev` as the validation split and doesn't know `iteration`: with `datasets` 5.1, loading a folder holding `iteration.jsonl` and `test.jsonl` silently returns only `test`. `dev`/`test` is also the usual vocabulary for researchers.
  - Labels are rebuilt from the snapshot's annotation exports, not copied from `synthetic/ground_truth_v2/`. Checked on snapshot `20261111T1636Z`: the rebuild, and a rebuild from the release-format annotations, both reproduce the bucket's files byte for byte, so the released annotations hold everything the labels depend on (see the writer section for what the script still needs before researchers can rerun it on them).
  - `splits.json` (transcript -> dev/test assignment, append-only) stays in the repo and is read from the code, pinned by the commit in each build's manifest; it is not snapshotted or shipped (decided 2026-11-11).
  - `annotators` and `annotator_roles` (parallel arrays) become one array of `{annotator, role}`, so a consumer can't separate a role from its annotator. Order is the tool's (two adjudicators: first to save first); it affects neither labels nor boundaries (checked on all 363 two-adjudicator moments).
  - Dropped: `moment_status` (always `adjudicated`) and `moment_created_at` (on the annotations row). `original_boundaries` is on every row, null where the selector's boundaries stand.
  - Known: 4 provider A moments have no turn after the cut (`cut_turn == end_turn`). They pass the current rules and ship unless a rule changes (benchmark-defining: needs sign-off).
- **Benchmark moments** (`benchmark/moments.jsonl`, `release/schemas/moments.schema.json`): one row per labelled moment, holding everything needed to replay it. Approved 2026-11-11:
  - A new v2 shape; v1's `moments.jsonl`, schema and runtime loader are untouched. No v1 `dimension`: the gold is `situation_type` (all five values) and `situation_votes`, the benchmark's target. The human tutor's action labels are the classifier's gold and stay only in `dev.jsonl` / `test.jsonl`.
  - Not frozen, versioned: the manifest gives the file's sha256 and runs should record it. Only the build a publication uses is pinned and signed off.
  - Self-contained: everything needed to run the benchmark is on the row, so nothing model-generated ships separately. The student persona and the student-state facets (`student_state.jsonl`) are not on it: the persona is an open research question and the facets are no longer used. Either goes on the row, with its model, prompt and settings, if the analysis adopts it.
  - `transcript` is the whole session, each turn and screen event copied unchanged from `transcripts/<source>.jsonl` (timing kept: wait time matters), events placed by `before_turn`. The code decides where to cut and how much continuation the simulated student gets.
  - `annotator_view` (provider A) is what the annotator whose boundaries the moment uses actually saw: their `transcript_selection`, with the cut where they drew it. Both placements ship so the code can compare them. Annotators could cut after a screen-event row, and `cut_turn` is then the tool's conversion to a turn number, one turn off in either direction: of the 797 provider A moments, 49 name the turn before the cut row, 22 the turn after it (so they include the tutor's decision turn), and 3 have screen events only.
  - Provider C rows carry the session's `problems`. Annotators saw the slide images, which cannot be released; `problem_text` stands in, and `learning_objective.year` is the year annotators saw.
  - Cuts stay where annotators drew them, with no v1-style role adjustment. In 245 moments (19%) the student speaks next, mostly because the tutor waited; the replay tutor will get a wait-time tool.

- **Assessments and Eedi reference tables** (`assessments/<provider>.jsonl`, `reference/eedi_*.jsonl`): the bucket files plus declared changes, re-checked every build as for transcripts. Release schemas are `assessments_<provider>.schema.json` (so they don't collide with the transcript schemas) and the bucket's `eedi_*.schema.json` names. Approved 2026-11-11/12:
  - Every student's assessments ship, not only annotated students'. The transcripts ship in full, and the annotated provider A sessions are the least connected to the quiz data (below).
  - Provider C ships only rows that join a released transcript: program assessments by `program_id`, exit cards (`post_session`) by `session_id`. This drops the assessments of the 490 sessions the transcript filters exclude (78 programs with no transcript at all, and 406 exit cards in other programs): 31,220 -> 30,574 rows. `batch` is pinned to `"2026-07-06"`.
  - Provider C's question HTML ships as authored, including images from the third-party LaTeX renderer `latex.codecogs.com`; each such image's `alt` holds its LaTeX, and the schema says rendering fetches from codecogs.
  - Provider A drops `batches` (the deliveries a quiz session appeared in) and keeps `batch` as the first delivery. Up to the 2026-11-11 snapshot the bucket's `batch` was always the earliest of `batches`, though its schema says the latest. The 2026-11-13 quiz file made it the latest (37,764 rows now list more than one delivery). Decided 2026-11-20: keep the release's meaning, so the writer sets `batch` to the earliest of `batches` rather than checking it. An empty `date_completed` (8,866 rows from the 2026-11-13 file, 9,340 before) becomes null; the bucket schema says `format: date`, which the build checks.
  - Provider A's `question_url` (a public image of the whole question on Eedi's host) is null unless the question is in `reference/eedi_diagnostic_questions.jsonl`, so the release links only questions Eedi has published, and null on unanswered entries (where it is the bare host). The reference table covers 745 of the 5,089 questions in the quizzes (16% of answers, 2026-11-13 file), and Eedi won't supply more. Eedi's feedback text (`explanation`) ships on every answer for now; it often restates the question, and we have asked Eedi whether that is acceptable for unpublished questions.
  - The reference tables ship unchanged. `eedi_diagnostic_questions` holds 988 questions in 1,971 versions (unique on `intervention_id`); a quiz answer names the question, not the version. Their licence is being confirmed with Eedi: the NeurIPS 2020 data is CC BY-NC-ND 4.0, and the tables merge it with the NeurIPS 2022 and Kaggle Mining Misconceptions data. If Eedi says no, they come out of the release and the card points to Eedi's downloads.
  - Known limitation, for the card: all annotated provider A transcripts are from the first delivery (2025-10-16), which has no session dates and no quiz links, so an annotated session can't be placed among its student's quizzes. The quiz data joins only by student.
  - Fixed upstream 2026-11-13: in 9,350 provider A quiz sessions, answers repeated exactly apart from `question_sequence` (60,202 of 299,614 entries). The repeats are now removed upstream (245,082 entries, none repeated) and `question_sequence` re-derived on those quizzes; the bucket doc describes this, so no card caveat is needed.

## Open decision: release workflow and provenance

Raised 2026-11-11, not yet decided. In v1 we lost track of which branch generated which data, what was run against it, and whether local code that changed bucket data was ever committed. v2 builds are traceable (snapshot, commit and per-file sha256 in each manifest), but the workflow around them is not settled:

- **Two paths produce benchmark data from the same annotation export.** The research path (the team's working `build_ground_truth.py`, `excerpts.py` and `decompose_situation.py`, run locally, writing `synthetic/ground_truth_v2/` and local `data/`) records no provenance. The release path (`release-v2/builds/<build_id>/`) does. They agree today, but the research code is changing and the release deliberately runs the merged version, so they will drift.
- **Builds can fork.** Builds sit side by side and nothing says which is current, so people working from different branches can each build and work against different releases without noticing.
- **Known gaps in the build command:** untracked files do not count as uncommitted changes (`code_state` uses `--untracked-files=no`), so a never-committed module can run in a build whose manifest names a commit without it; nothing checks the recorded commit is on GitHub; and rebasing the stacked PRs rewrites commits a build may name.

Options discussed:

- **A. The release is the only source.** Everyone reads builds; a rule change is a PR, then a build. One truth; slower research iteration.
- **B. Two paths with explicit roles.** Research iterates freely; anything shared or published (paper, website, Hugging Face) cites a release build made from merged code.
- **C. Research starts from a release build.** Research scripts read a pinned build's transcripts and annotations and record its ID in what they derive; rule changes stay local until merged and released.

Candidate mechanics for any option: canonical builds only from the integration branch (`v2`, later `main`), with branch builds marked as previews; one pointer (`release-v2/current.json`) naming the current build; a `release status` command saying whether a checkout contains the current build's commit; a pushed git tag per bucket build so its commit survives rebases; untracked files counted as uncommitted; `--allow-dirty` for local stores only.

Plan for now (2026-11-11): finish the stages, make one release, bring the research and release paths into sync on it, then log the release (build) ID with every experiment run so each result records exactly what data it used. Decide the workflow before the second release.

## Annotations writer (stage 2)

`release/annotations.py` builds `annotations/<provider>.jsonl` from the frozen `v2_annotations/source/<provider>_annotations.jsonl` (the nightly export) and `annotator_labels.json`. It fails on an annotator with no label and on any export field it does not list as kept or dropped, so upstream additions are reviewed before they ship. Builds also copy the packaged docs (`release/docs/`) to `docs/`.

## Benchmark labels writer (stage 3)

`release/benchmark.py` builds `benchmark/dev.jsonl` and `test.jsonl` by running `v2/build_ground_truth.py` (unchanged) on the snapshot's annotation exports with the committed `splits.json`, and writes them in the release shape (`release_record`, in the writer). Nothing is copied from `synthetic/ground_truth_v2/`. It then rebuilds them from the `annotations/` files the annotations writer just wrote, reshaped back to the export's form, and fails the build unless both runs give the same bytes, so it runs after `write_annotations`.

`build_ground_truth.py` is not changed for the release while the team's work on it is in progress (decided 2026-11-11): finish the data release first, then update the code. Before publishing, the code has to catch up with the schema's description, which says running the script on the released annotation files reproduces these files. Today it doesn't read the release format or write the release shape.

Checked 2026-11-11 without writing to the bucket: the annotations and benchmark writers run on a local copy of snapshot `20261111T1636Z`'s annotation inputs (sha256-checked against the snapshot), with #99's writer code (re-run on `b0ebd3e`, after the rebase onto the `transcript_selection` change), pass every build check and give dev 640 (A 401, C 239) and test 631 (A 396, C 235), with the same labels, boundaries and annotators as the working ground truth. The first bucket build with benchmark files is `20261123T2330Z-4091066` (below).

## Moments writer (stage 4)

`release/moments.py` builds `benchmark/moments.jsonl` only from files the same build has already written (`transcripts/`, `annotations/`, `benchmark/dev.jsonl` and `test.jsonl`), so it runs last and the release can always rebuild it. Rows follow the labels files, dev first. The view is the `transcript_selection` of the annotator whose boundaries the moment uses (the first saved where two match); where the moment's cut is not that annotator's own (two adjudicators drew different spans and the majority cut won), the cut goes after the last row holding `cut_turn`. The build fails if a view names a turn or screen event the transcript lacks, or if the schema's copies of the provider definitions drift from the provider schemas.

Checked 2026-11-11 without writing to the bucket: a full local build (transcripts, annotations, labels, moments) on snapshot `20261111T1636Z`'s inputs (sha256-checked against the snapshot) passes every check and `verify`, and gives 1,271 moments (110 MB): dev A 401 / C 239, test A 396 / C 235. All 797 provider A views are the boundaries' annotator's; 723 cut after the `cut_turn` row and 74 after a screen-event row.

## Assessments writer (stage 5)

`release/assessments.py` writes `assessments/<provider>.jsonl` from the snapshot's `source/tutoring_provider_a_eedi_quiz_sessions.jsonl` and `source/tutoring_provider_c_assessments.jsonl`, applying the declared changes above, and copies the four `source/eedi_*.jsonl` tables to `reference/` unchanged. Each bucket schema is drift-checked against its release schema with the declared changes applied (the check is shared with the transcripts writer). Provider A's question links are checked against the build's own reference questions; provider C's rows against the transcripts this build wrote, so the writer runs right after `write_transcripts`.

Checked 2026-11-12 without writing to the bucket: a local build of this writer on snapshot `20261111T1636Z`'s inputs (sha256-checked against the snapshot; the C transcripts staged as the transcripts writer would write them) passes every check and `verify`: provider A 50,489 rows (137 MB), provider C 30,574 (197 MB: 7,032 programs' pre- and post-program assessments and 16,510 exit cards, one per released transcript), and the reference tables (3,277 constructs, 1,971 question versions, 2,587 misconceptions, 755 subjects). 44,468 question links remain: the 44,814 answers on published questions less 346 unanswered.

## Processing the 2026-11-19 delivery

Branch `feature/v2-screenshots` from `v2`. Upstream changed six snapshot inputs since `20261111T1636Z`: provider A transcripts (2026-11-19) and quiz sessions (2026-11-13), with their schemas and docs. Annotation exports, all provider C files and the Eedi tables are byte-identical. The bucket also has a new `screenshots/tutoring_provider_a/` prefix.

What changed, checked 2026-11-20 against the 11-11 snapshot:

- Transcripts A: same 40,176 transcripts. New required `screenshots` array (`start_seconds`, `path`) on every row; 4,376 transcripts have frames, 10,118 in all, each path an existing object and no object unlisted. Separately, `eedi_content_link` lost exact duplicate links on 3,441 transcripts, gained links on 344 and lost one on 1; all 48,324 links resolve to a quiz session. No annotated transcript has screenshots or link changes (all annotated A transcripts are from the 2025-10-16 delivery; frames are from 2026-02-13 to 2026-06-24), so turns, screen events and the moments they feed are unchanged.
- Quiz sessions A: same 50,489 sessions. Repeated answers removed (above). `batch` is now the latest delivery and `batches` lists far more deliveries. Some values now appear to come from a later delivery: `actual_score` 601 rows, `predicted_score` 344, `lesson_length` 376, `date_completed` filled on 474. 1,020 sessions' `questions` changed beyond the repeat removal: upstream explained (2026-11-23) that an earlier single-delivery rebuild had dropped answers, now restored on 987 quizzes, and that completion fields now come from the finished lesson record. The other 33 swap only `explanation` or `question_url` text on some answers. Total answers 299,614 -> 245,082, matching upstream's figures. Schema change is description-only (`question_sequence`, `answer_sequence`), so drift checks pass.

A local build at `v2` (`1628bd5`) on a local snapshot of these inputs (`temp/release-local/`) fails in two places, both build guards working as intended: the provider A transcript drift check (`screenshots` not in the release schema) and the quiz `batch` check (37,764 rows). With both patched it passes every check, with unchanged counts: dev 640 / test 631, 1,271 moments, annotations A 1,261 / C 661, assessments A 50,489 / C 30,574.

Plan, red/green for each code change:

1. **Transcripts schema (stage 1).** Add `$defs.Screenshot` and required `screenshots` to `release/schemas/tutoring_provider_a.schema.json`: rules copied from the bucket, description in release wording (paths resolve against the release root; the bucket name never appears). For approval before merging. Tests: drift fixture includes `screenshots`; a row without it fails.
2. **Quiz `batch` (stage 5).** In `assessments.py`, set `batch` to the earliest of `batches` before dropping it, in place of the check. The release schema's `batch` rules don't change; its description stays "first arrived in". Update the `question_sequence` and `answer_sequence` descriptions to upstream's corrected meaning (re-derived numbering on de-duplicated quizzes; `answer_sequence` 0 = shown, not answered). Tests: replace `test_a_batch_that_is_not_the_first_delivery_fails` with one where a latest-delivery `batch` ships as the earliest.
3. **Screenshots (stage 6).** Add `screenshots/tutoring_provider_a/` to `SNAPSHOT_INPUTS` (about 1 GB). New writer `write_screenshots`, run after `write_transcripts`: server-side copy of every frame the released transcripts list to the same path in the build, failing on a listed path missing from the snapshot or a snapshotted frame no transcript lists. Frames are not data files, so they carry no schema; the manifest hashes them like every other build object. Also check the JPEGs carry no EXIF or other embedded metadata, and fail if they do. Tests on a `LocalStore`: copies, missing frame, orphan frame, metadata.
4. **Moments.** No change: moment rows carry no screenshots, and no benchmark moment has any. Revisit if a later annotation batch covers transcripts with frames.
5. **Local build** in `temp/release-local/` on a fresh local snapshot: every check and `verify` pass; `benchmark/` byte-identical to a build of the 11-11 snapshot; record the stage 5 numbers again (question links etc.).
6. **Status doc and card notes:** record the above; the card says the frames are a reviewed sample, not the video. It does not describe how frames were screened for content.

Steps 1-3 done 2026-11-23 (schema text approved, with the screening details left out and a note on the per-transcript folders). `write_screenshots` runs after `write_assessments`, since the assessments writer must directly follow the transcripts. Step 5 done 2026-11-23: local snapshot `20261123T1920Z`, build `20261123T1921Z-473acf7` passes every check and `verify`, including the metadata check on all 10,118 frames (about 1 GB). Against the 2026-11-20 local build (same inputs, both guards patched by hand): `benchmark/` dev, test and moments byte-identical; annotations, provider C transcripts and assessments identical; provider A assessments differ only in `batch`, on the 37,764 rows whose upstream `batch` is now the latest delivery. Counts unchanged (dev 640 / test 631, 1,271 moments, annotations A 1,261 / C 661, assessments A 50,489 / C 30,574). Provider A quiz answers 245,082; 38,139 question links (38,499 answers on published questions less 360 unanswered); 8,866 null `date_completed`.

Done 2026-11-23: bucket snapshot `20261123T2220Z` and build `20261123T2330Z-4091066` (#110; see Builds).

## Builds

All in the de-identified bucket, under `release-v2/`.

| Build | Snapshot | Code | Contents | Status |
|---|---|---|---|---|
| `20261111T1640Z-a12705f` | `20261111T1636Z` (30 inputs, incl. the 2026-11-11 nightly annotation export) | `a12705f` (PR #98), clean tree | `transcripts/` A 40,176 / C 16,510 rows; `annotations/` A 1,261 / C 661 rows; `docs/annotation_guide.md`; 4 schemas; manifest | All checks passed, manifest written. For review; not for publishing. Superseded by `20261123T2330Z-4091066`, so its `verify` (interrupted twice by S3 read timeouts) will not be rerun |
| `20261123T2330Z-4091066` | `20261123T2220Z` (10,148 inputs, 4.2 GB: the 30 data inputs plus 10,118 frames) | `4091066` (`v2` after #109, #111), clean tree | `transcripts/` A 40,176 / C 16,510; `assessments/` A 50,489 / C 30,574; `reference/` 4 Eedi tables; `annotations/` A 1,261 / C 661; `benchmark/` dev 640 / test 631 / moments 1,271; `screenshots/tutoring_provider_a/` 10,118 frames; schemas, docs, manifest (10,144 files) | All checks passed, manifest written. Every file byte-identical to the local build `20261123T1921Z-473acf7` (same inputs; passed `verify`). `verify` passed 2026-11-24 (run from a laptop, 47 minutes). Release candidate |

## Next

Decisions, in the order they unblock work:

1. **Stage 7:** the Hugging Face repo name, and the dataset's licence. If the Eedi reference tables ship, their terms (CC BY-NC-ND 4.0 for the NeurIPS 2020 data) carry through, so the licence waits on Eedi's answer.
2. **#101:** whether classifier excerpts cut where annotators drew the cut (some `cut_turn` values sit one turn from it; see "Benchmark moments" above). The team's call, for the research path.
3. **Release workflow and provenance** (open decision above): needed before the second release, not the first.

Waiting on others:

- Eedi: whether feedback text (`explanation`) can ship on unpublished questions, and the reference tables' licence. Each answer is a small follow-up writer change (null `explanation` on unlinked questions; drop the reference tables and point the card to Eedi's downloads).

Other work:

- Make the first release and bring the research path into sync with it; then every experiment run logs the release (build) ID it used.
- Runtime (separate from the release): a v2 moments loader, a wait-time tool for the replay tutor, runs recording the build ID and the moments file's sha256. Scorer: the over-scaffolding prompt is gated on the human tutor's gold scaffolding label, which a replay lacks.
- Before publishing: update `build_ground_truth.py` to read the released annotations and write the release shape, once the team's work on it allows.
