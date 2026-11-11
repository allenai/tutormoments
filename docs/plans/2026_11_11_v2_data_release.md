# v2 data release: implementation status

Status as of 2026-11-11. The release layout and principles live in the internal
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

## Stages

| # | Stage | Status | Open |
|---|---|---|---|
| 0 | Plumbing: snapshot, build, checks, manifest, verify | Done (`f9f4aac`); PR #97 into `v2` | Review + merge #97. `verify` needs retries on long S3 reads if it times out again |
| 1 | `transcripts/<provider>.jsonl` | Schemas approved 2026-11-04; writer done; first real build 2026-11-11 (below) | Screenshot fields wait on stage 6 |
| 2 | `annotations/<provider>.jsonl` | Schema, writer and guide in PR #98 (stacked on #97); first real build 2026-11-11 (below) | Review of #98 and the build by the annotation manager and Albert; one data-quality decision pending (internal). The `transcript_selection` change (2026-11-11) is not in that build |
| 3 | `benchmark/dev.jsonl`, `test.jsonl` | Schema approved 2026-11-11; schema and writer in PR #99 (stacked on #98) | The stage 2 data-quality decision changes provider C's labels, so no C build is reviewed before it |
| 4 | `benchmark/moments.jsonl` | Not started | Builder rules (benchmark-defining: needs sign-off), student persona, context format, runtime loader. Whether `student_state.jsonl` (model-generated facets used by the scorer prompts) ships |
| 5 | `assessments/`, `reference/eedi_*` | Not started | Full release vs annotated students only; C's third-party LaTeX renderer links |
| 6 | `screenshots/<transcript_id>/` | Waiting on the de-identified set | Layout, transcript `seconds`/`path` fields, OCR text |
| 7 | `manifest.json` schema, README dataset card | Not started (builds already write a manifest) | Card carries A's two dropped fields and the surrogate-name caveat |
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
  - Labels are rebuilt from the snapshot's annotation exports, not copied from `synthetic/ground_truth_v2/`. Checked on snapshot `20261111T1636Z`: the rebuild, and a rebuild from the release-format annotations, both reproduce the bucket's files byte for byte. So researchers can reproduce dev/test from the released annotations and the committed `splits.json`.
  - `annotators` and `annotator_roles` (parallel arrays) become one array of `{annotator, role}`, so a consumer can't separate a role from its annotator. Order is the tool's (two adjudicators: first to save first); it affects neither labels nor boundaries (checked on all 363 two-adjudicator moments).
  - Dropped: `moment_status` (always `adjudicated`) and `moment_created_at` (on the annotations row). `original_boundaries` is on every row, null where the selector's boundaries stand.
  - Known: 4 provider A moments have no turn after the cut (`cut_turn == end_turn`). They pass the current rules and ship unless a rule changes (benchmark-defining: needs sign-off).

## Annotations writer (stage 2)

`release/annotations.py` builds `annotations/<provider>.jsonl` from the frozen `v2_annotations/source/<provider>_annotations.jsonl` (the nightly export) and `annotator_labels.json`. It fails on an annotator with no label and on any export field it does not list as kept or dropped, so upstream additions are reviewed before they ship. Builds also copy the packaged docs (`release/docs/`) to `docs/`.

Open: the pending data-quality decision may change which entries ship; it will be applied in the writer and recorded here.

## Benchmark labels writer (stage 3)

`release/benchmark.py` builds `benchmark/dev.jsonl` and `test.jsonl` by running `v2/build_ground_truth.py` (unchanged) on the snapshot's annotation exports with the committed `splits.json`, and writes them in the release shape (`release_record`, in the writer). Nothing is copied from `synthetic/ground_truth_v2/`. It then rebuilds them from the `annotations/` files the annotations writer just wrote, reshaped back to the export's form, and fails the build unless both runs give the same bytes, so it runs after `write_annotations`.

`build_ground_truth.py` is not changed for the release while the team's work on it is in progress (decided 2026-11-11): finish the data release first, then update the code. Before publishing, the code has to catch up with the schema's description, which says running the script on the released annotation files reproduces these files. Today it doesn't read the release format or write the release shape.

## Builds

All in the de-identified bucket, under `release-v2/`.

| Build | Snapshot | Code | Contents | Status |
|---|---|---|---|---|
| `20261111T1640Z-a12705f` | `20261111T1636Z` (30 inputs, incl. the 2026-11-11 nightly annotation export) | `a12705f` (PR #98), clean tree | `transcripts/` A 40,176 / C 16,510 rows; `annotations/` A 1,261 / C 661 rows; `docs/annotation_guide.md`; 4 schemas; manifest | All checks passed, manifest written. For review; not for publishing (the pending decision may change it). `verify` not yet completed: the first run hit an S3 read timeout and the second was stopped on a slow connection; rerun from inside AWS |

## Next

1. Review #97, then #98 and build `20261111T1640Z-a12705f`.
2. Apply the pending data-quality decision in the annotations writer, then snapshot and build again.
3. Review the stage 3 schema PR #99 (stacked on #98), then write the benchmark labels writer: rebuild with `build_ground_truth.py` from the snapshot, write `dev`/`test`, and fail the build unless a rebuild from the released annotations matches.
