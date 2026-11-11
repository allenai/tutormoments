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
| 0 | Plumbing: snapshot, build, checks, manifest, verify | Done (`f9f4aac`) | Push + PR into `v2` |
| 1 | `transcripts/<provider>.jsonl` | Schemas approved 2026-11-04; writer done | First real snapshot + build. Screenshot fields wait on stage 6 |
| 2 | `annotations/<provider>.jsonl` | Schema approved and writer done 2026-11-11; builds ship `docs/annotation_guide.md` | One data-quality decision pending (internal) |
| 3 | `benchmark/dev.jsonl`, `test.jsonl` | Not started | Schema draft from `ground_truth_v2` output |
| 4 | `benchmark/moments.jsonl` | Not started | Builder rules (benchmark-defining: needs sign-off), student persona, context format, runtime loader |
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
  - Dropped: retracted moments (A 3, C 1); earlier saves of the same role by the same annotator (the last save is kept, as `build_ground_truth.py` does) and `revision`; A's turn-by-turn `transcript_selection`; the transcript fingerprints (`selection_source_sha256`, `corpus_revision`), which match no released transcript file; the tool-computed turn counts (`dialogue_turns` and variants), wrong on 17 early A moments.
  - Kept: thrown-out moments with their status, A's `enrichment_numbers`, and the v1 rigor hint fields, so researchers can account for annotators being shown hints.
  - No summary fields (outcome, agreement): resolved labels belong in `benchmark/`.
  - Field descriptions follow the annotators' guiding sheet.
- **Annotation guide** ships as `docs/annotation_guide.md` (`release/docs/annotation_guide.md`): the annotators' guiding sheet with its comments, feedback tab, images (crossed out in the source), internal notes and author line removed; the guidance text is otherwise word for word.

## Annotations writer (stage 2)

`release/annotations.py` builds `annotations/<provider>.jsonl` from the frozen `v2_annotations/source/<provider>_annotations.jsonl` (the nightly export) and `annotator_labels.json`. It fails on an annotator with no label and on any export field it does not list as kept or dropped, so upstream additions are reviewed before they ship. Builds also copy the packaged docs (`release/docs/`) to `docs/`.

Open: the pending data-quality decision may change which entries ship; it will be applied in the writer and recorded here.
