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
| 2 | `annotations/<provider>.jsonl` | Not started | Schema draft (below) |
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

## Next: annotations (stage 2)

Draft the schema for review. Known requirements:

- One schema for all providers; annotator names stripped (the export carries
  them); annotator IDs replaced by labels, failing on any unmapped ID.
- A `purpose` field so later kinds (e.g. gold replay classifications) are new
  values, not new schemas, plus `batch` for later passes. Decide whether replay
  classifications need a `replays/` file to join to.
- Proposed summary fields: `outcome`, `n_adjudicators`, `adjudicators_agree`,
  `duplicate_of`. C's `corpus_revision` becomes `transcript_revision`.
- Inputs: `v2_annotations/source/<provider>_annotations.jsonl` (refreshed nightly;
  builds read only the frozen snapshot copy).
