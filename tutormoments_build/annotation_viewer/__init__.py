"""Static site for reviewing the raters' key-moment annotations.

The page browses one transcript at a time, filtered by annotator, construct, outcome and
date, alongside distribution analytics for whatever the filters select and an agreement
table -- over the whole export, or over one rater's moments when the annotator filter
names one. It reads a pair of exported JSONL files instead of the annotation platform's
database, and needs no server.

Maintainer-only. The generated page carries real annotator names and student transcript
text, so it is written under `data/` (gitignored) and must never be committed or shared.
"""

from tutormoments_build.annotation_viewer.axes import (
    AXES,
    AXIS_KEYS,
    REPORTED_AXIS_KEYS,
    axis_pairs,
    coarse_axes,
    diff_axes,
    sar_of,
)
from tutormoments_build.annotation_viewer.records import (
    AGREEMENT,
    DISAGREEMENT,
    SINGLE_PASS,
    THROWN_OUT,
    annotator_roster,
    build_moments,
    by_role,
    latest_passes,
    no_key_moment_verdicts,
    read_records,
)
from tutormoments_build.annotation_viewer.site import build_payload, build_site
from tutormoments_build.annotation_viewer.transcripts import (
    TranscriptIndex,
    locate_span,
)

__all__ = [
    "AGREEMENT",
    "AXES",
    "AXIS_KEYS",
    "DISAGREEMENT",
    "REPORTED_AXIS_KEYS",
    "SINGLE_PASS",
    "THROWN_OUT",
    "TranscriptIndex",
    "annotator_roster",
    "axis_pairs",
    "build_moments",
    "build_payload",
    "build_site",
    "by_role",
    "coarse_axes",
    "diff_axes",
    "latest_passes",
    "locate_span",
    "no_key_moment_verdicts",
    "read_records",
    "sar_of",
]
