"""Loads the ground-truth moments that replays start from.

Moments come from both splits and are found in the transcripts the same way
`excerpts.py` finds them. Provider C moments also get the text of their
session's slides. See README.md, "Replays".
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass

from tutormoments_build.annotation_viewer.transcripts import TranscriptIndex
from tutormoments_build.v2 import excerpts

logger = logging.getLogger("tutormoments_build.v2.replay_data")

DEFAULT_SLIDES = "data/v2_annotations/source/tutoring_provider_c_lesson_slides.jsonl"


@dataclass(frozen=True)
class ReplayMoment:
    """One moment, with its transcript rows and where the cut and end fall."""

    moment: dict
    conversation_id: str
    rows: list[dict]
    cut_index: int
    end_index: int
    problems: list[dict]

    @property
    def moment_id(self) -> str:
        return self.moment["moment_id"]


def check_unique_ids(ground_truth: dict[str, list[dict]]) -> None:
    """Stop if a `moment_id` appears in both splits.

    Replay files don't record the split, so a repeated ID would be ambiguous.
    """
    seen: dict[str, str] = {}
    for split, records in ground_truth.items():
        for record in records:
            moment_id = record["moment_id"]
            if moment_id in seen:
                raise ValueError(
                    f"moment_id {moment_id} appears in both {seen[moment_id]!r} "
                    f"and {split!r}; replay paths and records carry no split, so "
                    "the two would collide"
                )
            seen[moment_id] = split


def load_slides(path: str = DEFAULT_SLIDES) -> dict[str, str]:
    """Slide text by `slide_id`, from the lesson-slides file."""
    slides = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                slides[str(record["slide_id"])] = record["problem_text"]
    return slides


def with_slide_text(moment_id: str, problems: list[dict], slides: dict) -> list:
    """Add each slide's text to its problem. Stops if a slide has no text."""
    out = []
    for problem in problems:
        text = (slides.get(str(problem["slide_id"])) or "").strip()
        if not text:
            raise ValueError(
                f"moment {moment_id}: no text for slide {problem['slide_id']} "
                "in the lesson-slides file"
            )
        out.append({**problem, "slide_text": text})
    return out


def load_moments(
    ground_truth_dir: str = excerpts.DEFAULT_GROUND_TRUTH_DIR,
    transcripts_paths=excerpts.DEFAULT_TRANSCRIPTS,
    slides_path: str | None = DEFAULT_SLIDES,
) -> tuple[list[ReplayMoment], Counter]:
    """Every moment found in the transcripts, and counts of those dropped, by reason.

    Pass `slides_path=None` to skip the slide text (personas don't use it).
    """
    slides = load_slides(slides_path) if slides_path else None
    ground_truth = excerpts.load_ground_truth(ground_truth_dir)
    check_unique_ids(ground_truth)
    moments = [m for records in ground_truth.values() for m in records]

    indexes: dict[str, TranscriptIndex] = {}
    for path in transcripts_paths:
        source = excerpts.source_of_transcripts(path)
        if source in indexes:
            raise ValueError(f"two transcripts files for provider {source!r}")
        wanted = {m["transcript_id"] for m in moments if m.get("source") == source}
        indexes[source] = TranscriptIndex(path, wanted)

    out: list[ReplayMoment] = []
    dropped: Counter = Counter()
    for moment in moments:
        index = indexes.get(moment.get("source"))
        transcript_id = moment["transcript_id"]
        if index is None:
            dropped["no_transcripts_for_source"] += 1
            logger.warning("no transcripts for %s", moment["moment_id"])
            continue
        if transcript_id not in index:
            dropped["transcript_not_in_export"] += 1
            logger.warning("transcript %s not in %s", transcript_id, index.path)
            continue
        try:
            bounds = excerpts.resolve_boundaries(moment, index.turn_rows(transcript_id))
        except IndexError as error:
            dropped["boundaries_out_of_range"] += 1
            logger.warning("moment %s: %s", moment["moment_id"], error)
            continue
        out.append(
            ReplayMoment(
                moment=moment,
                conversation_id=index.conversation_id(transcript_id) or transcript_id,
                rows=index.rows(transcript_id),
                cut_index=bounds["cut_index"],
                end_index=bounds["end_index"],
                problems=(
                    with_slide_text(
                        moment["moment_id"], index.problems(transcript_id), slides
                    )
                    if slides is not None
                    else index.problems(transcript_id)
                ),
            )
        )
    return out, dropped


def dialogue_prefix(moment: ReplayMoment) -> str:
    """The transcript up to and including the cut turn, dialogue rows only.

    Personas are generated from this. Screen-activity and pause rows are left out,
    as in v1.
    """
    return "\n".join(
        excerpts.format_row(row)
        for row in moment.rows[: moment.cut_index + 1]
        if row["turn_number"] is not None
    )
