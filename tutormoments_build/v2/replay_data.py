"""The v2 moments a replay starts from, with their transcript rows.

Replays (and the personas they read) cover both ground-truth splits: the split
distinction matters only for the human-human classification round. Split is
therefore left out of replay paths and records, which is only safe while
``moment_id`` is unique across splits -- so loading checks that and stops on a
collision.

Moments are located exactly as ``excerpts.build`` locates them: the same
transcripts files, the same ``resolve_boundaries``, the same reasons for
skipping one.
"""

import logging
from collections import Counter
from dataclasses import dataclass

from tutormoments_build.annotation_viewer.transcripts import TranscriptIndex
from tutormoments_build.v2 import excerpts

logger = logging.getLogger("tutormoments_build.v2.replay_data")


@dataclass(frozen=True)
class ReplayMoment:
    """One ground-truth moment, its rendered rows and resolved boundaries."""

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
    """Raise if any ``moment_id`` appears more than once across the splits."""
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


def load_moments(
    ground_truth_dir: str = excerpts.DEFAULT_GROUND_TRUTH_DIR,
    transcripts_paths=excerpts.DEFAULT_TRANSCRIPTS,
) -> tuple[list[ReplayMoment], Counter]:
    """Every locatable moment in both splits, and per-reason drop counts."""
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
                problems=index.problems(transcript_id),
            )
        )
    return out, dropped


def dialogue_prefix(moment: ReplayMoment) -> str:
    """The dialogue rows from the first through the cut row, one per line.

    Enrichments are left out (v1's personas were generated from dialogue
    alone); the cut turn is included, as v1's ``_prefix_turns_to_context``
    includes it. Rows render as the human-human excerpts render them.
    """
    return "\n".join(
        excerpts.format_row(row)
        for row in moment.rows[: moment.cut_index + 1]
        if row["turn_number"] is not None
    )
