"""Render transcript excerpts for the v2 ground-truth moments.

A ground-truth record (``data/ground_truth/{iteration,test}.jsonl``) says *where*
a moment sits; the transcripts export
(``data/v2_annotations/source/*_v2_transcripts.jsonl``) holds *what was said and
done* there. This module joins the two and writes the text an annotator or a
model would read: a lead-up window, then the moment itself, and nothing after it.

**Excerpts are cut on ``*_index``, not ``*_turn``.** The two numberings in a
ground-truth record are not interchangeable. ``start_turn``/``end_turn`` count
dialogue only; ``start_index``/``end_index`` are positions in the rendered row
list the annotator actually selected in -- dialogue *and* enrichments (screen
interactions, screen updates, pauses, problem changes) interleaved. The index
span is what the annotator drew; the turn span is a lossy projection of it, and
for 7 of the 474 released moments it is lossy enough to be wrong: those moments
lie entirely inside a silent stretch of screen activity, so all three of their
turn numbers collapse onto the one adjacent dialogue turn while their indices
carry the real extent. Cutting those on turns yields a single unrelated
utterance in place of the moment.

**Enrichments are content here, not decoration.** 384 of the 474 moments contain
at least one enrichment row inside their span, and the moments above contain
nothing else. They are rendered in place, keeping the ``[SCREEN INTERACTION]``
-style tag their text already carries.

The lead-up window is measured in *dialogue turns* (``context_turns``), not
rows: it reaches back to the first row of the Nth preceding dialogue turn, so
every enrichment interleaved with those turns comes along. That keeps the window
meaningful where screen activity is dense -- but it also means a small
``context_turns`` can pull in many rows (median 6 rows at N=5, but 88 in the
worst case), so ``max_context_rows`` can cap it.

Nothing after ``end_index`` is ever emitted: the excerpt stops at the moment's
last row, by design.

Usage::

    python -m tutormoments_build.v2.excerpts --dry-run
    python -m tutormoments_build.v2.excerpts --context-turns 8
    python -m tutormoments_build.v2.excerpts --print 02e89625-9d6b-5e00-b40d-5d38a6adb2b3
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

from tutormoments.logging_setup import logging_args_parent, setup_logging
from tutormoments_build.annotation_viewer.transcripts import TranscriptIndex

logger = logging.getLogger("tutormoments_build.v2.excerpts")

DEFAULT_GROUND_TRUTH_DIR = "data/ground_truth"
DEFAULT_TRANSCRIPTS = (
    "data/v2_annotations/source/tutoring_provider_a_v2_transcripts.jsonl"
)
DEFAULT_OUT_DIR = "data/excerpts"

# Ground-truth split stems, which are also the excerpt output stems.
SPLIT_STEMS = ("iteration", "test")

DEFAULT_CONTEXT_TURNS = 5

MOMENT_START = ">>> MOMENT START (row {row}, turn {turn}) <<<"
CUT_POINT = ">>> CUT POINT (row {row}, turn {turn}) <<<"
MOMENT_END = ">>> MOMENT END (row {row}, turn {turn}) <<<"


# ===========================================================================
# Reading
# ===========================================================================


def load_ground_truth(ground_truth_dir: str) -> dict[str, list[dict]]:
    """Return {split stem: [ground-truth record, ...]} in file order.

    A missing split file is not an error -- a round may have produced only one.
    """
    out: dict[str, list[dict]] = {}
    for stem in SPLIT_STEMS:
        path = os.path.join(ground_truth_dir, f"{stem}.jsonl")
        if not os.path.exists(path):
            logger.warning("no ground-truth file at %s; skipping split", path)
            continue
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        out[stem] = records
    if not out:
        raise FileNotFoundError(
            f"no ground-truth splits found in {ground_truth_dir}; run "
            "`python -m tutormoments_build.v2.build_ground_truth` first"
        )
    return out


# ===========================================================================
# Rendering
# ===========================================================================


def context_start(
    rows: list[dict],
    start_row: int,
    context_turns: int,
    max_context_rows: int | None = None,
) -> int:
    """First row of the lead-up window before ``start_row``.

    Reaches back to the first row of the ``context_turns``-th preceding dialogue
    turn, which carries every enrichment interleaved with those turns along with
    it. Enrichments sitting *before* that turn are left out, so the window opens
    on something someone said rather than mid-way through a run of screen
    activity. Runs out of transcript -> opens at row 0.

    ``max_context_rows`` clamps the result, for the dense-screen-activity case
    where a few turns of lead-up span a great many rows.
    """
    if context_turns <= 0:
        first = start_row
    else:
        seen = 0
        first = 0
        for index in range(start_row - 1, -1, -1):
            if rows[index]["turn_number"] is None:
                continue
            seen += 1
            first = index
            if seen == context_turns:
                break
    if max_context_rows is not None:
        first = max(first, start_row - max_context_rows)
    return first


def _strip_role_tag(text: str, role: str) -> str:
    """Drop the leading ``[TUTOR]`` / ``[STUDENT]`` tag a dialogue row repeats.

    The role is already printed in the line's own prefix, so keeping the tag
    would render it twice. Only the row's own role is stripped: a tag that
    disagrees with the role column is left in place to stay visible.
    """
    tag = f"[{role}]"
    if role and text.startswith(tag):
        return text[len(tag) :].lstrip()
    return text


def format_row(row: dict) -> str:
    """One transcript row as a line of excerpt text.

    Dialogue prints as ``Turn N. ROLE: text``, matching the convention the rest
    of the project formats transcripts with. An enrichment prints its text bare:
    it carries its own ``[SCREEN INTERACTION]``-style tag and has no turn number
    of its own, and inventing one would disagree with every number around it.
    """
    if row["turn_number"] is None:
        return row["text"]
    role = row["role"]
    return f"Turn {row['turn_number']}. {role}: {_strip_role_tag(row['text'], role)}"


def render_excerpt(
    rows: list[dict],
    boundaries: dict,
    *,
    context_turns: int = DEFAULT_CONTEXT_TURNS,
    max_context_rows: int | None = None,
) -> tuple[str, int]:
    """Render one moment's excerpt. Returns (text, first row rendered).

    ``boundaries`` supplies ``start_index``/``cut_index``/``end_index`` (the row
    positions that are cut on) and ``start_turn``/``cut_turn``/``end_turn``
    (printed in the markers, so a reader can find the moment in the export).

    The window runs from the lead-up through ``end_index`` and stops there --
    there is no trailing context and no trailing elision marker, because for
    this benchmark the moment's last row is where the transcript ends.
    """
    start_row = boundaries["start_index"]
    end_row = boundaries["end_index"]
    cut_row = boundaries["cut_index"]
    if not 0 <= start_row <= end_row < len(rows):
        raise IndexError(
            f"moment rows [{start_row}, {end_row}] fall outside the transcript's "
            f"{len(rows)} row(s)"
        )

    first = context_start(rows, start_row, context_turns, max_context_rows)

    lines = []
    if first > 0:
        lines += [f"[... {first} earlier row(s) omitted ...]", ""]

    for index in range(first, end_row + 1):
        if index == start_row:
            lines.append(
                MOMENT_START.format(row=start_row, turn=boundaries["start_turn"])
            )
        lines.append(format_row(rows[index]))
        if index == cut_row:
            lines.append(CUT_POINT.format(row=cut_row, turn=boundaries["cut_turn"]))
        if index == end_row:
            lines.append(MOMENT_END.format(row=end_row, turn=boundaries["end_turn"]))

    return "\n".join(lines), first


# ===========================================================================
# Assembly
# ===========================================================================


def build_record(
    moment: dict,
    rows: list[dict],
    conversation_id: str,
    *,
    context_turns: int = DEFAULT_CONTEXT_TURNS,
    max_context_rows: int | None = None,
) -> dict:
    """Assemble one excerpt record from a ground-truth record and its rows.

    The labels ride along so an excerpt file is usable on its own; everything
    else about the moment is recoverable by joining on ``moment_id``.
    """
    excerpt, first = render_excerpt(
        rows,
        moment,
        context_turns=context_turns,
        max_context_rows=max_context_rows,
    )
    span = rows[moment["start_index"] : moment["end_index"] + 1]
    enrichments = sum(1 for row in span if row["turn_number"] is None)

    return {
        "moment_id": moment["moment_id"],
        "transcript_id": moment["transcript_id"],
        "conversation_id": conversation_id,
        "split": moment["split"],
        "start_turn": moment["start_turn"],
        "cut_turn": moment["cut_turn"],
        "end_turn": moment["end_turn"],
        "start_index": moment["start_index"],
        "cut_index": moment["cut_index"],
        "end_index": moment["end_index"],
        "context_turns": context_turns,
        "context_start_index": first,
        "context_rows": moment["start_index"] - first,
        "moment_rows": len(span),
        "moment_dialogue_rows": len(span) - enrichments,
        "moment_enrichment_rows": enrichments,
        "labels": moment["labels"],
        "excerpt": excerpt,
    }


def _turn_span_disagrees(moment: dict, rows: list[dict]) -> bool:
    """Whether cutting this moment on turn numbers would lose annotated rows.

    True when the dialogue-turn span does not reach as far as the index span --
    the all-screen-activity moments the module docstring describes. Reported so
    the count stays visible rather than being silently absorbed.
    """
    spans: dict[int, list[int]] = {}
    for row in rows:
        number = row["turn_number"]
        if number is None:
            continue
        spans.setdefault(number, [row["turn_index"], row["turn_index"]])
        spans[number][1] = row["turn_index"]
    start, end = moment["start_turn"], moment["end_turn"]
    if start not in spans or end not in spans:
        return True
    return (
        spans[start][0] > moment["start_index"] or spans[end][1] < moment["end_index"]
    )


def build(
    ground_truth_dir: str,
    transcripts_path: str,
    *,
    context_turns: int = DEFAULT_CONTEXT_TURNS,
    max_context_rows: int | None = None,
) -> tuple[dict[str, list[dict]], Counter]:
    """Return ({output stem: [excerpt record, ...]}, per-reason drop counts)."""
    ground_truth = load_ground_truth(ground_truth_dir)
    wanted = {
        moment["transcript_id"]
        for records in ground_truth.values()
        for moment in records
    }
    index = TranscriptIndex(transcripts_path, wanted)
    logger.info(
        "%d of %d transcript(s) found in %s (%d in the file)",
        len(index),
        len(wanted),
        transcripts_path,
        index.total,
    )

    out: dict[str, list[dict]] = {stem: [] for stem in ground_truth}
    dropped: Counter = Counter()
    turn_span_lossy = 0

    for stem, records in ground_truth.items():
        for moment in records:
            transcript_id = moment["transcript_id"]
            if transcript_id not in index:
                dropped["transcript_not_in_export"] += 1
                logger.warning(
                    "transcript %s is not in %s; skipping moment %s",
                    transcript_id,
                    transcripts_path,
                    moment["moment_id"],
                )
                continue

            rows = index.rows(transcript_id)
            try:
                record = build_record(
                    moment,
                    rows,
                    index.conversation_id(transcript_id) or transcript_id,
                    context_turns=context_turns,
                    max_context_rows=max_context_rows,
                )
            except IndexError as error:
                dropped["boundaries_out_of_range"] += 1
                logger.warning("moment %s: %s", moment["moment_id"], error)
                continue

            if _turn_span_disagrees(moment, rows):
                turn_span_lossy += 1
            out[stem].append(record)

    if turn_span_lossy:
        logger.info(
            "%d moment(s) reach rows their dialogue-turn span does not cover; "
            "cut on *_index, as this module does, they are intact",
            turn_span_lossy,
        )
    return out, dropped


def write_split(out_dir: str, stem: str, records: list[dict]) -> str:
    """Write one split's excerpts JSONL atomically. Returns the path written."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.jsonl")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    os.replace(tmp, path)
    return path


# ===========================================================================
# Reporting
# ===========================================================================


def report(out: dict[str, list[dict]], dropped: Counter, dry_run: bool) -> str:
    lines = [
        "",
        "DRY RUN -- nothing written" if dry_run else "Excerpts written",
        "",
        f"  {'split':<12}{'moments':>9}{'lead-up rows':>15}{'moment rows':>14}"
        f"{'enrichment rows':>18}{'with enrichments':>19}",
    ]
    lines.append(f"  {'-' * (len(lines[-1]) - 2)}")

    for stem, records in out.items():
        if not records:
            lines.append(f"  {stem:<12}{0:>9}")
            continue
        context = sum(r["context_rows"] for r in records) / len(records)
        moment = sum(r["moment_rows"] for r in records) / len(records)
        enrich = sum(r["moment_enrichment_rows"] for r in records) / len(records)
        with_enrich = sum(1 for r in records if r["moment_enrichment_rows"])
        share = f"{with_enrich / len(records):.0%}"
        lines.append(
            f"  {stem:<12}{len(records):>9}{context:>15.1f}{moment:>14.1f}"
            f"{enrich:>18.1f}{f'{with_enrich} ({share})':>19}"
        )

    lines.append("")
    lines.append("  (row counts are means per moment)")

    if dropped:
        lines += ["", "  moments not emitted:"]
        for reason, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:<28}{count:>5}")
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tutormoments_build.v2.excerpts",
        description="Render transcript excerpts for the v2 ground-truth moments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[logging_args_parent()],
    )
    parser.add_argument(
        "--ground-truth", default=DEFAULT_GROUND_TRUTH_DIR, metavar="DIR"
    )
    parser.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS, metavar="FILE")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR")
    parser.add_argument(
        "--context-turns",
        type=int,
        default=DEFAULT_CONTEXT_TURNS,
        metavar="N",
        help="Dialogue turns of lead-up before the moment (0 for none). "
        "Enrichments interleaved with those turns are included.",
    )
    parser.add_argument(
        "--max-context-rows",
        type=int,
        default=None,
        metavar="N",
        help="Cap the lead-up at N rows, for stretches of dense screen activity",
    )
    parser.add_argument(
        "--print",
        dest="print_moment",
        metavar="MOMENT_ID",
        help="Print one moment's excerpt to stdout and exit (id prefix accepted)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without writing any file"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)

    out, dropped = build(
        args.ground_truth,
        args.transcripts,
        context_turns=args.context_turns,
        max_context_rows=args.max_context_rows,
    )

    if args.print_moment:
        for records in out.values():
            for record in records:
                if record["moment_id"].startswith(args.print_moment):
                    print(record["excerpt"])
                    return 0
        print(f"no moment matching {args.print_moment!r}", file=sys.stderr)
        return 1

    if not args.dry_run:
        for stem, records in out.items():
            logger.info(
                "wrote %d excerpt(s) to %s",
                len(records),
                write_split(args.out_dir, stem, records),
            )

    print(report(out, dropped, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
