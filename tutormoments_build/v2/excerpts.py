"""Render transcript excerpts for the v2 ground-truth moments.

This script combines ``data/ground_truth/{iteration,test}.jsonl`` with transcripts in
``data/v2_annotations/source/*_v2_transcripts.jsonl``.
The output is used to format transcript excerpts for action classification.

**Excerpts are cut on ``*_index``, not ``*_turn``.**
- ``start_turn``/``end_turn`` count dialogue only
- ``start_index``/``end_index`` are positions in the rendered row

The lead-up window is measured in *dialogue turns* (``context_turns``).
The window is measured back from the cut point, and from nothing else: it may
reach back past the moment's start, and on a long moment it may open *inside*
the moment, after ``start_index``. Every excerpt at a given width therefore
carries the same amount of lead-up, however the moment was drawn.

The excerpt stops at the moment's last row.

Usage:
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

DEFAULT_CONTEXT_TURNS = 20

# Lead-up widths every excerpt is rendered at. One moment yields one excerpt per
# width, and a consumer picks the one its prompt calls for -- the v2
# action-direction prompt reads 5 turns, over-scaffolding reads 20, and asking
# them to share a width would mean rebuilding this file to change either.
DEFAULT_CONTEXT_WIDTHS = (20, 5)

# The excerpt's one marker, and the literal string both v2 prompts name. It is
# bare: its position in the text is the whole signal. Row indices are a
# build-side artifact and turn numbers a lossy projection of them (see above),
# so printing either would add noise a reader cannot act on.
CUT_POINT = ">>> CUT POINT <<<"


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
    anchor_row: int,
    context_turns: int,
    max_context_rows: int | None = None,
) -> int:
    """First row of the lead-up window before ``anchor_row``.

    Reaches back to the first row of the ``context_turns``-th preceding dialogue
    turn, which carries every enrichment interleaved with those turns along with
    it. Enrichments sitting *before* that turn are left out, so the window opens
    on something someone said rather than mid-way through a run of screen
    activity. Runs out of transcript -> opens at row 0.

    ``max_context_rows`` clamps the result, for the dense-screen-activity case
    where a few turns of lead-up span a great many rows.

    ``render_excerpt`` anchors this on the *cut point*, not the moment start --
    see there for why.
    """
    if context_turns <= 0:
        first = anchor_row
    else:
        seen = 0
        first = 0
        for index in range(anchor_row - 1, -1, -1):
            if rows[index]["turn_number"] is None:
                continue
            seen += 1
            first = index
            if seen == context_turns:
                break
    if max_context_rows is not None:
        first = max(first, anchor_row - max_context_rows)
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

    ``boundaries`` supplies ``start_index``/``cut_index``/``end_index``, the row
    positions the excerpt is cut on. The ``*_turn`` values are not read here:
    the markers are bare, so what they mark is carried by where they sit in the
    text rather than by any number printed in them.

    The cut point is the only thing marked, because it is the only landmark
    either v2 prompt refers to. ``start_index`` and ``end_index`` still bound the
    window and the range check, but neither boundary is announced: the prompts
    ask only about what the tutor does *after* the cut, so where the annotated
    moment opened is not a distinction they act on, and its close needs no marker
    because nothing is rendered past it -- the excerpt simply ends there.

    The window runs from the lead-up through ``end_index`` and stops there --
    no trailing context, because for this benchmark the moment's last row is
    where the transcript ends. Its opening is measured back from ``cut_index``
    (see the module docstring) and from nothing else: ``context_turns`` turns
    before the cut is the whole rule, whether that reaches back past
    ``start_index`` or opens inside the moment. The teacher-drawn start is an
    annotation boundary, not a unit of context -- holding the window open to it
    would hand a long moment more lead-up than a short one at the same width,
    which is exactly the confound the cut-anchored window exists to remove.
    Elided lead-up is not announced either: the excerpt simply opens where the
    window opens. The returned first-row index still records how much came
    before, for the excerpt record.
    """
    start_row = boundaries["start_index"]
    end_row = boundaries["end_index"]
    cut_row = boundaries["cut_index"]
    if not 0 <= start_row <= end_row < len(rows):
        raise IndexError(
            f"moment rows [{start_row}, {end_row}] fall outside the transcript's "
            f"{len(rows)} row(s)"
        )

    # Anchored on the cut alone. May land after start_row, inside the moment.
    first = context_start(rows, cut_row, context_turns, max_context_rows)

    lines = []
    for index in range(first, end_row + 1):
        lines.append(format_row(rows[index]))
        if index == cut_row:
            lines.append(CUT_POINT)

    return "\n".join(lines), first


# ===========================================================================
# Assembly
# ===========================================================================


def build_record(
    moment: dict,
    rows: list[dict],
    conversation_id: str,
    *,
    context_widths: "tuple[int, ...]" = DEFAULT_CONTEXT_WIDTHS,
    max_context_rows: int | None = None,
) -> dict:
    """Assemble one excerpt record from a ground-truth record and its rows.

    The labels ride along so an excerpt file is usable on its own; everything
    else about the moment is recoverable by joining on ``moment_id``.

    ``excerpts`` holds one rendering per width in ``context_widths``, keyed by
    the width as a string (JSON object keys are strings, and round-tripping the
    file must not turn the key into something else). Only the lead-up differs
    between them, so everything width-independent -- boundaries, row counts,
    labels -- stays at the top level rather than being repeated per width.
    """
    excerpts = {}
    for width in context_widths:
        text, first = render_excerpt(
            rows,
            moment,
            context_turns=width,
            max_context_rows=max_context_rows,
        )
        excerpts[str(width)] = {
            "excerpt": text,
            "context_turns": width,
            "context_start_index": first,
            # Rows rendered before the cut -- the lead-up as the prompt sees it.
            # Measured from the cut, not the moment start, because the window is
            # the cut's: against start_index this count would go negative on a
            # window that opens inside the moment.
            "context_rows": moment["cut_index"] - first,
            # Whether the width was narrower than the moment's own pre-cut run.
            # Recorded per width so a later round can tell how much of the
            # annotated moment a given prompt actually saw.
            "opens_inside_moment": first > moment["start_index"],
        }
    span = rows[moment["start_index"] : moment["end_index"] + 1]
    enrichments = sum(1 for row in span if row["turn_number"] is None)

    # What lies after the cut is what a v2 prompt is actually asked to classify.
    # Screen activity counts: "[SCREEN INTERACTION] Tutor writes 3x7 on the
    # board" is a pedagogical move, so the two are counted separately rather
    # than folded together.
    post_cut = rows[moment["cut_index"] + 1 : moment["end_index"] + 1]
    post_cut_dialogue = sum(1 for row in post_cut if row["turn_number"] is not None)

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
        "moment_rows": len(span),
        "moment_dialogue_rows": len(span) - enrichments,
        "moment_enrichment_rows": enrichments,
        "post_cut_rows": len(post_cut),
        "post_cut_dialogue_rows": post_cut_dialogue,
        "labels": moment["labels"],
        "excerpts": excerpts,
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
    context_widths: "tuple[int, ...]" = DEFAULT_CONTEXT_WIDTHS,
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
                    context_widths=context_widths,
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


def _inside_moment_lines(out: dict[str, list[dict]], widths: list[int]) -> list[str]:
    """How often each width opened inside the moment rather than before it.

    Worth surfacing per run: it is the share of moments where the prompt saw
    less than the annotator drew, and it moves with the width, so a width that
    lands inside nearly every moment is a signal the window is too narrow to
    judge the question being asked.
    """
    records = [r for records in out.values() for r in records]
    if not records:
        return []
    lines = []
    for width in widths:
        inside = sum(
            1 for r in records if r["excerpts"][str(width)]["opens_inside_moment"]
        )
        lines.append(
            f"    @{width:<4}{inside:>5} of {len(records)} ({inside / len(records):.0%})"
        )
    return ["", "  windows opening inside the moment:", *lines] if lines else []


def report(out: dict[str, list[dict]], dropped: Counter, dry_run: bool) -> str:
    lines = [
        "",
        "DRY RUN -- nothing written" if dry_run else "Excerpts written",
        "",
    ]
    widths = sorted(
        (int(w) for records in out.values() for r in records for w in r["excerpts"]),
        reverse=True,
    )
    widths = list(dict.fromkeys(widths))

    lead_up = "".join(f"{f'lead-up @{w}':>15}" for w in widths)
    header = (
        f"  {'split':<12}{'moments':>9}{lead_up}{'moment rows':>14}"
        f"{'enrichment rows':>18}{'with enrichments':>19}"
    )
    lines += [header, f"  {'-' * (len(header) - 2)}"]

    for stem, records in out.items():
        if not records:
            lines.append(f"  {stem:<12}{0:>9}")
            continue
        per_width = "".join(
            f"{sum(r['excerpts'][str(w)]['context_rows'] for r in records) / len(records):>15.1f}"
            for w in widths
        )
        moment = sum(r["moment_rows"] for r in records) / len(records)
        enrich = sum(r["moment_enrichment_rows"] for r in records) / len(records)
        with_enrich = sum(1 for r in records if r["moment_enrichment_rows"])
        share = f"{with_enrich / len(records):.0%}"
        lines.append(
            f"  {stem:<12}{len(records):>9}{per_width}{moment:>14.1f}"
            f"{enrich:>18.1f}{f'{with_enrich} ({share})':>19}"
        )

    lines.append("")
    lines.append(
        "  (row counts are means per moment; lead-up @N is the N-turn window, "
        "counted back from the cut)"
    )

    lines += _inside_moment_lines(out, widths)

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
        nargs="+",
        default=list(DEFAULT_CONTEXT_WIDTHS),
        metavar="N",
        help="Dialogue turns of lead-up before the CUT POINT (0 for none). "
        "Enrichments interleaved with those turns are included. The window is "
        "measured from the cut alone: on a moment with a long pre-cut run it "
        "opens inside the moment, so every excerpt at a width carries the same "
        "lead-up. Pass several widths to render an excerpt at each; consumers "
        "pick the one their prompt calls for.",
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
        context_widths=tuple(args.context_turns),
        max_context_rows=args.max_context_rows,
    )

    if args.print_moment:
        for records in out.values():
            for record in records:
                if record["moment_id"].startswith(args.print_moment):
                    for width, rendered in record["excerpts"].items():
                        print(f"===== {width}-turn window =====\n")
                        print(rendered["excerpt"])
                        print()
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
