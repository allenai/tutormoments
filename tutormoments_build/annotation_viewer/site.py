"""Assemble the viewer payload and write the self-contained page.

Everything the page needs is embedded in it: the transcripts of the annotated
conversations, every pass on every moment, and the coarse axes those passes reduce to.
Filtering, the agreement table and the distribution bars are all recomputed in the
browser as the filters change, so the file can be opened straight off disk with no
server behind it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from tutormoments_build.annotation_viewer import axes as axes_mod
from tutormoments_build.annotation_viewer.records import (
    annotator_roster,
    build_moments,
    no_key_moment_verdicts,
    read_records,
)
from tutormoments_build.annotation_viewer.transcripts import (
    TranscriptIndex,
    locate_span,
)

TEMPLATE = Path(__file__).with_name("template.html")
PLACEHOLDER = "__VIEWER_DATA__"


def build_payload(annotations_path, transcripts_path, excluded=()):
    """Moments, transcripts and rosters for one annotations export."""
    records = read_records(annotations_path)
    moments = build_moments(records, excluded)
    verdicts = no_key_moment_verdicts(records, excluded)

    wanted = {m["transcript_id"] for m in moments} | {
        v["transcript_id"] for v in verdicts
    }
    index = TranscriptIndex(transcripts_path, wanted)
    _locate(moments, index)

    transcripts = {}
    for source in (moments, verdicts):
        for item in source:
            tid = item["transcript_id"]
            entry = transcripts.get(tid)
            if entry is None:
                found = tid in index
                # Only the transcripts file carries the id the annotation tool showed;
                # the export keys moments and verdicts by transcript alone.
                entry = transcripts[tid] = {
                    "transcript_id": tid,
                    "conversation_id": (index.conversation_id(tid) if found else "")
                    or tid,
                    "found": found,
                    "turns": index.rows(tid) if found else [],
                    "n_moments": 0,
                    "n_no_key_moments": 0,
                }
            entry["n_moments" if source is moments else "n_no_key_moments"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": {
            "annotations": str(annotations_path),
            "transcripts": str(transcripts_path),
        },
        "axes": list(axes_mod.AXES),
        "moments": moments,
        "no_key_moments": verdicts,
        "annotators": annotator_roster(moments, verdicts),
        "transcripts": sorted(transcripts.values(), key=lambda t: t["conversation_id"]),
        "transcripts_total": index.total,
    }


def _locate(moments, index):
    """Give every span the row positions the transcript pane highlights on.

    The export numbers a moment in dialogue turns; the pane draws dialogue and
    enrichments together, so where a turn number lands is a fact about the transcript
    and can only be resolved once it has been read. A moment whose transcript is
    missing keeps its turn numbers and gets no row positions -- there is nothing to
    draw it against.
    """
    for moment in moments:
        transcript_id = moment["transcript_id"]
        turn_rows = index.turn_rows(transcript_id) if transcript_id in index else {}
        for span in (moment["boundaries"], moment["original_boundaries"]):
            if span is not None:
                span.update(locate_span(turn_rows, span))


def build_site(annotations_path, transcripts_path, out_path, excluded=()):
    """Write the viewer to out_path and return the payload it embeds."""
    payload = build_payload(annotations_path, transcripts_path, excluded)
    # Embedded in a <script type="application/json"> block: escaping "<" keeps a stray
    # "</script>" inside transcript text from ending the block early.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page = TEMPLATE.read_text().replace(PLACEHOLDER, data)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return payload
