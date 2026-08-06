"""Assemble the viewer payload and write the self-contained page."""

import json
from collections import Counter
from pathlib import Path

from tutormoments_build.annotation_viewer.cases import (
    ACTION_FILTERS,
    action_tags,
    annotated_moments,
    build_cases,
)
from tutormoments_build.annotation_viewer.transcripts import (
    TranscriptIndex,
    moment_turns,
)

TEMPLATE = Path(__file__).with_name("template.html")
PLACEHOLDER = "__VIEWER_DATA__"
DEFAULT_MAX_TURNS = None  # no cap; the transcript box scrolls
PAGE_SIZE = 20


def _read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_payload(annotations_path, transcripts_path, max_turns=DEFAULT_MAX_TURNS, excluded=(),
                  include_single_pass=True):
    """Cases plus the transcript excerpt each one is about."""
    records = _read_records(annotations_path)
    paired = annotated_moments(records, excluded, include_single_pass=include_single_pass)

    index = TranscriptIndex(transcripts_path, {r["transcript_id"] for r in paired})

    moments = {}
    for r in paired:
        moment = r["moment"]
        turns = index.turns(r["transcript_id"]) if r["transcript_id"] in index else {}
        shown, before, after, missing = moment_turns(turns, moment, max_turns)
        moments[moment["id"]] = {
            "start_turn": moment["start_turn"],
            "end_turn": moment["end_turn"],
            "cut_turn": moment["cut_turn"],
            "created_by": moment["created_by"],
            "found": bool(turns),
            "elided_before": before,
            "elided_after": after,
            "missing": missing,
            "tags": action_tags(r),
            "turns": [
                {**turns[n], "cut": n == moment["cut_turn"]} for n in shown
            ],
        }

    cases = build_cases(paired)
    annotators = sorted(
        {c["first"]["annotator"] for c in cases}
        | {c["second"]["annotator"] for c in cases if c["second"]}
    )
    return {
        "moments": moments,
        "cases": cases,
        "annotators": annotators,
        "action_filters": _action_filter_options(moments),
        "page_size": PAGE_SIZE,
        "source": str(annotations_path),
    }


def _action_filter_options(moments):
    """The action values actually present, grouped for the filter, with moment counts."""
    counts = Counter(tag for m in moments.values() for tag in m["tags"])
    groups = []
    for caption, field, _ in ACTION_FILTERS:
        options = sorted(
            (
                {"tag": tag, "label": tag.split(":", 1)[1].replace("_", " "), "count": n}
                for tag, n in counts.items()
                if tag.startswith(f"{field}:")
            ),
            key=lambda o: (-o["count"], o["label"]),
        )
        if options:
            groups.append({"caption": caption, "options": options})
    return groups


def build_site(annotations_path, transcripts_path, out_path,
               max_turns=DEFAULT_MAX_TURNS, excluded=(), include_single_pass=True):
    """Write the viewer to out_path and return the payload it embeds."""
    payload = build_payload(
        annotations_path, transcripts_path, max_turns, excluded, include_single_pass
    )
    # Embedded in a <script type="application/json"> block: escaping "<" keeps a stray
    # "</script>" inside transcript text from ending the block early.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    page = TEMPLATE.read_text().replace(PLACEHOLDER, data)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return payload
