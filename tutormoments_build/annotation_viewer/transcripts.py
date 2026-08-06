"""Read the transcript rows that moments are numbered against.

Moments record positions in the row list the annotation tool rendered: dialogue turns and
enrichments (pauses, screen updates) interleaved, numbered by `index`. That numbering is
what `start_turn`, `cut_turn` and `end_turn` mean, so only an export carrying `index` can
locate a moment. Numbering by dialogue turns alone drifts further out of step with every
enrichment that precedes the moment.
"""

import json
import re

_TRANSCRIPT_ID = re.compile(rb'"transcript_id":\s*"([^"]+)"')


class TranscriptIndex:
    """Byte offsets of selected transcripts, with rows read and normalised on demand."""

    def __init__(self, path, wanted):
        self.path = path
        self._offsets = {}
        self._cache = {}
        wanted = set(wanted)
        with open(path, "rb") as fh:
            position = 0
            for raw in fh:
                found = _TRANSCRIPT_ID.search(raw, 0, 200)
                if found:
                    tid = found.group(1).decode()
                    if tid in wanted:
                        self._offsets[tid] = (position, len(raw))
                position += len(raw)

    def __contains__(self, transcript_id):
        return transcript_id in self._offsets

    def __len__(self):
        return len(self._offsets)

    def turns(self, transcript_id):
        """One transcript's rows, keyed by the index moments are numbered against."""
        if transcript_id not in self._cache:
            position, length = self._offsets[transcript_id]
            with open(self.path, "rb") as fh:
                fh.seek(position)
                record = json.loads(fh.read(length))
            self._cache[transcript_id] = _rows(record, self.path)
        return self._cache[transcript_id]


def _rows(record, path):
    rows = record.get("turns") or []
    if rows and "index" not in rows[0]:
        raise ValueError(
            f"{path}: transcript rows carry no 'index'. Moments are numbered against the "
            "rendered row list (dialogue plus enrichments), which only the v2 transcripts "
            "export provides. An export numbering dialogue turns alone silently mis-locates "
            "every moment that follows an enrichment."
        )
    return {row["index"]: _normalise(row) for row in rows}


def _normalise(row):
    """One row as the viewer needs it: speaker separated from text, enrichments marked.

    Row text repeats its own speaker or enrichment type inline ("[TUTOR] Hi there"), which
    would read as a stutter next to a speaker column, so the prefix is stripped off.
    """
    kind = row.get("type") or "dialogue"
    enrichment = kind != "dialogue"
    label = kind.strip("[]").lower() if enrichment else (row.get("role") or "").title()
    text = str(row.get("text") or "")
    prefix = kind if enrichment else f"[{row.get('role')}]"
    if text.startswith(prefix):
        text = text[len(prefix):].lstrip()
    return {
        "n": row["index"],
        "role": label,
        "text": text,
        "enrichment": enrichment,
        "timestamp": row.get("timestamp") or "",
    }


def moment_turns(turns, moment, max_turns=None):
    """The rows to show for a moment, windowed on the cut row only if a cap is given.

    Returns (row indices, elided before, elided after, missing). Uncapped by default: the
    transcript box scrolls, so a cap hides content without making the card any shorter.
    `missing` counts rows the span names that the transcript does not have, so a short
    excerpt can say so rather than look complete.
    """
    span = range(moment["start_turn"], moment["end_turn"] + 1)
    present = [n for n in span if n in turns]
    missing = len(span) - len(present)
    if max_turns is None or len(present) <= max_turns:
        return present, 0, 0, missing
    cut = moment["cut_turn"]
    centre = present.index(cut) if cut in present else len(present) // 2
    start = max(0, min(centre - max_turns // 2, len(present) - max_turns))
    return present[start : start + max_turns], start, len(present) - start - max_turns, missing
