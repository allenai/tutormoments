"""Read the transcript rows that moments are located in.

A moment's `start_turn`, `cut_turn` and `end_turn` are dialogue turn numbers: they
count dialogue only, matching each row's `turn_number`, and that is the numbering the
annotator saw, the export records, and the ground-truth builder cuts on. The transcript
pane draws something wider -- the row list the annotation tool rendered, dialogue and
enrichments (pauses, screen updates) interleaved and numbered by `index`. The two
numberings coincide only until the first enrichment, so a moment is placed in the pane
by mapping its turn numbers onto rows (`locate_span`), never by reading them as row
positions.

The file holds every transcript in the collection and runs to tens of megabytes, but
only the annotated ones are ever rendered, so rows are read by byte offset on demand.
"""

import json
import re

_TRANSCRIPT_ID = re.compile(rb'"transcript_id":\s*"([^"]+)"')


class TranscriptIndex:
    """Byte offsets of selected transcripts, with rows read and normalised on demand."""

    def __init__(self, path, wanted):
        self.path = path
        self.total = 0  # transcripts in the file, annotated or not
        self._offsets = {}
        self._cache = {}
        wanted = set(wanted)
        with open(path, "rb") as fh:
            position = 0
            for raw in fh:
                if raw.strip():
                    self.total += 1
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

    def _record(self, transcript_id):
        position, length = self._offsets[transcript_id]
        with open(self.path, "rb") as fh:
            fh.seek(position)
            return json.loads(fh.read(length))

    def conversation_id(self, transcript_id):
        """The id the annotation tool used, which moments are keyed by."""
        return self._record(transcript_id).get("conversation_id")

    def rows(self, transcript_id):
        """One transcript's rows in order, as the pane draws them."""
        if transcript_id not in self._cache:
            self._cache[transcript_id] = _rows(self._record(transcript_id), self.path)
        return self._cache[transcript_id]

    def turn_rows(self, transcript_id):
        """Each dialogue turn number mapped to (first row, last row) carrying it.

        A turn is usually one row, but the v2 numbering sometimes gives two rows the
        same number (and skips the next), so both ends are kept: an edge lands on the
        row that keeps the whole turn inside the moment.
        """
        return _turn_rows(self.rows(transcript_id))


def _rows(record, path):
    rows = record.get("turns") or []
    if rows and "index" not in rows[0]:
        raise ValueError(
            f"{path}: transcript rows carry no 'index'. The pane draws the rendered row "
            "list -- dialogue plus enrichments -- and each row's position in it is what "
            "a moment's turn numbers are resolved to, which only the v2 transcripts "
            "export provides."
        )
    return [_normalise(row) for row in rows]


def _turn_rows(rows):
    spans = {}
    for row in rows:
        number = row["turn_number"]
        if number is None:
            continue
        first, _ = spans.get(number, (row["turn_index"], None))
        spans[number] = (first, row["turn_index"])
    return spans


def locate_span(turn_rows, span):
    """A span's three turn numbers as row positions in the rendered transcript.

    The start resolves to the first row of its turn and the cut and the end to the last
    row of theirs, so an enrichment sitting between the rows of a turn stays on the
    inside of the moment rather than being cut off mid-turn.

    An edge can name a number no row carries -- the v2 numbering skips one wherever it
    repeats one -- so each edge falls back to the nearest row on its own side of the
    span: the start looks forward, the cut and the end look back. A transcript with no
    dialogue rows at all leaves the positions None, and the pane highlights nothing.
    """
    numbers = sorted(turn_rows)
    return {
        "start_row": _first_row_at_or_after(turn_rows, numbers, span["start_turn"]),
        "cut_row": _last_row_at_or_before(turn_rows, numbers, span["cut_turn"]),
        "end_row": _last_row_at_or_before(turn_rows, numbers, span["end_turn"]),
    }


def _first_row_at_or_after(turn_rows, numbers, turn):
    later = [n for n in numbers if n >= turn]
    return turn_rows[later[0]][0] if later else None


def _last_row_at_or_before(turn_rows, numbers, turn):
    earlier = [n for n in numbers if n <= turn]
    return turn_rows[earlier[-1]][1] if earlier else None


def _normalise(row):
    """One row, trimmed to what the transcript pane draws.

    The text keeps its leading `[TUTOR]` / `[SCREEN UPDATE]` tag: the page strips the
    speaker tag it already shows in the role column and renders the rest as inline
    chips, so a screen event that also carries dialogue is never demoted to a label.

    `turn_number` is None on enrichment rows: they sit between turns and the annotator
    never saw a number on them, so the page leaves theirs blank rather than inventing
    one that would disagree with every number around it.
    """
    return {
        "turn_index": row["index"],
        "turn_number": row.get("turn_number"),
        "role": row.get("role") or "",
        "text": str(row.get("text") or ""),
        "type": row.get("type") or "dialogue",
        "timestamp": row.get("timestamp") or "",
    }
