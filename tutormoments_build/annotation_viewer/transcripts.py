"""Read the transcript rows that moments are located in.

A moment's `start_turn`, `cut_turn` and `end_turn` are dialogue turn numbers: they
count dialogue only, matching each row's `turn_number`, and that is the numbering the
annotator saw, the export records, and the ground-truth builder cuts on. The transcript
pane draws something wider -- the row list the annotation tool rendered, dialogue and
enrichments (pauses, screen updates) interleaved and numbered by `index`. The two
numberings coincide only until the first enrichment, so a moment is placed in the pane
by mapping its turn numbers onto rows (`locate_span`), never by reading them as row
positions.

The file holds every transcript in the collection and runs to gigabytes, but
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

    def problems(self, transcript_id):
        """The session's problems deck by deck, for the cards the page flips through.

        Only some providers record them -- tutoring_provider_c lists each slide the
        session covered with its learning objective -- and none record the problem text
        itself or when in the session a slide was up, so a card says what was covered
        and nothing about which turns it covers.

        A slide's position counts within its learning objective's deck, so the cards
        keep the decks in the order the provider lists them and sort slides only
        within one: sorting on position alone would shuffle two decks together.
        """
        problems = self._record(transcript_id).get("problems") or []
        decks = {}
        for p in problems:
            decks.setdefault(_objective_id(p), len(decks))
        return [
            _problem(p)
            for p in sorted(
                problems,
                key=lambda p: (decks[_objective_id(p)], p.get("slide_position") or 0),
            )
        ]

    def turn_rows(self, transcript_id):
        """Each dialogue turn number mapped to (first row, last row) carrying it.

        A turn is usually one row, but the v2 numbering sometimes gives two rows the
        same number (and skips the next), so both ends are kept: an edge lands on the
        row that keeps the whole turn inside the moment.
        """
        return _turn_rows(self.rows(transcript_id))


def _rows(record, path):
    rows = record.get("turns") or []
    if rows and _plain_dialogue(rows):
        rows = _interleave(rows, record.get("enrichments") or [])
    if rows and "index" not in rows[0]:
        raise ValueError(
            f"{path}: transcript rows carry no 'index'. The pane draws the rendered row "
            "list -- dialogue plus enrichments -- and each row's position in it is what "
            "a moment's turn numbers are resolved to, which only the v2 transcripts "
            "export provides."
        )
    return [_normalise(row) for row in rows]


def _plain_dialogue(rows):
    """Whether `turns` holds dialogue alone, as in the per-provider exports.

    There enrichments, where a provider has them, sit in a list of their own beside
    `turns` (tutoring_provider_a's) or are absent altogether (tutoring_provider_c's),
    and the rendered row list is rebuilt from the two (`_interleave`). They are told
    apart by shape -- a `start_seconds` and no `type` or `index` -- rather than accepted
    for merely lacking an `index`: the older provider A export lacks one too, and it is
    missing the enrichment rows its annotators saw.
    """
    return all(
        "start_seconds" in row and "type" not in row and "index" not in row
        for row in rows
    )


def _interleave(turns, enrichments):
    """Dialogue and enrichments as one row list, numbered in the order they are drawn.

    An enrichment is drawn just before the dialogue turn its `before_turn` names, in
    enrichment-number order, and one naming no turn -- whatever followed the last line
    of dialogue -- after all of them. The annotation tool folded some screen
    interactions into the turn before them instead of giving them a row; the export
    does not say which, so here every enrichment gets its own row. Moments are placed
    by turn number, so this changes what the pane shows, not where a moment lands.
    """
    waiting = {}
    for enrichment in sorted(enrichments, key=lambda e: e["enrichment_number"]):
        waiting.setdefault(enrichment.get("before_turn"), []).append(enrichment)
    ordered = []
    for turn in turns:
        ordered += [_enrichment_row(e) for e in waiting.pop(turn["turn_number"], [])]
        ordered.append(_dialogue_row(turn))
    # Left over: those after the last turn, and any naming a turn the dialogue lacks.
    ordered += [
        _enrichment_row(e)
        for key in sorted(waiting, key=lambda k: (k is None, k or 0))
        for e in waiting[key]
    ]
    return [row | {"index": position} for position, row in enumerate(ordered)]


def _dialogue_row(row):
    """A dialogue turn in the v2 export's shape."""
    return row | {
        "role": str(row.get("role") or "").upper(),
        "type": "dialogue",
        "timestamp": _clock(row.get("start_seconds")),
    }


def _enrichment_row(enrichment):
    """An enrichment in the v2 export's shape: a bracketed tag the page draws as a chip.

    The export names no speaker for an enrichment, so the row takes neither speaker's
    colour; who acted is in the text ("Tutor types ...").
    """
    tag = f"[{str(enrichment.get('type') or 'enrichment').replace('_', ' ').upper()}]"
    label = str(enrichment.get("label") or "").strip()
    content = str(enrichment.get("content") or "").strip()
    start, end = enrichment.get("start_seconds"), enrichment.get("end_seconds")
    return {
        "turn_number": None,
        "role": "",
        "type": tag,
        "text": " ".join(part for part in (tag, label + (" —" if label and content else ""), content) if part),
        "timestamp": _clock(start)
        + (f"-{_clock(end)}" if end is not None and end != start else ""),
    }


def _clock(seconds):
    return (
        f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}" if seconds is not None else ""
    )


def _objective_id(problem):
    return (problem.get("learning_objective") or {}).get("id")


def _problem(problem):
    objective = problem.get("learning_objective") or {}
    return {
        "slide_id": problem.get("slide_id"),
        "slide_position": problem.get("slide_position"),
        "objective": str(objective.get("title") or "").strip(),
        "objective_id": objective.get("id"),
        "year": objective.get("year"),
    }


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
