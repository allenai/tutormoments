"""Turn the raw annotations JSONL into the moments and rosters the page renders.

One line of the export is one moment with every annotation saved against it, or a
"no key moments" verdict on a whole transcript. What the viewer needs from that is
per moment: the latest pass of every rater who worked it, each pass reduced to its
coarse axes, where the two passes differ, and where the moment actually sits in the
transcript once a reannotator has moved its boundaries.
"""

import json
from pathlib import Path

from tutormoments_build.annotation_viewer import axes as axes_mod

AGREEMENT = "agreement"
DISAGREEMENT = "disagreement"
THROWN_OUT = "thrown out on review"
SINGLE_PASS = "single pass"

# Moments the tool retracted were never really annotated; showing them would put
# withdrawn judgments next to live ones.
RETRACTED = "retracted"


def read_records(path):
    """Every JSON line of an annotations export."""
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def latest_passes(annotations):
    """The newest pass of every rater on a moment, in the order the card shows them.

    Re-saving appends a revision rather than replacing one, so only the highest
    revision a rater filed is their standing judgment. Ties break on the save time,
    which matters only for exports where revisions were not bumped.

    Keyed by rater rather than by role alone: a moment can go to two adjudicators, and
    keying on the role would let the second one quietly replace the first instead of
    standing beside it.

    Within a role they come in the order they first ruled, so the card can number two
    adjudicators 1 and 2. That order is taken from the first pass a rater filed and not
    the one that stands, so revising a judgment does not renumber the columns.
    """
    newest = {}
    first_ruled = {}
    for a in annotations:
        role = a.get("role")
        if role not in axes_mod.ROLES:
            continue
        key = (role, a["annotator_id"])
        created = a["created_at"] or ""
        first_ruled[key] = min(first_ruled.get(key, created), created)
        prior = newest.get(key)
        if prior is None or (a["revision"], a["created_at"]) > (
            prior["revision"],
            prior["created_at"],
        ):
            newest[key] = a
    return [
        newest[key]
        for key in sorted(
            newest, key=lambda k: (axes_mod.ROLES.index(k[0]), first_ruled[k], k[1])
        )
    ]


def by_role(passes):
    """The passes of each role, in filing order. Only adjudicators run to more than one."""
    grouped = {role: [] for role in axes_mod.ROLES}
    for pass_ in passes:
        grouped[pass_["role"]].append(pass_)
    return grouped


def _sole(grouped, role):
    """The pass that anchors agreement for a role filed once per moment.

    Selection and reannotation are single-rater passes -- the first pass and the second
    pass are exactly what agreement is measured between. Were an export ever to carry
    two of either, the earlier one anchors the comparison and both still get a column.
    """
    passes = grouped[role]
    return passes[0] if passes else None


def _name(row):
    """Who this is, as a reader recognises them.

    Ids stay the join key everywhere -- filtering, the roster, kappa pooling -- but a
    page of UUIDs tells a maintainer nothing about who did the work. Exports predating
    the name field fall back to the id.
    """
    return row.get("annotator_name") or row["annotator_id"]


def _pass(annotation):
    """One rater's pass as the page consumes it.

    `axes` is None when the pass recorded no judgment at all -- a reannotator who
    threw the moment out. That is not a judgment to compare, and the card says so
    rather than rendering a column of hollow "not appropriate" chips.

    `sar` is the judgment itself, already unwrapped: an adjudicator files theirs under
    `final`, so a page reading `payload.situation` would find nothing on them and
    render an empty column. Everything the card shows about what a rater judged reads
    `sar`; only `meta`, which stays at the payload root for every role, reads `payload`.
    """
    payload = annotation["payload"] or {}
    role = annotation["role"]
    sar = axes_mod.sar_of(role, payload)
    return {
        "role": role,
        "annotator_id": annotation["annotator_id"],
        "annotator_name": _name(annotation),
        "revision": annotation["revision"],
        "created_at": annotation["created_at"],
        "date": (annotation["created_at"] or "")[:10],
        "is_test": int(annotation.get("is_test") or 0),
        "payload": payload,
        "sar": sar,
        "observations": _observations(role, payload),
        # How the adjudicator resolved the two passes, and why. Only they record it.
        "rationale": (payload.get("rationale") or "").strip(),
        "decisions": payload.get("decisions") or None,
        "axes": axes_mod.coarse_axes(sar) if sar else None,
    }


def _observations(role, payload):
    """Free-text observations, which an adjudicator files under `final` like the rest."""
    source = payload.get("final") if role == "adjudicator" else payload
    return ((source or {}).get("other_observations") or "").strip()


def _outcome(first, second):
    """How the second pass came out, or that there was not one."""
    if first is None or second is None or first["axes"] is None:
        return SINGLE_PASS
    if second["axes"] is None:
        return THROWN_OUT
    return (
        DISAGREEMENT if axes_mod.diff_axes(first["axes"], second["axes"]) else AGREEMENT
    )


def _meta(grouped):
    """The reannotator's (else an adjudicator's) meta block, which moves boundaries.

    Where two adjudicators ruled, the first of them to record a block is the one whose
    boundaries stand -- the same reading as before, which is that the earliest later
    pass carrying one wins.
    """
    for role in ("reannotator", "adjudicator"):
        for pass_ in grouped[role]:
            meta = pass_.get("payload", {}).get("meta")
            if meta:
                return meta
    return {}


_SPAN = ("start", "end", "cut")


def _span(source, suffix):
    """The three positions of a span, in one of the two numberings."""
    return {f"{edge}_{suffix}": source[f"{edge}_{suffix}"] for edge in _SPAN}


def _boundaries(moment, grouped):
    """Where the moment sits now, and where it sat before a reannotator moved it.

    Both are in the dialogue turn numbering the export records and the cards print.
    Where that lands in the transcript pane depends on the transcript itself, so the
    row positions are filled in later, once it has been read (`locate_span`).

    A reannotator can redraw the span or the cut point without the export rewriting the
    moment row, so the moved boundaries live in their meta block -- and only the edges
    that actually moved are named there.
    """
    meta = _meta(grouped)
    original = _span(moment, "turn")
    moved = dict(original)
    if meta.get("changed_boundaries"):
        for edge in ("start", "end"):
            _move(moved, meta, edge)
    if meta.get("redrew_cut_point"):
        _move(moved, meta, "cut")
    return moved, (original if moved != original else None)


def _move(moved, meta, edge):
    """One edge of the span, wherever the reannotator redrew it."""
    if meta.get(f"new_{edge}_turn") is not None:
        moved[f"{edge}_turn"] = meta[f"new_{edge}_turn"]


def _thrown_out(moment, grouped):
    """Whether any later pass marked the moment for removal."""
    if moment.get("status") == "thrown_out":
        return True
    return any(
        (pass_.get("payload", {}).get("meta") or {}).get("throw_out")
        for role in ("reannotator", "adjudicator")
        for pass_ in grouped[role]
    )


def build_moments(records, excluded=()):
    """Every annotated moment, newest pass per role, ready to filter and render."""
    excluded = set(excluded)
    moments = []
    for record in records:
        moment = record.get("moment")
        if not moment or moment.get("status") == RETRACTED:
            continue
        kept = [
            a
            for a in record.get("annotations") or []
            if a["annotator_id"] not in excluded
        ]
        passes = [_pass(a) for a in latest_passes(kept)]
        if not passes:
            continue
        grouped = by_role(passes)
        first, second = _sole(grouped, "selector"), _sole(grouped, "reannotator")
        moved, original = _boundaries(moment, grouped)
        moments.append(
            {
                "id": moment["moment_id"],
                "transcript_id": record["transcript_id"],
                "start_turn": moment["start_turn"],
                "end_turn": moment["end_turn"],
                "cut_turn": moment["cut_turn"],
                "status": moment.get("status") or "",
                # The name is what a reader recognises; the id is the fallback for
                # exports predating it.
                "created_by": moment.get("created_by_name")
                or moment.get("created_by")
                or "",
                "is_test": int(moment.get("is_test") or 0),
                "passes": passes,
                "outcome": _outcome(first, second),
                "thrown_out": _thrown_out(moment, grouped),
                "boundaries": moved,
                "original_boundaries": original,
                "diff": sorted(
                    axes_mod.diff_axes(first["axes"], second["axes"])
                    if first and second and first["axes"] and second["axes"]
                    else ()
                ),
                # Rater-vs-rater values, pooled across moments for the kappa table.
                "pairs": (
                    axes_mod.axis_pairs(first["axes"], second["axes"])
                    if first and second and first["axes"] and second["axes"]
                    else {}
                ),
            }
        )
    return moments


def no_key_moment_verdicts(records, excluded=()):
    """Transcripts a rater read through and found no key moment in.

    They carry no moment of their own, so nothing else in the viewer would show that
    the transcript was looked at rather than skipped.
    """
    excluded = set(excluded)
    out = []
    for record in records:
        verdict = record.get("no_key_moments_record")
        if not verdict or verdict["annotator_id"] in excluded:
            continue
        out.append(
            {
                "transcript_id": record["transcript_id"],
                "annotator_id": verdict["annotator_id"],
                "annotator_name": _name(verdict),
                "created_at": verdict["created_at"],
                "date": (verdict["created_at"] or "")[:10],
                "is_test": int(verdict.get("is_test") or 0),
                "note": (verdict.get("payload") or {}).get("note") or "",
            }
        )
    return out


def annotator_roster(moments, verdicts):
    """Everyone who annotated, with what they did -- the annotator filter's options.

    Counts are of passes, not moments: a rater who reviewed a moment someone else cut
    contributed work to it, and their reannotation is what the count is about.
    """
    roster = {}

    def entry(annotator_id, name):
        return roster.setdefault(
            annotator_id,
            {
                "id": annotator_id,
                "name": name,
                "selector": 0,
                "reannotator": 0,
                "adjudicator": 0,
                "no_key_moments": 0,
                "total": 0,
                "is_test": 0,
            },
        )

    for moment in moments:
        for pass_ in moment["passes"]:
            row = entry(pass_["annotator_id"], pass_["annotator_name"])
            row[pass_["role"]] += 1
            row["total"] += 1
            row["is_test"] = max(row["is_test"], pass_["is_test"])
    for verdict in verdicts:
        row = entry(verdict["annotator_id"], verdict["annotator_name"])
        row["no_key_moments"] += 1
        row["total"] += 1
        row["is_test"] = max(row["is_test"], verdict["is_test"])
    # The filter menu reads as a list of people, so it sorts like one.
    return sorted(roster.values(), key=lambda r: (r["name"], r["id"]))
