"""Turn the raw annotations JSONL into the moments and rosters the page renders.

One line of the export is one moment with every annotation saved against it, or a
"no key moments" verdict on a whole transcript. What the viewer needs from that is
per moment: the latest pass of each role, each pass reduced to its coarse axes,
where the two passes differ, and where the moment actually sits in the transcript
once a reannotator has moved its boundaries.
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


def latest_by_role(annotations):
    """The newest pass of each role.

    Re-saving appends a revision rather than replacing one, so only the highest
    revision of a role is that rater's standing judgment. Ties break on the save
    time, which matters only for exports where revisions were not bumped.
    """
    newest = {}
    for a in annotations:
        role = a.get("role")
        if role not in axes_mod.ROLES:
            continue
        prior = newest.get(role)
        if prior is None or (a["revision"], a["created_at"]) > (
            prior["revision"],
            prior["created_at"],
        ):
            newest[role] = a
    return newest


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
    """
    sar = axes_mod.sar_of(annotation["role"], annotation["payload"])
    return {
        "role": annotation["role"],
        "annotator_id": annotation["annotator_id"],
        "annotator_name": _name(annotation),
        "revision": annotation["revision"],
        "created_at": annotation["created_at"],
        "date": (annotation["created_at"] or "")[:10],
        "is_test": int(annotation.get("is_test") or 0),
        "payload": annotation["payload"],
        "axes": axes_mod.coarse_axes(sar) if sar else None,
    }


def _outcome(passes):
    """How the second pass came out, or that there was not one."""
    first = passes.get("selector")
    second = passes.get("reannotator")
    if first is None or second is None or first["axes"] is None:
        return SINGLE_PASS
    if second["axes"] is None:
        return THROWN_OUT
    return (
        DISAGREEMENT if axes_mod.diff_axes(first["axes"], second["axes"]) else AGREEMENT
    )


def _meta(passes):
    """The reannotator's (else adjudicator's) meta block, which moves boundaries."""
    for role in ("reannotator", "adjudicator"):
        pass_ = passes.get(role)
        meta = (pass_ or {}).get("payload", {}).get("meta")
        if meta:
            return meta
    return {}


_SPAN = ("start", "end", "cut")


def _span(source, suffix):
    """The three positions of a span, in one of the two numberings."""
    return {f"{edge}_{suffix}": source[f"{edge}_{suffix}"] for edge in _SPAN}


def _boundaries(moment, passes):
    """Where the moment sits now, and where it sat before a reannotator moved it.

    Both are in the dialogue turn numbering the export records and the cards print.
    Where that lands in the transcript pane depends on the transcript itself, so the
    row positions are filled in later, once it has been read (`locate_span`).

    A reannotator can redraw the span or the cut point without the export rewriting the
    moment row, so the moved boundaries live in their meta block -- and only the edges
    that actually moved are named there.
    """
    meta = _meta(passes)
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


def _thrown_out(moment, passes):
    """Whether any later pass marked the moment for removal."""
    if moment.get("status") == "thrown_out":
        return True
    return any(
        (passes.get(role) or {}).get("payload", {}).get("meta", {}).get("throw_out")
        for role in ("reannotator", "adjudicator")
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
        passes = {role: _pass(a) for role, a in latest_by_role(kept).items()}
        if not passes:
            continue
        first, second = passes.get("selector"), passes.get("reannotator")
        moved, original = _boundaries(moment, passes)
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
                "passes": [passes[role] for role in axes_mod.ROLES if role in passes],
                "outcome": _outcome(passes),
                "thrown_out": _thrown_out(moment, passes),
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
