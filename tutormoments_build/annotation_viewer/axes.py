"""Coarse agreement axes -- the single definition of what a "disagreement" is.

Ported from the annotation platform the raters used (edu_dense_annotation
`v2/backend/axes.py` and its frontend mirror) so that a moment counts as a
disagreement here exactly when it did there. Keep the two in lockstep: changing an
axis changes every agreement number the viewer reports.

The axes are deliberately coarse. Strategy and reason checkboxes are shown on the
cards but never compared: they are unlocked by the yes/no field above them, so two
raters who ticked different strategies would otherwise be counted as disagreeing
twice over the same judgment.
"""

# (key, label, construct) for every compared axis. `construct` groups an axis under
# the situation / action / result row it belongs to.
#
# `reported` decides whether the axis gets a row in the agreement table. The result
# axes are set False: they are still compared, so two raters who split on them still
# make the moment a disagreement on its card, but their kappa is not a number the team
# is tracking, and reporting it invited reading too much into it.
AXES = (
    {
        "key": "scaffolding_appropriate",
        "label": "scaffolding appropriate",
        "construct": "sit",
        "reported": True,
    },
    {
        "key": "rigor_appropriate",
        "label": "rigor appropriate",
        "construct": "sit",
        "reported": True,
    },
    {
        "key": "scaffolding_present",
        "label": "scaffolding present",
        "construct": "act",
        "reported": True,
    },
    {
        "key": "over_scaffolding",
        "label": "over-scaffolding",
        "construct": "act",
        "reported": True,
    },
    {
        "key": "rigor_present",
        "label": "rigor present",
        "construct": "act",
        "reported": True,
    },
    {
        "key": "meaningful_success",
        "label": "meaningful success",
        "construct": "res",
        "reported": False,
    },
    {
        "key": "high_engagement",
        "label": "high engagement",
        "construct": "res",
        "reported": False,
    },
)

AXIS_KEYS = tuple(a["key"] for a in AXES)
REPORTED_AXIS_KEYS = tuple(a["key"] for a in AXES if a["reported"])

CONSTRUCTS = ("sit", "act", "res")
CONSTRUCT_LABELS = {"sit": "Situation", "act": "Action", "res": "Result"}

ROLES = ("selector", "reannotator", "adjudicator")


def sar_of(role, payload):
    """The situation/action/result a pass recorded, or None when it recorded none.

    An adjudicator wraps their final call under `final`. A reannotator who threw the
    moment out saves only `meta`, leaving no judgment to compare against.
    """
    if not payload:
        return None
    if role == "adjudicator":
        return payload.get("final") or None
    sar = {k: payload.get(k) for k in ("situation", "action", "result")}
    if sar["situation"] is None and sar["action"] is None and sar["result"] is None:
        return None
    return sar


def coarse_axes(sar):
    """One pass's judgment reduced to the booleans agreement is measured on.

    `over_scaffolding` is None when the rater saw no scaffolding at all: they were
    never asked how much of it there was, so there is nothing to agree about.
    """
    if not sar:
        return dict.fromkeys(AXIS_KEYS)
    sit = sar.get("situation") or {}
    act = sar.get("action") or {}
    res = sar.get("result") or {}
    present = bool(act.get("scaffolding_present"))
    return {
        "scaffolding_appropriate": bool(sit.get("scaffolding_appropriate")),
        "rigor_appropriate": bool(sit.get("rigor_appropriate")),
        "scaffolding_present": present,
        "over_scaffolding": (act.get("scaffolding_amount") == "over_scaffolding")
        if present
        else None,
        "rigor_present": bool(act.get("rigor_present")),
        "meaningful_success": res.get("problem_success") == "meaningful_success",
        "high_engagement": res.get("cognitive_engagement") == "high_or_good",
    }


def diff_axes(a, b):
    """The axes two passes judged differently, skipping ones either left unjudged."""
    out = set()
    for key in AXIS_KEYS:
        if key == "over_scaffolding" and not (
            a.get("scaffolding_present") and b.get("scaffolding_present")
        ):
            continue
        av, bv = a.get(key), b.get(key)
        if av is not None and bv is not None and av != bv:
            out.add(key)
    return out


def axis_pairs(first, second):
    """Rater-vs-rater values per axis, for the axes both raters judged.

    Cohen's kappa is these pairs pooled across whichever moments the filters select,
    which is why the page computes it in the browser and this module only ships the
    pairs: there is no Python running once the file is open.
    """
    pairs = {}
    for key in AXIS_KEYS:
        if key == "over_scaffolding" and not (
            first.get("scaffolding_present") and second.get("scaffolding_present")
        ):
            continue
        av, bv = first.get(key), second.get(key)
        if av is not None and bv is not None:
            pairs[key] = [av, bv]
    return pairs
