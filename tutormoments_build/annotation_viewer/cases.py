"""Turn raw annotation records into per-construct comparison cases.

A *case* is one moment seen through one construct (situation, action, result): the first
pass's judgment, the second pass's judgment, and whether they agreed. The three constructs
are kept separate because raters can agree about what the tutor did and still disagree
about whether it worked.
"""

# Fields compared to decide agreement. Multi-selects are shown but deliberately excluded:
# they are unlocked by the fields above them, so two raters who chose different amounts would
# otherwise be counted as disagreeing twice.
CONSTRUCTS = {
    "situation": ("scaffolding_appropriate", "rigor_appropriate"),
    "action": (
        "scaffolding_present",
        "scaffolding_amount",
        "rigor_present",
        "rigor_amount",
    ),
    "result": ("problem_success", "cognitive_engagement"),
}

# Checkbox groups displayed under each construct: (caption, list field, free-text "other" field).
MULTI_SELECTS = {
    "situation": (),
    "action": (
        ("scaffolding strategies", "scaffolding_strategies", "scaffolding_strategy_other"),
        ("over-scaffolding reasons", "over_scaffolding_reasons", None),
        ("rigor strategies", "rigor_strategies", "rigor_strategy_other"),
    ),
    "result": (),
}

# Action values offered as filters: (caption, payload field, is a multi-select).
ACTION_FILTERS = (
    ("Scaffolding amount", "scaffolding_amount", False),
    ("Scaffolding strategy", "scaffolding_strategies", True),
    ("Over-scaffolding reason", "over_scaffolding_reasons", True),
    ("Rigor amount", "rigor_amount", False),
    ("Rigor strategy", "rigor_strategies", True),
)

# The rater's own prose for each construct.
FREE_TEXT = {"situation": "why", "action": "explanation", "result": "explanation"}

AGREEMENT = "agreement"
DISAGREEMENT = "disagreement"
NO_SECOND_JUDGMENT = "no second judgment"
SINGLE_PASS = "single pass"


def latest_revisions(annotations):
    """Keep one annotation per annotator: their highest revision.

    Re-saving appends a new revision rather than replacing the old one, and no annotator
    holds two roles on the same moment, so annotator_id alone keys this cleanly.
    """
    newest = {}
    for a in annotations:
        prior = newest.get(a["annotator_id"])
        if prior is None or (a["revision"], a["created_at"]) > (
            prior["revision"],
            prior["created_at"],
        ):
            newest[a["annotator_id"]] = a
    return list(newest.values())


def annotated_moments(records, excluded=(), include_single_pass=False):
    """Moments with a first pass, each carrying its second pass when one exists.

    Most moments were never reviewed, so `second` is None far more often than not. Including
    them is what lets the viewer cover every annotator rather than only the eight whose work
    happened to be sampled for a second opinion.
    """
    kept = []
    for r in records:
        if "moment" not in r or r["moment"]["status"] == "retracted":
            continue
        current = [
            a for a in latest_revisions(r["annotations"]) if a["annotator_id"] not in excluded
        ]
        first = next((a for a in current if a["role"] == "selector"), None)
        second = next((a for a in current if a["role"] == "reannotator"), None)
        if first and (second or include_single_pass):
            kept.append({**r, "first": first, "second": second})
    return kept


def paired_moments(records, excluded=()):
    """Moments carrying both a first and a second pass."""
    return annotated_moments(records, excluded, include_single_pass=False)


def action_tags(record):
    """Every action value either pass recorded on a moment, as "field:value" tags.

    Tags belong to the moment rather than to one case: the reason to filter by, say,
    over-scaffolding is usually to read the situation and result of those moments too, not
    only their action panels. Either pass counts -- a value one rater saw is worth finding
    even when the other did not.
    """
    tags = set()
    for annotation in (record.get("first"), record.get("second")):
        action = (annotation or {}).get("payload", {}).get("action") or {}
        for _, field, multi in ACTION_FILTERS:
            recorded = action.get(field)
            for value in (recorded or []) if multi else ([recorded] if recorded else []):
                tags.add(f"{field}:{value}")
    return sorted(tags)


def _judgment(annotation, construct):
    """One rater's section of the payload, or None when they recorded none.

    A reannotator who threw the moment out leaves the section empty; that is not a
    disagreement about the construct, it is the absence of a second opinion.
    """
    return (annotation["payload"].get(construct) or None) if annotation else None


def _boxes(judgment, field, other_field):
    """Every box ticked in one checkbox group, the free-text "other" entry included."""
    ticked = list((judgment or {}).get(field) or [])
    if judgment and other_field and judgment.get(other_field):
        ticked.append(f'other: "{judgment[other_field]}"')
    return ticked


def _side(judgment, other, construct, annotation):
    """One rater's half of a case: compared fields, checkbox groups, and their prose."""
    fields = [
        {
            "name": name,
            "value": judgment.get(name),
            # With no opposite judgment there is nothing to differ from, so nothing is flagged.
            "agrees": other is None or judgment.get(name) == other.get(name),
        }
        for name in CONSTRUCTS[construct]
    ]
    multi = []
    for caption, field, other_field in MULTI_SELECTS[construct]:
        ticked = _boxes(judgment, field, other_field)
        if not ticked:
            continue
        shared = set(_boxes(other, field, other_field))
        multi.append(
            {"caption": caption, "boxes": [{"box": b, "shared": b in shared} for b in ticked]}
        )
    return {
        "annotator": annotation["annotator_id"],
        "judged": True,
        "fields": fields,
        "multi": multi,
        "text": judgment.get(FREE_TEXT[construct]) or "",
    }


def _no_judgment(annotation):
    """A pass that recorded nothing for this construct, having thrown the moment out.

    Kept distinct from a judgment with no prose: the panel must say the rater left no
    judgment rather than look like they reviewed it and had nothing to add.
    """
    return {
        "annotator": annotation["annotator_id"],
        "judged": False,
        "fields": [],
        "multi": [],
        "text": "",
    }


def build_cases(paired):
    """One case per (moment, construct), classified as agreement or disagreement."""
    cases = []
    for r in paired:
        moment = r["moment"]
        for construct in CONSTRUCTS:
            first = _judgment(r["first"], construct)
            second = _judgment(r["second"], construct)
            if first is None:
                continue  # a first pass that recorded nothing is not a case to review
            if r["second"] is None:
                outcome = SINGLE_PASS  # never reviewed, so there is nothing to compare
            elif second is None:
                outcome = NO_SECOND_JUDGMENT  # reviewed, but the reviewer threw it out
            elif all(first.get(n) == second.get(n) for n in CONSTRUCTS[construct]):
                outcome = AGREEMENT
            else:
                outcome = DISAGREEMENT
            cases.append(
                {
                    "moment_id": moment["id"],
                    "construct": construct,
                    "outcome": outcome,
                    "first": _side(first, second, construct, r["first"]),
                    # None renders as a single full-width panel; a reviewer who left no
                    # judgment still gets a panel saying so.
                    "second": (
                        None
                        if r["second"] is None
                        else _side(second, first, construct, r["second"])
                        if second is not None
                        else _no_judgment(r["second"])
                    ),
                }
            )
    return cases
