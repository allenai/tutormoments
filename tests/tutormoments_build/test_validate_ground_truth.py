"""The ground-truth input contract must be enforced loudly, not silently defaulted.

annotate.py needs `situation_label_agg`; eval.py needs `strategy_label`,
`situation_label_agg`, `action_direction_agg`, `student_outcome_agg`. These
aggregates are scaffolding-specific (rapport moments legitimately lack them), so
the validator must be annotation_type-aware: a blanket "every moment needs all
four" check would wrongly reject every standard ground-truth file.
"""

import pytest

from tutormoments_build.groundtruth import validate_ground_truth

SCAFFOLDING = {
    "annotation_type": "scaffolding",
    "strategy_label": "scaffold",
    "situation_label_agg": "scaffolding",
    "action_direction_agg": "toward",
    "student_outcome_agg": "pos",
    "turn_start": 1,
}
RAPPORT = {"annotation_type": "rapport", "strategy_label": "warm", "turn_start": 4}


def _gt(*moments):
    return {"conversations": {"c1": {"num_turns": 5, "key_moments": list(moments)}}}


EVAL_KW = dict(
    all_moments=("strategy_label",),
    scaffolding_only=(
        "situation_label_agg",
        "action_direction_agg",
        "student_outcome_agg",
    ),
)


def test_passes_when_all_required_keys_present():
    validate_ground_truth(_gt(dict(SCAFFOLDING), dict(RAPPORT)), **EVAL_KW)  # no raise


def test_rapport_not_required_to_carry_scaffolding_aggregates():
    # RAPPORT has no situation_label_agg/action_direction_agg/student_outcome_agg.
    validate_ground_truth(_gt(dict(RAPPORT)), **EVAL_KW)  # no raise


def test_raises_when_scaffolding_missing_situation_label_agg():
    m = dict(SCAFFOLDING)
    del m["situation_label_agg"]
    with pytest.raises(ValueError, match="situation_label_agg"):
        validate_ground_truth(_gt(m), **EVAL_KW)


def test_raises_when_scaffolding_missing_action_direction_agg():
    m = dict(SCAFFOLDING)
    del m["action_direction_agg"]
    with pytest.raises(ValueError, match="action_direction_agg"):
        validate_ground_truth(_gt(m), **EVAL_KW)


def test_raises_when_any_moment_missing_strategy_label():
    m = dict(RAPPORT)
    del m["strategy_label"]
    with pytest.raises(ValueError, match="strategy_label"):
        validate_ground_truth(_gt(m), **EVAL_KW)


def test_error_message_reports_count_and_example_location():
    m = dict(SCAFFOLDING)
    del m["student_outcome_agg"]
    with pytest.raises(ValueError, match=r"c1"):
        validate_ground_truth(_gt(m), **EVAL_KW)
