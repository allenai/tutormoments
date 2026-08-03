"""Tests for over-scaffolding decomposition caching in build_ground_truth.py.

Over-scaffolding decomposition works like action/result decomposition: a content
hash keys the cache so a moment is reused when unchanged and re-decomposed when
its text changes. The over-scaffold prompt reads situation + action + result, so
its key must hash all three (unlike action_decompose_key, which hashes only the
action). These tests pin that down.
"""

from tutorsim_build.groundtruth import (
    action_decompose_key,
    overscaffold_decompose_key,
)


def _moment(**extra):
    base = {
        "annotator_id": "ann1",
        "turn_start": 1,
        "turn_end": 10,
        "annotation_type": "scaffolding",
        "situation": "The student is stuck.",
        "action": "The tutor explains the whole answer.",
        "result": "The student copies it down.",
    }
    base.update(extra)
    return base


def test_identical_moments_share_a_key():
    assert overscaffold_decompose_key(_moment()) == overscaffold_decompose_key(
        _moment()
    )


def test_changing_result_changes_the_key():
    # action_decompose_key (action only) would NOT change here -- the over-scaffold
    # key must, because the prompt consumes the result text too.
    a = _moment()
    b = _moment(result="The student realizes their mistake independently.")
    assert overscaffold_decompose_key(a) != overscaffold_decompose_key(b)
    assert action_decompose_key(a) == action_decompose_key(b)


def test_changing_situation_changes_the_key():
    a = _moment()
    b = _moment(situation="The student already solved it once.")
    assert overscaffold_decompose_key(a) != overscaffold_decompose_key(b)


def test_changing_action_changes_the_key():
    a = _moment()
    b = _moment(action="The tutor asks a probing question.")
    assert overscaffold_decompose_key(a) != overscaffold_decompose_key(b)
