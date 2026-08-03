"""Tests for the per-cluster action/result facet aggregation in build_ground_truth.py.

Covers the custom business logic added for action_direction_agg / student_outcome_agg:
clustering scaffolding moments, unifying their decomposed facets (one contribution
per annotator), planning reuse-vs-classify decisions against the cache, and
reconstructing that cache from existing ground truth files. Also smoke-tests that
refactoring compute_situation_label_agg onto the shared cluster helper preserved
its behavior.
"""

import json

from tutormoments_build.groundtruth import (
    DEFAULT_RESULT_LABEL,
    _invalidate_agg_cache,
    _scaffolding_clusters,
    _unify_facets,
    compute_situation_label_agg,
    load_existing_action_result_agg,
    moment_key,
    plan_action_result_agg,
)


def _moment(
    annotator_id,
    turn_start,
    turn_end,
    annotation_type="scaffolding",
    action_decomposed=None,
    result_decomposed=None,
    situation_label=None,
    result="r",
):
    m = {
        "annotator_id": annotator_id,
        "turn_start": turn_start,
        "turn_end": turn_end,
        "annotation_type": annotation_type,
        "result": result,
        "action_decomposed": action_decomposed if action_decomposed is not None else [],
        "result_decomposed": result_decomposed if result_decomposed is not None else [],
    }
    if situation_label is not None:
        m["situation_label"] = situation_label
    return m


# --- _scaffolding_clusters -------------------------------------------------


def test_scaffolding_clusters_ignores_rapport_moments():
    moments = [
        _moment("t1", 1, 5, annotation_type="rapport"),
        _moment("t2", 1, 5, annotation_type="scaffolding"),
    ]
    clusters = _scaffolding_clusters(moments)
    assert len(clusters) == 1
    indices, cluster_moments = clusters[0]
    assert indices == [1]
    assert cluster_moments == [moments[1]]


def test_scaffolding_clusters_groups_overlapping_different_annotators():
    moments = [
        _moment("t1", 1, 10),
        _moment("t2", 1, 10),  # identical range as t1 -- IoU == 1.0
        _moment("t3", 50, 60),  # disjoint -- separate cluster
    ]
    clusters = _scaffolding_clusters(moments)
    cluster_sets = sorted(
        [frozenset(idxs) for idxs, _ in clusters], key=lambda s: min(s)
    )
    assert cluster_sets == [frozenset({0, 1}), frozenset({2})]


def test_scaffolding_clusters_does_not_directly_link_same_annotator():
    # Same annotator, overlapping ranges -- _cluster_by_iou skips direct same-annotator
    # links, and with no third annotator to bridge them they land in separate clusters.
    moments = [
        _moment("t1", 1, 10),
        _moment("t1", 2, 10),
    ]
    clusters = _scaffolding_clusters(moments)
    assert len(clusters) == 2


# --- _unify_facets ----------------------------------------------------------


def test_unify_facets_concatenates_across_annotators_in_order():
    cluster_moments = [
        _moment("t1", 1, 10, action_decomposed=["a1", "a2"], result_decomposed=["r1"]),
        _moment("t2", 2, 10, action_decomposed=["b1"], result_decomposed=["r2", "r3"]),
    ]
    unified_action, unified_result = _unify_facets(cluster_moments)
    assert unified_action == ["a1", "a2", "b1"]
    assert unified_result == ["r1", "r2", "r3"]


def test_unify_facets_keeps_only_first_occurrence_per_annotator():
    # t1 appears twice (e.g. transitively grouped via t2) -- only the first
    # contributes, mirroring compute_situation_label_agg's annotator dedup.
    cluster_moments = [
        _moment("t1", 1, 10, action_decomposed=["a1"], result_decomposed=["r1"]),
        _moment("t2", 5, 14, action_decomposed=["b1"], result_decomposed=["r2"]),
        _moment(
            "t1",
            9,
            18,
            action_decomposed=["a2-should-be-dropped"],
            result_decomposed=["r2-dropped"],
        ),
    ]
    unified_action, unified_result = _unify_facets(cluster_moments)
    assert unified_action == ["a1", "b1"]
    assert unified_result == ["r1", "r2"]


# --- compute_situation_label_agg (regression after refactor onto _scaffolding_clusters) ---


def test_compute_situation_label_agg_majority_votes_overlapping_cluster():
    moments = [
        _moment("t1", 1, 10, situation_label={"scaffolding": "yes", "rigor": "no"}),
        _moment("t2", 1, 10, situation_label={"scaffolding": "yes", "rigor": "no"}),
        _moment("t3", 1, 10, situation_label={"scaffolding": "no", "rigor": "yes"}),
        _moment("t4", 80, 90, annotation_type="rapport"),
    ]
    agg = compute_situation_label_agg(moments)
    assert agg == {0: "scaffolding", 1: "scaffolding", 2: "scaffolding"}
    assert 3 not in agg


def test_compute_situation_label_agg_unknown_when_no_signal():
    moments = [
        _moment(
            "t1",
            1,
            10,
            situation_label={"scaffolding": "no_mention", "rigor": "unclear"},
        ),
        _moment(
            "t2", 2, 10, situation_label={"scaffolding": None, "rigor": "no_mention"}
        ),
    ]
    agg = compute_situation_label_agg(moments)
    assert agg == {0: "unknown", 1: "unknown"}


# --- plan_action_result_agg --------------------------------------------------


def _plan_for(moments, cached_agg=None):
    cached_agg = cached_agg or {}
    to_action, to_result = [], []
    plan = plan_action_result_agg("conv1", moments, cached_agg, to_action, to_result)
    return plan, to_action, to_result


def test_plan_defaults_when_cluster_has_no_facets():
    moments = [_moment("t1", 1, 10), _moment("t2", 1, 10)]
    plan, to_action, to_result = _plan_for(moments)
    assert len(plan) == 1
    _, action_item, result_item = plan[0]
    assert action_item == ("default", "unknown")
    assert result_item == ("default", DEFAULT_RESULT_LABEL)
    assert to_action == []
    assert to_result == []


def test_plan_queues_classification_when_no_cache_entry():
    moments = [
        _moment("t1", 1, 10, action_decomposed=["a1"], result_decomposed=["r1"]),
        _moment("t2", 1, 10, action_decomposed=["b1"], result_decomposed=["r2"]),
    ]
    plan, to_action, to_result = _plan_for(moments)
    assert len(plan) == 1
    cluster_indices, action_item, result_item = plan[0]
    assert cluster_indices == [0, 1]
    assert action_item[0] == "classify"
    assert result_item[0] == "classify"
    assert to_action == [{"key": action_item[1], "facets": ["a1", "b1"]}]
    assert to_result == [{"key": result_item[1], "facets": ["r1", "r2"]}]


def test_plan_reuses_cache_when_unified_facets_unchanged():
    moments = [
        _moment("t1", 1, 10, action_decomposed=["a1"], result_decomposed=["r1"]),
        _moment("t2", 1, 10, action_decomposed=["b1"], result_decomposed=["r2"]),
    ]
    sig = frozenset(moment_key(m) for m in moments)
    cached_agg = {
        "conv1": {
            sig: {
                "unified_action": ["a1", "b1"],
                "unified_result": ["r1", "r2"],
                "action_direction_agg": "scaffolding",
                "student_outcome_agg": "pos",
            }
        }
    }
    plan, to_action, to_result = _plan_for(moments, cached_agg)
    _, action_item, result_item = plan[0]
    assert action_item == ("reuse", "scaffolding")
    assert result_item == ("reuse", "pos")
    assert to_action == []
    assert to_result == []


def test_plan_reclassifies_when_unified_facets_changed():
    moments = [
        _moment("t1", 1, 10, action_decomposed=["a1-edited"], result_decomposed=["r1"]),
        _moment("t2", 1, 10, action_decomposed=["b1"], result_decomposed=["r2"]),
    ]
    sig = frozenset(moment_key(m) for m in moments)
    # Cached entry exists for this exact cluster, but the action facets drifted
    # (e.g. re-decomposition changed the text) -- it should be reclassified while
    # the unchanged result facets are still reused.
    cached_agg = {
        "conv1": {
            sig: {
                "unified_action": ["a1-old"],
                "unified_result": ["r1", "r2"],
                "action_direction_agg": "rigor",
                "student_outcome_agg": "neg",
            }
        }
    }
    plan, to_action, to_result = _plan_for(moments, cached_agg)
    _, action_item, result_item = plan[0]
    assert action_item[0] == "classify"
    assert to_action == [{"key": action_item[1], "facets": ["a1-edited", "b1"]}]
    assert result_item == ("reuse", "neg")
    assert to_result == []


# --- _invalidate_agg_cache (selective --refresh-agg) -------------------------


def _full_cache(moments):
    sig = frozenset(moment_key(m) for m in moments)
    return {
        "conv1": {
            sig: {
                "unified_action": ["a1", "b1"],
                "unified_result": ["r1", "r2"],
                "action_direction_agg": "scaffolding",
                "student_outcome_agg": "pos",
            }
        }
    }


def _cached_moments():
    return [
        _moment("t1", 1, 10, action_decomposed=["a1"], result_decomposed=["r1"]),
        _moment("t2", 1, 10, action_decomposed=["b1"], result_decomposed=["r2"]),
    ]


def test_invalidate_agg_cache_action_reclassifies_action_reuses_result():
    moments = _cached_moments()
    invalidated = _invalidate_agg_cache(_full_cache(moments), "action")
    plan, to_action, to_result = _plan_for(moments, invalidated)
    _, action_item, result_item = plan[0]
    assert action_item[0] == "classify"  # action label cleared -> reclassify
    assert to_action == [{"key": action_item[1], "facets": ["a1", "b1"]}]
    assert result_item == ("reuse", "pos")  # result untouched -> still reused
    assert to_result == []


def test_invalidate_agg_cache_result_reclassifies_result_reuses_action():
    moments = _cached_moments()
    invalidated = _invalidate_agg_cache(_full_cache(moments), "result")
    plan, to_action, to_result = _plan_for(moments, invalidated)
    _, action_item, result_item = plan[0]
    assert action_item == ("reuse", "scaffolding")  # action untouched -> still reused
    assert to_action == []
    assert result_item[0] == "classify"  # result label cleared -> reclassify
    assert to_result == [{"key": result_item[1], "facets": ["r1", "r2"]}]


def test_invalidate_agg_cache_both_wipes_entire_cache():
    moments = _cached_moments()
    assert _invalidate_agg_cache(_full_cache(moments), "both") == {}


def test_invalidate_agg_cache_does_not_mutate_input():
    moments = _cached_moments()
    cache = _full_cache(moments)
    sig = next(iter(cache["conv1"]))
    _invalidate_agg_cache(cache, "action")
    assert cache["conv1"][sig]["action_direction_agg"] == "scaffolding"


# --- load_existing_action_result_agg -----------------------------------------


def test_load_existing_action_result_agg_reconstructs_cache(tmp_path):
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()

    moments = [
        _moment("t1", 1, 10, action_decomposed=["a1"], result_decomposed=["r1"]),
        _moment("t2", 1, 10, action_decomposed=["b1"], result_decomposed=["r2"]),
        _moment("t3", 80, 90, annotation_type="rapport"),
    ]
    moments[0]["action_direction_agg"] = "scaffolding"
    moments[0]["student_outcome_agg"] = "pos"
    moments[1]["action_direction_agg"] = "scaffolding"
    moments[1]["student_outcome_agg"] = "pos"

    (gt_dir / "conv1.json").write_text(
        json.dumps(
            {"conversation_id": "conv1", "num_turns": 100, "key_moments": moments}
        ),
        encoding="utf-8",
    )

    cache = load_existing_action_result_agg(gt_dir)
    assert list(cache.keys()) == ["conv1"]
    sig = frozenset(moment_key(m) for m in moments[:2])
    assert sig in cache["conv1"]
    entry = cache["conv1"][sig]
    assert entry["unified_action"] == ["a1", "b1"]
    assert entry["unified_result"] == ["r1", "r2"]
    assert entry["action_direction_agg"] == "scaffolding"
    assert entry["student_outcome_agg"] == "pos"


def test_load_existing_action_result_agg_skips_clusters_without_agg_labels(tmp_path):
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()

    moments = [_moment("t1", 1, 10, action_decomposed=["a1"], result_decomposed=["r1"])]
    (gt_dir / "conv1.json").write_text(
        json.dumps(
            {"conversation_id": "conv1", "num_turns": 10, "key_moments": moments}
        ),
        encoding="utf-8",
    )

    cache = load_existing_action_result_agg(gt_dir)
    assert cache == {}
