"""Tests for --refresh-decomp cache invalidation in build_ground_truth.py.

Covers _invalidate_decomp_cache, which selectively clears the cached action /
result decompositions so a run re-decomposes them (e.g. after editing
decompose_action.md / decompose_result.md, whose changes the content-hash decomp
cache otherwise silently ignores). Mirrors the _invalidate_agg_cache tests.
"""

from tutorsim_build.groundtruth import _invalidate_decomp_cache


def _full_cache():
    # Shape mirrors load_existing_decompositions(): {conv: {"action": {...}, "result": {...}}}
    return {
        "conv1": {
            "action": {("t1", 1, 10, "scaffolding", "abc"): ["a1"]},
            "result": {("t1", 1, 10, "scaffolding", "def"): ["r1"]},
        }
    }


def test_invalidate_decomp_cache_action_clears_action_keeps_result():
    invalidated = _invalidate_decomp_cache(_full_cache(), "action")
    assert invalidated["conv1"]["action"] == {}
    assert invalidated["conv1"]["result"] == {
        ("t1", 1, 10, "scaffolding", "def"): ["r1"]
    }


def test_invalidate_decomp_cache_result_clears_result_keeps_action():
    invalidated = _invalidate_decomp_cache(_full_cache(), "result")
    assert invalidated["conv1"]["result"] == {}
    assert invalidated["conv1"]["action"] == {
        ("t1", 1, 10, "scaffolding", "abc"): ["a1"]
    }


def test_invalidate_decomp_cache_both_wipes_entire_cache():
    assert _invalidate_decomp_cache(_full_cache(), "both") == {}


def test_invalidate_decomp_cache_does_not_mutate_input():
    cache = _full_cache()
    _invalidate_decomp_cache(cache, "action")
    assert cache["conv1"]["action"] == {("t1", 1, 10, "scaffolding", "abc"): ["a1"]}
