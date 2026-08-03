"""Unit tests for the tutor cost aggregation in analysis/benchmark_perf_cost.py.

Only the pure aggregation (summarize_exchanges) is tested: the id-set filter,
de-duplication by scenario_id, per-turn latency mean, and output-tokens-per-turn.
File walking and the canonical scores loader are thin I/O and not unit-tested.
"""

import sys
from pathlib import Path

import pytest

# benchmark_perf_cost imports matplotlib/pandas (the figure/analysis extras).
# Skip the module rather than hard-erroring collection when they're absent.
pytest.importorskip("matplotlib")
pytest.importorskip("pandas")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "analysis" / "working-paper-20260630")
)
import benchmark_perf_cost as bpc  # noqa: E402


def _ex(sid, lats, out_tok, total_tok):
    return {
        "scenario_id": sid,
        "tutor_latencies": lats,
        "tutor_usage": {"output_tokens": out_tok, "total_tokens": total_tok},
    }


def test_filters_to_id_set_and_dedupes():
    exchanges = [
        _ex("a", [2.0, 4.0], 100, 1000),  # in set
        _ex("a", [9.0], 999, 9999),  # duplicate scenario_id -> ignored
        _ex("b", [6.0], 60, 600),  # in set
        _ex("z", [100.0], 1, 1),  # NOT in set -> ignored
    ]
    s = bpc.summarize_exchanges(exchanges, id_set={"a", "b"})
    assert s["n_scenarios"] == 2
    # turns: a=[2,4], b=[6] -> 3 turns, mean = 12/3 = 4.0
    assert s["tutor_latency_mean_s"] == pytest.approx(4.0)
    # output per turn: a=100/2=50, b=60/1=60 -> mean 55
    assert s["tutor_output_tokens_per_turn"] == pytest.approx(55.0)


def test_no_id_set_counts_all():
    exchanges = [_ex("a", [1.0], 10, 100), _ex("b", [3.0], 30, 300)]
    s = bpc.summarize_exchanges(exchanges)
    assert s["n_scenarios"] == 2
    assert s["tutor_latency_mean_s"] == pytest.approx(2.0)


def test_empty_yields_none():
    s = bpc.summarize_exchanges([], id_set={"a"})
    assert s["n_scenarios"] == 0
    assert s["tutor_latency_mean_s"] is None
    assert s["tutor_output_tokens_per_turn"] is None
