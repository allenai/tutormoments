"""Validation tests for the frozen balanced_520 release.

These tests assert hard invariants on the built release dir's moments.jsonl
and exercise the load_moments API round-trip. All counts are exact; any
deviation is a failure, not a warning.

The canonical id list lives at tutorsim_build/balanced_520_ids.json (committed
— deidentified UUID surrogates only). The built release dir is NOT committed
(distributed via the dataset release); moments tests skip when it is absent.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from tutorsim.moments import load_moments

# Repo root relative to this file: tests/tutorsim/ -> ../../
_REPO_ROOT = Path(__file__).parent.parent.parent
_RELEASE_DIR = _REPO_ROOT / "data" / "balanced_520_release"
_MOMENTS_FILE = _RELEASE_DIR / "moments.jsonl"
_IDS_FILE = _REPO_ROOT / "tutorsim_build" / "balanced_520_ids.json"

# The built release is not committed to git (distributed via the dataset
# release). These validation tests run only when it has been built or
# downloaded locally; otherwise they skip rather than fail.
pytestmark = pytest.mark.skipif(
    not (_MOMENTS_FILE.exists() and _IDS_FILE.exists()),
    reason="balanced_520 release not present locally (build with tutorsim-build dataset build)",
)


@pytest.fixture(scope="module")
def raw_moments() -> list[dict]:
    """Load moments as raw dicts for cheap attribute checks."""
    assert _MOMENTS_FILE.exists(), f"Missing: {_MOMENTS_FILE}"
    records = []
    with open(_MOMENTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def canonical_ids() -> list[str]:
    """Load the canonical ordered id list."""
    assert _IDS_FILE.exists(), f"Missing: {_IDS_FILE}"
    return json.load(open(_IDS_FILE, encoding="utf-8"))


class TestExactCounts:
    def test_total_count(self, raw_moments):
        assert len(raw_moments) == 520, f"Expected 520, got {len(raw_moments)}"

    def test_dimension_split(self, raw_moments):
        counts = Counter(s["dimension"] for s in raw_moments)
        assert counts["scaffolding"] == 260, (
            f"Expected 260 scaffolding, got {counts['scaffolding']}"
        )
        assert counts["rigor"] == 260, f"Expected 260 rigor, got {counts['rigor']}"

    def test_all_have_non_empty_context(self, raw_moments):
        empty = [s["id"] for s in raw_moments if not s.get("context")]
        assert not empty, f"Moments with empty context: {empty}"

    def test_all_have_rubric_gold(self, raw_moments):
        missing = [s["id"] for s in raw_moments if not s.get("rubric", {}).get("gold")]
        assert not missing, f"Moments missing rubric.gold: {missing}"

    def test_ids_are_unique(self, raw_moments):
        all_ids = [s["id"] for s in raw_moments]
        assert len(set(all_ids)) == len(all_ids), "Duplicate moment ids found"

    def test_ids_are_namespaced(self, raw_moments):
        bad = [s["id"] for s in raw_moments if not s["id"].startswith("balanced_520:")]
        assert not bad, f"IDs not namespaced with 'balanced_520:': {bad[:5]}"

    def test_reference_counts(self, raw_moments):
        """Report reference presence. Empty is allowed only when real convo ended at cut."""
        non_empty = sum(
            1 for s in raw_moments if s.get("student", {}).get("reference", "")
        )
        empty = len(raw_moments) - non_empty
        # Both empty and non-empty are acceptable; report the split as an assertion
        # so test output shows the numbers even when passing.
        assert non_empty + empty == 520
        # Log counts via assertion message for visibility
        print(f"reference non-empty: {non_empty}, empty: {empty}")

    def test_cut_votes_are_release_form(self, raw_moments):
        """Released records carry Arrow-friendly cut_votes (list of pairs)."""
        bad = [
            s["id"]
            for s in raw_moments
            if not isinstance(s["provenance"].get("cut_votes"), list)
        ]
        assert not bad, f"Moments with non-list cut_votes: {bad[:5]}"


class TestOrderMatchesIdList:
    def test_order_matches_id_list(self, raw_moments, canonical_ids):
        """Moment order must match tutorsim_build/balanced_520_ids.json."""
        # Build the expected prefixed ids in the order they appear in the id-list
        actual_ids = [s["id"] for s in raw_moments]
        actual_set = set(actual_ids)

        expected_ordered = [
            f"balanced_520:{raw_id}"
            for raw_id in canonical_ids
            if f"balanced_520:{raw_id}" in actual_set
        ]

        assert actual_ids == expected_ordered, (
            "Moment order does not match canonical id-list order"
        )


class TestLoadMomentsRoundTrip:
    def test_load_moments_count(self):
        """load_moments returns exactly 520 Moment objects."""
        from tutorsim.moments import Moment

        moments, source = load_moments(data_path=str(_RELEASE_DIR))
        assert len(moments) == 520, f"Expected 520, got {len(moments)}"
        assert all(isinstance(m, Moment) for m in moments)
        assert source["record_count"] == 520

    def test_load_moments_first_last_ids(self, canonical_ids):
        """First and last moment ids match canonical id-list order."""
        moments, _ = load_moments(data_path=str(_RELEASE_DIR))
        expected_first = f"balanced_520:{canonical_ids[0]}"
        expected_last = f"balanced_520:{canonical_ids[-1]}"
        assert moments[0].id == expected_first, (
            f"First id mismatch: {moments[0].id} != {expected_first}"
        )
        assert moments[-1].id == expected_last, (
            f"Last id mismatch: {moments[-1].id} != {expected_last}"
        )

    def test_manifest_hash_matches(self):
        """validate_dataset passes on the built release dir."""
        from tutorsim.moments import validate_dataset

        report = validate_dataset(_RELEASE_DIR)
        assert report["record_count"] == 520
