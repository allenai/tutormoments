"""Tests for the append-only v2 iterate/heldout split assignment."""

import json
import os

import pytest

from tutormoments_build.v2 import splits as S


def _annotations_file(tmp_path, rows):
    path = tmp_path / "annotations.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def _moment_row(tid, i=0):
    return {
        "transcript_id": tid,
        "moment": {"moment_id": f"{tid}-{i}"},
        "annotations": [],
    }


def _no_key_row(tid):
    return {
        "transcript_id": tid,
        "no_key_moments_record": {"payload": {"no_key_moments": True}},
    }


def _corpus(n_transcripts, moments_each=3, seed=0):
    """{tid: summary} without going through disk."""
    return {
        f"t{i:04d}": {"n_moments": moments_each, "no_key_moments": False}
        for i in range(n_transcripts)
    }


def _empty_manifest():
    return S.load_manifest(os.path.join("does", "not", "exist.json"))


def _assign(corpus, manifest, **kw):
    kw.setdefault("seed", 1234)
    kw.setdefault("heldout_fraction", 0.5)
    kw.setdefault("balance_by", "moments")
    return S.assign_new(corpus, manifest, **kw)


# ---------------------------------------------------------------------------
# reading annotations
# ---------------------------------------------------------------------------


def test_summarize_counts_moments_per_transcript(tmp_path):
    path = _annotations_file(
        tmp_path,
        [
            _moment_row("a", 0),
            _moment_row("a", 1),
            _moment_row("b", 0),
            _no_key_row("c"),
        ],
    )
    got = S.summarize_transcripts(path)
    assert got["a"]["n_moments"] == 2
    assert got["b"]["n_moments"] == 1
    assert got["c"] == {"n_moments": 0, "no_key_moments": True}


def test_summarize_rejects_unrecognised_row(tmp_path):
    path = _annotations_file(tmp_path, [{"transcript_id": "a", "something_else": {}}])
    with pytest.raises(ValueError, match="neither 'moment'"):
        S.summarize_transcripts(path)


def test_summarize_rejects_row_without_transcript_id(tmp_path):
    path = _annotations_file(tmp_path, [{"moment": {}}])
    with pytest.raises(ValueError, match="no transcript_id"):
        S.summarize_transcripts(path)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_same_seed_gives_same_assignment():
    corpus = _corpus(50)
    first, _ = _assign(corpus, _empty_manifest())
    second, _ = _assign(corpus, _empty_manifest())
    assert first == second


def test_different_seed_gives_different_assignment():
    corpus = _corpus(50)
    first, _ = _assign(corpus, _empty_manifest(), seed=1)
    second, _ = _assign(corpus, _empty_manifest(), seed=2)
    assert first != second


def test_assignment_is_independent_of_input_order():
    corpus = _corpus(40)
    shuffled = dict(reversed(list(corpus.items())))
    assert (
        _assign(corpus, _empty_manifest())[0] == _assign(shuffled, _empty_manifest())[0]
    )


# ---------------------------------------------------------------------------
# append-only behaviour -- the point of the whole script
# ---------------------------------------------------------------------------


def test_existing_assignments_are_never_revisited():
    manifest = _empty_manifest()
    round_one = _corpus(30)
    new, _ = _assign(round_one, manifest)
    manifest["assignments"].update(new)
    manifest["batches"].append({"batch": 1})
    frozen = {tid: rec["split"] for tid, rec in manifest["assignments"].items()}

    round_two = _corpus(60)  # the original 30 plus 30 more
    new2, new_ids = _assign(round_two, manifest)

    assert set(new2).isdisjoint(frozen), "already-assigned transcripts were reassigned"
    assert len(new_ids) == 30
    manifest["assignments"].update(new2)
    for tid, split in frozen.items():
        assert manifest["assignments"][tid]["split"] == split


def test_new_batch_is_tagged_with_its_batch_number():
    manifest = _empty_manifest()
    manifest["assignments"].update(_assign(_corpus(10), manifest)[0])
    manifest["batches"].append({"batch": 1})
    new, _ = _assign(_corpus(20), manifest)
    assert {rec["batch"] for rec in new.values()} == {2}


def test_ratio_stays_on_target_across_many_rounds():
    manifest = _empty_manifest()
    for round_no, size in enumerate([20, 55, 90, 140, 200], start=1):
        new, _ = _assign(_corpus(size), manifest)
        manifest["assignments"].update(new)
        manifest["batches"].append({"batch": round_no})
        totals = S._totals(manifest["assignments"])
        moments = {s: totals[s]["moments"] for s in S.SPLITS}
        share = moments["heldout"] / sum(moments.values())
        assert 0.45 <= share <= 0.55, f"round {round_no} drifted to {share:.2%}"


def test_transcripts_missing_from_annotations_keep_their_assignment():
    manifest = _empty_manifest()
    manifest["assignments"].update(_assign(_corpus(20), manifest)[0])
    manifest["batches"].append({"batch": 1})

    shrunk = {k: v for k, v in _corpus(20).items() if k != "t0000"}
    new, _ = _assign(shrunk, manifest)
    assert "t0000" in manifest["assignments"]
    assert "t0000" not in new


# ---------------------------------------------------------------------------
# balancing
# ---------------------------------------------------------------------------


def test_balance_by_moments_evens_moment_counts_not_transcript_counts():
    # Two heavyweight transcripts against many lightweight ones: balancing by
    # moments must not simply halve the transcript count.
    corpus = {f"t{i:04d}": {"n_moments": 1, "no_key_moments": False} for i in range(20)}
    corpus["heavy0"] = {"n_moments": 10, "no_key_moments": False}
    corpus["heavy1"] = {"n_moments": 10, "no_key_moments": False}

    new, _ = _assign(corpus, _empty_manifest(), balance_by="moments")
    per_split = {s: 0 for s in S.SPLITS}
    for tid, rec in new.items():
        per_split[rec["split"]] += corpus[tid]["n_moments"]
    assert abs(per_split["iterate"] - per_split["heldout"]) <= 1


def test_balance_by_transcripts_evens_transcript_counts():
    corpus = {
        f"t{i:04d}": {"n_moments": i % 7, "no_key_moments": False} for i in range(41)
    }
    new, _ = _assign(corpus, _empty_manifest(), balance_by="transcripts")
    counts = {s: 0 for s in S.SPLITS}
    for rec in new.values():
        counts[rec["split"]] += 1
    assert abs(counts["iterate"] - counts["heldout"]) <= 1


def test_no_key_moment_transcripts_are_split_not_bunched():
    # Zero-weight under --balance-by moments; the count tie-break must still
    # spread them across both splits.
    corpus = {f"n{i:04d}": {"n_moments": 0, "no_key_moments": True} for i in range(20)}
    new, _ = _assign(corpus, _empty_manifest(), balance_by="moments")
    counts = {s: 0 for s in S.SPLITS}
    for rec in new.values():
        counts[rec["split"]] += 1
    assert abs(counts["iterate"] - counts["heldout"]) <= 1


def test_heldout_fraction_is_respected():
    new, _ = _assign(
        _corpus(200, moments_each=1), _empty_manifest(), heldout_fraction=0.25
    )
    heldout = sum(1 for rec in new.values() if rec["split"] == "heldout")
    assert abs(heldout / 200 - 0.25) < 0.02


# ---------------------------------------------------------------------------
# manifest I/O and CLI
# ---------------------------------------------------------------------------


def test_write_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "splits.json")
    manifest = _empty_manifest()
    manifest.update({"seed": 1, "heldout_fraction": 0.5, "balance_by": "moments"})
    manifest["assignments"].update(_assign(_corpus(8), manifest)[0])
    S.write_manifest(path, manifest)

    reloaded = S.load_manifest(path)
    assert reloaded["assignments"] == manifest["assignments"]
    assert list(reloaded["assignments"]) == sorted(reloaded["assignments"])
    assert S.read_splits(path) == {
        tid: rec["split"] for tid, rec in manifest["assignments"].items()
    }


def test_load_manifest_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"schema_version": "99.0", "assignments": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version"):
        S.load_manifest(str(path))


def _cli(tmp_path, rows, *argv):
    ann = _annotations_file(tmp_path, rows)
    manifest = str(tmp_path / "splits.json")
    code = S.main(["--annotations", ann, "--manifest", manifest, *argv])
    return code, manifest


def test_cli_writes_manifest_and_is_idempotent(tmp_path):
    rows = [_moment_row(f"t{i}", j) for i in range(12) for j in range(2)]
    code, manifest = _cli(tmp_path, rows)
    assert code == 0
    first = S.load_manifest(manifest)
    assert len(first["assignments"]) == 12
    assert len(first["batches"]) == 1

    # Re-running with no new transcripts must not add a batch or change anything.
    S.main(["--annotations", _annotations_file(tmp_path, rows), "--manifest", manifest])
    second = S.load_manifest(manifest)
    assert second["assignments"] == first["assignments"]
    assert len(second["batches"]) == 1


def test_cli_dry_run_writes_nothing(tmp_path):
    rows = [_moment_row(f"t{i}") for i in range(6)]
    code, manifest = _cli(tmp_path, rows, "--dry-run")
    assert code == 0
    assert not os.path.exists(manifest)


def test_cli_rejects_param_change_unless_forced(tmp_path):
    rows = [_moment_row(f"t{i}") for i in range(6)]
    ann = _annotations_file(tmp_path, rows)
    manifest = str(tmp_path / "splits.json")
    assert S.main(["--annotations", ann, "--manifest", manifest]) == 0

    more = _annotations_file(tmp_path, rows + [_moment_row("extra")])
    args = ["--annotations", more, "--manifest", manifest, "--seed", "999"]
    assert S.main(args) == 2
    assert S.main([*args, "--allow-param-change"]) == 0


def test_cli_rejects_out_of_range_fraction(tmp_path):
    rows = [_moment_row("t0")]
    code, _ = _cli(tmp_path, rows, "--heldout-fraction", "1.5")
    assert code == 2


def test_cli_second_round_only_assigns_new_transcripts(tmp_path):
    rows = [_moment_row(f"t{i}") for i in range(10)]
    _, manifest = _cli(tmp_path, rows)
    before = S.read_splits(manifest)

    rows += [_moment_row(f"t{i}") for i in range(10, 18)]
    S.main(["--annotations", _annotations_file(tmp_path, rows), "--manifest", manifest])

    after = S.read_splits(manifest)
    assert len(after) == 18
    assert all(after[tid] == split for tid, split in before.items())
    assert len(S.load_manifest(manifest)["batches"]) == 2


def test_no_key_moment_transcripts_split_alongside_weighted_ones():
    # Regression: zero-weight transcripts cannot change the moment balance, so
    # ranking them by moment deficit sent every one of them to whichever split
    # happened to be a moment behind. They must be spread on transcript count.
    corpus = {
        f"t{i:04d}": {"n_moments": 1 + i % 5, "no_key_moments": False}
        for i in range(60)
    }
    corpus.update(
        {f"n{i:04d}": {"n_moments": 0, "no_key_moments": True} for i in range(27)}
    )

    new, _ = _assign(corpus, _empty_manifest(), balance_by="moments")
    empty = {s: 0 for s in S.SPLITS}
    for tid, rec in new.items():
        if corpus[tid]["no_key_moments"]:
            empty[rec["split"]] += 1
    assert min(empty.values()) >= 10, f"no-key-moment transcripts bunched: {empty}"

    moments = {s: 0 for s in S.SPLITS}
    for tid, rec in new.items():
        moments[rec["split"]] += corpus[tid]["n_moments"]
    assert abs(moments["iterate"] - moments["heldout"]) <= 1
