"""`release pull`: a local copy of one build, sha256-checked, sharing files with
builds already held locally.

Runs against a local-directory store; no AWS calls. All data is synthetic.
"""

import hashlib
import json
import os

import pytest

from tutormoments_build.release import pull as P
from tutormoments_build.release.store import LocalStore, dump_json

RELEASE = "release-v2"
OLD = "20261101T0000Z-aaaaaaa"
NEW = "20261124T2322Z-081a944"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(store_root, build_id, files: dict[str, bytes]) -> str:
    """Write a finished build into the store; return its manifest's sha256."""
    prefix = store_root / RELEASE / "builds" / build_id
    for path, data in files.items():
        target = prefix / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = dump_json(
        {
            "build_id": build_id,
            "files": {
                p: {"sha256": _sha(d), "bytes": len(d)} for p, d in files.items()
            },
        }
    )
    (prefix / "manifest.json").write_bytes(manifest)
    return _sha(manifest)


@pytest.fixture
def store_root(tmp_path):
    root = tmp_path / "bucket"
    root.mkdir()
    return root


@pytest.fixture
def local(tmp_path):
    return tmp_path / "local"


SHARED = b"transcripts that did not change\n"


def pull(store_root, local, build_id, sha, **kwargs):
    return P.pull_build(
        LocalStore(store_root),
        build_id,
        local,
        manifest_sha256=sha,
        release=RELEASE,
        **kwargs,
    )


def test_pull_copies_every_file_and_writes_the_manifest(store_root, local):
    sha = publish(
        store_root, NEW, {"benchmark/dev.jsonl": b"d\n", "transcripts/a.jsonl": SHARED}
    )
    summary = pull(store_root, local, NEW, sha)

    build = local / "builds" / NEW
    assert (build / "benchmark/dev.jsonl").read_bytes() == b"d\n"
    assert (build / "transcripts/a.jsonl").read_bytes() == SHARED
    assert json.loads((build / "manifest.json").read_text())["build_id"] == NEW
    assert summary["downloaded"] == 2 and summary["linked"] == 0


def test_a_manifest_that_does_not_match_the_pin_is_refused(store_root, local):
    publish(store_root, NEW, {"benchmark/dev.jsonl": b"d\n"})
    with pytest.raises(P.PullError, match="manifest"):
        pull(store_root, local, NEW, "0" * 64)
    assert not (local / "builds" / NEW).exists()


def test_files_already_held_by_another_local_build_are_hard_linked(store_root, local):
    old_sha = publish(store_root, OLD, {"transcripts/a.jsonl": SHARED})
    pull(store_root, local, OLD, old_sha)
    new_sha = publish(
        store_root, NEW, {"transcripts/a.jsonl": SHARED, "benchmark/dev.jsonl": b"d\n"}
    )

    summary = pull(store_root, local, NEW, new_sha)

    old_file = local / "builds" / OLD / "transcripts/a.jsonl"
    new_file = local / "builds" / NEW / "transcripts/a.jsonl"
    assert os.stat(old_file).st_ino == os.stat(new_file).st_ino
    assert summary["linked"] == 1 and summary["downloaded"] == 1


def test_an_object_that_does_not_match_its_manifest_is_refused(store_root, local):
    sha = publish(store_root, NEW, {"benchmark/dev.jsonl": b"d\n"})
    (store_root / RELEASE / "builds" / NEW / "benchmark/dev.jsonl").write_bytes(
        b"bad\n"
    )

    with pytest.raises(P.PullError, match="benchmark/dev.jsonl"):
        pull(store_root, local, NEW, sha)
    # No manifest, so research code treats the build as not local.
    assert not (local / "builds" / NEW / "manifest.json").exists()


def test_pulling_again_keeps_what_is_already_there(store_root, local):
    sha = publish(store_root, NEW, {"benchmark/dev.jsonl": b"d\n"})
    pull(store_root, local, NEW, sha)
    summary = pull(store_root, local, NEW, sha)
    assert summary == {"downloaded": 0, "linked": 0, "kept": 1, "pruned": []}


def test_a_locally_edited_file_is_replaced(store_root, local):
    sha = publish(store_root, NEW, {"benchmark/dev.jsonl": b"d\n"})
    pull(store_root, local, NEW, sha)
    target = local / "builds" / NEW / "benchmark/dev.jsonl"
    target.write_bytes(b"edited\n")

    summary = pull(store_root, local, NEW, sha)
    assert target.read_bytes() == b"d\n"
    assert summary["downloaded"] == 1


def test_only_limits_the_pull_to_some_prefixes(store_root, local):
    sha = publish(
        store_root, NEW, {"benchmark/dev.jsonl": b"d\n", "screenshots/x/1.jpg": b"img"}
    )
    pull(store_root, local, NEW, sha, only=["benchmark/"])
    build = local / "builds" / NEW
    assert (build / "benchmark/dev.jsonl").exists()
    assert not (build / "screenshots").exists()
    assert (build / "manifest.json").exists()


def test_prune_removes_every_other_local_build(store_root, local):
    old_sha = publish(store_root, OLD, {"transcripts/a.jsonl": SHARED})
    pull(store_root, local, OLD, old_sha)
    new_sha = publish(store_root, NEW, {"transcripts/a.jsonl": SHARED})

    summary = pull(store_root, local, NEW, new_sha, prune=True)

    assert sorted(os.listdir(local / "builds")) == [NEW]
    assert summary["pruned"] == [OLD]
    # The shared file survives through its hard link.
    assert (local / "builds" / NEW / "transcripts/a.jsonl").read_bytes() == SHARED
