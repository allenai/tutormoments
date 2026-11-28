"""Finding the release build research reads, and recording what was read."""

import hashlib
import json
import os

import pytest

from tutormoments_build.v2 import release_data as R

PINNED = "20261124T2322Z-081a944"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_build(root, build_id, files: dict[str, bytes]):
    """A local build folder: the files plus a manifest listing their sha256."""
    build = root / "builds" / build_id
    for path, data in files.items():
        target = build / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {
        "build_id": build_id,
        "files": {p: {"sha256": _sha(d), "bytes": len(d)} for p, d in files.items()},
    }
    (build / R.MANIFEST_FILE).write_text(json.dumps(manifest))
    return build


@pytest.fixture
def pointer(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"build_id": PINNED, "manifest_sha256": "0" * 64}))
    return str(path)


class TestCommittedPointer:
    def test_names_a_build_and_its_manifest_hash(self):
        pinned = R.pinned_build()
        assert pinned["build_id"]
        assert len(pinned["manifest_sha256"]) == 64
        int(pinned["manifest_sha256"], 16)


class TestDefaultBuildDir:
    def test_pointer_names_the_local_build(self, pointer, monkeypatch):
        monkeypatch.delenv(R.RELEASE_ENV, raising=False)
        assert R.default_build_dir(pointer, local_root="cache") == os.path.join(
            "cache", "builds", PINNED
        )

    def test_environment_overrides_the_pointer(self, pointer, monkeypatch):
        monkeypatch.setenv(R.RELEASE_ENV, "/elsewhere/builds/other")
        assert R.default_build_dir(pointer) == "/elsewhere/builds/other"


class TestRequireLocal:
    def test_a_build_that_was_never_pulled_says_how_to_get_it(self, tmp_path):
        missing = tmp_path / "builds" / PINNED / "benchmark"
        with pytest.raises(FileNotFoundError, match="release pull"):
            R.require_local([str(missing)])

    def test_a_build_without_its_manifest_is_not_local(self, tmp_path):
        # A pull writes the manifest last, so a folder without one is partial.
        partial = tmp_path / "builds" / PINNED / "benchmark"
        partial.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="release pull"):
            R.require_local([str(partial)])

    def test_a_path_outside_any_build_is_checked_plainly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.jsonl"):
            R.require_local([str(tmp_path / "nope.jsonl")])

    def test_present_paths_pass(self, tmp_path):
        build = make_build(tmp_path, PINNED, {"benchmark/dev.jsonl": b"{}\n"})
        R.require_local([str(build / "benchmark"), str(build / "benchmark/dev.jsonl")])


class TestProvenance:
    def test_records_the_build_and_each_file_by_its_release_path(
        self, tmp_path, pointer
    ):
        build = make_build(
            tmp_path,
            PINNED,
            {
                "benchmark/dev.jsonl": b"d\n",
                "transcripts/tutoring_provider_a.jsonl": b"t\n",
            },
        )
        block = R.provenance(
            [
                build / "benchmark/dev.jsonl",
                build / "transcripts/tutoring_provider_a.jsonl",
            ],
            pointer_path=pointer,
        )
        assert block == {
            "build_id": PINNED,
            "pinned": True,
            "files": {
                "benchmark/dev.jsonl": _sha(b"d\n"),
                "transcripts/tutoring_provider_a.jsonl": _sha(b"t\n"),
            },
        }

    def test_a_build_other_than_the_pinned_one_is_marked(
        self, tmp_path, pointer, caplog
    ):
        build = make_build(
            tmp_path, "20261201T0000Z-1234567", {"benchmark/dev.jsonl": b"d\n"}
        )
        block = R.provenance([build / "benchmark/dev.jsonl"], pointer_path=pointer)
        assert block["build_id"] == "20261201T0000Z-1234567"
        assert block["pinned"] is False
        assert "not the pinned build" in caplog.text

    def test_a_changed_file_is_refused(self, tmp_path, pointer):
        build = make_build(tmp_path, PINNED, {"benchmark/dev.jsonl": b"d\n"})
        (build / "benchmark/dev.jsonl").write_bytes(b"edited\n")
        with pytest.raises(R.ReleaseDataError, match="sha256"):
            R.provenance([build / "benchmark/dev.jsonl"], pointer_path=pointer)

    def test_a_file_the_manifest_does_not_list_is_refused(self, tmp_path, pointer):
        build = make_build(tmp_path, PINNED, {"benchmark/dev.jsonl": b"d\n"})
        (build / "benchmark/extra.jsonl").write_bytes(b"x\n")
        with pytest.raises(R.ReleaseDataError, match="not in"):
            R.provenance([build / "benchmark/extra.jsonl"], pointer_path=pointer)

    def test_files_outside_a_build_are_hashed_under_their_own_path(
        self, tmp_path, pointer
    ):
        loose = tmp_path / "student_state.jsonl"
        loose.write_bytes(b"s\n")
        block = R.provenance([loose], pointer_path=pointer)
        assert block == {
            "build_id": None,
            "pinned": False,
            "files": {str(loose): _sha(b"s\n")},
        }

    def test_mixing_two_builds_is_refused(self, tmp_path, pointer):
        one = make_build(tmp_path, PINNED, {"benchmark/dev.jsonl": b"d\n"})
        two = make_build(
            tmp_path, "20261201T0000Z-1234567", {"transcripts/a.jsonl": b"t\n"}
        )
        with pytest.raises(R.ReleaseDataError, match="two builds"):
            R.provenance(
                [one / "benchmark/dev.jsonl", two / "transcripts/a.jsonl"],
                pointer_path=pointer,
            )
