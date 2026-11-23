"""Screenshot writer: frames ship where their transcripts say, and only clean JPEGs.

All frames are synthetic byte strings built from JPEG markers; the writer reads
only the header, never the image data.
"""

import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("jsonschema")

from tutormoments_build.release import build as rb
from tutormoments_build.release.screenshots import embedded_metadata, write_screenshots
from tutormoments_build.release.snapshot import SNAPSHOT_INPUTS, create_snapshot
from tutormoments_build.release.store import LocalStore

CLEAN = {"commit": "abc1234def5678", "dirty": False}
NOW = datetime(2026, 11, 23, 9, 0, tzinfo=timezone.utc)
T1 = "00000000-0000-5000-8000-000000000001"
T2 = "ab000000-0000-5000-8000-000000000002"
FRAMES = "screenshots/tutoring_provider_a/"


def segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def jfif(thumbnail: bytes = b"") -> bytes:
    width, height = (1, 1) if thumbnail else (0, 0)
    return segment(
        0xE0,
        b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01" + bytes([width, height]) + thumbnail,
    )


def jpeg(*segments: bytes) -> bytes:
    """SOI, the given header segments, a start-of-scan, image bytes, EOI."""
    return (
        b"\xff\xd8" + b"".join(segments) + segment(0xDA, b"\x00") + b"\x12\x34\xff\xd9"
    )


CLEAN_JPEG = jpeg(jfif(), segment(0xDB, b"\x00" * 65))


def frame(transcript_id: str, seconds: str) -> str:
    return f"{FRAMES}{transcript_id[:2]}/{transcript_id}/{seconds}.jpg"


def transcript(transcript_id, *paths):
    return {
        "transcript_id": transcript_id,
        "screenshots": [
            {"start_seconds": float(p.rsplit("/", 1)[1][:-4]), "path": p} for p in paths
        ],
    }


def jsonl(*rows):
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


# The released transcripts, as write_transcripts would leave them in the build.
RELEASED = {
    "tutoring_provider_a": [
        transcript(T1, frame(T1, "1.5"), frame(T1, "20.25")),
        transcript(T2, frame(T2, "300")),
        transcript("00000000-0000-5000-8000-000000000003"),
    ],
    "tutoring_provider_c": [{"transcript_id": "c"}],  # C has no screenshots field
}


def released_transcripts(ctx):
    for provider, rows in RELEASED.items():
        ctx.store.write_bytes(
            f"{ctx.build_prefix}transcripts/{provider}.jsonl", jsonl(*rows)
        )
    return []


@pytest.fixture
def store(tmp_path):
    s = LocalStore(tmp_path)
    for rows in RELEASED.values():
        for row in rows:
            for shot in row.get("screenshots", []):
                s.write_bytes(shot["path"], CLEAN_JPEG)
    return s


def build(store):
    create_snapshot(store, snapshot_id="s1", inputs=(FRAMES,))
    return rb.run_build(
        store,
        "s1",
        writers=(released_transcripts, write_screenshots),
        code=CLEAN,
        now=NOW,
    )


def test_every_listed_frame_ships_at_its_path_and_is_hashed(store):
    manifest = build(store)
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    shipped = sorted(p for p in manifest["files"] if p.startswith("screenshots/"))
    assert shipped == sorted([frame(T1, "1.5"), frame(T1, "20.25"), frame(T2, "300")])
    for path in shipped:
        assert store.read_bytes(prefix + path) == CLEAN_JPEG
        entry = manifest["files"][path]
        assert entry["sha256"] == store.digest(prefix + path)["sha256"]
        assert "schemas" not in entry  # frames are not data files


def test_listed_frame_missing_from_the_snapshot_fails(store, tmp_path):
    (tmp_path / frame(T2, "300")).unlink()
    with pytest.raises(SystemExit, match="not in the snapshot"):
        build(store)


def test_frame_no_transcript_lists_fails(store):
    store.write_bytes(frame(T1, "99"), CLEAN_JPEG)
    with pytest.raises(SystemExit, match="no released transcript lists"):
        build(store)


def test_frame_with_embedded_metadata_fails_and_nothing_is_copied(store):
    exif = jpeg(jfif(), segment(0xE1, b"Exif\x00\x00II*\x00"))
    store.write_bytes(frame(T1, "1.5"), exif)
    with pytest.raises(SystemExit, match="embedded APP1"):
        build(store)
    assert not [k for k in store.list("release-v2/builds/") if "/screenshots/" in k]


@pytest.mark.parametrize(
    "data,found",
    [
        (CLEAN_JPEG, []),
        (jpeg(jfif(), segment(0xE1, b"Exif\x00\x00")), ["APP1"]),  # EXIF or XMP
        (jpeg(jfif(), segment(0xE2, b"ICC_PROFILE\x00")), ["APP2"]),
        (jpeg(jfif(), segment(0xFE, b"made by a camera")), ["COM"]),
        (jpeg(jfif(thumbnail=b"\x01\x02\x03")), ["JFIF thumbnail"]),
        (jpeg(segment(0xE0, b"JFXX\x00\x10")), ["APP0 JFXX"]),
        # Fill bytes before a marker are legal and not metadata.
        (b"\xff\xd8\xff" + jfif() + segment(0xDA, b"\x00"), []),
        # Anything after the scan starts is image data, not header.
        (jpeg(jfif()) + b"\xff\xfe\x00\x04hi", []),
        (b"\x89PNG\r\n", ["not a JPEG"]),
        (b"\xff\xd8\xff\xe1\x00", ["truncated header"]),
    ],
)
def test_embedded_metadata(data, found):
    assert embedded_metadata(data) == found


def test_frames_are_snapshotted_and_written_after_the_transcripts():
    assert FRAMES in SNAPSHOT_INPUTS
    names = [w.__name__ for w in rb.default_writers()]
    assert names.index("write_screenshots") > names.index("write_transcripts")
