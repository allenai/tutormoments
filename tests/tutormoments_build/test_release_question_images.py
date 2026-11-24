"""Question-image writer: images ship where the quizzes say, named by their bytes'
SHA-256, carrying only header segments someone reviewed.

Images are synthetic byte strings built from JPEG markers, as in the screenshot
tests; the EXIF block is the real one the delivered images carry.
"""

import hashlib
import json

import pytest

pytest.importorskip("jsonschema")

from tests.tutormoments_build.test_release_screenshots import (
    CLEAN,
    CLEAN_JPEG,
    NOW,
    jfif,
    jpeg,
    jsonl,
    segment,
)
from tutormoments_build.release import build as rb
from tutormoments_build.release.question_images import (
    ALLOWED_SEGMENTS,
    IMAGES_PREFIX,
    QUIZZES,
    write_question_images,
)
from tutormoments_build.release.screenshots import embedded_metadata
from tutormoments_build.release.snapshot import SNAPSHOT_INPUTS, create_snapshot
from tutormoments_build.release.store import LocalStore

# The EXIF block Windows GDI+ writes: gamma 2.2, sRGB, 96 dpi (3779 px/m), and
# nothing else. Every EXIF-carrying image in the 2026-11-24 delivery has it.
GDI_EXIF = bytes.fromhex(
    "4578696600004d4d002a00000008000503010005000000010000004a0303000100000001"
    "00000000511000010000000101000000511100040000000100000ec35112000400000001"
    "00000ec300000000000186a00000b18f"
)


def image_path(data: bytes) -> str:
    return f"{IMAGES_PREFIX}{hashlib.sha256(data).hexdigest()}.jpg"


PLAIN = CLEAN_JPEG
WITH_EXIF = jpeg(jfif(), segment(0xE1, GDI_EXIF))


def entry(path=None):
    return {"parent_question_id": "1"} | ({"question_image_path": path} if path else {})


def quizzes(*paths):
    """Released quiz rows, as write_assessments would leave them in the build."""
    rows = [{"questions": [entry(p), entry()]} for p in paths] + [{"questions": []}]

    def released_quizzes(ctx):
        ctx.store.write_bytes(ctx.build_prefix + QUIZZES, jsonl(*rows))
        return []

    return released_quizzes


@pytest.fixture
def store(tmp_path):
    s = LocalStore(tmp_path)
    for data in (PLAIN, WITH_EXIF):
        s.write_bytes(image_path(data), data)
    return s


def build(store, *paths):
    create_snapshot(store, snapshot_id="s1", inputs=(IMAGES_PREFIX,))
    return rb.run_build(
        store,
        "s1",
        writers=(quizzes(*paths), write_question_images),
        code=CLEAN,
        now=NOW,
    )


def test_every_listed_image_ships_at_its_path_and_is_hashed(store):
    paths = [image_path(PLAIN), image_path(WITH_EXIF), image_path(PLAIN)]
    manifest = build(store, *paths)
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    shipped = sorted(p for p in manifest["files"] if p.startswith(IMAGES_PREFIX))
    assert shipped == sorted({image_path(PLAIN), image_path(WITH_EXIF)})
    assert store.read_bytes(prefix + image_path(WITH_EXIF)) == WITH_EXIF
    assert "schemas" not in manifest["files"][image_path(PLAIN)]


def test_listed_image_missing_from_the_snapshot_fails(store):
    with pytest.raises(SystemExit, match="not in the snapshot"):
        build(store, image_path(PLAIN), image_path(WITH_EXIF), image_path(b"x"))


def test_image_no_quiz_lists_fails(store):
    with pytest.raises(SystemExit, match="no released quiz lists"):
        build(store, image_path(PLAIN))


def test_image_whose_name_is_not_its_hash_fails(store):
    store.write_bytes(image_path(PLAIN), WITH_EXIF)
    with pytest.raises(SystemExit, match="name is not the SHA-256"):
        build(store, image_path(PLAIN), image_path(WITH_EXIF))


@pytest.mark.parametrize(
    "extra",
    [
        segment(0xE1, GDI_EXIF[:-1] + b"\x00"),  # any change to the reviewed bytes
        segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta/>"),
        segment(0xFE, b"made by a camera"),
    ],
)
def test_unreviewed_metadata_fails_and_nothing_is_copied(tmp_path, extra):
    store = LocalStore(tmp_path)
    data = jpeg(jfif(), extra)
    store.write_bytes(image_path(data), data)
    with pytest.raises(SystemExit, match="embedded"):
        build(store, image_path(data))
    assert not [k for k in store.list("release-v2/builds/") if IMAGES_PREFIX in k]


def test_only_the_pinned_segments_are_allowed():
    assert hashlib.sha256(GDI_EXIF).hexdigest() in ALLOWED_SEGMENTS
    assert embedded_metadata(WITH_EXIF, ALLOWED_SEGMENTS) == []
    # The screenshot check allows none of them.
    assert embedded_metadata(WITH_EXIF) == ["APP1"]


def test_images_are_snapshotted_and_written_after_the_quizzes():
    assert IMAGES_PREFIX in SNAPSHOT_INPUTS
    names = [w.__name__ for w in rb.default_writers()]
    assert names.index("write_question_images") > names.index("write_assessments")


def test_quiz_schema_paths_match_the_shipped_layout():
    schema = json.loads(
        rb.packaged_schemas()["assessments_tutoring_provider_a.schema.json"]
    )
    pattern = schema["$defs"]["Question"]["properties"]["question_image_path"][
        "pattern"
    ]
    assert pattern.startswith("^" + IMAGES_PREFIX)
