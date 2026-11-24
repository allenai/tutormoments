"""Write ``eedi_question_images/`` from the frozen bucket images.

A provider A quiz entry names the image of the question the student saw
(``questions[].question_image_path``), relative to the dataset root, and each image
ships at that path. Every image the released quizzes list must be in the snapshot,
and every snapshotted image must be listed. An image is named by the SHA-256 of its
bytes, which is checked too.

The images are Eedi's question content, not session data. Beyond a plain JFIF
header, an image may carry only the exact header segments pinned in
``ALLOWED_SEGMENTS``, reviewed 2026-11-24: technical, nothing about who made the
image or when. Any other segment fails the build until someone reviews it.

Images are not data files, so this writer names no schema; the manifest hashes
them like every other build object. Runs after ``write_assessments``, whose output
it reads.
"""

import hashlib
import json
import logging
from pathlib import PurePosixPath

from tutormoments_build.release.build import BuildContext, Output
from tutormoments_build.release.screenshots import (
    embedded_metadata,
    ship_listed_files,
)

logger = logging.getLogger("tutormoments_build.release.question_images")

IMAGES_PREFIX = "eedi_question_images/"
QUIZZES = "assessments/tutoring_provider_a.jsonl"

# SHA-256 of each reviewed segment payload.
ALLOWED_SEGMENTS = frozenset(
    {
        # APP1 EXIF as Windows GDI+ writes it: gamma 2.2, sRGB, 96 dpi. 965 images.
        "d3e900451b43acafa1908910601cd7dfb143913a2dd2a954914e9f7cf23c0874",
        # APP2 ICC profile: Google's sRGB profile, four byte variants. 55 images.
        "43f0eebabbfc0a90e5faab31fca21ed9002e2a09b2fb1bac5da66c8bb4b8774f",
        "c3bb12de30d7357252ec3a5ec781bd2f8a6dd8c69dd7d3de97bbac262d9e1fd4",
        "db12f8c7a699fa6d0459f03b52266f5f643266bdbefaec1a68cdc5734498ac0b",
        "43f980bcf3fe7684b317e2511014a3e6739ddc9e1686d31b0fe24347b8cf1e16",
    }
)


def listed_images(ctx: BuildContext) -> set[str]:
    """Every image path a quiz entry in this build lists."""
    paths = set()
    for line in ctx.store.iter_lines(ctx.build_prefix + QUIZZES):
        for question in json.loads(line)["questions"]:
            if "question_image_path" in question:
                paths.add(question["question_image_path"])
    return paths


def _clean_image(path: str, data: bytes) -> list[str]:
    problems = []
    if hashlib.sha256(data).hexdigest() != PurePosixPath(path).stem:
        problems.append("name is not the SHA-256 of its bytes")
    found = embedded_metadata(data, ALLOWED_SEGMENTS)
    if found:
        problems.append(f"embedded {', '.join(found)}")
    return problems


def write_question_images(ctx: BuildContext) -> list[Output]:
    listed = listed_images(ctx)
    ship_listed_files(
        ctx, "question images", listed, IMAGES_PREFIX, "released quiz", _clean_image
    )
    logger.info("wrote %d question images", len(listed))
    return []
