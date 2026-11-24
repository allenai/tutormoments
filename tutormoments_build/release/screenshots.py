"""Write ``screenshots/<provider>/`` from the frozen bucket frames.

Each frame ships at the path its transcript lists (``screenshots[].path``), so a
path resolves against the dataset root. Every frame a released transcript lists
must be in the snapshot, and every snapshotted frame must be listed by one: a
mismatch means the transcripts and frames came from different deliveries.

Each frame must also be a JPEG carrying nothing beyond the image: no EXIF, XMP,
ICC profile, comment or thumbnail. The human review checked what a frame shows,
not what its file carries. Every frame is checked before any is copied.

Frames are not data files, so this writer names no schema; the manifest hashes
them like every other build object. Runs after ``write_transcripts``, whose
output it reads.
"""

import hashlib
import json
import logging
from typing import Callable

from tutormoments_build.release.build import BuildContext, Output

logger = logging.getLogger("tutormoments_build.release.screenshots")

FRAMES_PREFIX = "screenshots/"
TRANSCRIPTS_DIR = "transcripts/"

# Show this many problems in the failure message; the count stays exact.
MAX_EXAMPLES = 5

_SOI = b"\xff\xd8"
_SOS, _EOI, _COM = 0xDA, 0xD9, 0xFE
# Markers that stand alone, with no length or payload.
_STANDALONE = {0x01, *range(0xD0, 0xD8)}


def embedded_metadata(data: bytes, allowed: frozenset[str] = frozenset()) -> list[str]:
    """What a JPEG's header carries beyond the image; empty for a clean frame.

    Walks the marker segments up to the start of the scan. A plain JFIF ``APP0``
    is allowed; a JFIF thumbnail, any other ``APP0``, ``APP1``-``APP15`` (EXIF,
    XMP, ICC, vendor data) and ``COM`` comments are reported, unless the SHA-256
    of the segment's payload is in ``allowed`` (exact bytes someone reviewed).
    """
    if not data.startswith(_SOI):
        return ["not a JPEG"]
    found = []
    i = len(_SOI)
    while True:
        if i + 2 > len(data):
            return found + ["truncated header"]
        if data[i] != 0xFF:
            return found + ["malformed header"]
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte before a marker
            i += 1
            continue
        if marker in _STANDALONE:
            i += 2
            continue
        if marker == _EOI:
            return found
        if i + 4 > len(data):
            return found + ["truncated header"]
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if length < 2 or i + 2 + length > len(data):
            return found + ["truncated header"]
        if marker == _SOS:
            return found
        payload = data[i + 4 : i + 2 + length]
        kind = None
        if marker == 0xE0:
            if not payload.startswith(b"JFIF\x00"):
                kind = f"APP0 {payload[:4].decode('latin-1')}"
            elif len(payload) >= 14 and (payload[12] or payload[13]):
                kind = "JFIF thumbnail"
        elif 0xE1 <= marker <= 0xEF:
            kind = f"APP{marker - 0xE0}"
        elif marker == _COM:
            kind = "COM"
        if kind and hashlib.sha256(payload).hexdigest() not in allowed:
            found.append(kind)
        i += 2 + length


def listed_frames(ctx: BuildContext) -> set[str]:
    """Every frame path a transcript in this build lists."""
    paths = set()
    for key in ctx.store.list(ctx.build_prefix + TRANSCRIPTS_DIR):
        for line in ctx.store.iter_lines(key):
            for shot in json.loads(line).get("screenshots", []):
                paths.add(shot["path"])
    return paths


def ship_listed_files(
    ctx: BuildContext,
    label: str,
    listed: set[str],
    prefix: str,
    lister: str,
    check: Callable[[str, bytes], list[str]],
) -> None:
    """Copy each listed snapshot file to the same path in the build.

    Fails before copying anything on a listed path missing from the snapshot, a
    snapshotted file under ``prefix`` that no ``lister`` lists, or a file ``check``
    (given its path and bytes) reports problems with.
    """
    frozen = {key for key in ctx.snapshot["inputs"] if key.startswith(prefix)}
    problems = [f"listed but not in the snapshot: {p}" for p in sorted(listed - frozen)]
    problems += [
        f"in the snapshot but no {lister} lists it: {p}"
        for p in sorted(frozen - listed)
    ]
    for path in sorted(listed & frozen):
        data = ctx.store.read_bytes(ctx.snapshot_key(path))
        problems += [f"{path}: {problem}" for problem in check(path, data)]
    if problems:
        raise SystemExit(
            f"{label}: {len(problems)} problem(s), e.g. "
            + "; ".join(problems[:MAX_EXAMPLES])
        )
    for path in sorted(listed):
        ctx.copy_from_snapshot(path, path)


def _clean_frame(path: str, data: bytes) -> list[str]:
    found = embedded_metadata(data)
    return [f"embedded {', '.join(found)}"] if found else []


def write_screenshots(ctx: BuildContext) -> list[Output]:
    listed = listed_frames(ctx)
    ship_listed_files(
        ctx, "screenshots", listed, FRAMES_PREFIX, "released transcript", _clean_frame
    )
    logger.info("wrote %d screenshots", len(listed))
    return []
