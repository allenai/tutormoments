"""Which release build the v2 research code reads, and a record of what it read.

The build is named by ``release.json`` beside this file: its ``build_id`` and the
sha256 of its manifest. Moving research to a new build is a PR changing that
file, so a commit always says which data it ran on. ``tutormoments-build release
pull`` fetches the named build into ``temp/release/builds/<build_id>/``.
``TUTORMOMENTS_V2_RELEASE`` points at another build folder instead, to try a
build before its PR.

Every script stamps its outputs with ``provenance``: the build ID, whether it
is the pinned one, and the sha256 of each input read. Files inside a build are
checked against its manifest on the way, so an edited local copy can't pass
as the release.
"""

import hashlib
import json
import logging
import os

logger = logging.getLogger("tutormoments_build.v2.release_data")

POINTER_PATH = os.path.join(os.path.dirname(__file__), "release.json")
LOCAL_ROOT = os.path.join("temp", "release")
RELEASE_ENV = "TUTORMOMENTS_V2_RELEASE"
MANIFEST_FILE = "manifest.json"

PULL_HINT = "run `tutormoments-build release pull` to fetch it"

_CHUNK = 8 * 1024 * 1024


class ReleaseDataError(ValueError):
    """An input doesn't match the release build it claims to come from."""


def pinned_build(pointer_path: str = POINTER_PATH) -> dict:
    """``{"build_id", "manifest_sha256"}`` from the committed pointer."""
    with open(pointer_path, encoding="utf-8") as fh:
        return json.load(fh)


def default_build_dir(
    pointer_path: str = POINTER_PATH, local_root: str = LOCAL_ROOT
) -> str:
    """``$TUTORMOMENTS_V2_RELEASE`` if set, else the pinned build's local folder."""
    override = os.environ.get(RELEASE_ENV)
    if override:
        return override
    return os.path.join(local_root, "builds", pinned_build(pointer_path)["build_id"])


def build_path(*parts: str) -> str:
    """A path inside the default build, e.g. ``build_path("benchmark")``."""
    return os.path.join(default_build_dir(), *parts)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _build_root(path: str) -> str | None:
    """The ``.../builds/<build_id>`` folder ``path`` sits in, if any.

    Found by name rather than by a manifest, so a build that was never pulled
    (or is half pulled) is still recognised as one.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "builds" and i + 1 < len(parts):
            return os.sep.join(parts[: i + 2]) or os.sep
    return None


def require_local(paths) -> None:
    """Fail clearly if any input is missing, saying how to fetch a missing build."""
    for path in map(str, paths):
        root = _build_root(path)
        if root and not os.path.isfile(os.path.join(root, MANIFEST_FILE)):
            raise FileNotFoundError(
                f"release build {os.path.basename(root)} is not local "
                f"(no {MANIFEST_FILE} in {root}); {PULL_HINT}"
            )
        if not os.path.exists(path):
            raise FileNotFoundError(f"no such input: {path}")


def _load_manifest(root: str) -> dict:
    manifest_path = os.path.join(root, MANIFEST_FILE)
    if not os.path.isfile(manifest_path):
        raise ReleaseDataError(f"{root} has no {MANIFEST_FILE}; {PULL_HINT}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def provenance(paths, pointer_path: str = POINTER_PATH) -> dict:
    """``{"build_id", "pinned", "files": {path: sha256}}`` for the inputs read.

    A file inside a build is keyed by its release path (``benchmark/dev.jsonl``)
    and must match the build's manifest. A file outside any build (a research
    input such as ``student_state.jsonl``) is keyed by the path as given.
    Inputs from two different builds are refused: one output, one release.
    """
    build_id = None
    files: dict[str, str] = {}
    manifests: dict[str, dict] = {}
    for path in map(str, paths):
        sha = _sha256(path)
        root = _build_root(path)
        if root is None:
            files[path] = sha
            continue
        if root not in manifests:
            manifests[root] = _load_manifest(root)
        manifest = manifests[root]
        if build_id is not None and manifest["build_id"] != build_id:
            raise ReleaseDataError(
                f"inputs come from two builds: {build_id} and {manifest['build_id']}"
            )
        build_id = manifest["build_id"]
        release_path = os.path.relpath(os.path.abspath(path), root).replace(os.sep, "/")
        entry = manifest["files"].get(release_path)
        if entry is None:
            raise ReleaseDataError(
                f"{release_path} is not in build {build_id}'s {MANIFEST_FILE}"
            )
        if entry["sha256"] != sha:
            raise ReleaseDataError(
                f"{path}: sha256 does not match build {build_id}'s {MANIFEST_FILE}; "
                f"the local copy was changed. {PULL_HINT}"
            )
        files[release_path] = sha

    pinned = pinned_build(pointer_path)["build_id"]
    if build_id is not None and build_id != pinned:
        logger.warning(
            "reading build %s, not the pinned build %s (%s)",
            build_id,
            pinned,
            os.path.basename(pointer_path),
        )
    return {
        "build_id": build_id,
        "pinned": build_id is not None and build_id == pinned,
        "files": files,
    }
