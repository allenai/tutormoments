"""Fetch one finished build into a local folder: ``<dest>/builds/<build_id>/``.

Every file is sha256-checked against the build's manifest, and the manifest
against the hash pinned in the repo, so a local copy is exactly the release.
A file already held by another local build (same sha256) is hard-linked rather
than downloaded: builds mostly share files, so a new one costs only what
changed. The manifest is written last, so an interrupted pull leaves a folder
research code treats as not there.
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from tutormoments_build.release.build import (
    DEFAULT_RELEASE,
    MANIFEST_FILE,
    build_prefix,
)
from tutormoments_build.release.store import Store, digest_file

logger = logging.getLogger("tutormoments_build.release.pull")


class PullError(RuntimeError):
    """The store's copy doesn't match what the pin or manifest says."""


def _local_index(dest: Path, skip: str) -> dict[str, Path]:
    """{sha256: path} for files in other local builds, from their manifests."""
    index: dict[str, Path] = {}
    builds = dest / "builds"
    if not builds.is_dir():
        return index
    for build in sorted(builds.iterdir()):
        manifest_path = build / MANIFEST_FILE
        if build.name == skip or not manifest_path.is_file():
            continue
        listed = json.loads(manifest_path.read_text())["files"]
        for path, entry in listed.items():
            if (build / path).is_file():
                index.setdefault(entry["sha256"], build / path)
    return index


def pull_build(
    store: Store,
    build_id: str,
    dest: Path,
    *,
    manifest_sha256: str | None,
    release: str = DEFAULT_RELEASE,
    only: list[str] | None = None,
    prune: bool = False,
) -> dict:
    """Make ``<dest>/builds/<build_id>`` a checked copy of the build.

    ``only`` limits the files to some release-path prefixes (the manifest is
    still written whole). ``prune`` then deletes every other local build.
    Returns ``{"downloaded", "linked", "kept", "pruned"}``.
    """
    dest = Path(dest)
    prefix = build_prefix(release, build_id)
    manifest_bytes = store.read_bytes(prefix + MANIFEST_FILE)
    if manifest_sha256 is not None:
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != manifest_sha256:
            raise PullError(
                f"build {build_id}: manifest sha256 {actual} does not match the "
                f"pinned {manifest_sha256}"
            )
    else:
        logger.warning("build %s: no pinned manifest hash to check", build_id)
    listed = json.loads(manifest_bytes)["files"]

    target = dest / "builds" / build_id
    manifest_path = target / MANIFEST_FILE
    # Until the new manifest lands, this folder is incomplete.
    manifest_path.unlink(missing_ok=True)
    index = _local_index(dest, skip=build_id)

    counts = {"downloaded": 0, "linked": 0, "kept": 0}
    for path, entry in sorted(listed.items()):
        if only and not any(path.startswith(p) for p in only):
            continue
        want = entry["sha256"]
        local = target / path
        if local.is_file() and digest_file(local)["sha256"] == want:
            counts["kept"] += 1
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".part")
        tmp.unlink(missing_ok=True)
        source = index.get(want)
        if source is not None and digest_file(source)["sha256"] == want:
            os.link(source, tmp)
            kind = "linked"
        else:
            store.download(prefix + path, tmp)
            kind = "downloaded"
        if digest_file(tmp)["sha256"] != want:
            tmp.unlink()
            raise PullError(
                f"build {build_id}: {path} does not match its manifest sha256"
            )
        # Replace rather than write through: the old file may be a hard link
        # shared with another build.
        local.unlink(missing_ok=True)
        os.replace(tmp, local)
        counts[kind] += 1

    manifest_path.write_bytes(manifest_bytes)

    pruned = []
    if prune:
        for other in sorted((dest / "builds").iterdir()):
            if other.name != build_id and other.is_dir():
                shutil.rmtree(other)
                pruned.append(other.name)
    logger.info(
        "build %s at %s: %d downloaded, %d linked, %d kept; pruned %s",
        build_id,
        target,
        counts["downloaded"],
        counts["linked"],
        counts["kept"],
        pruned or "nothing",
    )
    return {**counts, "pruned": pruned}
