"""Build a release from a frozen snapshot into ``<release>/builds/<build_id>/``.

A build copies the packaged schemas and docs, runs each writer in
``default_writers()``, checks every data file it wrote (``checks.py``), and only
then writes ``manifest.json``. A build folder without
a manifest failed or is incomplete and must not be published. A failed build
leaves ``checks.json`` beside its files saying why.

``build_id`` is the UTC minute plus the short code commit, so the folder name
alone says which code built it. A dirty working tree is refused unless
``allow_dirty`` is set, and the manifest records that it was.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Callable

from tutormoments_build.release import checks
from tutormoments_build.release.snapshot import (
    DEFAULT_RELEASE,
    SNAPSHOT_FILE,
    load_snapshot,
    new_snapshot_id,
    snapshot_prefix,
)
from tutormoments_build.release.store import Store, dump_json

logger = logging.getLogger("tutormoments_build.release.build")

MANIFEST_FILE = "manifest.json"
CHECKS_FILE = "checks.json"
SCHEMA_DIR = "schemas/"
DOCS_DIR = "docs/"
MANIFEST_VERSION = "1"


@dataclass
class BuildContext:
    store: Store
    release: str
    snapshot: dict
    snapshot_prefix: str
    build_prefix: str
    workdir: Path

    def snapshot_key(self, key: str) -> str:
        """Store key of a frozen input, given its original bucket key."""
        return self.snapshot_prefix + key

    def write(self, path: str, src: Path) -> None:
        self.store.upload(src, self.build_prefix + path)

    def copy_from_snapshot(self, key: str, path: str) -> None:
        self.store.copy(self.snapshot_key(key), self.build_prefix + path)


@dataclass(frozen=True)
class Output:
    """One data file a writer produced, and the schemas (in ``schemas/``) it must pass."""

    path: str
    schemas: tuple[str, ...]


Writer = Callable[[BuildContext], list[Output]]


def default_writers() -> tuple[Writer, ...]:
    """Release writers, run in order. Each stage adds its writer once its schema
    is approved. Imported here because writers import this module."""
    from tutormoments_build.release.annotations import write_annotations
    from tutormoments_build.release.benchmark import write_benchmark_labels
    from tutormoments_build.release.transcripts import write_transcripts

    return (write_transcripts, write_annotations, write_benchmark_labels)


def build_prefix(release: str, build_id: str) -> str:
    return f"{release}/builds/{build_id}/"


def code_state() -> dict:
    """The commit this code is running from, and whether tracked files differ from it."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def packaged_schemas() -> dict[str, bytes]:
    """The release-owned schemas shipped with this package (e.g. the core contracts)."""
    folder = resources.files("tutormoments_build.release") / "schemas"
    return {
        entry.name: entry.read_bytes()
        for entry in folder.iterdir()
        if entry.name.endswith(".schema.json")
    }


def packaged_docs() -> dict[str, bytes]:
    """The release-owned docs (e.g. the annotation guide), shipped under ``docs/``."""
    folder = resources.files("tutormoments_build.release") / "docs"
    return {
        entry.name: entry.read_bytes()
        for entry in folder.iterdir()
        if entry.name.endswith(".md")
    }


def run_build(
    store: Store,
    snapshot_id: str,
    *,
    release: str = DEFAULT_RELEASE,
    writers: tuple[Writer, ...] | None = None,
    allow_dirty: bool = False,
    code: dict | None = None,
    now: datetime | None = None,
) -> dict:
    """Build, check and (if every check passes) write the manifest. Returns it."""
    snapshot = load_snapshot(store, release, snapshot_id)
    code = code or code_state()
    if code["dirty"] and not allow_dirty:
        raise SystemExit(
            "working tree has uncommitted changes; commit them or pass --allow-dirty"
        )

    build_id = f"{new_snapshot_id(now)}-{code['commit'][:7]}"
    prefix = build_prefix(release, build_id)
    if store.list(prefix):
        raise SystemExit(f"build {build_id} already exists")

    if writers is None:
        writers = default_writers()
    for name, data in sorted(packaged_schemas().items()):
        store.write_bytes(prefix + SCHEMA_DIR + name, data)
    for name, data in sorted(packaged_docs().items()):
        store.write_bytes(prefix + DOCS_DIR + name, data)

    with tempfile.TemporaryDirectory(prefix="tutormoments-release-") as tmp:
        ctx = BuildContext(
            store=store,
            release=release,
            snapshot=snapshot,
            snapshot_prefix=snapshot_prefix(release, snapshot_id),
            build_prefix=prefix,
            workdir=Path(tmp),
        )
        outputs = []
        for writer in writers:
            logger.info("running %s", writer.__name__)
            outputs.extend(writer(ctx))

    reports = check_outputs(store, prefix, outputs)
    failed = [r for r in reports if r.errors]
    if failed:
        store.write_bytes(
            prefix + CHECKS_FILE, dump_json([r.__dict__ for r in reports])
        )
        for report in failed:
            logger.error(
                "%s: %d problem(s), e.g. %s",
                report.path,
                report.errors,
                report.examples[:3],
            )
        raise SystemExit(
            f"build {build_id} failed its checks; see {CHECKS_FILE}. No manifest written."
        )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "release": release,
        "build_id": build_id,
        "created_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "code": code,
        "snapshot": {
            "snapshot_id": snapshot_id,
            "snapshot_json_sha256": store.digest(
                snapshot_prefix(release, snapshot_id) + SNAPSHOT_FILE
            )["sha256"],
        },
        "files": describe_files(store, prefix, outputs, reports),
    }
    store.write_bytes(prefix + MANIFEST_FILE, dump_json(manifest))
    logger.info("build %s complete: %d files", build_id, len(manifest["files"]))
    return manifest


def check_outputs(
    store: Store, prefix: str, outputs: list[Output]
) -> list[checks.FileReport]:
    """Run every check on every data file; a file naming no schema fails."""
    validators: dict[str, checks.Draft202012Validator] = {}
    reports = []
    for output in outputs:
        if not output.schemas:
            report = checks.FileReport(output.path)
            report.add("names no schema; every data file needs one")
            reports.append(report)
            continue
        for name in output.schemas:
            if name not in validators:
                schema = json.loads(store.read_bytes(prefix + SCHEMA_DIR + name))
                validators[name] = checks.load_validator(schema)
        reports.append(
            checks.check_file(
                store,
                prefix + output.path,
                output.path,
                [validators[name] for name in output.schemas],
            )
        )
    return reports


def describe_files(
    store: Store, prefix: str, outputs: list[Output], reports: list[checks.FileReport]
) -> dict:
    """Hash every object in the build; data files also carry rows and schemas."""
    data = {o.path: (o, r) for o, r in zip(outputs, reports)}
    files = {}
    for key in store.list(prefix):
        path = key[len(prefix) :]
        if path in (MANIFEST_FILE, CHECKS_FILE):
            continue
        entry = store.digest(key)
        entry.pop("lines")
        if path in data:
            output, report = data[path]
            entry["rows"] = report.rows
            entry["schemas"] = list(output.schemas)
        files[path] = entry
    return files


def verify_build(
    store: Store, build_id: str, *, release: str = DEFAULT_RELEASE
) -> list[str]:
    """Re-check a finished build against its manifest. Returns problems (empty = good)."""
    prefix = build_prefix(release, build_id)
    if not store.exists(prefix + MANIFEST_FILE):
        return [f"build {build_id} has no {MANIFEST_FILE}; it failed or is incomplete"]
    manifest = json.loads(store.read_bytes(prefix + MANIFEST_FILE))
    listed = manifest["files"]
    present = {key[len(prefix) :] for key in store.list(prefix)} - {
        MANIFEST_FILE,
        CHECKS_FILE,
    }

    problems = [f"not in manifest: {p}" for p in sorted(present - listed.keys())]
    problems += [f"missing: {p}" for p in sorted(listed.keys() - present)]
    for path in sorted(present & listed.keys()):
        if store.digest(prefix + path)["sha256"] != listed[path]["sha256"]:
            problems.append(f"hash mismatch: {path}")

    outputs = [
        Output(path, tuple(entry["schemas"]))
        for path, entry in sorted(listed.items())
        if "schemas" in entry and path in present
    ]
    for report in check_outputs(store, prefix, outputs):
        problems += [f"{report.path}: {msg}" for msg in report.examples]
        if report.errors > len(report.examples):
            problems.append(
                f"{report.path}: {report.errors - len(report.examples)} more"
            )
    return problems
