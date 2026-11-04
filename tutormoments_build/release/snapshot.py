"""Freeze the release inputs into ``<release>/snapshots/<snapshot_id>/``.

Each input is copied server-side under its own bucket key, then hashed from the
copy, so ``snapshot.json`` describes exactly the bytes a build will read.
``snapshot.json`` is written last: a snapshot folder without it is incomplete
and builds refuse it. An existing snapshot is never overwritten.
"""

import json
import logging
from datetime import datetime, timezone

from tutormoments_build.release.store import Store, dump_json

logger = logging.getLogger("tutormoments_build.release.snapshot")

DEFAULT_RELEASE = "release-v2"
SNAPSHOT_FILE = "snapshot.json"

# Bucket keys a release build reads. A key ending in "/" is a prefix: every
# object under it is frozen. Provider B adds its keys here; A and C's stay put.
SNAPSHOT_INPUTS = (
    "source/tutoring_provider_a.jsonl",
    "source/tutoring_provider_c.jsonl",
    "source/tutoring_provider_a_eedi_quiz_sessions.jsonl",
    "source/tutoring_provider_c_assessments.jsonl",
    "source/eedi_constructs.jsonl",
    "source/eedi_diagnostic_questions.jsonl",
    "source/eedi_misconceptions.jsonl",
    "source/eedi_subjects.jsonl",
    "v2_annotations/source/tutoring_provider_a_annotations.jsonl",
    "v2_annotations/source/tutoring_provider_c_annotations.jsonl",
    "v2_annotations/annotator_labels.json",
    "synthetic/ground_truth_v2/",
    "schemas/tutoring_provider_a.schema.json",
    "schemas/tutoring_provider_c.schema.json",
    "schemas/tutoring_provider_a_eedi_quiz_sessions.schema.json",
    "schemas/tutoring_provider_c_assessments.schema.json",
    "schemas/eedi_constructs.schema.json",
    "schemas/eedi_diagnostic_questions.schema.json",
    "schemas/eedi_misconceptions.schema.json",
    "schemas/eedi_subjects.schema.json",
    "docs/tutoring_provider_a.md",
    "docs/tutoring_provider_c.md",
    "docs/tutoring_provider_a_eedi_quiz_sessions.md",
    "docs/tutoring_provider_c_assessments.md",
    "docs/eedi_constructs.md",
    "docs/eedi_diagnostic_questions.md",
    "docs/eedi_misconceptions.md",
    "docs/eedi_subjects.md",
)


def snapshot_prefix(release: str, snapshot_id: str) -> str:
    return f"{release}/snapshots/{snapshot_id}/"


def new_snapshot_id(now: datetime | None = None) -> str:
    """UTC minute stamp, e.g. ``20261103T1502Z``."""
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%MZ")


def expand_inputs(store: Store, inputs: tuple[str, ...]) -> list[str]:
    """Resolve prefixes to their keys. A missing input is an error, not a skip."""
    keys = []
    for item in inputs:
        if item.endswith("/"):
            found = store.list(item)
        else:
            found = [item] if store.exists(item) else []
        if not found:
            raise SystemExit(f"snapshot input missing from the bucket: {item}")
        keys.extend(found)
    return keys


def create_snapshot(
    store: Store,
    *,
    release: str = DEFAULT_RELEASE,
    snapshot_id: str | None = None,
    inputs: tuple[str, ...] = SNAPSHOT_INPUTS,
    dry_run: bool = False,
) -> dict:
    """Copy and hash every input; return the snapshot record."""
    snapshot_id = snapshot_id or new_snapshot_id()
    prefix = snapshot_prefix(release, snapshot_id)
    if store.list(prefix):
        raise SystemExit(
            f"snapshot {snapshot_id} already exists; snapshots are never overwritten"
        )

    keys = expand_inputs(store, inputs)
    if dry_run:
        logger.info("dry run: would freeze %d objects into %s", len(keys), prefix)
        return {"snapshot_id": snapshot_id, "inputs": {key: None for key in keys}}

    record = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {},
    }
    for key in keys:
        store.copy(key, prefix + key)
        record["inputs"][key] = store.digest(prefix + key)
        logger.info("froze %s (%d bytes)", key, record["inputs"][key]["bytes"])
    store.write_bytes(prefix + SNAPSHOT_FILE, dump_json(record))
    logger.info("snapshot %s complete: %d objects", snapshot_id, len(keys))
    return record


def load_snapshot(store: Store, release: str, snapshot_id: str) -> dict:
    """Return a complete snapshot's record; refuse an incomplete one."""
    key = snapshot_prefix(release, snapshot_id) + SNAPSHOT_FILE
    if not store.exists(key):
        raise SystemExit(
            f"snapshot {snapshot_id} has no {SNAPSHOT_FILE}; it is missing or incomplete"
        )
    return json.loads(store.read_bytes(key))
