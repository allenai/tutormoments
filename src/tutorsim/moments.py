"""Moment dataclass and released-dataset loading for tutorsim.

A Moment is the runnable unit of the benchmark: a human-annotated key moment
frozen at its cut point, packaged with the pre-cut transcript context, the
gold dimension, and the real human continuation (the oracle student's
reference). Moments are consumed from a released dataset — either a Hugging
Face dataset (``dataset=``) or a local release directory (``data_path=``) —
and are never constructed at runtime (see ``tutorsim_build/`` for that).
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# v2 (2026-07-03): student.trait — the frozen paper-run persona — is required
# on every record; the runtime consumes it and no longer generates traits.
DATASET_SCHEMA_VERSION = 2
DEFAULT_DATASET_CONFIG = "moments"
# Filenames follow the release's flat `{thing}.{kind}` convention, so a full
# release download is itself a valid --data_path directory.
MOMENTS_FILENAME = "moments.jsonl"
MANIFEST_FILENAME = "moments.manifest.json"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the requested moments dataset cannot be found."""


def _missing_dataset_message(path: Path) -> str:
    return (
        f"Moments dataset not found: expected {path}. Point --data_path at a "
        "local release directory containing moments.jsonl, or pass --dataset "
        "with a published Hugging Face dataset id."
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records_content_hash(records: list[dict]) -> str:
    """SHA-256 over the canonical JSON serialization of loaded records.

    Defined over records (sorted keys, newline-joined), not files, so the
    Hugging Face path and the local-file path verify against the same value.
    """
    digest = hashlib.sha256()
    for rec in records:
        digest.update(
            json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Moment
# ---------------------------------------------------------------------------


def _normalize_cut_votes(value: Any) -> dict:
    """Normalize provenance.cut_votes to its internal dict form.

    The released schema encodes cut_votes as a list of {cut_turn, votes}
    pairs because Arrow cannot represent variable-key maps faithfully; the
    internal form is {str(cut_turn): votes}. Accepts either form.
    """
    if isinstance(value, list):
        return {str(pair["cut_turn"]): pair["votes"] for pair in value}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


@dataclass
class Moment:
    """A declarative, self-contained key moment for tutor simulation.

    context: list of prefix turns, each {"turn_number": int, "role": str, "text": str}.
    turn_number is the REAL (non-sequential) turn number from the source transcript.
    role is lowercase ("tutor" or "student") as a structured value.
    """

    id: str
    context: list[dict]
    dimension: str
    student: dict
    rubric: dict
    provenance: dict

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict, preserving all fields (no dropping)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Moment":
        """Reconstruct a Moment from a dict (released or internal form)."""
        provenance = dict(d["provenance"])
        if "cut_votes" in provenance:
            provenance["cut_votes"] = _normalize_cut_votes(provenance["cut_votes"])
        return cls(
            id=d["id"],
            context=d["context"],
            dimension=d["dimension"],
            student=d["student"],
            rubric=d["rubric"],
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_moments_jsonl(path: str | Path) -> list[Moment]:
    """Read a moments.jsonl file into Moment objects, in file order."""
    moments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                moments.append(Moment.from_dict(json.loads(line)))
    return moments


def load_manifest(release_dir: str | Path) -> dict | None:
    """Load a release directory's moments.manifest.json if present."""
    path = Path(release_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_moments(
    *,
    dataset: str | None = None,
    data_path: str | Path | None = None,
    revision: str | None = None,
    config: str = DEFAULT_DATASET_CONFIG,
) -> tuple[list[Moment], dict]:
    """Load the runnable moments set from a released dataset.

    Exactly one source must be provided; data_path wins when both are set.

    Args:
        dataset: Hugging Face dataset id (normal-user path). Loaded via
            `datasets.load_dataset`, relying on the HF cache.
        data_path: Local release directory containing moments.jsonl
            (developer path; no network).
        revision: Pinned dataset revision (HF path only).
        config: Dataset config name holding the runnable set.

    Returns:
        (moments, source_meta) where source_meta records the resolved source
        and the record-level content hash for the run's config.json.
    """
    if data_path:
        path = Path(data_path) / MOMENTS_FILENAME
        if not path.exists():
            raise DatasetNotFoundError(_missing_dataset_message(path))
        moments = _read_moments_jsonl(path)
        source_meta = {
            "dataset_id": None,
            "revision": None,
            "data_path": str(Path(data_path)),
            "config": config,
        }
    elif dataset:
        from datasets import load_dataset  # heavy import; only on the HF path

        ds = load_dataset(dataset, name=config, revision=revision, split="train")
        moments = [Moment.from_dict(dict(row)) for row in ds]
        source_meta = {
            "dataset_id": dataset,
            "revision": revision,
            "data_path": None,
            "config": config,
        }
    else:
        raise ValueError(
            "No dataset source: pass --dataset <hf id> or --data_path <release dir> "
            "(or set dataset.id in the config)."
        )

    if not moments:
        raise DatasetNotFoundError(
            f"Dataset source {source_meta} yielded zero moments."
        )

    source_meta["record_count"] = len(moments)
    source_meta["content_hash"] = records_content_hash([m.to_dict() for m in moments])
    return moments, source_meta


def validate_dataset(release_dir: str | Path) -> dict:
    """Validate a local release directory's moments.jsonl against its manifest.

    Returns a small validation report. Raises ValueError/FileNotFoundError for
    missing files, count mismatches, hash mismatches, or malformed manifests.
    """
    release_dir = Path(release_dir)
    jsonl_path = release_dir / MOMENTS_FILENAME
    if not jsonl_path.exists():
        raise DatasetNotFoundError(_missing_dataset_message(jsonl_path))

    moments = _read_moments_jsonl(jsonl_path)
    manifest = load_manifest(release_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"Dataset manifest not found: {release_dir / MANIFEST_FILENAME}"
        )

    schema_version = manifest.get("schema_version")
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema_version {schema_version!r}; "
            f"expected {DATASET_SCHEMA_VERSION}"
        )

    expected_count = manifest.get("record_count", manifest.get("n"))
    if expected_count is not None and int(expected_count) != len(moments):
        raise ValueError(
            f"Dataset count mismatch: manifest={expected_count}, loaded={len(moments)}"
        )

    content_hash = records_content_hash([m.to_dict() for m in moments])
    expected_hash = manifest.get("content_hash")
    if expected_hash and expected_hash != content_hash:
        raise ValueError(
            f"Dataset content hash mismatch: manifest={expected_hash}, actual={content_hash}"
        )

    expected_file_hash = manifest.get("file_sha256")
    actual_file_hash = file_sha256(jsonl_path)
    if expected_file_hash and expected_file_hash != actual_file_hash:
        raise ValueError(
            f"Dataset file hash mismatch: manifest={expected_file_hash}, "
            f"actual={actual_file_hash}"
        )

    # Schema v2: every record must carry the frozen student persona.
    traitless = [
        m.id for m in moments if not (m.student.get("trait") or {}).get("persona")
    ]
    if traitless:
        raise ValueError(
            f"{len(traitless)} moment(s) have no student.trait persona "
            f"(first: {traitless[0]}); this release is not runnable against "
            "the paper's frozen student population."
        )

    return {
        "name": manifest.get("name"),
        "path": str(jsonl_path),
        "record_count": len(moments),
        "content_hash": content_hash,
        "file_sha256": actual_file_hash,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# Reference transcript (runtime fallback for batch mode; reused by build)
# ---------------------------------------------------------------------------


def _build_reference_transcript(conversation: dict, cut_turn: int) -> str:
    """Format the post-cut real human turns from a full conversation.

    Returns a newline-joined string of 'Turn N. ROLE: text' lines for every
    turn whose turn_number > cut_turn. Empty string if no post-cut turns.
    """
    lines = []
    for turn in conversation.get("turns", []):
        n = turn.get("turn_number")
        if n is None or n <= cut_turn:
            continue
        role = turn.get("role", "")
        text = turn.get("text", "")
        lines.append(f"Turn {n}. {role}: {text}")
    return "\n".join(lines)
