"""Read and write release objects in S3 or, for tests, a local directory.

Both backends address objects by a ``/``-separated key relative to the root.
Large files move through local temp files rather than memory.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT_ENV = "TUTORMOMENTS_RELEASE_ROOT"

_CHUNK = 8 * 1024 * 1024


def resolve_root(arg_root: str | None) -> str:
    """Return the store root: ``--root`` wins, else ``TUTORMOMENTS_RELEASE_ROOT``."""
    root = arg_root or os.environ.get(ROOT_ENV)
    if not root:
        raise SystemExit(f"No release root given. Pass --root or set {ROOT_ENV}.")
    return root


def open_store(root: str) -> "Store":
    """``s3://bucket[/prefix]`` -> S3Store; anything else is a local directory."""
    if root.startswith("s3://"):
        bucket, _, prefix = root[len("s3://") :].partition("/")
        return S3Store(bucket, prefix.strip("/"))
    return LocalStore(Path(root))


def dump_json(obj) -> bytes:
    """Stable, diff-friendly JSON for records the release writes (manifests etc.)."""
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest_stream(read) -> dict:
    """Hash a stream read in chunks: {"sha256", "bytes", "lines"} (newline count)."""
    digest = hashlib.sha256()
    size = lines = 0
    while chunk := read(_CHUNK):
        digest.update(chunk)
        size += len(chunk)
        lines += chunk.count(b"\n")
    return {"sha256": digest.hexdigest(), "bytes": size, "lines": lines}


def digest_file(path: Path) -> dict:
    with open(path, "rb") as fh:
        return digest_stream(fh.read)


class Store:
    """Interface shared by both backends."""

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        """Every key under ``prefix``, sorted."""
        raise NotImplementedError

    def download(self, key: str, dest: Path) -> Path:
        raise NotImplementedError

    def upload(self, src: Path, key: str) -> None:
        raise NotImplementedError

    def copy(self, src_key: str, dest_key: str) -> None:
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def write_bytes(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def digest(self, key: str) -> dict:
        """Hash an object by streaming it; nothing is written locally."""
        raise NotImplementedError

    def iter_lines(self, key: str):
        """Yield an object's lines as bytes, streamed."""
        raise NotImplementedError


class LocalStore(Store):
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file()
        )

    def download(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), dest)
        return dest

    def upload(self, src: Path, key: str) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    def copy(self, src_key: str, dest_key: str) -> None:
        self.upload(self._path(src_key), dest_key)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def digest(self, key: str) -> dict:
        return digest_file(self._path(key))

    def iter_lines(self, key: str):
        with open(self._path(key), "rb") as fh:
            yield from fh


class S3Store(Store):
    def __init__(self, bucket: str, prefix: str = ""):
        import boto3

        self.bucket = bucket
        self.prefix = prefix
        self.client = boto3.client("s3")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def list(self, prefix: str) -> list[str]:
        strip = len(self._key("")) if self.prefix else 0
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys.extend(obj["Key"][strip:] for obj in page.get("Contents", []))
        return sorted(keys)

    def download(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(key), str(dest))
        return dest

    def upload(self, src: Path, key: str) -> None:
        self.client.upload_file(str(src), self.bucket, self._key(key))

    def copy(self, src_key: str, dest_key: str) -> None:
        # Managed copy: multipart above 5 GB, server-side, never through this machine.
        self.client.copy(
            {"Bucket": self.bucket, "Key": self._key(src_key)},
            self.bucket,
            self._key(dest_key),
        )

    def read_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self._key(key))[
            "Body"
        ].read()

    def write_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def digest(self, key: str) -> dict:
        body = self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"]
        return digest_stream(body.read)

    def iter_lines(self, key: str):
        body = self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"]
        yield from body.iter_lines(chunk_size=_CHUNK, keepends=True)
