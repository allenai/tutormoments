#!/usr/bin/env python3
"""Publish a built ground-truth file to S3 as part of the data release.

TEMPORARY one-off release helper -- not part of the tutormoments package. Uploads the
consolidated ground_truth_hybrid.jsonl (or any file) to S3 using the standard AWS
credential chain (env vars / shared config / instance profile). Bucket and prefix
come from CLI args or env vars; nothing is hardcoded and no S3 vars are added to
.env.example.

Usage:
    python scripts/publish_ground_truth_s3.py data/ground_truth_hybrid/ground_truth_hybrid.jsonl \
        --bucket my-bucket --prefix releases/2026-06-30

Env fallbacks (CLI args take precedence):
    GROUND_TRUTH_S3_BUCKET   -- default bucket when --bucket is omitted
    GROUND_TRUTH_S3_PREFIX   -- default key prefix when --prefix is omitted
"""

import argparse
import os
import sys
from pathlib import Path

import boto3


def resolve_bucket(arg_bucket):
    """Return the target bucket: CLI arg wins, else GROUND_TRUTH_S3_BUCKET env."""
    bucket = arg_bucket or os.environ.get("GROUND_TRUTH_S3_BUCKET")
    if not bucket:
        raise SystemExit(
            "No S3 bucket given. Pass --bucket or set GROUND_TRUTH_S3_BUCKET."
        )
    return bucket


def resolve_prefix(arg_prefix):
    """Return the key prefix: CLI arg wins, else GROUND_TRUTH_S3_PREFIX env, else ''."""
    if arg_prefix is not None:
        return arg_prefix
    return os.environ.get("GROUND_TRUTH_S3_PREFIX", "")


def publish(file_path, *, bucket, prefix):
    """Upload file_path to s3://{bucket}/{prefix}/{basename}. Returns the object key."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"file to publish not found: {path}")
    key = f"{prefix.rstrip('/')}/{path.name}" if prefix else path.name
    s3 = boto3.client("s3")
    s3.upload_file(str(path), bucket, key)
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", help="Path to the ground-truth file to upload")
    parser.add_argument(
        "--bucket",
        default=None,
        help="Target S3 bucket (or set GROUND_TRUTH_S3_BUCKET)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Key prefix (or set GROUND_TRUTH_S3_PREFIX; default none)",
    )
    args = parser.parse_args(argv)

    bucket = resolve_bucket(args.bucket)
    prefix = resolve_prefix(args.prefix)
    key = publish(args.file, bucket=bucket, prefix=prefix)
    print(f"Uploaded {args.file} -> s3://{bucket}/{key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
