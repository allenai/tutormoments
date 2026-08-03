"""Tests for the temporary ground-truth S3 publish script.

Uses moto[s3] (a dev dependency) to mock S3 -- no real AWS calls. The script is a
one-off release helper, not part of the runtime package; it lives under tutorsim_build/.
"""

import importlib.util
from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from moto import mock_aws

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tutorsim_build"
    / "publish_ground_truth_s3.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("publish_ground_truth_s3", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gt_file(tmp_path):
    f = tmp_path / "ground_truth_hybrid.jsonl"
    f.write_text('{"conversation_id": "c1", "key_moments": []}\n', encoding="utf-8")
    return f


@mock_aws
def test_publish_uploads_file_to_bucket_prefix(gt_file):
    mod = _load_module()
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="my-bucket")

    key = mod.publish(str(gt_file), bucket="my-bucket", prefix="releases/2026")

    assert key == "releases/2026/ground_truth_hybrid.jsonl"
    body = s3.get_object(Bucket="my-bucket", Key=key)["Body"].read().decode("utf-8")
    assert body == gt_file.read_text(encoding="utf-8")


@mock_aws
def test_publish_handles_empty_prefix(gt_file):
    mod = _load_module()
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="bucket-two")

    key = mod.publish(str(gt_file), bucket="bucket-two", prefix="")

    assert key == "ground_truth_hybrid.jsonl"
    assert s3.get_object(Bucket="bucket-two", Key=key)["Body"].read()


def test_publish_missing_file_raises(tmp_path):
    mod = _load_module()
    with pytest.raises(FileNotFoundError):
        mod.publish(str(tmp_path / "nope.jsonl"), bucket="b", prefix="p")


def test_resolve_bucket_prefers_arg_over_env(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("GROUND_TRUTH_S3_BUCKET", "env-bucket")
    assert mod.resolve_bucket("arg-bucket") == "arg-bucket"
    assert mod.resolve_bucket(None) == "env-bucket"


def test_resolve_bucket_errors_when_unset(monkeypatch):
    mod = _load_module()
    monkeypatch.delenv("GROUND_TRUTH_S3_BUCKET", raising=False)
    with pytest.raises(SystemExit):
        mod.resolve_bucket(None)
