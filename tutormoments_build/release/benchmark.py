"""Write ``benchmark/dev.jsonl`` and ``test.jsonl``: the gold labels for every moment.

The labels are built by ``tutormoments_build.v2.build_ground_truth`` from the
snapshot's annotation exports and the committed ``splits.json``, and written in
their release shape (``release_splits``). Nothing is copied from the working
ground truth in the bucket.

The release promises that running that script on the released annotation files
gives these files exactly. Every build checks it: the labels are built a second
time from the ``annotations/`` files the annotations writer just wrote, and the
build fails unless both runs give the same bytes. So this writer must run after
``write_annotations``.

Staff are excluded upstream, so no exclusion list is read; a staff annotator
would have no release label and fail the annotations writer first.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from tutormoments_build.release.annotations import LABELS, PROVIDERS
from tutormoments_build.release.build import BuildContext, Output
from tutormoments_build.v2 import build_ground_truth as gt

logger = logging.getLogger("tutormoments_build.release.benchmark")

SCHEMA = "benchmark_labels.schema.json"
SPLITS = gt.DEFAULT_MANIFEST
ANNOTATIONS_DIR = "annotations/"
BENCHMARK_DIR = "benchmark/"


def export_key(provider: str) -> str:
    return f"v2_annotations/source/{provider}_annotations.jsonl"


def labels_files(
    annotations: list[Path], labels: Path, no_staff: Path, out_dir: Path
) -> tuple[dict[str, Path], Counter]:
    """Build the release label files into ``out_dir``: {release name: path}."""
    out, dropped = gt.build(
        [str(p) for p in annotations], SPLITS, str(labels), str(no_staff)
    )
    files = {
        name: Path(gt.write_split(str(out_dir), name, records))
        for name, records in gt.release_splits(out).items()
    }
    return files, dropped


def write_benchmark_labels(ctx: BuildContext) -> list[Output]:
    work = ctx.workdir / "benchmark"
    export_dir, released_dir = work / "export", work / "released"
    export_dir.mkdir(parents=True)
    released_dir.mkdir()

    exports = [
        ctx.store.download(
            ctx.snapshot_key(export_key(p)), export_dir / f"{p}_annotations.jsonl"
        )
        for p in PROVIDERS
    ]
    labels = ctx.store.download(
        ctx.snapshot_key(LABELS), work / "annotator_labels.json"
    )
    no_staff = work / "no_staff.json"
    no_staff.write_text(json.dumps([]))

    built, dropped = labels_files(exports, labels, no_staff, work / "from_export")
    logger.info(
        "benchmark labels: %s; not emitted: %s",
        {name: sum(1 for _ in open(p, encoding="utf-8")) for name, p in built.items()},
        dict(dropped),
    )

    released = []
    for provider in PROVIDERS:
        key = ctx.build_prefix + ANNOTATIONS_DIR + f"{provider}.jsonl"
        if not ctx.store.exists(key):
            raise SystemExit(
                f"{ANNOTATIONS_DIR}{provider}.jsonl is not in the build; the benchmark "
                "labels are checked against it, so write_annotations must run first"
            )
        released.append(ctx.store.download(key, released_dir / f"{provider}.jsonl"))
    rebuilt, _ = labels_files(released, labels, no_staff, work / "from_release")

    outputs = []
    for name, path in built.items():
        release_path = f"{BENCHMARK_DIR}{name}.jsonl"
        if path.read_bytes() != rebuilt[name].read_bytes():
            raise SystemExit(
                f"the released annotations do not reproduce {release_path}; the "
                "annotations release has lost something the labels depend on"
            )
        ctx.write(release_path, path)
        outputs.append(Output(release_path, (SCHEMA,)))
    return outputs
