"""Write ``benchmark/dev.jsonl`` and ``test.jsonl``: the gold labels for every moment.

The labels come from ``tutormoments_build.v2.build_ground_truth``, run unchanged
on the snapshot's annotation exports with ``splits.json``, and are written in
their release shape (``release_record``). Nothing is copied from the working
ground truth in the bucket. ``build_ground_truth.py`` itself is not changed for
the release while the team's work on it is in progress, so the release shape
lives here.

Every build also checks that the released annotations hold everything the labels
depend on: the labels are built a second time from the ``annotations/`` files the
annotations writer just wrote (reshaped back to the export's form by
``as_export_row``), and the build fails unless both runs give the same bytes. So
this writer must run after ``write_annotations``.

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

# build_ground_truth output stem -> release file name. Hugging Face reads "dev"
# as the validation split; it does not know "iteration".
RELEASE_SPLITS = {"iteration": "dev", "test": "test"}

BOUNDARY_FIELDS = ("start_turn", "end_turn", "cut_turn")


def export_key(provider: str) -> str:
    return f"v2_annotations/source/{provider}_annotations.jsonl"


def release_record(record: dict) -> dict:
    """One ground-truth record in its release shape (``benchmark_labels.schema.json``).

    The split takes its release name, each annotator is paired with their role
    in one object rather than two parallel lists, ``original_boundaries`` is on
    every row (null where the selector's stand), and the moment's status (always
    adjudicated here) and selection time (on the annotations row) are dropped.
    """
    return {
        "moment_id": record["moment_id"],
        "source": record["source"],
        "transcript_id": record["transcript_id"],
        "split": RELEASE_SPLITS[record["split"]],
        **{f: record[f] for f in BOUNDARY_FIELDS},
        "boundaries_source": record["boundaries_source"],
        "original_boundaries": record.get("original_boundaries"),
        "situation_type": record["situation_type"],
        "situation_votes": record["situation_votes"],
        "labels": record["labels"],
        "action_ties": record["action_ties"],
        "n_votes": record["n_votes"],
        "annotators": [
            {"annotator": annotator, "role": role}
            for annotator, role in zip(
                record["annotators"], record["annotator_roles"], strict=True
            )
        ],
    }


def as_export_row(row: dict) -> dict:
    """A released annotations row in the annotation tool's export shape.

    The release names each annotator by label (``annotator``) where the export
    has ``annotator_id``, and records a transcript with no key moments as
    ``moment: null`` with one entry where the export has ``no_key_moments_record``.
    """
    entries = [
        {
            "annotator_id": e["annotator"],
            **{k: v for k, v in e.items() if k != "annotator"},
        }
        for e in row["annotations"]
    ]
    shaped = {k: v for k, v in row.items() if k not in ("moment", "annotations")}
    if row["moment"] is None:
        (shaped["no_key_moments_record"],) = entries
    else:
        shaped["moment"] = row["moment"]
        shaped["annotations"] = entries
    return shaped


def labels_files(
    annotations: list[Path], labels: Path, no_staff: Path, out_dir: Path
) -> tuple[dict[str, Path], Counter]:
    """Build the release label files into ``out_dir``: {release name: path}."""
    out, dropped = gt.build(
        [str(p) for p in annotations], SPLITS, str(labels), str(no_staff)
    )
    files = {
        RELEASE_SPLITS[stem]: Path(
            gt.write_split(
                str(out_dir),
                RELEASE_SPLITS[stem],
                [release_record(r) for r in records],
            )
        )
        for stem, records in out.items()
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

    # The released annotations, reshaped for build_ground_truth. Their annotators
    # are labels already, so the label map they need is labels -> themselves.
    released = []
    for provider in PROVIDERS:
        key = ctx.build_prefix + ANNOTATIONS_DIR + f"{provider}.jsonl"
        if not ctx.store.exists(key):
            raise SystemExit(
                f"{ANNOTATIONS_DIR}{provider}.jsonl is not in the build; the benchmark "
                "labels are checked against it, so write_annotations must run first"
            )
        path = released_dir / f"{provider}_annotations.jsonl"
        with open(path, "w", encoding="utf-8") as out:
            for line in ctx.store.iter_lines(key):
                out.write(json.dumps(as_export_row(json.loads(line))) + "\n")
        released.append(path)
    own_labels = work / "release_labels.json"
    own_labels.write_text(
        json.dumps({label: label for label in json.loads(labels.read_text()).values()})
    )
    rebuilt, _ = labels_files(released, own_labels, no_staff, work / "from_release")

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
