"""Snapshot tests for the human-round classification (`classify_excerpts`).

For a fixed set of excerpt records, these pin everything the human round sends
and writes at every prompt version: batch requests, counts, parsed records,
output paths and reports. They make sure changes made for replay classification
can't silently change the human round.

The latest prompt version is stored in full so a diff is readable; older
versions are stored as hashes. To regenerate after an intended change:

    python tests/tutormoments_build/test_v2_classify_excerpts_golden.py
"""

import hashlib
import json
import os

from tutormoments_build.v2 import classify_excerpts as C

GOLDEN = os.path.join(
    os.path.dirname(__file__), "fixtures", "v2_classify_excerpts_golden.json"
)

CFG = C.V2Spec(
    model="vendor/golden-model",
    provider="gemini",
    thinking={"thinking_level": "high", "include_thoughts": True},
    poll_interval=60,
)


def _record(moment_id, split, labels, situation, post_cut_rows=2, facets=("f1",)):
    return {
        "moment_id": moment_id,
        "transcript_id": f"t-{moment_id}",
        "conversation_id": f"c-{moment_id}",
        "split": split,
        "post_cut_rows": post_cut_rows,
        "post_cut_dialogue_rows": 1,
        "excerpts": {
            width: {"excerpt": f"{width} excerpt of {moment_id} {{not a slot}}"}
            for width in ("full", "20")
        },
        "labels": labels,
        "situation_type": situation,
        C.DESCRIPTION_KEY: {
            "moment_id": moment_id,
            "adjudicators": ["J01", "J02"],
            "situation_method": "majority",
            "model": "m",
            "prompt": "p",
            "facets": list(facets),
        },
    }


def _labels(scaffolding, rigor, over=False):
    return {
        "scaffolding_present": scaffolding,
        "rigor_present": rigor,
        "over_scaffolding_present": over,
    }


RECORDS = {
    "iteration": [
        _record("aaa", "iteration", _labels(True, False, True), "scaffold_only"),
        _record("bbb", "iteration", _labels(False, True), "rigor_maj"),
        _record("ccc", "iteration", _labels(True, True), "fifty_fifty", facets=()),
    ],
    "test": [
        _record("ddd", "test", _labels(True, False), "scaffold_maj", post_cut_rows=0),
        _record("eee", "test", _labels(True, False), "rigor_only"),
    ],
}

RAW = {
    "action__aaa": {
        "text": json.dumps({"description": "a", "scaffolding": "yes", "rigor": "no"}),
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    },
    "overscaffold__aaa": {
        "text": json.dumps({"description": "o", "over-scaffolding": "yes"}),
        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    },
    "action__bbb": {"text": "not json", "usage": {"total_tokens": 3}},
    "action__ccc": {"text": "", "error": "overloaded"},
    "overscaffold__ccc": {"text": '{"over_scaffolding": "no"}'},
    "action__eee": {"text": '[{"scaffolding": "no", "rigor": "yes"}]'},
}

PASS_SETS = (C.PASSES, ("over_scaffolding",))


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def snapshot_version(version: str) -> dict:
    """Everything the human round produces for RECORDS at one prompt version."""
    import copy

    prompts = C.load_prompt_set(version)
    flat = [r for records in RECORDS.values() for r in records]
    out = {}
    for passes in PASS_SETS:
        for width in ("20", "full"):
            name = f"{'+'.join(passes)}@{width}"
            records = copy.deepcopy(RECORDS)
            entries, counts = C.build_entries(
                copy.deepcopy(flat), prompts, passes, width
            )
            records_out = {
                stem: [
                    C.build_record(r, RAW, CFG, prompts, passes, width)
                    for r in split_records
                ]
                for stem, split_records in records.items()
            }
            out[name] = {
                "entries": entries,
                "counts": dict(sorted(counts.items())),
                "records": records_out,
                "report": C.report(
                    records_out,
                    counts,
                    dry_run=False,
                    cfg=CFG,
                    prompts=prompts,
                    out_dir="out",
                    passes=passes,
                    width=width,
                ),
                "dry_run_report": C.report(
                    {},
                    counts,
                    dry_run=True,
                    cfg=CFG,
                    prompts=prompts,
                    out_dir="out",
                    passes=passes,
                    width=width,
                ),
                "prediction_dir": C.prediction_dir("out", CFG.model, version, width),
            }
    return out


def snapshot() -> dict:
    versions = C.available_prompt_versions()
    latest = C.latest_prompt_version()
    return {
        "latest_version": latest,
        "latest": snapshot_version(latest),
        "hashes": {v: _hash(snapshot_version(v)) for v in versions},
    }


def _load():
    with open(GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


def test_latest_prompt_version_output_is_unchanged():
    golden = _load()
    assert C.latest_prompt_version() == golden["latest_version"], (
        "a new prompt version was added; regenerate the golden file"
    )
    # Round-trip through JSON so tuples compare as the stored lists.
    fresh = json.loads(json.dumps(snapshot_version(golden["latest_version"])))
    for name, expected in golden["latest"].items():
        for part, value in expected.items():
            assert fresh[name][part] == value, f"{name}: {part} changed"
    assert fresh.keys() == golden["latest"].keys()


def test_every_prompt_version_output_is_unchanged():
    golden = _load()
    for version, expected in golden["hashes"].items():
        assert _hash(snapshot_version(version)) == expected, f"v{version} changed"


if __name__ == "__main__":
    with open(GOLDEN, "w", encoding="utf-8") as fh:
        json.dump(snapshot(), fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN}")
