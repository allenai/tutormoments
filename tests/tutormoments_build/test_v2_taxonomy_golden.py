"""Snapshot tests for the human-round taxonomy (`taxonomy.py`).

For a fixed set of excerpt records, these pin what the human round sends and
writes at every prompt version: batch requests, counts, parsed records and
reports. They make sure changes made for replay taxonomy can't silently change
the human round.

The latest prompt version is stored in full so a diff is readable; older
versions are stored as hashes. To regenerate after an intended change:

    python tests/tutormoments_build/test_v2_taxonomy_golden.py
"""

import copy
import hashlib
import json
import os

from tutormoments_build.v2 import taxonomy as T
from tutormoments_build.v2.classify_excerpts import V2Spec, available_prompt_versions

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "v2_taxonomy_golden.json")

CFG = V2Spec(
    model="vendor/golden-model",
    provider="anthropic",
    thinking={"thinking": {"type": "adaptive"}, "effort": "high"},
    poll_interval=60,
)


def _record(moment_id, split, scaffolding, rigor, situation, post_cut_rows=2):
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
        "labels": {"scaffolding_present": scaffolding, "rigor_present": rigor},
        "situation_type": situation,
    }


RECORDS = {
    "iteration": [
        _record("aaa", "iteration", True, False, "scaffold_only"),
        _record("bbb", "iteration", False, True, "rigor_maj"),
        _record("ccc", "iteration", True, True, "fifty_fifty"),
        _record("fff", "iteration", False, False, "rigor_only"),
    ],
    "test": [
        _record("ddd", "test", True, False, "scaffold_maj", post_cut_rows=0),
        _record("eee", "test", False, True, "rigor_only"),
    ],
}

RAW = {
    "taxonomy__aaa": {
        "text": '[{"evidence": "3", "category": "hint"}, {"evidence": "4", '
        '"category": "made_up"}]',
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    },
    "taxonomy__bbb": {"text": "not json", "usage": {"total_tokens": 3}},
    "taxonomy__ccc": {"text": "", "error": "overloaded"},
    "taxonomy__eee": {
        "text": '```json\n[{"evidence": "5", "category": "define_term"}]\n```'
    },
}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def snapshot_version(version: str) -> dict:
    """Everything the human round produces for RECORDS at one prompt version."""
    prompt = T.load_prompt(version)
    categories = T.parse_categories(prompt.template)
    flat = [r for records in RECORDS.values() for r in copy.deepcopy(records)]
    out = {}
    for width in ("20", "full"):
        entries, counts = T.build_entries(flat, prompt.template, width)
        records_out = {
            stem: [
                T.build_record(r, RAW, CFG, prompt, categories, width)
                for r in copy.deepcopy(split_records)
            ]
            for stem, split_records in RECORDS.items()
        }
        for records in records_out.values():
            for record in records:
                if record["taxonomy"] and record["taxonomy"]["parse_error"]:
                    counts["unparsed"] += 1
        out[width] = {
            "entries": entries,
            "counts": dict(sorted(counts.items())),
            "records": records_out,
            "report": T.report(
                records_out,
                counts,
                dry_run=False,
                cfg=CFG,
                out_dir="out",
                width=width,
                prompt=prompt,
            ),
            "dry_run_report": T.report(
                {},
                counts,
                dry_run=True,
                cfg=CFG,
                out_dir="out",
                width=width,
                prompt=prompt,
            ),
        }
    return out


def snapshot() -> dict:
    versions = available_prompt_versions(T.PROMPT_ROOT)
    latest = T.default_prompt_version()
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
    assert T.default_prompt_version() == golden["latest_version"], (
        "a new prompt version was added; regenerate the golden file"
    )
    fresh = json.loads(json.dumps(snapshot_version(golden["latest_version"])))
    assert fresh.keys() == golden["latest"].keys()
    for width, expected in golden["latest"].items():
        for part, value in expected.items():
            assert fresh[width][part] == value, f"@{width}: {part} changed"


def test_every_prompt_version_output_is_unchanged():
    golden = _load()
    for version, expected in golden["hashes"].items():
        assert _hash(snapshot_version(version)) == expected, f"v{version} changed"


if __name__ == "__main__":
    with open(GOLDEN, "w", encoding="utf-8") as fh:
        json.dump(snapshot(), fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {GOLDEN}")
