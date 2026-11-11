"""Annotations writer: the annotation tool's export -> ``annotations/<provider>.jsonl``.

All rows are synthetic, shaped like the nightly export. The build runs against a local
store, so every output also passes the packaged annotations schema and the name check.
"""

import copy
import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("jsonschema")

from tutormoments_build.release import build as rb
from tutormoments_build.release.annotations import write_annotations
from tutormoments_build.release.snapshot import create_snapshot
from tutormoments_build.release.store import LocalStore

CLEAN = {"commit": "abc1234def5678", "dirty": False}
NOW = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)
U = "00000000-0000-5000-8000-0000000000{:02d}".format
SHA = "ab" * 32

# Raw annotator ids -> release labels.
SELECTOR, REANNOTATOR, ADJUDICATOR = U(10), U(11), U(12)
LABELS = {SELECTOR: "A01", REANNOTATOR: "A02", ADJUDICATOR: "A03"}

INPUTS = (
    "v2_annotations/source/tutoring_provider_a_annotations.jsonl",
    "v2_annotations/source/tutoring_provider_c_annotations.jsonl",
    "v2_annotations/annotator_labels.json",
)


def sar():
    return {
        "situation": {
            "scaffolding_appropriate": True,
            "rigor_appropriate": False,
            "why": "Stuck.",
        },
        "action": {
            "scaffolding_present": True,
            "scaffolding_amount": "appropriate",
            "scaffolding_strategies": ["hint"],
            "scaffolding_strategy_other": "",
            "over_scaffolding_reasons": [],
            "rigor_present": False,
            "rigor_amount": None,
            "rigor_strategies": [],
            "rigor_strategy_other": "",
            "explanation": "Gives a starting point.",
        },
        "result": {
            "problem_success": "meaningful_success",
            "cognitive_engagement": "high_or_good",
            "explanation": "Solves it.",
        },
    }


def raw_entry(
    annotator_id,
    role,
    payload,
    revision=1,
    created_at="2026-10-01T12:00:00Z",
    provider="a",
):
    entry = {
        "annotator_id": annotator_id,
        "annotator_name": "Pat",
        "role": role,
        "revision": revision,
        "created_at": created_at,
        "interface_version": "2.34.5",
        "schema_version": "2.7",
        "payload": payload,
    }
    if provider == "a":
        entry["selection_source_sha256"] = SHA
        entry["transcript_selection"] = {
            "rows": [{"turn_number": 4, "enrichment_numbers": [2], "is_cut": True}]
        }
        entry["enrichment_numbers"] = [2]
    return entry


def selector(revision=1, **kw):
    return raw_entry(
        SELECTOR,
        "selector",
        {**sar(), "other_observations": ""},
        revision=revision,
        **kw,
    )


def reannotator(**kw):
    meta = {
        "throw_out": False,
        "throw_out_reason": "",
        "changed_boundaries": True,
        "new_start_turn": 3,
        "new_end_turn": 6,
        "new_dialogue_turns": 4,
        "redrew_cut_point": False,
        "new_cut_turn": None,
    }
    return raw_entry(
        REANNOTATOR,
        "reannotator",
        {**sar(), "other_observations": "", "meta": meta},
        **kw,
    )


def adjudicator(**kw):
    payload = {
        "final": {**sar(), "other_observations": ""},
        "decisions": dict.fromkeys(
            ["situation", "action", "result", "boundaries", "other_observations"],
            "agreed",
        ),
        "final_start_turn": 3,
        "final_cut_turn": 4,
        "final_end_turn": 6,
        "final_dialogue_turns": 4,
        "rationale": "Both agree.",
        "reverted_throw_out": False,
        "meta": {
            "throw_out": False,
            "throw_out_category": None,
            "throw_out_reason": "",
        },
    }
    return raw_entry(ADJUDICATOR, "adjudicator", payload, **kw)


def moment(status="adjudicated", moment_id=U(2)):
    return {
        "moment_id": moment_id,
        "start_turn": 2,
        "cut_turn": 4,
        "end_turn": 6,
        "dialogue_turns": 5,
        "status": status,
        "created_by": SELECTOR,
        "created_by_name": "Pat",
        "created_at": "2026-10-01T11:00:00Z",
        "in_v1_rigor_region": None,
    }


def raw_a(annotations=None, **row):
    return {
        "transcript_id": U(1),
        "v1_rigor_hinted": False,
        "v1_rigor_regions": [],
        "moment": moment(),
        "annotations": annotations
        if annotations is not None
        else [selector(), reannotator(), adjudicator()],
        **row,
    }


def raw_c(annotations=None, **row):
    return {
        "transcript_id": U(3),
        "source": "tutoring_provider_c",
        "corpus_revision": SHA,
        "v1_rigor_hinted": False,
        "v1_rigor_regions": [],
        "moment": moment(moment_id=U(4)),
        "annotations": annotations
        if annotations is not None
        else [
            selector(provider="c"),
            reannotator(provider="c"),
            adjudicator(provider="c"),
        ],
        **row,
    }


def no_key_a():
    record = raw_entry(
        SELECTOR, "selector", {"no_key_moments": True, "note": "Reading lesson."}
    )
    for key in (
        "selection_source_sha256",
        "transcript_selection",
        "enrichment_numbers",
    ):
        record.pop(key)
    return {
        "transcript_id": U(5),
        "v1_rigor_hinted": False,
        "v1_rigor_regions": [],
        "no_key_moments_record": record,
    }


def jsonl(*rows):
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def store_with(tmp_path, a_rows, c_rows, labels=LABELS):
    s = LocalStore(tmp_path)
    s.write_bytes(INPUTS[0], jsonl(*a_rows))
    s.write_bytes(INPUTS[1], jsonl(*c_rows))
    s.write_bytes(INPUTS[2], json.dumps(labels).encode())
    return s


def build(store):
    create_snapshot(store, snapshot_id="s1", inputs=INPUTS)
    return rb.run_build(store, "s1", writers=(write_annotations,), code=CLEAN, now=NOW)


def released(store, manifest, provider):
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    lines = store.read_bytes(f"{prefix}annotations/{provider}.jsonl").splitlines()
    return [json.loads(line) for line in lines]


def test_moment_row_ships_labels_and_answers_without_dropped_fields(tmp_path):
    store = store_with(tmp_path, [raw_a()], [raw_c()])
    manifest = build(store)
    [row] = released(store, manifest, "tutoring_provider_a")

    assert row["source"] == "tutoring_provider_a"
    assert (row["purpose"], row["batch"]) == ("rigor_scaffolding", "v2")
    assert row["moment"] == {
        "moment_id": U(2),
        "start_turn": 2,
        "cut_turn": 4,
        "end_turn": 6,
        "status": "adjudicated",
        "selected_by": "A01",
        "created_at": "2026-10-01T11:00:00Z",
        "in_v1_rigor_region": None,
    }
    assert [(e["annotator"], e["role"]) for e in row["annotations"]] == [
        ("A01", "selector"),
        ("A02", "reannotator"),
        ("A03", "adjudicator"),
    ]
    selector_entry = row["annotations"][0]
    assert set(selector_entry) == {
        "annotator",
        "role",
        "created_at",
        "interface_version",
        "schema_version",
        "enrichment_numbers",
        "payload",
    }
    assert selector_entry["enrichment_numbers"] == [2]
    assert selector_entry["payload"] == {**sar(), "other_observations": ""}
    assert "new_dialogue_turns" not in row["annotations"][1]["payload"]["meta"]
    assert row["annotations"][1]["payload"]["meta"]["new_start_turn"] == 3
    assert "final_dialogue_turns" not in row["annotations"][2]["payload"]

    text = json.dumps(
        released(store, manifest, "tutoring_provider_a")
        + released(store, manifest, "tutoring_provider_c")
    )
    for raw in (*LABELS, "Pat", SHA):
        assert raw not in text
    assert manifest["files"]["annotations/tutoring_provider_a.jsonl"]["schemas"] == [
        "annotations.schema.json"
    ]


def test_c_rows_carry_no_provider_a_fields(tmp_path):
    store = store_with(tmp_path, [raw_a()], [raw_c()])
    [row] = released(store, build(store), "tutoring_provider_c")
    assert row["source"] == "tutoring_provider_c"
    assert all("enrichment_numbers" not in e for e in row["annotations"])


def test_retracted_moments_are_dropped(tmp_path):
    retracted = raw_a(
        moment=moment(status="retracted", moment_id=U(6)), annotations=[selector()]
    )
    store = store_with(tmp_path, [retracted, raw_a()], [raw_c()])
    manifest = build(store)
    assert [
        r["moment"]["moment_id"]
        for r in released(store, manifest, "tutoring_provider_a")
    ] == [U(2)]
    assert manifest["files"]["annotations/tutoring_provider_a.jsonl"]["rows"] == 1


def test_only_the_last_save_of_a_role_is_kept(tmp_path):
    later = selector(revision=2, created_at="2026-10-02T12:00:00Z")
    later["payload"]["other_observations"] = "Second thoughts."
    store = store_with(
        tmp_path,
        [
            raw_a(
                annotations=[later, selector(revision=1), reannotator(), adjudicator()]
            )
        ],
        [raw_c()],
    )
    [row] = released(store, build(store), "tutoring_provider_a")
    selectors = [e for e in row["annotations"] if e["role"] == "selector"]
    assert [e["payload"]["other_observations"] for e in selectors] == [
        "Second thoughts."
    ]
    assert len(row["annotations"]) == 3


def test_two_adjudicators_are_both_kept(tmp_path):
    second = adjudicator()
    second["annotator_id"] = U(13)
    labels = {**LABELS, U(13): "A04"}
    store = store_with(
        tmp_path,
        [raw_a(annotations=[selector(), reannotator(), adjudicator(), second])],
        [raw_c()],
        labels,
    )
    [row] = released(store, build(store), "tutoring_provider_a")
    assert [
        e["annotator"] for e in row["annotations"] if e["role"] == "adjudicator"
    ] == ["A03", "A04"]


def test_no_key_moments_record_becomes_a_row_with_no_moment(tmp_path):
    store = store_with(tmp_path, [no_key_a()], [raw_c()])
    [row] = released(store, build(store), "tutoring_provider_a")
    assert row["moment"] is None
    assert row["annotations"] == [
        {
            "annotator": "A01",
            "role": "selector",
            "created_at": "2026-10-01T12:00:00Z",
            "interface_version": "2.34.5",
            "schema_version": "2.7",
            "payload": {"no_key_moments": True, "note": "Reading lesson."},
        }
    ]


def test_unmapped_annotator_fails_the_build(tmp_path):
    stranger = reannotator()
    stranger["annotator_id"] = U(99)
    store = store_with(
        tmp_path, [raw_a(annotations=[selector(), stranger, adjudicator()])], [raw_c()]
    )
    with pytest.raises(SystemExit, match=U(99)):
        build(store)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda r: r.update(new_flag=True), id="row field"),
        pytest.param(lambda r: r["moment"].update(priority=1), id="moment field"),
        pytest.param(
            lambda r: r["annotations"][0].update(device="mac"), id="entry field"
        ),
    ],
)
def test_unknown_upstream_field_fails_instead_of_shipping_or_dropping(tmp_path, change):
    row = copy.deepcopy(raw_a())
    change(row)
    store = store_with(tmp_path, [row], [raw_c()])
    with pytest.raises(SystemExit, match="unknown"):
        build(store)


def test_unknown_moment_status_fails(tmp_path):
    store = store_with(tmp_path, [raw_a(moment=moment(status="pending"))], [raw_c()])
    with pytest.raises(SystemExit, match="failed its checks"):
        build(store)


def test_build_ships_the_annotation_guide(tmp_path):
    store = store_with(tmp_path, [raw_a()], [raw_c()])
    manifest = build(store)
    prefix = rb.build_prefix("release-v2", manifest["build_id"])
    assert "docs/annotation_guide.md" in manifest["files"]
    assert store.read_bytes(prefix + "docs/annotation_guide.md").startswith(
        b"# Guiding Sheet"
    )
    assert rb.verify_build(store, manifest["build_id"]) == []
