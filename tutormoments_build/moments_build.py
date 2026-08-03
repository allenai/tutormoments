"""Build the released moments set from ground truth + transcripts.

Maintainer-only dataset construction: this module hydrates key-moment ids
into frozen, self-contained Moment records and writes the release artifacts
(moments.jsonl + moments.manifest.json). The cut-point logic here (modal vote, role
adjustment) is benchmark-defining — it runs once at build time and its output
is published; the runtime never re-runs it.
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from tutormoments.moments import (
    DATASET_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    MOMENTS_FILENAME,
    Moment,
    _build_reference_transcript,
    _read_moments_jsonl,
    file_sha256,
    records_content_hash,
    validate_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The authoritative JSON Schema for released moment records, shipped with the
# release next to moments.jsonl (same `{thing}.{kind}` convention).
SCHEMA_FILENAME = "moments.schema.json"

_SCAFF_AGGS = {"scaffolding", "rigor"}

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_CONV_ID_PREFIX_RE = re.compile(r"^\d{4}-t\d+_\d{4}-s\d+_")


# ---------------------------------------------------------------------------
# Local-only data loaders
# Read from explicit directories; no S3.
# ---------------------------------------------------------------------------


def _conv_id_to_uuid(conv_id: str) -> str:
    """Extract the transcript-UUID component from a full conv_id.

    Accepts bare UUIDs, legacy composites, and bench composites.
    """
    matches = _UUID_RE.findall(conv_id)
    if matches:
        return matches[-1]
    stripped = _CONV_ID_PREFIX_RE.sub("", conv_id)
    if stripped != conv_id:
        return stripped
    return conv_id


def _load_ground_truth_files(ground_truth_dir: str) -> list[dict]:
    """Load all ground-truth JSON files from a local directory, sorted by filename."""
    gt_dir = Path(ground_truth_dir)
    files = []
    for fpath in sorted(gt_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if data is not None:
            files.append(data)
    return files


def _load_transcripts(transcripts_dir: str) -> dict[str, dict]:
    """Load all transcript JSON files from a local directory, sorted by filename.

    Returns {conv_id: conversation_dict}.
    """
    tx_dir = Path(transcripts_dir)
    transcripts = {}
    for fpath in sorted(tx_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        if data and "conversation_id" in data:
            transcripts[data["conversation_id"]] = data
        elif data:
            transcripts[fpath.stem] = data
    return transcripts


# ---------------------------------------------------------------------------
# JSONL transcript loader
# ---------------------------------------------------------------------------


def _transform_normalized_record(rec: dict) -> dict:
    """Transform an S3 normalized JSONL record to internal transcript format.

    S3 format: turns[] (dialogue only) + enrichments[] (non-dialogue, with before_turn).
    Enrichments are inserted before their associated dialogue turn with
    is_enrichment=True so they appear in excerpts as context but don't shift
    turn numbering.

    Omitted: _annotate_turns_with_start_seconds (applied separately if needed).
    """
    from collections import defaultdict as _defaultdict

    sess = rec.get("session", {})
    source_id = rec.get("transcript_id", "") or rec.get("source_id", "")
    tutor_id = sess.get("tutor_id", "") or sess.get("source_tutor_id", "")
    student_id = sess.get("student_id", "") or sess.get("source_student_id", "")

    # source_id format varies by batch:
    #   older batches: UUID only (e.g. "69b80b21-...")
    #   newer batches: full conv_id (e.g. "2025-t27247_2025-s12069_69b80b21-...")
    if tutor_id and tutor_id in source_id:
        conv_id = source_id  # already includes tutor_student prefix
    else:
        conv_id = f"{tutor_id}_{student_id}_{source_id}"

    # Index dialogue turns by their original turn_number
    raw_turns = rec.get("turns", [])
    max_dialogue_turn = max((t["turn_number"] for t in raw_turns), default=0)

    # Group enrichments by the dialogue turn they precede
    enrichments_by_turn = _defaultdict(list)
    trailing_enrichments = []
    for e in rec.get("enrichments", []):
        etype = e.get("type", "").upper().replace(" ", "_")
        label = e.get("label", "")
        content = e.get("content", "")
        text = f"[{etype}]"
        if label:
            text = f"[{etype}: {label}]"
        if content:
            text = f"{text} {content}"
        ss = float(e.get("start_seconds", 0) or 0)
        entry = {
            "role": "TUTOR",
            "text": text,
            "type": etype,
            "timestamp": f"{ss}s",
            "start_seconds": ss,
            "is_enrichment": True,
        }
        before = e.get("before_turn") or 1  # None/0 -> before turn 1
        if before <= max_dialogue_turn:
            enrichments_by_turn[before].append(entry)
        else:
            trailing_enrichments.append(entry)

    # Build final list: enrichments before each dialogue turn, preserving original numbering
    final_turns = []
    for t in raw_turns:
        turn_num = t["turn_number"]
        for e_entry in enrichments_by_turn.get(turn_num, []):
            final_turns.append({**e_entry, "turn_number": turn_num})
        ss = float(t.get("start_seconds", 0) or 0)
        final_turns.append(
            {
                "turn_number": turn_num,
                "role": t["role"].upper(),
                "text": t["text"],
                "type": "DIALOGUE",
                "timestamp": f"{ss}s",
                "start_seconds": ss,
                "is_enrichment": False,
            }
        )
    for e_entry in trailing_enrichments:
        final_turns.append({**e_entry, "turn_number": max_dialogue_turn})

    # Build context from demographics if available
    demo = rec.get("demographics", {})
    student = demo.get("student", {})
    context_parts = []
    if student.get("grade"):
        context_parts.append(f"Grade {student['grade']}")
    if student.get("subject"):
        context_parts.append(student["subject"])
    context = ", ".join(context_parts)

    return {
        "conversation_id": conv_id,
        "transcript_id": source_id,
        "tutor_id": tutor_id,
        "student_id": student_id,
        "context": context,
        "platform": rec.get("source", "tutoring_provider_a"),
        "num_turns": len(raw_turns),  # dialogue turns only
        "turns": final_turns,
    }


def _load_jsonl_index(path: str) -> dict[str, dict]:
    """Load a local JSONL file, transform each record, and index by conv_id.

    Local-only variant: no S3/backend, no caching across calls.
    Returns {conv_id: conversation_dict}.
    """
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        logger.warning("JSONL file not found: %s", path)
        return {}

    index: dict[str, dict] = {}
    errors = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                transformed = _transform_normalized_record(rec)
                conv_id = transformed["conversation_id"]
                index[conv_id] = transformed
            except Exception:
                errors += 1

    logger.info(
        "Indexed %d transcripts from %s (%d parse errors)", len(index), path, errors
    )
    return index


# ---------------------------------------------------------------------------
# Cut/extraction helpers
# ---------------------------------------------------------------------------


def _pick_modal_cut(votes: list[int]) -> "int | None":
    """Return the most-voted cut. On tie, return the smallest.

    Returns None when votes is empty.
    """
    if not votes:
        return None
    counts = Counter(votes)
    max_count = max(counts.values())
    winners = [c for c, n in counts.items() if n == max_count]
    return min(winners)


def _role_adjust_cut(cut_turn: int, conversation: dict) -> "int | None":
    """Adjust cut_turn based on the role of the turn at cut_turn.

    - STUDENT: cut stays as-is (prefix includes the student turn).
    - TUTOR: cut_turn -= 1 (prefix excludes the human tutor turn; AI replaces it).
    - turn not found OR adjustment falls below 1: returns None (caller drops the cluster).
    """
    turns_by_n = {t["turn_number"]: t for t in conversation.get("turns", [])}
    turn = turns_by_n.get(cut_turn)
    if turn is None:
        return None
    if turn.get("role") == "TUTOR":
        adjusted = cut_turn - 1
        if adjusted < 1:
            return None
        return adjusted
    return cut_turn


def _pick_representative_member(members: list[dict], chosen_cut: int) -> dict:
    """Return the member to use for the moment's detection payload.

    Preference: members whose own cut_turn equals the chosen (modal) cut,
    smallest annotator_id (lexicographic) on tie. Falls back to smallest
    annotator_id overall if none voted the chosen cut.
    """
    matching = [m for m in members if m.get("cut_turn") == chosen_cut]
    pool = matching if matching else members
    return min(pool, key=lambda m: m.get("annotator_id") or "")


def _resolve_cluster(
    members: list[dict], conversation: dict, turn_start: int, turn_end: int
) -> "dict | None":
    """Return resolved cluster data, or None if the cluster should be dropped.

    Applies vote filtering (cut in [ts, te]), modal selection, role adjustment,
    and representative-member pick.
    """
    votes: list[int] = []
    for m in members:
        cut = m.get("cut_turn")
        if not isinstance(cut, int):
            continue
        if cut < turn_start or cut > turn_end:
            continue
        votes.append(cut)
    chosen = _pick_modal_cut(votes)
    if chosen is None:
        return None
    adjusted = _role_adjust_cut(chosen, conversation)
    if adjusted is None:
        return None
    rep = _pick_representative_member(members, chosen)
    return {
        "cut_turn": adjusted,
        "chosen_cut_turn": chosen,
        "cut_votes": dict(Counter(votes)),
        "cluster_size": len(members),
        "representative": rep,
    }


def _format_prefix(conversation: dict, cut_turn: int) -> str:
    """Format transcript turns up to and including cut_turn."""
    lines = []
    for turn in conversation["turns"]:
        if turn["turn_number"] > cut_turn:
            break
        n = turn["turn_number"]
        role = turn["role"]
        text = turn["text"]
        lines.append(f"Turn {n}. {role}: {text}")
    return "\n".join(lines)


def _get_student_context(conversation: dict) -> str:
    """Extract student context from conversation metadata."""
    context = conversation.get("context", "")
    if context:
        return context
    parts = []
    if conversation.get("platform"):
        parts.append(f"Platform: {conversation['platform']}")
    return "; ".join(parts) if parts else "K-12 tutoring session"


def _last_student_msg(conversation: dict, cut_turn: int) -> str:
    """Find the last STUDENT message at or before cut_turn."""
    last = ""
    for turn in conversation["turns"]:
        if turn["turn_number"] > cut_turn:
            break
        if turn["role"] == "STUDENT":
            last = turn["text"]
    return last


def _prefix_turns_to_context(conversation: dict, cut_turn: int) -> list[dict]:
    """Convert prefix turns into the Moment context format: [{turn_number, role, text}].

    turn_number is preserved verbatim from the source turn (real, non-sequential).
    Role is lowercased (TUTOR -> tutor, STUDENT -> student).
    Only includes turns up to and including cut_turn.
    """
    result = []
    for turn in conversation["turns"]:
        if turn["turn_number"] > cut_turn:
            break
        result.append(
            {
                "turn_number": turn["turn_number"],
                "role": turn["role"].lower(),
                "text": turn["text"],
            }
        )
    return result


# ---------------------------------------------------------------------------
# Public API: build_moments
# ---------------------------------------------------------------------------


def build_moments(
    *,
    set_name: str,
    ids: list[str],
    ground_truth_dir: str,
    transcripts_dir: str,
    tutoring_provider_a_jsonl: "str | None" = None,
) -> "list[Moment]":
    """Hydrate a list of moment ids into Moment objects.

    Each id must be a string of the form '{conv_id}__hum_{ts}_{te}'.
    Moments are returned in the order given by `ids`.

    Args:
        set_name: Prefix used for Moment.id, e.g. "balanced_520".
        ids: Ordered list of moment id strings to hydrate.
        ground_truth_dir: Path to directory of per-conversation GT JSON files.
        transcripts_dir: Path to directory of per-conversation transcript JSON files.
        tutoring_provider_a_jsonl: Optional path to a normalized JSONL transcript file (e.g.
            tutoring_provider_a.jsonl). When provided, JSONL-loaded convs are merged
            into the transcript pool alongside per-file *.json files. JSONL entries
            take lower precedence: per-file JSON wins on conv_id collision.

    Returns:
        List of Moment objects in the same order as `ids`. IDs with no
        matching ground-truth cluster are silently skipped.
    """
    # Load data
    gt_files = _load_ground_truth_files(ground_truth_dir)
    transcripts = _load_transcripts(transcripts_dir)

    # Merge JSONL transcripts (lower precedence: per-file JSON wins on collision)
    if tutoring_provider_a_jsonl is not None:
        jsonl_index = _load_jsonl_index(tutoring_provider_a_jsonl)
        for conv_id, conv in jsonl_index.items():
            if conv_id not in transcripts:
                transcripts[conv_id] = conv

    # Build UUID->conv_id lookup
    uuid_to_conv = {_conv_id_to_uuid(cid): cid for cid in transcripts}

    # First pass: build cluster map {(conv_id, ts, te): [members]}
    clusters: dict[tuple, list[dict]] = {}
    for gt in gt_files:
        gt_uuid = gt.get("conversation_id")
        full_conv_id = uuid_to_conv.get(_conv_id_to_uuid(gt_uuid or ""))
        if not full_conv_id:
            continue
        for m in gt.get("key_moments", []):
            if m.get("situation_label_agg") not in _SCAFF_AGGS:
                continue
            ts = m.get("turn_start")
            te = m.get("turn_end")
            if ts is None or te is None:
                continue
            key = (full_conv_id, ts, te)
            clusters.setdefault(key, []).append(m)

    # Resolve every cluster once (resolve_cluster may return None -> dropped)
    resolved: dict[tuple, dict] = {}
    for (conv_id, ts, te), members in clusters.items():
        if conv_id not in transcripts:
            continue
        conversation = transcripts[conv_id]
        r = _resolve_cluster(members, conversation, ts, te)
        if r is not None:
            resolved[(conv_id, ts, te)] = r

    # Parse requested ids and build an index for O(1) lookup
    # id format: "{conv_id}__hum_{ts}_{te}"
    _HUM_RE = re.compile(r"^(.+)__hum_(\d+)_(\d+)$")

    moments_by_key: dict[tuple, Moment] = {}
    for moment_id_str in ids:
        m = _HUM_RE.match(moment_id_str)
        if not m:
            logger.warning("Unrecognized moment id format: %s", moment_id_str)
            continue
        conv_id, ts_s, te_s = m.group(1), int(m.group(2)), int(m.group(3))
        key = (conv_id, ts_s, te_s)

        if key in moments_by_key:
            continue  # already built (duplicate in ids list)

        r = resolved.get(key)
        if r is None:
            logger.warning("No resolved cluster for moment id: %s", moment_id_str)
            continue

        if conv_id not in transcripts:
            logger.warning("Transcript not found for conv_id: %s", conv_id)
            continue

        conversation = transcripts[conv_id]
        cut_turn = r["cut_turn"]
        rep = r["representative"]

        context_turns = _prefix_turns_to_context(conversation, cut_turn)
        if not context_turns:
            logger.warning("Empty prefix for %s, skipping", moment_id_str)
            continue

        reference = _build_reference_transcript(conversation, cut_turn)
        student_ctx = _get_student_context(conversation)
        dimension = rep.get("situation_label_agg")

        moment = Moment(
            id=f"{set_name}:{moment_id_str}",
            context=context_turns,
            dimension=dimension,
            student={
                "mode": "oracle",
                "reference": reference,
                "context": student_ctx,
            },
            rubric={
                "gold": dimension,
                "hint": rep.get("situation", ""),
            },
            provenance={
                "conv_id": conv_id,
                "cut_turn": cut_turn,
                "turn_start": ts_s,
                "turn_end": te_s,
                "moment_id": rep.get("moment_id"),
                "annotator_id": rep.get("annotator_id"),
                "chosen_cut_turn": r["chosen_cut_turn"],
                "cut_votes": r["cut_votes"],
                "cluster_size": r["cluster_size"],
            },
        )
        moments_by_key[key] = moment

    # Return in the requested order (skipping unknown ids)
    result = []
    seen_keys = set()
    for moment_id_str in ids:
        m = _HUM_RE.match(moment_id_str)
        if not m:
            continue
        key = (m.group(1), int(m.group(2)), int(m.group(3)))
        if key in moments_by_key and key not in seen_keys:
            result.append(moments_by_key[key])
            seen_keys.add(key)
    return result


# ---------------------------------------------------------------------------
# Public API: build_moments_from_reference_run
# ---------------------------------------------------------------------------


def build_moments_from_reference_run(
    *,
    set_name: str,
    reference_run_jsonl: str,
    transcripts_dir: str,
    tutoring_provider_a_jsonl: "str | None" = None,
) -> "list[Moment]":
    """Reconstruct the frozen moments set from a published benchmark run.

    Each run record embeds the benchmark-time detection payload (labels, cut
    votes, chosen cut) per scenario. When later ground-truth rebuilds drift
    (lost cut votes, flipped aggregate labels), the run file is the only
    surviving record of the selection — this rebuilds exactly what ran.

    Args:
        set_name: Prefix used for Moment.id, e.g. "balanced_520".
        reference_run_jsonl: Path to the published run JSONL (one row per
            model x prompt x moment replay; deduped by scenario_id here).
        transcripts_dir: Path to directory of per-conversation transcript JSON files.
        tutoring_provider_a_jsonl: Optional normalized JSONL transcript file, merged
            into the pool at lower precedence (per-file JSON wins).

    Returns:
        List of Moment objects sorted by scenario_id. Scenarios whose
        transcript is missing are skipped with a warning.
    """
    transcripts = _load_transcripts(transcripts_dir)
    if tutoring_provider_a_jsonl is not None:
        jsonl_index = _load_jsonl_index(tutoring_provider_a_jsonl)
        for conv_id, conv in jsonl_index.items():
            if conv_id not in transcripts:
                transcripts[conv_id] = conv

    # One record per scenario_id (detection payload is identical across cells)
    records: dict[str, dict] = {}
    with open(reference_run_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.setdefault(rec["scenario_id"], rec)

    moments = []
    for scenario_id in sorted(records):
        rec = records[scenario_id]
        conv_id = rec["conv_id"]
        cut_turn = rec["cut_turn"]
        det = rec["detection"]

        conversation = transcripts.get(conv_id)
        if conversation is None:
            logger.warning("Transcript not found for conv_id: %s", conv_id)
            continue

        context_turns = _prefix_turns_to_context(conversation, cut_turn)
        if not context_turns:
            logger.warning("Empty prefix for %s, skipping", scenario_id)
            continue

        # The frozen persona the paper's simulated student ran with. Required:
        # a release without it is a different (non-reproducible) student
        # population, so refuse loudly rather than skip.
        trait = rec.get("student_trait")
        if not trait or not trait.get("persona"):
            raise ValueError(
                f"Run record for {scenario_id} carries no student_trait; "
                "the reference run must embed the frozen personas "
                "(re-pull the updated benchmark run file)."
            )

        dimension = det["situation_label_agg"]
        moments.append(
            Moment(
                id=f"{set_name}:{scenario_id}",
                context=context_turns,
                dimension=dimension,
                student={
                    "mode": "oracle",
                    "reference": _build_reference_transcript(conversation, cut_turn),
                    "context": _get_student_context(conversation),
                    "trait": {
                        "persona": trait["persona"],
                        "trait_mode": trait.get("trait_mode", "joined-3"),
                        "generator_model": trait.get("generator_model", ""),
                        "generated_at": trait.get("generated_at", ""),
                    },
                },
                rubric={
                    "gold": dimension,
                    "hint": det.get("situation", ""),
                },
                provenance={
                    "conv_id": conv_id,
                    "cut_turn": cut_turn,
                    "turn_start": det["turn_start"],
                    "turn_end": det["turn_end"],
                    "moment_id": det.get("moment_id"),
                    "annotator_id": det.get("annotator_id"),
                    "chosen_cut_turn": det.get("chosen_cut_turn"),
                    "cut_votes": det.get("cut_votes") or {},
                    "cluster_size": det.get("cluster_size"),
                },
            )
        )
    return moments


# ---------------------------------------------------------------------------
# Release serialization
# ---------------------------------------------------------------------------


def _moment_to_release_dict(moment: Moment) -> dict:
    """Convert a Moment to the released (Arrow-friendly) record form.

    provenance.cut_votes becomes a list of {cut_turn, votes} pairs sorted by
    cut_turn — Arrow cannot represent variable-key maps faithfully, and the
    runtime's Moment.from_dict normalizes it back to the internal dict.
    """
    d = moment.to_dict()
    votes = d["provenance"].get("cut_votes") or {}
    d["provenance"]["cut_votes"] = [
        {"cut_turn": int(k), "votes": v}
        for k, v in sorted(votes.items(), key=lambda kv: int(kv[0]))
    ]
    return d


def validate_records_against_schema(records: "list[dict]") -> None:
    """jsonschema-validate release-form records against the packaged schema.

    The schema is load-bearing: anything write_release publishes (and anything
    `tutormoments-build dataset validate` accepts) must conform to it. Raises
    ValueError naming the first offending record and error. `format` keywords
    (e.g. annotator_id's uuid) are annotations, not assertions — standard
    jsonschema semantics.
    """
    from importlib.resources import files

    import jsonschema

    schema = json.loads(
        (files("tutormoments_build") / SCHEMA_FILENAME).read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    for i, rec in enumerate(records):
        errors = sorted(validator.iter_errors(rec), key=str)
        if errors:
            e = errors[0]
            path = "/".join(str(p) for p in e.absolute_path) or "<record>"
            raise ValueError(
                f"Record {i} ({rec.get('id', '?')}) violates {SCHEMA_FILENAME} "
                f"at {path}: {e.message}"
            )


def write_release(
    moments: "list[Moment]",
    out_dir: str | Path,
    *,
    set_name: str,
    version: str = "0",
    created: str = "",
    provenance: dict | None = None,
) -> dict:
    """Write moments.jsonl + moments.manifest.json + moments.schema.json
    into a release directory.

    The manifest carries two hashes: content_hash (record-level, canonical
    JSON of the normalized records — comparable across the HF and local load
    paths) and file_sha256 (file-level integrity of moments.jsonl).
    """
    from importlib.resources import files

    # Release contract: every moment carries the frozen student persona.
    traitless = [
        m.id for m in moments if not (m.student.get("trait") or {}).get("persona")
    ]
    if traitless:
        raise ValueError(
            f"{len(traitless)} moment(s) have no student.trait persona "
            f"(first: {traitless[0]}); a release without frozen traits is a "
            "different student population and cannot be published."
        )

    # The shipped schema is enforced, not documentation: refuse to publish
    # any record that does not conform.
    release_dicts = [_moment_to_release_dict(m) for m in moments]
    validate_records_against_schema(release_dicts)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / MOMENTS_FILENAME
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in release_dicts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Hash the records exactly as the runtime will load them (round-trip
    # through Moment.from_dict) so both load paths verify against this value.
    loaded = _read_moments_jsonl(jsonl_path)
    content_hash = records_content_hash([m.to_dict() for m in loaded])

    manifest = {
        "name": set_name,
        "version": version,
        "schema_version": DATASET_SCHEMA_VERSION,
        "record_count": len(moments),
        "content_hash": content_hash,
        "file_sha256": file_sha256(jsonl_path),
        "provenance": provenance or {},
        "created": created,
    }
    with open(out_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Ship the record schema alongside the data (authoritative shape doc).
    schema_text = (files("tutormoments_build") / SCHEMA_FILENAME).read_text(
        encoding="utf-8"
    )
    (out_dir / SCHEMA_FILENAME).write_text(schema_text, encoding="utf-8")

    return manifest


# ---------------------------------------------------------------------------
# CLI entry points (dispatched from tutormoments_build.cli)
# ---------------------------------------------------------------------------


def _cli_build(args: argparse.Namespace) -> None:
    """Build a release directory: <out>/{moments.jsonl, moments.manifest.json, moments.schema.json}."""
    ids_path = Path(args.ids)
    with open(ids_path, encoding="utf-8") as f:
        ids = json.load(f)

    moments = build_moments(
        set_name=args.set,
        ids=ids,
        ground_truth_dir=args.ground_truth,
        transcripts_dir=args.transcripts,
        tutoring_provider_a_jsonl=getattr(args, "tutoring_provider_a_jsonl", None),
    )

    # New sets need their frozen personas generated at build time (the runtime
    # only consumes student.trait). Generator = the configured student model,
    # matching the paper's provenance (the student model authored its personas).
    from tutormoments.client import ModelClient
    from tutormoments.config import student_spec
    from tutormoments_build.traits import generate_traits_for_moments

    spec = student_spec()
    n_generated = generate_traits_for_moments(
        moments,
        model_client=ModelClient(spec["model"]),
        model_name=spec["model"],
    )
    print(f"Generated {n_generated} student traits with {spec['model']}")

    manifest = write_release(
        moments,
        args.out,
        set_name=args.set,
        version=getattr(args, "version", None) or "0",
        created=args.created,
        provenance={
            "ids_file": str(ids_path),
            "ground_truth_dir": args.ground_truth,
            "transcripts_dir": args.transcripts,
            "cut_logic": "modal vote; tie->smallest; TUTOR-role decrement",
            "source_tags": ["human_annotated", "hybrid_ground_truth"],
        },
    )
    print(
        f"Wrote {manifest['record_count']} moments to {Path(args.out) / MOMENTS_FILENAME}"
    )
    print(f"Wrote manifest to {Path(args.out) / MANIFEST_FILENAME}")
    print(f"  content_hash: {manifest['content_hash']}")


def _cli_build_from_run(args: argparse.Namespace) -> None:
    """Build a release directory from a published benchmark run."""
    import tempfile

    transcripts_dir = args.transcripts
    if transcripts_dir is None:
        # Builder requires a directory; an empty one means "JSONL pool only".
        transcripts_dir = tempfile.mkdtemp(prefix="tutormoments_empty_tx_")

    moments = build_moments_from_reference_run(
        set_name=args.set,
        reference_run_jsonl=args.reference_run,
        transcripts_dir=transcripts_dir,
        tutoring_provider_a_jsonl=args.tutoring_provider_a_jsonl,
    )

    if args.ids:
        with open(args.ids, encoding="utf-8") as f:
            expected = set(json.load(f))
        got = {m.id.split(":", 1)[1] for m in moments}
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        if missing or extra:
            raise SystemExit(
                f"Coverage mismatch vs {args.ids}: "
                f"{len(missing)} missing, {len(extra)} extra. "
                f"First missing: {missing[:3]}"
            )
        print(f"Coverage check passed: all {len(expected)} ids present")

    manifest = write_release(
        moments,
        args.out,
        set_name=args.set,
        version=getattr(args, "version", None) or "0",
        created=args.created,
        provenance={
            "reference_run": str(args.reference_run),
            "ids_file": str(args.ids) if args.ids else None,
            "cut_logic": "frozen from reference run (benchmark-time detections)",
            "source_tags": ["human_annotated", "reference_run"],
        },
    )
    print(
        f"Wrote {manifest['record_count']} moments to {Path(args.out) / MOMENTS_FILENAME}"
    )
    print(f"Wrote manifest to {Path(args.out) / MANIFEST_FILENAME}")
    print(f"  content_hash: {manifest['content_hash']}")


def _cli_validate(args: argparse.Namespace) -> None:
    """Validate a release directory's {moments.jsonl, moments.manifest.json}."""
    report = validate_dataset(args.data_path)

    # Manifest counts/hashes are necessary but not sufficient — also
    # jsonschema-validate the raw records against the packaged schema.
    with open(Path(args.data_path) / MOMENTS_FILENAME, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    validate_records_against_schema(records)

    print(f"Dataset valid: {report['name']}")
    print(f"  records: {report['record_count']} (schema-conformant)")
    print(f"  content_hash: {report['content_hash']}")
    print(f"  file_sha256 : {report['file_sha256']}")
