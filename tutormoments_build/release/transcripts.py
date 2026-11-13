"""Write ``transcripts/<provider>.jsonl`` from the frozen bucket transcripts.

Rows ship as the bucket has them, except for the field drops declared in
``PROVIDERS``. Each file is checked against its provider's release schema and
the shared ``transcript_core.schema.json``.

The release schemas are reviewed copies of the bucket's: public-facing text,
same validation rules apart from the declared changes. Every build re-derives
that: it applies the declared changes to the frozen bucket schema and requires
the rules to match the packaged release schema exactly. An upstream rule change
therefore fails the build until someone reviews it, rather than drifting.
"""

import copy
import json
import logging
from dataclasses import dataclass
from typing import Callable

from tutormoments_build.release.build import BuildContext, Output, packaged_schemas

logger = logging.getLogger("tutormoments_build.release.transcripts")

CORE_SCHEMA = "transcript_core.schema.json"

# Schema keys that carry human-facing text rather than validation rules.
TEXT_KEYS = frozenset({"$id", "$comment", "title", "description", "examples"})


def _drop_a_fields(row: dict) -> dict:
    """Provider A: drop the batch-specific pair count and the constant link type."""
    row["session"].pop("prior_session_count", None)
    for link in row["eedi_content_link"]:
        kind = link.pop("type")
        if kind != "quiz_session":
            raise SystemExit(
                f"{row['transcript_id']}: eedi_content_link type {kind!r}; the release "
                "drops this field on the assumption it is always 'quiz_session'"
            )
    return row


def _a_schema_changes(rules: dict) -> None:
    del rules["$defs"]["Session"]["properties"]["prior_session_count"]
    link = rules["$defs"]["EediContentLink"]
    del link["properties"]["type"]
    link["required"].remove("type")


def _c_schema_changes(rules: dict) -> None:
    # The partner delivered once and will not deliver again.
    rules["properties"]["batch"]["const"] = "2026-07-06"


@dataclass(frozen=True)
class Provider:
    transform: Callable[[dict], dict] | None
    schema_changes: Callable[[dict], None]


# Provider B adds an entry here, with its own release schema.
PROVIDERS = {
    "tutoring_provider_a": Provider(_drop_a_fields, _a_schema_changes),
    "tutoring_provider_c": Provider(None, _c_schema_changes),
}


def schema_rules(schema):
    """A schema minus its human-facing text: what validation actually uses."""
    if isinstance(schema, dict):
        return {k: schema_rules(v) for k, v in schema.items() if k not in TEXT_KEYS}
    if isinstance(schema, list):
        return [schema_rules(item) for item in schema]
    return schema


def check_declared_changes(
    bucket_schema: dict,
    release_schema: dict,
    schema_changes: Callable[[dict], None],
    label: str,
    module: str,
) -> None:
    """Fail unless bucket rules + declared changes == release rules."""
    expected = copy.deepcopy(schema_rules(bucket_schema))
    schema_changes(expected)
    if expected != schema_rules(release_schema):
        raise SystemExit(
            f"{label}: the bucket schema's rules no longer match the reviewed release "
            "schema. Review the upstream change, then update the release schema "
            f"(or the declared changes in {module})."
        )


def check_schema_drift(
    bucket_schema: dict, release_schema: dict, provider: str
) -> None:
    """Fail unless the provider's bucket rules + declared changes == release rules."""
    check_declared_changes(
        bucket_schema,
        release_schema,
        PROVIDERS[provider].schema_changes,
        provider,
        "transcripts.py",
    )


def write_transcripts(ctx: BuildContext) -> list[Output]:
    release_schemas = packaged_schemas()
    outputs = []
    for provider, spec in PROVIDERS.items():
        schema_name = f"{provider}.schema.json"
        bucket_schema = json.loads(
            ctx.store.read_bytes(ctx.snapshot_key(f"schemas/{schema_name}"))
        )
        check_schema_drift(
            bucket_schema, json.loads(release_schemas[schema_name]), provider
        )

        source = f"source/{provider}.jsonl"
        path = f"transcripts/{provider}.jsonl"
        if spec.transform is None:
            ctx.copy_from_snapshot(source, path)
        else:
            local = ctx.workdir / f"{provider}.jsonl"
            with open(local, "w", encoding="utf-8") as out:
                for line in ctx.store.iter_lines(ctx.snapshot_key(source)):
                    row = spec.transform(json.loads(line))
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            ctx.write(path, local)
            local.unlink()
        logger.info("wrote %s", path)
        outputs.append(Output(path, (schema_name, CORE_SCHEMA)))
    return outputs
