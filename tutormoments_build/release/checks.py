"""Checks every build must pass before it can be published.

1. Every data file names at least one schema, and every row validates against
   each of them (strict provider schemas and shared core contracts alike).
2. No row carries a person's name. The check is list-free: it never needs real
   names to compare against. Any key ending in ``name`` fails unless it is a
   known non-person field, so a new ``*_name`` field fails closed until someone
   reviews it.
"""

import json
import re
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator, FormatChecker

from tutormoments_build.release.store import Store

NAME_KEY = re.compile(r"(^|_)name$")

# Keys ending in "name" that hold no person's name, by release path ("*" = any).
ALLOWED_NAME_KEYS = {
    "*": frozenset({"construct_name", "misconception_name", "quiz_name"}),
    "reference/eedi_subjects.jsonl": frozenset({"name"}),
}

# Stop collecting after this many problems per file; the count stays exact.
MAX_EXAMPLES = 20


@dataclass
class FileReport:
    path: str
    rows: int = 0
    errors: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.errors += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(message)


def name_keys(obj, allowed: frozenset[str], trail: str = "") -> list[str]:
    """Every key path in ``obj`` whose key looks like a name and is not allowed."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{trail}.{key}" if trail else key
            if NAME_KEY.search(key) and key not in allowed:
                found.append(here)
            found.extend(name_keys(value, allowed, here))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(name_keys(item, allowed, f"{trail}[]"))
    return found


def allowed_name_keys(path: str) -> frozenset[str]:
    return ALLOWED_NAME_KEYS["*"] | ALLOWED_NAME_KEYS.get(path, frozenset())


def check_file(
    store: Store, key: str, path: str, validators: list[Draft202012Validator]
) -> FileReport:
    """Validate one JSONL file (``key`` in the store, ``path`` in the release)."""
    report = FileReport(path)
    allowed = allowed_name_keys(path)
    for line_no, line in enumerate(store.iter_lines(key), start=1):
        if not line.strip():
            report.add(f"line {line_no}: blank line")
            continue
        report.rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            report.add(f"line {line_no}: not JSON ({exc.msg})")
            continue
        for validator in validators:
            error = next(validator.iter_errors(row), None)
            if error is not None:
                where = "/".join(str(p) for p in error.absolute_path) or "<row>"
                report.add(f"line {line_no}: {where}: {error.message[:200]}")
        for hit in name_keys(row, allowed):
            report.add(f"line {line_no}: name-like key {hit}")
    return report


def load_validator(schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())
