"""Every prompt file ships in the installed package.

Prompts are loaded from package resources at call time, so a file the
package-data globs miss works in an editable checkout and is absent from a
wheel. Setuptools globs do not cross directories: ``prompts/v2/*.md`` matches
nothing under ``prompts/v2/1/``.
"""

import tomllib
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = {
    "tutormoments": ROOT / "src" / "tutormoments",
    "tutormoments_build": ROOT / "tutormoments_build",
}


def _package_data() -> dict[str, list[str]]:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["tool"]["setuptools"]["package-data"]


def _matches(relative: PurePosixPath, pattern: str) -> bool:
    """Setuptools-style match: ``*`` stays within one path segment."""
    parts = PurePosixPath(pattern).parts
    return len(parts) == len(relative.parts) and all(
        fnmatchcase(name, glob) for name, glob in zip(relative.parts, parts)
    )


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_every_prompt_file_is_package_data(package):
    root = PACKAGES[package]
    patterns = _package_data()[package]
    prompts = [
        PurePosixPath(path.relative_to(root).as_posix())
        for path in (root / "prompts").rglob("*")
        if path.is_file() and path.suffix in {".md", ".txt"}
    ]

    assert prompts
    missing = [str(p) for p in prompts if not any(_matches(p, g) for g in patterns)]
    assert not missing, f"not in {package} package-data: {missing}"
