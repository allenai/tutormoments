"""Read packaged resource files (prompts) from the tutormoments_build package.

Mirrors tutormoments.resources but anchored at files("tutormoments_build") — the
runtime helper is hard-wired to the tutormoments package and cannot read
build-only prompts.
"""

from importlib.resources import files


def resource_text(relative_path: str) -> str:
    """Return the text of a resource file inside the tutormoments_build package."""
    return (files("tutormoments_build") / relative_path).read_text(encoding="utf-8")
