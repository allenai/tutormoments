"""Read packaged resource files (prompts) from the tutormoments_build package.

Mirrors tutormoments.resources but anchored at files("tutormoments_build") — the
runtime helper is hard-wired to the tutormoments package and cannot read
build-only prompts.
"""

from importlib.resources import files


def resource_text(relative_path: str) -> str:
    """Return the text of a resource file inside the tutormoments_build package."""
    return (files("tutormoments_build") / relative_path).read_text(encoding="utf-8")


def resource_path(relative_path: str):
    """Return the traversable for a resource *directory* inside the package.

    Needed where a caller has to enumerate what is there rather than read one
    known file -- listing the available prompt versions, say. Returns a
    Traversable, not a str: a packaged resource need not exist on the
    filesystem.
    """
    return files("tutormoments_build") / relative_path
