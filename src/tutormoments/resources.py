"""Access packaged tutormoments runtime assets."""

from importlib.resources import files


def resource_text(relative_path: str) -> str:
    """Read a UTF-8 text resource from the tutormoments package."""
    return (files("tutormoments") / relative_path).read_text(encoding="utf-8")
