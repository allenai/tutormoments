"""Shared fixtures for the runtime test suite."""

import pytest

from tutormoments.client import _reset_client_cache


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """Isolate the shared ModelClient cache between tests.

    resolve_tutor/resolve_student memoize one client per model id so
    conversations reuse a warm connection pool. Without clearing it, a client
    built under one test's patched SDK constructor would leak into the next
    test and be asserted against the wrong mock.
    """
    _reset_client_cache()
    yield
    _reset_client_cache()


@pytest.fixture(autouse=True)
def _no_human_reference_download(monkeypatch):
    """Keep `report` / `view` offline: the human KL reference downloads from
    the Hub, so it fails here and the row is omitted. Tests that exercise it
    pass their own `download` (or patch `human_reference`)."""
    from tutormoments import taxonomy

    def _offline(*_a, **_kw):
        raise RuntimeError("network disabled in the offline suite")

    monkeypatch.setattr(taxonomy, "_hf_download", _offline)
