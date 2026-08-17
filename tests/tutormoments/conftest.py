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
