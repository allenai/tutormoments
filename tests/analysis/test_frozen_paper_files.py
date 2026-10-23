"""The June 2026 paper's frozen numbers must never change.

`kl_divergence_table.tex` (locked to the paper's KL values) and
`v1_action_taxonomy_distribution.csv` (the Fig 4 distribution) are the paper's
canonical record. The classifications behind them are lost and their pools
can't be rebuilt from the release, so nothing can regenerate them: any edit
is a silent change to published results. Like `balanced_520_ids.json`, they
are committed once and pinned here. Runs report their own KL (`tutormoments
report`); this guard only protects the paper's.
"""

import hashlib
from pathlib import Path

import pytest

PAPER_DIR = Path(__file__).resolve().parents[2] / "analysis" / "working-paper-20260630"

FROZEN_SHA256 = {
    "kl_divergence_table.tex": (
        "7933733e9a50766bd3db70c09d9c9e97635b1c34e05cc92ee1181d10204e0bfb"
    ),
    "v1_action_taxonomy_distribution.csv": (
        "f28f0fdff26eac94bc5bbd651bad70449ac9640f758a96dde144dedabb88871a"
    ),
}


@pytest.mark.parametrize("name", sorted(FROZEN_SHA256))
def test_paper_file_is_unchanged(name):
    # Hash LF-normalized bytes so a CRLF (autocrlf) checkout doesn't fail.
    data = (PAPER_DIR / name).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA256[name], (
        f"{name} is a frozen paper artifact and must not be edited or regenerated"
    )
