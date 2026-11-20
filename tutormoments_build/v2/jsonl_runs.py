"""Resumable, write-as-you-go JSONL runs keyed by ``moment_id``.

Shared by the persona and replay scripts: each record is appended as soon as
it is made, so an interrupted run loses at most the moments in flight, and a
rerun skips moments already in the file. A moment whose work raises is logged
and left out, so the next run retries it.

``overwrite`` regenerates the given moments into ``<path>.partial``, carrying
over records for moments not given, and replaces the file only once every
record is in -- a failure leaves the old file untouched.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("tutormoments_build.v2.jsonl_runs")


def load_by_id(path: str) -> dict[str, dict]:
    """``{moment_id: record}``; empty when the file does not exist."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                out[record["moment_id"]] = record
    return out


def pending(moments, path: str, overwrite: bool = False) -> list:
    """The moments a run would generate: all with ``overwrite``, else the missing."""
    existing = load_by_id(path)
    return [m for m in moments if overwrite or m.moment_id not in existing]


def run(moments, path: str, work, *, overwrite=False, concurrency=4) -> list[dict]:
    """Apply ``work(moment) -> record`` to each pending moment, writing as it goes.

    Returns the records written by this call.
    """
    existing = load_by_id(path)
    todo = pending(moments, path, overwrite)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if overwrite:
        redo = {m.moment_id for m in todo}
        write_path = path + ".partial"
        with open(write_path, "w", encoding="utf-8") as fh:
            for moment_id, record in existing.items():
                if moment_id not in redo:
                    fh.write(json.dumps(record) + "\n")
    else:
        write_path = path

    lock = threading.Lock()
    written: list[dict] = []
    failed: list[str] = []
    with open(write_path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(work, m): m for m in todo}
            for future in as_completed(futures):
                moment_id = futures[future].moment_id
                try:
                    record = future.result()
                except Exception as error:
                    failed.append(moment_id)
                    logger.error("%s failed: %s", moment_id, error)
                    continue
                with lock:
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()
                    written.append(record)
                logger.info("wrote %s", moment_id)

    if failed:
        logger.error("%d moment(s) failed; rerun to retry them", len(failed))
        if overwrite:
            raise RuntimeError(
                f"{len(failed)} moment(s) failed under --overwrite; {path} left "
                f"unchanged, new records in {write_path}"
            )
    if overwrite:
        os.replace(write_path, path)
    return written
