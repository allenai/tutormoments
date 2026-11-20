"""Runs that write one JSON line per moment as they go, so they can resume.

Used by the persona and replay scripts. Each record is written as soon as it's
made, so an interrupted run loses only the moments in progress. A rerun skips
moments already in the file. A moment that raises is logged and left out, so
the next run retries it.

With `overwrite`, new records go to `<path>.partial` and replace the file only
once all of them succeed. If any fail, the old file is left as it was.
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
    """The moments a run would do: all with `overwrite`, else only the missing ones."""
    existing = load_by_id(path)
    return [m for m in moments if overwrite or m.moment_id not in existing]


def run(moments, path: str, work, *, overwrite=False, concurrency=4) -> list[dict]:
    """Run `work(moment) -> record` for each pending moment, writing as it goes.

    Returns the records this call wrote.
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
