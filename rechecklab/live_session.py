"""Session-scoped live call deduplication, including overlapping Streamlit reruns."""
from __future__ import annotations

import hashlib
import json
from threading import Lock


def live_input_key(question, model_signature, host, backend):
    payload = dict(question=question.strip(), artifact=model_signature,
                   host=host.rstrip("/"), protocol=backend.get("protocol_fingerprint"),
                   model=backend.get("model"), digest=backend.get("model_digest"),
                   settings=backend.get("generation_settings"))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def execute_once(state, key, operation):
    """Cache completed or failed jobs; a busy session never starts another call.

    Input-change callbacks clear the displayed result, not these jobs or lock.
    A failed attempt is retained too: retrying a button must not repeat an
    initial call whose recheck may already have started on the server.
    """
    lock = state.setdefault("_live_lock", Lock())
    if not lock.acquire(blocking=False):
        return {"status": "busy"}
    try:
        jobs = state.setdefault("_live_jobs", {})
        if key in jobs:
            return jobs[key]
        # A Streamlit stop/rerun can interrupt a script after the server has
        # received its request. Keep a tombstone rather than risk replaying it.
        jobs[key] = {"status": "busy"}
        try:
            job = {"status": "complete", "result": operation()}
        except Exception as exc:
            job = {"status": "failed", "error": f"Live execution failed: {exc}"}
        jobs[key] = job
        # Retain every request in this browser session, including failed calls.
        # Eviction could make returning to a previous question call the LLM twice.
        return job
    finally:
        lock.release()
