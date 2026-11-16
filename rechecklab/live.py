"""Question-only live Ollama execution: an initial call followed by zero or one recheck.

The public gate can be demonstrated with its named solver under an explicitly
experimental collector protocol. Its released prompts and checkpoint digest
are unavailable, so this live use is not the evaluated replay protocol.
Custom gates require a complete, matching collector protocol and checkpoint.
There are no answer keys or correctness-based overrides in this engine.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from .collector import (
    INITIAL_INSTRUCTION, RECHECK_INSTRUCTION, actual_tokens,
    ollama_chat, parse_answer, response_text,
)
from .data import COLLECTION_SETTINGS, collection_protocol
from .features import INITIAL_REQUIRED, validate_initial_record
from .predictor import ArtifactLoadError, decide_recheck, load_artifact

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 120
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "outputs" / "v0.2" / "models" / "source_gate.joblib"
DEFAULT_SETTINGS = {
    "protocol_version": "0.2", "budget": 768, "seed": 42, "context_window": 4096,
    "temperature": 0.0, "thinking": False,
    "initial_instruction": INITIAL_INSTRUCTION,
    "recheck_instruction": RECHECK_INSTRUCTION,
}


def load_live_artifact(path=DEFAULT_ARTIFACT, *, expected_protocol=None, expected_sha256=None):
    """Freshly deserialize the production bundle; never cache a class instance.

    Streamlit can evict imported modules while retaining resource-cache objects.
    Loading from disk resolves OutcomePredictor through its current canonical
    import, preserving strict isinstance and all fitted/schema checks.
    """
    return load_artifact(path, expected_protocol=expected_protocol, expected_sha256=expected_sha256)


def protocol_fingerprint(host, model, model_digest, settings):
    """Scope a UI run identity to the actual endpoint, solver, and call settings."""
    payload = dict(host=host.rstrip("/"), model=model, model_digest=model_digest,
                   generation_settings=settings)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _host(host):
    if not isinstance(host, str):
        raise ValueError("The Ollama endpoint must be an HTTP URL.")
    parsed = urlparse(host)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("The Ollama endpoint must be an HTTP URL without embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("The Ollama endpoint cannot contain a query or fragment.")
    return host.rstrip("/")


def _timeout(value):
    if isinstance(value, bool):
        raise ValueError("Backend timeout must be between 1 and 120 seconds.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Backend timeout must be between 1 and 120 seconds.") from exc
    if not math.isfinite(value) or not 1 <= value <= DEFAULT_TIMEOUT:
        raise ValueError("Backend timeout must be between 1 and 120 seconds.")
    return value


def _live_settings(protocol):
    """Use saved numeric settings, but only the collector's implemented prompts."""
    saved = protocol.get("generation_settings")
    settings = dict(DEFAULT_SETTINGS)
    mismatches = []
    if isinstance(saved, dict):
        for key in ("budget", "seed", "temperature", "thinking", "context_window"):
            if key in saved:
                settings[key] = saved[key]
        if saved.get("protocol_version") in ("0.1", "0.2"):
            settings["protocol_version"] = saved["protocol_version"]
        # This validates the actual settings before any inference call.
        _, settings = collection_protocol(settings)
        for key in (*COLLECTION_SETTINGS, "context_window"):
            if key not in saved:
                mismatches.append(f"Saved generation setting {key} is missing.")
            elif saved[key] != settings[key]:
                mismatches.append(f"Saved {key} differs from the live collector protocol.")
        for key in ("initial_seed", "recheck_seed"):
            actual = settings["seed"] if key == "initial_seed" else settings["seed"] + 1
            if key in saved and saved[key] != actual:
                mismatches.append(f"Saved {key} differs from the live collector protocol.")
        # Collector protocols v0.1/v0.2 explicitly execute no finalizer call.
        # The absence of this extra field has that documented version meaning.
        for key in ("finalizer", "finalizer_used"):
            if key in saved and saved[key] is not False:
                mismatches.append("The saved protocol uses a finalizer; live collection does not.")
    else:
        mismatches.append("The released gate does not contain complete generation prompts and settings.")
    protocol_id, settings = collection_protocol(settings)
    if protocol.get("recheck_protocol") != protocol_id:
        mismatches.append("The live collector protocol fingerprint differs from the saved recheck protocol.")
    if protocol.get("action") != "recheck":
        mismatches.append("The saved action is not the collector's explicit recheck action.")
    if not protocol.get("model_digest"):
        mismatches.append("The saved gate does not identify a checkpoint digest.")
    published = protocol.get("published_generation_settings")
    if isinstance(published, dict):
        mismatches.append(f"Released context-window settings are unavailable; live explicitly uses {settings['context_window']} tokens.")
        if published.get("base_generation_limit") != settings["budget"]:
            mismatches.append(f"Released initial generation limit {published.get('base_generation_limit')} differs from live limit {settings['budget']}.")
        if published.get("finalizer_max_tokens"):
            mismatches.append(f"Released calls may use a {published['finalizer_max_tokens']}-token finalizer; live calls execute no finalizer.")
    actual = {**settings, "initial_seed": settings["seed"], "recheck_seed": settings["seed"] + 1,
              "finalizer_used": False, "recheck_protocol": protocol_id}
    return actual, mismatches


def probe_backend(bundle, host=DEFAULT_HOST, timeout=5):
    """Return actual server/model availability and fail closed on solver changes."""
    status = dict(ready=False, status="unavailable", message="", model=None,
                  model_digest=None, experimental=False, protocol_match=False,
                  generation_settings=None, mismatches=[], protocol_fingerprint=None,
                  installed_models=[], live_protocol=None, error_stage=None)
    try:
        # The loader and probe must use the same active class even when only
        # the predictor module was evicted and this live module was retained.
        importlib.import_module("rechecklab.predictor").validate_artifact(bundle)
    except (TypeError, ValueError, KeyError) as exc:
        status.update(status="predictor_load_error", error_stage="predictor",
                      message=f"Saved predictor validation failed: {exc}")
        return status
    if "live_study_protocol" in bundle["protocol"]:
        return _probe_study_backend(bundle, host, timeout, status)
    try:
        host = _host(host)
        timeout = _timeout(timeout)
        protocol = bundle["protocol"]
        status["model"] = protocol["model"]
        settings, mismatches = _live_settings(protocol)
        status["generation_settings"] = settings
        status["mismatches"] = mismatches
    except (TypeError, ValueError) as exc:
        status.update(status="invalid_configuration", error_stage="configuration", message=str(exc))
        return status
    try:
        with urlopen(host + "/api/tags", timeout=timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise ValueError("Ollama did not return a valid installed-model list.")
        installed = {}
        for entry in payload["models"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
                raise ValueError("Ollama returned an invalid model identity.")
            digest = entry.get("digest")
            if digest is not None and (not isinstance(digest, str) or not digest.strip()):
                raise ValueError("Ollama returned an invalid checkpoint digest.")
            if entry["name"] in installed and installed[entry["name"]] != digest:
                raise ValueError("Ollama returned conflicting checkpoint identities.")
            installed[entry["name"]] = digest
        status["installed_models"] = sorted(installed)
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        status.update(status="unavailable", error_stage="backend", message=f"Ollama is unavailable at {host}: {exc}")
        return status
    except (ValueError, TypeError) as exc:
        status.update(status="invalid_backend_response", error_stage="backend", message=str(exc))
        return status
    model = protocol["model"]
    if model not in installed:
        status.update(status="model_missing", message=f"The saved gate requires {model}. Install that exact model with: ollama pull {model}")
        return status
    digest = installed[model]
    status["model_digest"] = digest
    if not digest:
        status.update(status="checkpoint_unknown", message="Ollama did not identify the installed model's checkpoint digest. Live prediction is blocked.")
        return status
    expected_digest = protocol.get("model_digest")
    if expected_digest is not None and digest != expected_digest:
        status.update(status="checkpoint_mismatch", message="The installed checkpoint digest differs from the saved gate. Live prediction is blocked.")
        return status
    status["protocol_fingerprint"] = protocol_fingerprint(host, model, digest, settings)
    status["live_protocol"] = dict(model=model, model_digest=digest,
                                   recheck_protocol=settings["recheck_protocol"], action="recheck",
                                   generation_settings=collection_protocol(settings)[1],
                                   initial_seed=settings["initial_seed"], recheck_seed=settings["recheck_seed"],
                                   finalizer_used=False)
    is_public = protocol["recheck_protocol"].startswith("sevra:") and "published_generation_settings" in protocol
    if mismatches:
        if not is_public:
            status.update(status="protocol_mismatch", message="The saved gate and live collector settings are incompatible. Use a matching gate before running live.")
            return status
        status.update(ready=True, status="experimental", experimental=True,
                      message=(f"Ollama is ready with {model}. Experimental live use: the public checkpoint digest and full prompts are unavailable. "
                               f"This collector uses a {settings['budget']}-token output limit, {settings['context_window']}-token context, thinking={settings['thinking']}, and no finalizer; its live behavior has not been evaluated by the replay pilot."))
        return status
    status.update(ready=True, status="compatible", protocol_match=True,
                  message=f"Ollama is ready with the saved {model} checkpoint and matching collector settings.")
    return status


def _probe_study_backend(bundle, host, timeout, status):
    """Validate a future study artifact against the exact collection protocol."""
    from .study_protocol import protocol_hash, validate_protocol, verify_backend
    contract = bundle["protocol"]
    try:
        host = _host(host)
        _timeout(timeout)
        protocol = validate_protocol(contract["live_study_protocol"])
        digest = protocol_hash(protocol)
        expected = {"model": protocol["model"], "model_digest": protocol["model_digest"],
                    "action": "recheck", "recheck_protocol": "live-study:" + digest,
                    "study_protocol_hash": digest}
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("Study artifact identity/hash disagrees with its full frozen protocol.")
        status.update(model=protocol["model"], model_digest=protocol["model_digest"],
                      generation_settings=protocol["generation"], live_protocol=contract,
                      protocol_fingerprint=protocol_fingerprint(host, protocol["model"], protocol["model_digest"], protocol))
    except (TypeError, ValueError, KeyError) as exc:
        status.update(status="protocol_mismatch", error_stage="configuration", message=str(exc))
        return status
    try:
        verify_backend(protocol, host)
    except (URLError, TimeoutError, ConnectionError, OSError) as exc:
        status.update(status="unavailable", error_stage="backend", message=f"Ollama study backend is unavailable: {exc}")
        return status
    except (ValueError, TypeError, KeyError) as exc:
        status.update(status="protocol_mismatch", error_stage="backend", message=str(exc))
        return status
    status.update(ready=True, protocol_match=True, experimental=True, status="experimental",
                  message="Ollama matches the frozen local-study checkpoint and protocol. Study predictions remain experimental; matching generation settings alone does not establish accuracy.")
    return status


def _answer(content, answer_type):
    if answer_type == "open":
        # Open questions have no trusted key and no task-specific scoring rule.
        # Use only an explicit final marker; never invent an answer from reasoning.
        marked = re.findall(r"\bFINAL(?:\s+ANSWER)?\s*:\s*([^\n]+)", content, re.I)
        return marked[-1].strip().strip(" *`") if marked else ""
    if answer_type not in ("number", "boolean", "choice"):
        raise ValueError("answer_type must be open, number, boolean, or choice.")
    return parse_answer(content, answer_type) or ""


def _call_output(response, model, question, answer_type):
    if not isinstance(response, dict) or response.get("done") is not True:
        raise ValueError("Ollama returned an incomplete response.")
    if response.get("model") != model:
        raise ValueError("Ollama's actual response model differs from the saved gate.")
    message = response.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Ollama did not provide response text.")
    if "thinking" in message and not isinstance(message["thinking"], str):
        raise ValueError("Ollama returned invalid thinking text.")
    if not isinstance(response.get("done_reason"), str) or response["done_reason"].strip().casefold() not in ("stop", "length"):
        raise ValueError("Ollama did not provide a supported actual completion status (stop or length).")
    record = validate_initial_record(dict(
        query=question, base_response=response_text(response),
        base_answer=_answer(message["content"], answer_type),
        base_actual_tokens=actual_tokens(response), base_done_reason=response["done_reason"],
        base_finalizer_used=False,  # This protocol executed no finalizer call.
    ))
    output = dict(response=record["base_response"], answer=record["base_answer"],
                  tokens=int(record["base_actual_tokens"]), done_reason=record["base_done_reason"],
                  finalizer_used=False, parse_valid=bool(record["base_answer"]),
                  prompt_eval_count=int(float(response["prompt_eval_count"])),
                  eval_count=int(float(response["eval_count"])),
                  model=model, wall_seconds=response.get("wall_seconds"))
    return record, output, message["content"] or record["base_response"]


def _reported_attempt(response):
    """Retain returned text/metadata for failure audit without validating it as input."""
    if not isinstance(response, dict):
        return {"response": None, "reported_metadata": {"response_type": type(response).__name__},
                "validated": False}
    message = response.get("message")
    text = None
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        text = message["content"]
        if isinstance(message.get("thinking"), str) and message["thinking"]:
            text = message["thinking"] + "\n\n" + text
    # Repr preserves missing/invalid token metadata without emitting NaN JSON.
    metadata = {key: repr(response[key]) if key in response else "<missing>"
                for key in ("model", "done", "done_reason", "prompt_eval_count", "eval_count")}
    return {"response": text, "reported_metadata": metadata, "validated": False}


def run_live_question(question, bundle, host=DEFAULT_HOST, timeout=DEFAULT_TIMEOUT,
                      allow_experimental=False, on_progress=None, answer_type="open"):
    """Generate an initial answer, predict using initial signals, then obey the gate.

No trusted key can be passed into this function. Correctness and the actual
HELP/HURT/NO_CHANGE outcome remain unknown without separate offline scoring.
"""
    result = dict(status="blocked", initial=None, probabilities=None, decision=None,
                  recheck_attempted=False, rechecked=False, revised=None,
                  selected_answer=None, selected_response=None, selected_source=None,
                  error=None, error_stage=None, backend=None, live_protocol=None, initial_record=None,
                  initial_attempt=None, recheck_attempt=None,
                  actual_outcome="Unknown", correctness="Unknown")

    def progress(stage, message):
        if on_progress is not None:
            on_progress(stage, message)

    if not isinstance(question, str) or not question.strip():
        result.update(status="input_error", error="Enter a nonempty question.")
        return result
    try:
        timeout = _timeout(timeout)
        if answer_type not in ("open", "number", "boolean", "choice"):
            raise ValueError("answer_type must be open, number, boolean, or choice.")
    except ValueError as exc:
        result.update(status="input_error", error=str(exc))
        return result
    progress("checking_backend", "Checking the actual solver and saved gate compatibility.")
    backend = probe_backend(bundle, host=host)
    result["backend"] = backend
    result["live_protocol"] = backend.get("live_protocol")
    if not backend["ready"]:
        result["error"] = backend["message"]
        result["error_stage"] = backend.get("error_stage")
        if backend["status"] == "predictor_load_error":
            result["status"] = "predictor_load_error"
        progress("failed", result["error"])
        return result
    if backend["experimental"] and not allow_experimental:
        result["error"] = "This public gate requires explicitly identified experimental live use before inference."
        progress("failed", result["error"])
        return result
    if "live_study_protocol" in bundle["protocol"]:
        return _run_study_question(question, answer_type, bundle, backend, host, timeout, progress, result)
    settings, model = backend["generation_settings"], backend["model"]
    messages = [dict(role="user", content=INITIAL_INSTRUCTION + "\n\n" + question)]
    progress("generating_initial", "Generating the initial answer with the saved solver.")
    try:
        initial_response = ollama_chat(model, messages, host, settings["budget"], settings["initial_seed"],
                                       settings["temperature"], settings["thinking"], timeout=timeout,
                                       context_window=settings["context_window"])
        result["initial_attempt"] = _reported_attempt(initial_response)
        record, initial, previous = _call_output(initial_response, model, question, answer_type)
        result["initial_attempt"]["validated"] = True
        actual_identity = {key: backend["live_protocol"][key] for key in ("model", "recheck_protocol", "action")}
        if backend["model_digest"] is not None:
            actual_identity["model_digest"] = backend["model_digest"]
        result["initial_record"] = {**record, **actual_identity}
        result["initial"] = initial
        result.update(selected_answer=initial["answer"], selected_response=initial["response"], selected_source="initial")
    except (URLError, TimeoutError, ConnectionError, OSError, TypeError, ValueError, KeyError) as exc:
        if result["initial_attempt"] is None and hasattr(exc, "response"):
            result["initial_attempt"] = _reported_attempt(exc.response)
        result.update(status="initial_failed", error_stage="backend", error=f"Initial generation failed: {exc}")
        progress("failed", result["error"])
        return result
    progress("predicting", "Estimating HELP, NO_CHANGE, and HURT from the initial response only.")
    try:
        # Backend identity was checked above. A public experimental run keeps
        # its actual protocol separate instead of relabeling it as the replay.
        if backend["protocol_match"]:
            frame = pd.DataFrame([result["initial_record"]])
        else:
            frame = pd.DataFrame([{**{key: record[key] for key in INITIAL_REQUIRED}, "model": model}])
        probabilities = bundle["predictor"].probabilities(frame)
        recheck = bool(decide_recheck(probabilities, bundle["threshold"], bundle["harm_weight"])[0])
        result["probabilities"] = probabilities[0].tolist()
        result["decision"] = "Recheck" if recheck else "Skip recheck"
    except (TypeError, ValueError, KeyError) as exc:
        result.update(status="prediction_failed", error_stage="predictor", error=f"Prediction failed; the initial answer is retained: {exc}")
        progress("failed", result["error"])
        return result
    if not recheck:
        result.update(status="complete", recheck_tokens=0, total_tokens=initial["tokens"])
        progress("complete", "The gate skipped the recheck and retained the initial answer. Correctness is unknown.")
        return result
    result["recheck_attempted"] = True
    progress("rechecking", "The gate chose one recheck; checking the fixed initial answer.")
    try:
        revision_response = ollama_chat(model, messages + [dict(role="assistant", content=previous),
                                       dict(role="user", content=RECHECK_INSTRUCTION)], host,
                                       settings["budget"], settings["recheck_seed"], settings["temperature"],
                                       settings["thinking"], timeout=timeout, context_window=settings["context_window"])
        result["recheck_attempt"] = _reported_attempt(revision_response)
        _, revised, _ = _call_output(revision_response, model, question, answer_type)
        result["recheck_attempt"]["validated"] = True
        result.update(status="complete", revised=revised, rechecked=True,
                      selected_answer=revised["answer"], selected_response=revised["response"], selected_source="revised",
                      recheck_tokens=revised["tokens"], total_tokens=initial["tokens"] + revised["tokens"])
        progress("complete", "One recheck completed. Its correctness and outcome require a trusted answer key.")
    except (URLError, TimeoutError, ConnectionError, OSError, TypeError, ValueError, KeyError) as exc:
        if result["recheck_attempt"] is None and hasattr(exc, "response"):
            result["recheck_attempt"] = _reported_attempt(exc.response)
        result.update(status="recheck_failed", selected_source="initial_fallback",
                      error_stage="backend",
                      recheck_tokens=None, total_tokens=None,
                      error=f"The gate chose a recheck, but it failed; the initial answer is retained: {exc}")
        progress("failed", result["error"])
    return result


def _run_study_question(question, answer_type, bundle, backend, host, timeout, progress, result):
    """Use exactly the same call, parser and feature record as offline collection."""
    from .study_protocol import execute_call
    protocol = bundle["protocol"]["live_study_protocol"]
    if answer_type == "boolean":
        result.update(status="input_error", error="This local-study protocol supports numeric, choice or open questions.")
        return result
    started = time.perf_counter()
    progress("generating_initial", "Generating under the frozen local-study protocol.")
    try:
        raw, record, initial = execute_call(protocol, question, answer_type, "initial", host=host, timeout=timeout)
        result["initial_attempt"] = {**_reported_attempt(raw), "validated": True}
        identity = {key: bundle["protocol"][key] for key in ("model", "model_digest", "action", "recheck_protocol", "study_protocol_hash")}
        result.update(initial=initial, initial_record={**record, **identity}, selected_answer=initial["answer"],
                      selected_response=initial["response"], selected_source="initial")
    except (URLError, TimeoutError, ConnectionError, OSError, TypeError, ValueError, KeyError) as exc:
        result.update(status="initial_failed", error_stage="backend", error=f"Study initial generation failed: {exc}")
        if hasattr(exc, "response"):
            result["initial_attempt"] = _reported_attempt(exc.response)
        progress("failed", result["error"])
        return result
    progress("predicting", "Estimating recheck effects from initial signals only.")
    prediction_started = time.perf_counter()
    try:
        probabilities = bundle["predictor"].probabilities(pd.DataFrame([result["initial_record"]]))
        recheck = bool(decide_recheck(probabilities, bundle["threshold"], bundle["harm_weight"])[0])
        result.update(probabilities=probabilities[0].tolist(), decision="Recheck" if recheck else "Skip recheck",
                      predictor_wall_seconds=time.perf_counter() - prediction_started)
    except (TypeError, ValueError, KeyError) as exc:
        result.update(status="prediction_failed", error_stage="predictor", error=f"Study prediction failed; initial answer retained: {exc}")
        progress("failed", result["error"])
        return result
    if not recheck:
        result.update(status="complete", recheck_tokens=0, total_tokens=initial["tokens"],
                      live_generation_prediction_wall_seconds=time.perf_counter() - started)
        progress("complete", "Skipped recheck; correctness remains unknown.")
        return result
    result["recheck_attempted"] = True
    progress("rechecking", "Rechecking the same question and full saved initial response.")
    try:
        raw, _, revised = execute_call(protocol, question, answer_type, "recheck", initial["response"], host=host, timeout=timeout)
        result["recheck_attempt"] = {**_reported_attempt(raw), "validated": True}
        result.update(status="complete", revised=revised, rechecked=True, selected_answer=revised["answer"],
                      selected_response=revised["response"], selected_source="revised", recheck_tokens=revised["tokens"],
                      total_tokens=initial["tokens"] + revised["tokens"],
                      live_generation_prediction_wall_seconds=time.perf_counter() - started)
        progress("complete", "One protocol-matched recheck completed; actual correctness requires a separate trusted key.")
    except (URLError, TimeoutError, ConnectionError, OSError, TypeError, ValueError, KeyError) as exc:
        result.update(status="recheck_failed", selected_source="initial_fallback", error_stage="backend",
                      error=f"Study recheck failed; initial answer retained: {exc}",
                      live_generation_prediction_wall_seconds=time.perf_counter() - started)
        if hasattr(exc, "response"):
            result["recheck_attempt"] = _reported_attempt(exc.response)
        progress("failed", result["error"])
    return result
