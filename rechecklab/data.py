"""Validate paired outcomes and keep every rollout of a question together."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .features import INITIAL_REQUIRED, validate_initial_record

LABELS = ["HELP", "NO_CHANGE", "HURT"]
REQUIRED = [*INITIAL_REQUIRED, "base_correct", "final_correct", "action_actual_tokens",
            "action", "model", "recheck_protocol"]
PUBLIC_REVISION = "10411137fcaa02b3bd549a2285b43ffe315417f6"
COLLECTION_SETTINGS = ("protocol_version", "budget", "seed", "temperature", "thinking",
                       "initial_instruction", "recheck_instruction")


def as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if str(value).strip().lower() in ("true", "1"):
        return True
    if str(value).strip().lower() in ("false", "0"):
        return False
    raise ValueError(f"Expected a boolean correctness label, got {value!r}")


def transition_label(base_correct, final_correct):
    base, final = as_bool(base_correct), as_bool(final_correct)
    return "HELP" if not base and final else "HURT" if base and not final else "NO_CHANGE"


def question_hash(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Question text must be a nonempty string.")
    canonical = " ".join(unicodedata.normalize("NFKC", query).casefold().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_text_column(frame, key):
    if key not in frame:
        raise ValueError(f"{key} is required to identify the fixed model/recheck protocol.")
    values = frame[key]
    if not values.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError(f"{key} must contain explicit nonempty strings.")


def _single_value(frame, key):
    _require_text_column(frame, key)
    values = frame[key].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Use one fixed {key} per experiment; found {len(values)} values.")
    return values[0]


def _token_counts(values, key):
    if values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"Invalid token counts in {key}: booleans are not token counts.")
    try:
        counts = pd.to_numeric(values, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid token counts in {key}.") from exc
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError(f"Invalid token counts in {key}: require finite nonnegative integers.")
    return counts


def validate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Paired outcomes must be a nonempty DataFrame.")
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    out = frame.copy()
    # This is the same initial-input validation used by training and the demo.
    initial = [validate_initial_record(row) for row in out.to_dict("records")]
    for key in INITIAL_REQUIRED:
        out[key] = [row[key] for row in initial]
    for key in ("base_correct", "final_correct"):
        out[key] = out[key].map(as_bool)
    out["label"] = [transition_label(a, b) for a, b in zip(out.base_correct, out.final_correct)]
    out["group_id"] = out["query"].map(question_hash)
    for key in ("base_actual_tokens", "action_actual_tokens"):
        out[key] = _token_counts(out[key], key)
    for key in ("action", "model", "recheck_protocol"):
        _single_value(out, key)
    if out["action"].iloc[0] not in ("recheck", "active_verify"):
        raise ValueError("Only one explicit recheck/active_verify action belongs in a paired rechecking experiment.")
    if "model_digest" in out:
        _single_value(out, "model_digest")
    if "rollout" not in out:
        out["rollout"] = 0
    out["rollout"] = _token_counts(out["rollout"], "rollout")
    if "example_id" not in out:
        out["example_id"] = out["group_id"].str[:16]
    if "dataset" not in out:
        out["dataset"] = "user_pairs"
    for key in ("example_id", "dataset"):
        if out[key].isna().any() or out[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"{key} cannot be missing or empty.")
        out[key] = out[key].astype(str)
    # All observations sharing an attempt must describe one fixed initial call.
    out["attempt_id"] = out.dataset.astype(str) + ":" + out.example_id.astype(str)
    initial_keys = ["group_id", *INITIAL_REQUIRED, "base_correct"]
    # Question case/spacing is canonicalized; initial response metadata stays exact.
    initial_keys.remove("query")
    consistency = out.groupby(["dataset", "example_id"])[initial_keys].nunique(dropna=False)
    if (consistency > 1).any().any():
        raise ValueError("A repeated attempt has inconsistent initial responses or metadata. Fix its rollout IDs.")
    return out.reset_index(drop=True)


def collection_protocol(metadata):
    """Identify actual generation settings; legacy hashes omit unspecified context."""
    missing = set(COLLECTION_SETTINGS) - set(metadata)
    if missing:
        raise ValueError("Collection metadata is missing generation settings: " + ", ".join(sorted(missing)))
    settings = {key: metadata[key] for key in COLLECTION_SETTINGS}
    for key in ("protocol_version", "initial_instruction", "recheck_instruction"):
        if not isinstance(settings[key], str) or not settings[key].strip():
            raise ValueError(f"Collection metadata requires a nonempty {key}.")
    for key in ("budget", "seed"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int):
            raise ValueError(f"Collection metadata {key} must be an integer.")
    if settings["budget"] < 1 or not isinstance(settings["thinking"], bool):
        raise ValueError("Collection metadata needs a positive budget and explicit thinking boolean.")
    try:
        temperature = float(settings["temperature"])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Collection temperature must be finite and nonnegative.") from exc
    if isinstance(settings["temperature"], bool) or not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Collection temperature must be finite and nonnegative.")
    settings["temperature"] = temperature
    if "context_window" in metadata:
        context_window = metadata["context_window"]
        if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window < 1:
            raise ValueError("Collection metadata context_window must be a positive integer.")
        settings["context_window"] = context_window
    serialized = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "ollama:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest(), settings


def _attach_collection_metadata(frame, path):
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return frame
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Collection protocol metadata must be an object.")
    protocol, settings = collection_protocol(metadata)
    out = frame.copy()
    identities = {"model": metadata.get("model"), "recheck_protocol": protocol}
    if "model_digest" in metadata:
        identities["model_digest"] = metadata["model_digest"]
    for key, value in identities.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Collection metadata requires an explicit {key}.")
        if key in out and not out[key].eq(value).all():
            raise ValueError(f"Row {key} conflicts with its collection protocol metadata.")
        out[key] = value
    out.attrs["collection_metadata"] = metadata
    out.attrs["generation_settings"] = settings
    return out


def load_pairs(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".jsonl":
        frame = pd.read_json(path, lines=True, dtype=False)
    else:
        # Answer/ID strings may contain significant leading zeroes. Numeric
        # token fields are validated separately after loading.
        frame = pd.read_csv(path, keep_default_na=False,
                            dtype={key: str for key in ("base_answer", "final_answer", "gold_answer", "example_id")})
    return validate_pairs(_attach_collection_metadata(frame, path))


def validate_transfer_compatibility(source, target):
    """Reject solver/protocol changes before fitting a transferred gate."""
    assert_disjoint(source, target)
    identity = {}
    for key in ("model", "recheck_protocol", "action"):
        left, right = _single_value(source, key), _single_value(target, key)
        if left != right:
            raise ValueError(f"Source/target {key} compatibility mismatch: {left!r} != {right!r}.")
        identity[key] = left
    if "model_digest" in source or "model_digest" in target:
        left, right = _single_value(source, "model_digest"), _single_value(target, "model_digest")
        if left != right:
            raise ValueError("Source/target model checkpoint compatibility mismatch.")
        identity["model_digest"] = left
    if "action_description" in source or "action_description" in target:
        left, right = _single_value(source, "action_description"), _single_value(target, "action_description")
        if left != right:
            raise ValueError("Source/target recheck action descriptions differ.")
        identity["action_description_sha256"] = hashlib.sha256(left.encode("utf-8")).hexdigest()
    settings = source.attrs.get("generation_settings")
    target_settings = target.attrs.get("generation_settings")
    if settings is not None or target_settings is not None:
        if target_settings != settings:
            raise ValueError("Source/target generation settings differ.")
        identity["generation_settings"] = settings
    public_settings = source.attrs.get("public_generation_settings")
    target_public_settings = target.attrs.get("public_generation_settings")
    if public_settings is not None or target_public_settings is not None:
        if target_public_settings != public_settings:
            raise ValueError("Source/target public generation settings differ.")
        identity["published_generation_settings"] = public_settings
        identity["identity_validation_limit"] = (
            "Public release identifies model family and action, but omits the checkpoint digest and full prompts."
        )
    elif "model_digest" not in identity:
        identity["identity_validation_limit"] = "Model name and declared protocol match; no checkpoint digest was supplied."
    return identity


def _with_canonical_groups(frame):
    if "query" not in frame:
        raise ValueError("Question text is required for independent-question grouping.")
    out = frame.copy()
    out["group_id"] = out["query"].map(question_hash)
    return out


def group_partition(frame, fractions, seed=42):
    if not fractions or not np.isfinite(fractions).all() or not np.isclose(sum(fractions), 1) or any(f <= 0 for f in fractions):
        raise ValueError("Positive partition fractions must sum to one.")
    frame = _with_canonical_groups(frame)
    groups = frame.group_id.drop_duplicates().to_numpy()
    np.random.default_rng(seed).shuffle(groups)
    bounds = [0] + [int(len(groups) * sum(fractions[:i])) for i in range(1, len(fractions))] + [len(groups)]
    parts = [frame[frame.group_id.isin(groups[bounds[i]:bounds[i + 1]])].copy().reset_index(drop=True)
             for i in range(len(fractions))]
    if any(p.empty for p in parts):
        raise ValueError("Too few independent questions for the requested split.")
    assert_disjoint(*parts)
    return parts


def assert_disjoint(*frames):
    # Never trust cached/user-provided hashes when proving no question leakage.
    if any("query" not in f for f in frames):
        raise ValueError("Question text is required to check partition leakage.")
    groups = [set(f["query"].map(question_hash)) for f in frames]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i] & groups[j]:
                raise ValueError("Question leakage between experimental partitions.")


def public_frames(root: Path):
    raw = Path(root) / "data" / "raw"
    source = pd.read_parquet(raw / "train.parquet")
    target = pd.read_parquet(raw / "commonsenseqa.parquet")
    source = source[source.action == "active_verify"].copy()
    target = target[target.action == "active_verify"].copy()
    source["dataset"], target["dataset"] = "SEVRA_MATH_train", "SEVRA_CommonsenseQA"
    for frame in (source, target):
        if _single_value(frame, "model") != "qwen3:4b":
            raise ValueError("Included public data model differs from its pinned Qwen3-4B attribution.")
        frame["recheck_protocol"] = f"sevra:{PUBLIC_REVISION}:active_verify"
        # Published unavailable completion statuses become a training-time category.
        # Raw files remain untouched; the feature builder flags unknown completion.
        unknown = frame["base_done_reason"].eq("")
        frame.loc[unknown, "base_done_reason"] = "unknown"
        frame.attrs["unknown_initial_completion_rows"] = int(unknown.sum())
        frame.attrs["public_generation_settings"] = {
            "dataset_revision": PUBLIC_REVISION, "base_generation_limit": 4096,
            "finalizer_max_tokens": 512, "finalizer": "non-reasoning final answer when needed",
            "protocol_evidence": "Pinned public dataset card; complete prompts/checkpoint digest not released",
        }
    return validate_pairs(source), validate_pairs(target)


def write_manifest(path, partitions, metadata):
    payload = dict(metadata)
    payload["partitions"] = {
        name: {"questions": int(f.group_id.nunique()), "rows": len(f),
               "group_ids": sorted(f.group_id.unique().tolist()), "label_counts": f.label.value_counts().to_dict()}
        for name, f in partitions.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
