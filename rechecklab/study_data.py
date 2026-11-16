"""Pin benchmarks and freeze key-free question partitions before live collection."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .data import question_hash

DATA_SCHEMA_VERSION = "live-study-questions-v1"
PARTITIONS = ("source_train", "source_development", "source_test", "target_adaptation_pool", "target_test")
QUESTION_FIELDS = {
    "example_id", "dataset", "domain", "partition", "group_id", "query", "answer_type",
    "source_dataset", "source_revision", "source_split", "source_question_id",
}
BENCHMARKS = {
    "gsm8k": dict(repository="openai/gsm8k", domain="maths", answer_type="number", license="MIT",
                  files={"train": "main/train-00000-of-00001.parquet", "test": "main/test-00000-of-00001.parquet"},
                  authors="Cobbe et al. (2021), Training Verifiers to Solve Math Word Problems",
                  paper="https://arxiv.org/abs/2110.14168", homepage="https://github.com/openai/grade-school-math"),
    "commonsenseqa": dict(repository="tau/commonsense_qa", domain="commonsenseqa", answer_type="choice", license="MIT",
                          files={"train": "data/train-00000-of-00001.parquet", "validation": "data/validation-00000-of-00001.parquet"},
                          authors="Talmor, Herzig, Lourie and Berant (2019), CommonsenseQA",
                          paper="https://aclanthology.org/N19-1421/", homepage="https://www.tau-nlp.org/commonsenseqa"),
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def planned_counts(maths_target=1000, target_target=500):
    for name, value, minimum in (("maths_target", maths_target, 5), ("target_target", target_target, 3)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum} to provide nonempty reserved partitions.")
    train, development = 3 * maths_target // 5, maths_target // 5
    adaptation = 2 * target_target // 5
    return dict(source_train=int(train), source_development=int(development),
                source_test=int(maths_target - train - development),
                target_adaptation_pool=int(adaptation), target_test=int(target_target - adaptation))


def _fetch(url, retries=3):
    errors = []
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "RecheckLab-live-study/0.3"})
            with urlopen(request, timeout=90) as response:
                return response.read(), attempt
        except OSError as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(f"Could not download pinned benchmark resource {url}: " + "; ".join(errors))


def resolve_snapshot(repository):
    """Resolve main once; every subsequent resource URL uses the returned commit."""
    raw, _ = _fetch(f"https://huggingface.co/api/datasets/{repository}?blobs=true")
    payload = json.loads(raw)
    revision = payload.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Hugging Face did not identify a full commit for {repository}.")
    siblings = {item["rfilename"]: item for item in payload.get("siblings", []) if isinstance(item, dict) and "rfilename" in item}
    return dict(repository=repository, revision=revision, files=siblings,
                resolved_utc=datetime.now(timezone.utc).isoformat())


def download_snapshot(root, dataset, snapshot):
    config = BENCHMARKS[dataset]
    folder = Path(root) / "raw" / dataset / snapshot["revision"]
    folder.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, remote in {**config["files"], "dataset_card": "README.md"}.items():
        if remote not in snapshot["files"]:
            raise ValueError(f"Pinned {config['repository']} snapshot lacks {remote}.")
        url = f"https://huggingface.co/datasets/{config['repository']}/resolve/{snapshot['revision']}/{remote}?download=true"
        path = folder / (name + (".md" if name == "dataset_card" else ".parquet"))
        sibling = snapshot["files"][remote]
        lfs = sibling.get("lfs") or {}
        expected = lfs.get("sha256") or lfs.get("oid")
        attempts = 0
        if not path.exists():
            content, attempts = _fetch(url)
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(content)
            actual = sha256_file(temporary)
            if expected and actual != expected:
                raise ValueError(f"Pinned download checksum mismatch for {remote}: {actual} != {expected}.")
            if name != "dataset_card":
                pd.read_parquet(temporary)  # Reject HTML/error payloads before committing cache.
            temporary.replace(path)
        actual = sha256_file(path)
        if expected and actual != expected:
            raise ValueError(f"Cached pinned benchmark checksum mismatch: {path}.")
        entries[name] = dict(path=str(path.relative_to(Path(root))), source_file=remote, url=url,
                             sha256=actual, bytes=path.stat().st_size, download_attempts=attempts)
    return entries


def _numeric_key(solution):
    if not isinstance(solution, str) or "####" not in solution:
        raise ValueError("GSM8K solution is missing its published #### answer delimiter.")
    key = solution.rsplit("####", 1)[1].strip()
    if not re.fullmatch(r"[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", key):
        raise ValueError(f"GSM8K published answer is not a strict numeric key: {key!r}.")
    try:
        if not Decimal(key.replace(",", "")).is_finite():
            raise ValueError("GSM8K answer key must be finite.")
    except InvalidOperation as exc:
        raise ValueError("Invalid GSM8K numeric answer key.") from exc
    return key


def benchmark_candidates(dataset, split, frame, revision):
    """Adapt benchmark schemas; gold stays private until a separate key file."""
    if dataset not in BENCHMARKS or split not in BENCHMARKS[dataset]["files"]:
        raise ValueError("Only the declared labeled benchmark splits can be selected.")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Benchmark questions require a full pinned revision.")
    config = BENCHMARKS[dataset]
    candidates = []
    for index, source in enumerate(frame.to_dict("records")):
        stem = source.get("question")
        if not isinstance(stem, str) or not stem.strip():
            raise ValueError(f"Missing benchmark question in {dataset}/{split}/{index}.")
        if dataset == "gsm8k":
            query, identifier, key = stem.strip(), str(index), _numeric_key(source.get("answer"))
        else:
            identifier = source.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"CommonsenseQA row {index} lacks its published question ID.")
            choices = source.get("choices")
            if isinstance(choices, dict):
                labels, texts = list(choices.get("label", [])), list(choices.get("text", []))
            elif isinstance(choices, (list, np.ndarray)):
                labels, texts = [item.get("label") for item in choices], [item.get("text") for item in choices]
            else:
                raise ValueError("CommonsenseQA choices must contain labeled answer options.")
            if len(labels) != 5 or len(texts) != 5 or set(labels) != set("ABCDE") or not all(isinstance(text, str) and text.strip() for text in texts):
                raise ValueError("CommonsenseQA requires exactly five nonempty A-E choices.")
            options = dict(zip(labels, texts))
            query = stem.strip() + "\n\nOptions:\n" + "\n".join(f"{label}. {options[label].strip()}" for label in "ABCDE")
            query += "\n\nAnswer with one option letter (A-E)."
            key = source.get("answerKey")
            if key not in tuple("ABCDE"):
                raise ValueError(f"CommonsenseQA {split}/{identifier} has no valid public answer key.")
        candidates.append(dict(example_id=f"{dataset}:{split}:{identifier}", dataset=dataset, domain=config["domain"],
            group_id=question_hash(query), query=query, answer_type=config["answer_type"],
            source_dataset=config["repository"], source_revision=revision, source_split=split,
            source_question_id=identifier, _gold_answer=key))
    if len({row["example_id"] for row in candidates}) != len(candidates):
        raise ValueError(f"Duplicate published question IDs in {dataset}/{split}.")
    return candidates


def deduplicate_candidates(pools):
    """Exclude cross-boundary overlaps globally, then choose one within-pool row."""
    locations = defaultdict(set)
    for name, rows in pools.items():
        for row in rows:
            locations[row["group_id"]].add(name)
    overlapping = {group for group, names in locations.items() if len(names) > 1}
    clean, details = {}, {}
    for name, rows in pools.items():
        kept, seen, duplicate_rows, excluded_overlap_rows = [], set(), 0, 0
        representative_keys = {}
        for row in rows:
            if row["group_id"] in overlapping:
                excluded_overlap_rows += 1
            elif row["group_id"] in seen:
                current = Decimal(row["_gold_answer"].replace(",", "")) if row["answer_type"] == "number" else row["_gold_answer"]
                if representative_keys[row["group_id"]] != current:
                    raise ValueError("Conflicting trusted keys for a canonical duplicate question.")
                duplicate_rows += 1
            else:
                kept.append(row)
                seen.add(row["group_id"])
                representative_keys[row["group_id"]] = Decimal(row["_gold_answer"].replace(",", "")) if row["answer_type"] == "number" else row["_gold_answer"]
        clean[name] = kept
        details[name] = dict(original_rows=len(rows), retained_unique_questions=len(kept),
                             removed_within_pool_duplicate_rows=duplicate_rows,
                             excluded_cross_boundary_overlap_rows=excluded_overlap_rows)
    return clean, dict(cross_boundary_overlapping_question_groups=len(overlapping), pools=details,
                      exclusion_rule="Exclude every cross-original-split/domain overlapping group; keep first pinned row for within-split duplicates before seeded selection.")


def select_questions(pools, counts, seed=42):
    clean, duplicate_audit = deduplicate_candidates(pools)
    requested = dict(gsm8k_train=counts["source_train"] + counts["source_development"], gsm8k_test=counts["source_test"],
                     commonsenseqa_train=counts["target_adaptation_pool"], commonsenseqa_validation=counts["target_test"])
    selected = {}
    seeds = {name: seed + offset for offset, name in enumerate(requested)}
    for name, number in requested.items():
        if len(clean.get(name, [])) < number:
            raise ValueError(f"{name} has too few independent questions for planned target {number} after overlap exclusion.")
        order = np.random.default_rng(seeds[name]).permutation(len(clean[name]))[:number]
        selected[name] = [dict(clean[name][index]) for index in order]
    assignments = (
        ("source_train", selected["gsm8k_train"][:counts["source_train"]]),
        ("source_development", selected["gsm8k_train"][counts["source_train"]:]),
        ("source_test", selected["gsm8k_test"]),
        ("target_adaptation_pool", selected["commonsenseqa_train"]),
        ("target_test", selected["commonsenseqa_validation"]),
    )
    questions, keys = [], []
    for partition, rows in assignments:
        for row in rows:
            key = row.pop("_gold_answer")
            row["partition"] = partition
            questions.append(row)
            keys.append(dict(example_id=row["example_id"], gold_answer=key, answer_type=row["answer_type"]))
    validate_question_manifest(questions, keys, counts)
    return questions, keys, duplicate_audit, seeds


def validate_question_manifest(questions, keys, expected_counts=None):
    if not questions:
        raise ValueError("Prepared question manifest cannot be empty.")
    identifiers, groups, counts = set(), set(), Counter()
    split_rules = {"source_train": ("gsm8k", "train"), "source_development": ("gsm8k", "train"),
                   "source_test": ("gsm8k", "test"), "target_adaptation_pool": ("commonsenseqa", "train"),
                   "target_test": ("commonsenseqa", "validation")}
    for row in questions:
        if set(row) != QUESTION_FIELDS:
            raise ValueError("Question manifest schema differs or contains an answer/future field.")
        if row["partition"] not in split_rules or (row["dataset"], row["source_split"]) != split_rules[row["partition"]]:
            raise ValueError("Question assignment crosses its reserved original benchmark split.")
        config = BENCHMARKS[row["dataset"]]
        if row["domain"] != config["domain"] or row["answer_type"] != config["answer_type"] or row["source_dataset"] != config["repository"]:
            raise ValueError("Question manifest benchmark metadata is incompatible.")
        if not re.fullmatch(r"[0-9a-f]{40}", row["source_revision"]):
            raise ValueError("Question manifest must pin a full benchmark revision.")
        if row["group_id"] != question_hash(row["query"]):
            raise ValueError("Question manifest contains a stale or forged canonical group.")
        if row["example_id"] in identifiers or row["group_id"] in groups:
            raise ValueError("Duplicate/overlapping question leaks into the fixed manifest.")
        identifiers.add(row["example_id"])
        groups.add(row["group_id"])
        counts[row["partition"]] += 1
    if expected_counts is not None and dict(counts) != expected_counts:
        raise ValueError(f"Prepared partition counts do not match the full planned targets: {dict(counts)} != {expected_counts}.")
    if len(keys) != len(questions) or {key.get("example_id") for key in keys} != identifiers:
        raise ValueError("Separate trusted keys must match every selected example exactly once.")
    types = {row["example_id"]: row["answer_type"] for row in questions}
    for key in keys:
        if set(key) != {"example_id", "gold_answer", "answer_type"} or key["answer_type"] != types[key["example_id"]]:
            raise ValueError("Trusted-key schema/type differs from its fixed question.")
        if key["answer_type"] == "number":
            _numeric_key("#### " + str(key["gold_answer"]))
        elif key["gold_answer"] not in tuple("ABCDE"):
            raise ValueError("Invalid trusted choice answer key.")
    return dict(counts)


def _read_jsonl(path):
    # JSONL boundaries are literal LF. Unicode separators inside a benchmark
    # question are valid string data and must never become record boundaries.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def _write_frozen(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to replace a different frozen study file: {path}.")
        return
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def prepare_study(root, maths_target=1000, target_target=500, seed=42):
    """Prepare exactly the planned unique questions, or fail without shrinking targets."""
    root = Path(root).resolve()
    project = Path(__file__).resolve().parents[1]
    preserved = [project / name for name in ("data", "models", "reports", "outputs/v0.2")]
    if root == project or any(root == path.resolve() or path.resolve() in root.parents for path in preserved):
        raise ValueError("Use a separate versioned live-study directory; preserve the included public pilot.")
    counts = planned_counts(maths_target, target_target)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Selection seed must be a nonnegative integer.")
    root.mkdir(parents=True, exist_ok=True)
    plan_path, question_path, key_path = root / "study_plan.json", root / "questions.jsonl", root / "trusted_keys.jsonl"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("data_schema_version") != DATA_SCHEMA_VERSION or plan.get("seed") != int(seed) or plan.get("planned_questions_by_partition") != counts:
            raise ValueError("Study data are frozen for different targets/seed/schema; choose a new versioned run directory.")
        if plan.get("planned_total_by_domain") != dict(maths=int(maths_target), commonsenseqa=int(target_target)) or plan.get("selected_questions_by_partition") != counts:
            raise ValueError("Frozen study plan domain/partition totals were altered.")
        identity = {key: plan[key] for key in ("data_schema_version", "seed", "planned_questions_by_partition", "manifest_sha256")}
        if plan.get("study_plan_hash") != hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest():
            raise ValueError("Frozen study plan selection hash was altered.")
        for name, path in (("questions.jsonl", question_path), ("trusted_keys.jsonl", key_path)):
            if not path.exists() or sha256_file(path) != plan["manifest_sha256"][name]:
                raise ValueError(f"Frozen study manifest missing or altered: {path}.")
        questions = _read_jsonl(question_path)
        validate_question_manifest(questions, _read_jsonl(key_path), counts)
        for row in questions:
            if plan["datasets"][row["dataset"]]["revision"] != row["source_revision"]:
                raise ValueError("Frozen study plan provenance revision differs from its questions.")
        return plan
    snapshots, pools = {}, {}
    for dataset, config in BENCHMARKS.items():
        snapshot = resolve_snapshot(config["repository"])
        files = download_snapshot(root, dataset, snapshot)
        snapshots[dataset] = dict(repository=config["repository"], revision=snapshot["revision"],
            resolved_utc=snapshot["resolved_utc"], files=files, license=config["license"],
            authors=config["authors"], paper=config["paper"], homepage=config["homepage"],
            pinned_dataset_card=f"https://huggingface.co/datasets/{config['repository']}/blob/{snapshot['revision']}/README.md")
        for split in config["files"]:
            frame = pd.read_parquet(root / files[split]["path"])
            pools[dataset + "_" + split] = benchmark_candidates(dataset, split, frame, snapshot["revision"])
    questions, keys, duplicates, seeds = select_questions(pools, counts, int(seed))
    question_bytes = ("".join(_canonical_json(row) + "\n" for row in questions)).encode("utf-8")
    key_bytes = ("".join(_canonical_json(row) + "\n" for row in keys)).encode("utf-8")
    plan = dict(data_schema_version=DATA_SCHEMA_VERSION, status="PREPARED", seed=int(seed),
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        planned_total_by_domain=dict(maths=int(maths_target), commonsenseqa=int(target_target)),
        planned_questions_by_partition=counts, selected_questions_by_partition=counts,
        manifest_paths=dict(questions="questions.jsonl", trusted_keys="trusted_keys.jsonl"),
        manifest_sha256={"questions.jsonl": hashlib.sha256(question_bytes).hexdigest(), "trusted_keys.jsonl": hashlib.sha256(key_bytes).hexdigest()},
        datasets=snapshots, question_selection_seeds=seeds, duplicate_overlap_audit=duplicates,
        selection_rule="Select by seeded permutation of independent canonical question groups before any model outcomes; one initial/recheck pair per selected question.",
        canonicalization="Unicode NFKC, casefold, collapse whitespace; SHA256 of full question including all choice options.",
        source_question_ids="GSM8K has no released question ID: pinned Parquet row offset; CommonsenseQA uses its published id.",
        rounding_rule="Floor maths train=3N/5 and development=N/5; remainder test. Floor target adaptation=2N/5; remainder test.",
        benchmark_boundaries={"source_train": "GSM8K main/train", "source_development": "GSM8K main/train",
            "source_test": "GSM8K main/test", "target_adaptation_pool": "CommonsenseQA/train", "target_test": "CommonsenseQA/validation"},
        repartition_notes=["GSM8K train is repartitioned into study training/development without using outcomes; its original test remains held out.",
            "CommonsenseQA labeled training questions supply adaptation only. Labeled validation supplies this study's fixed target test; official test answer keys are withheld and that split is not used.",
            "This is a new fixed-solver self-rechecking experiment, not reproduction of the original benchmark leaderboard or SEVRA outcomes.",
            "GSM8K replaces the public pilot's MATH source because its public numeric keys support transparent deterministic offline scoring.",
            "The planned sample sizes are collection targets and do not guarantee enough rare HURT events or statistical reliability."],
        target_adaptation=dict(budgets=[25, 50, 100, 200], repeats=5, subset_seeds=[int(seed) + 100 + i for i in range(5)]),
        smoke_allowed_partitions=["source_train", "target_adaptation_pool"],
        question_format_version="GSM8K original question; CommonsenseQA stem plus labeled A-E options and one-letter response instruction-v1",
        key_separation="Question manifest contains no answers/solutions/correctness. Trusted keys are used offline only and are never passed to model prompts or predictor inputs.",
        raw_downloads="Local raw/ snapshots are provenance caches and must be excluded from diagnostic bundles.")
    selection_identity = {key: plan[key] for key in ("data_schema_version", "seed", "planned_questions_by_partition", "manifest_sha256")}
    plan["study_plan_hash"] = hashlib.sha256(_canonical_json(selection_identity).encode("utf-8")).hexdigest()
    _write_frozen(question_path, question_bytes)
    _write_frozen(key_path, key_bytes)
    _write_frozen(plan_path, (json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    return plan
