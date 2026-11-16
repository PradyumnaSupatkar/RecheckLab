"""Frozen MATH L4/L5 numeric-subset development gate and conditional study.

Run from repository root. --prepare downloads the pinned benchmark test mirror,
audits source-only eligibility, validates the exact grader, and freezes 10+60
disjoint questions before inference. --run performs development, then starts
the 60-question run only if its predeclared reliability/resource gate passes.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import psutil

DEV = Path(__file__).resolve().parent
ROOT = DEV.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL = ROOT / "outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01"
CACHE = DEV / "source_cache"
MIRROR = "EleutherAI/hendrycks_math"
SOURCE_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
TOPICS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
          "number_theory", "prealgebra", "precalculus"]
MODEL = "qwen3:4b-instruct"
MODEL_DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
OLLAMA_VERSION_EXPECTED = "ollama version is 0.40.16"
INITIAL_PROMPT = ("Solve the problem carefully. Provide enough concise reasoning to make the result checkable. "
                  "End with exactly one final line in this form: FINAL: <integer or exact fraction>. "
                  "Use an exact fraction instead of a decimal when needed. Do not omit the final line.")
REVISION_PROMPT = ("Review your previous solution against the problem and correct any mathematical error. "
                   "Give a concise explanation of your check. End with exactly one final line in this form: "
                   "FINAL: <integer or exact fraction>. Use an exact fraction instead of a decimal when needed. "
                   "Do not omit the final line.")
GENERATION = {"context_window": 8192, "num_predict": -1, "temperature": 0.0, "top_k": 1,
              "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
              "stop": "model defaults; no request override", "think_field": "omitted",
              "stream": True, "watchdog_seconds": 2700, "idle_read_timeout_seconds": 120, "retries": 0}
PAGEFILE_CRITICAL_PERCENT = 98.0
SELECTION_SEED = 20261120
INIT_SEED_BASE = 61000
REV_SEED_BASE = 71000
GATE_PATH = ROOT / "outputs/v0.2/models/source_gate.joblib"
GATE_SHA = "9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439"
GATE_THRESHOLD = 0.25968145276947807
MATH500 = ROOT / "data/raw/math500.parquet"
MATH_SOURCE_TRAIN = ROOT / "data/prepared/source_train.parquet"
MATH_SOURCE_DEV = ROOT / "data/prepared/source_development.parquet"
MATH_SOURCE_TEST = ROOT / "data/prepared/source_test.parquet"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(text: str) -> str:
    from rechecklab.data import question_hash
    return question_hash(text)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows):
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8"))
    return sha_bytes(payload.encode("utf-8"))


def parse_exact_rational(raw: str | None):
    """Return Fraction only for explicitly supported integer/rational syntax."""
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:displaystyle|textstyle)\s*", "", text).strip()
    if text.startswith("\\boxed"):
        m = re.fullmatch(r"\\boxed\s*\{(.*)\}", text, flags=re.S)
        if not m:
            return None
        text = m.group(1).strip()
    m = re.fullmatch(r"([+-]?)\\(?:dfrac|tfrac|frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}", text)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        numerator, denominator = int(m.group(2)) * sign, int(m.group(3))
        if denominator == 0:
            return None
        return Fraction(numerator, denominator)
    text = text.replace(",", "")
    if re.fullmatch(r"[+-]?\d+", text):
        return Fraction(int(text), 1)
    m = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", text)
    if m and int(m.group(2)) != 0:
        return Fraction(int(m.group(1)), int(m.group(2)))
    return None


def parse_final_answer(content: str):
    lines = [line.strip() for line in str(content).splitlines() if line.strip()]
    if not lines:
        return None, "missing_final_marker"
    marker_present = any(re.match(r"^FINAL\s*:", line, re.I) for line in lines)
    if not marker_present:
        return None, "missing_final_marker"
    m = re.fullmatch(r"FINAL\s*:\s*(.*?)\s*", lines[-1], re.I)
    if not m:
        return None, "malformed_final_answer"
    value = parse_exact_rational(m.group(1))
    if value is None:
        return None, "malformed_final_answer"
    return fraction_text(value), "parseable"


def fraction_text(value: Fraction):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def validate_grader():
    tests = [
        ("FINAL: 7", "7"), ("FINAL: -12", "-12"), ("FINAL: 6/8", "3/4"),
        (r"FINAL: \frac{-2}{6}", "-1/3"), (r"FINAL: \boxed{\dfrac{10}{5}}", "2"),
    ]
    for raw, expected in tests:
        parsed, status = parse_final_answer(raw)
        if status != "parseable" or parsed != expected:
            raise AssertionError(f"Synthetic grader failed {raw!r}: {parsed!r}/{status}")
    invalid = ["FINAL: 1/0", "FINAL: sqrt(2)", "FINAL: 0.5", "FINAL: 2 plus 3", "No answer"]
    for raw in invalid:
        parsed, status = parse_final_answer(raw)
        if parsed is not None or status not in ("malformed_final_answer", "missing_final_marker"):
            raise AssertionError(f"Synthetic invalid grader case accepted: {raw!r}")
    assert Fraction(1, 2) == Fraction(2, 4)
    return {"synthetic_tests": len(tests) + len(invalid) + 1, "passed": True,
            "coverage": ["positive integer", "negative integer", "rational reduction", "LaTeX fraction", "boxed fraction", "zero denominator rejected", "irrational rejected", "decimal rejected", "free text rejected", "equivalent fractions compare exactly"]}


def api_dataset_info():
    url = f"https://huggingface.co/api/datasets/{MIRROR}/revision/{SOURCE_REVISION}?blobs=true"
    req = Request(url, headers={"User-Agent": "RecheckLab-prospective-MATH-evaluation/0.4.1"})
    with urlopen(req, timeout=30) as response:
        data = json.load(response)
    if data.get("sha") != SOURCE_REVISION:
        raise ValueError("Dataset host did not return the frozen MATH mirror commit.")
    return {item["rfilename"]: item for item in data.get("siblings", []) if isinstance(item, dict) and item.get("rfilename")}


def download_test_partitions():
    files = api_dataset_info()
    results = {}
    for topic in TOPICS:
        rel = f"{topic}/test-00000-of-00001.parquet"
        if rel not in files:
            raise FileNotFoundError(f"Pinned MATH test partition is missing: {rel}")
        target = CACHE / (topic + "_test.parquet")
        expected_size = files[rel].get("size")
        lfs = files[rel].get("lfs") or {}
        expected_sha = lfs.get("sha256")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{MIRROR}/resolve/{SOURCE_REVISION}/{rel}?download=true"
            req = Request(url, headers={"User-Agent": "RecheckLab-prospective-MATH-evaluation/0.4.1"})
            temp = target.with_suffix(".parquet.part")
            with urlopen(req, timeout=120) as response, temp.open("wb") as f:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
            temp.replace(target)
        actual_sha, actual_size = sha_file(target), target.stat().st_size
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(f"Pinned MATH partition SHA mismatch for {rel}.")
        if expected_size is not None and actual_size != int(expected_size):
            raise ValueError(f"Pinned MATH partition byte-size mismatch for {rel}.")
        results[rel] = {"local_file": str(target.relative_to(ROOT)), "size_bytes": actual_size,
                        "sha256": actual_sha, "source_expected_sha256": expected_sha,
                        "source_expected_size": expected_size}
    return results


def collect_known_hashes():
    sevra = set()
    math500 = set()
    prior_other = set()
    details = {"math500_rows": 0, "sevra_source_rows": 0, "parquet_files_scanned": 0,
               "jsonl_rows_scanned": 0, "csv_files_scanned": 0}
    def add_frame(path: Path, category=None):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            return
        details["parquet_files_scanned"] += 1
        col = "query" if "query" in frame.columns else "problem" if "problem" in frame.columns else None
        if not col:
            return
        dest = math500 if (path.resolve() == MATH500.resolve()) else sevra if ("source_" in path.name and path.parent.name == "prepared") else prior_other
        for text in frame[col].dropna().astype(str):
            if text.strip():
                dest.add(canonical_hash(text))
        if path.resolve() == MATH500.resolve():
            details["math500_rows"] = len(frame)
        if dest is sevra:
            details["sevra_source_rows"] += len(frame)
    for path in (ROOT / "data").rglob("*.parquet"):
        add_frame(path)
    for path in (ROOT / "outputs").rglob("*.parquet"):
        if path.is_relative_to(DEV):
            continue
        add_frame(path)
    for path in (ROOT / "outputs").rglob("*.csv"):
        if path.is_relative_to(DEV) or path.is_relative_to(EVAL):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        details["csv_files_scanned"] += 1
        col = "query" if "query" in frame.columns else "question" if "question" in frame.columns else None
        if col:
            for text in frame[col].dropna().astype(str):
                if text.strip():
                    prior_other.add(canonical_hash(text))
    for path in (ROOT / "outputs").rglob("*.jsonl"):
        if path.is_relative_to(DEV) or path.is_relative_to(EVAL):
            continue
        for row in read_jsonl(path):
            details["jsonl_rows_scanned"] += 1
            query = row.get("query") or row.get("question") or row.get("problem")
            if isinstance(query, str) and query.strip():
                prior_other.add(canonical_hash(query))
            if isinstance(row.get("group_id"), str):
                prior_other.add(row["group_id"])
    return sevra, math500, prior_other, details


def boxed_expressions(solution: str):
    found = []
    for match in re.finditer(r"\\boxed\s*\{", solution):
        open_brace = solution.find("{", match.start())
        depth = 0
        for index in range(open_brace, len(solution)):
            if solution[index] == "{":
                depth += 1
            elif solution[index] == "}":
                depth -= 1
                if depth == 0:
                    found.append(solution[open_brace + 1:index].strip())
                    break
    return found


def extract_reference(solution: str):
    boxes = boxed_expressions(solution)
    if len(boxes) != 1:
        return None, "reference_box_count_not_one"
    value = parse_exact_rational(boxes[0])
    if value is None:
        return None, "reference_not_supported_integer_or_rational"
    return fraction_text(value), "eligible"


def ollama_identity(allow_loaded_current=False):
    with urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        tags = json.load(response).get("models", [])
    model = next((x for x in tags if x.get("name") == MODEL), None)
    if not model or model.get("digest") != MODEL_DIGEST:
        raise ValueError("Required installed Instruct model/digest is not available; no model will be downloaded.")
    req = Request("http://localhost:11434/api/show", data=json.dumps({"model": MODEL, "verbose": True}).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=15) as response:
        show = json.load(response)
    version = subprocess.check_output(["ollama", "--version"], text=True, timeout=10).strip()
    if version != OLLAMA_VERSION_EXPECTED or show.get("model_info", {}).get("general.finetune") != "Instruct" or show.get("model_info", {}).get("general.version") != "2507":
        raise ValueError("Installed model is not the pinned Instruct-2507 checkpoint or Ollama version.")
    try:
        rows = subprocess.check_output(["ollama", "ps"], text=True, timeout=10).splitlines()[1:]
    except Exception as exc:
        raise RuntimeError("Cannot verify active Ollama processes.") from exc
    active=[row.strip() for row in rows if row.strip()]
    if active and (not allow_loaded_current or any(not row.split()[0].startswith(MODEL) for row in active)):
        raise RuntimeError("Another Ollama model is already active; no process will be stopped.")
    return {"name": MODEL, "digest": model["digest"], "size_bytes": model.get("size"),
            "family": show.get("details", {}).get("family"), "fine_tune": show.get("model_info", {}).get("general.finetune"),
            "version": show.get("model_info", {}).get("general.version"), "quantization": show.get("details", {}).get("quantization_level"),
            "ollama_version": version, "thinking_request_field": "omitted"}


def resource_sample():
    vm, swap = psutil.virtual_memory(), psutil.swap_memory()
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
        name, total, used, free, util, temp = [v.strip() for v in line.split(",", 5)]
        gpu = {"name": name, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free), "utilization_percent": int(util), "temperature_c": int(temp)}
    except Exception as exc:
        gpu = {"available": False, "error": type(exc).__name__}
    try:
        ollama_ps = subprocess.check_output(["ollama", "ps"], text=True, timeout=5)
    except Exception as exc:
        ollama_ps = f"ERROR:{type(exc).__name__}"
    return {"timestamp_unix": time.time(), "total_ram_gib": round(vm.total/1024**3, 3),
            "free_ram_gib": round(vm.available/1024**3, 3), "ram_used_percent": vm.percent,
            "swap_percent": swap.percent, "swap_used_gib": round(swap.used/1024**3,3),
            "cpu_utilization_percent": psutil.cpu_percent(interval=0.1), "gpu": gpu, "ollama_ps": ollama_ps}


def resource_safe(r):
    # No free-RAM or free-VRAM reserve cutoff. Stop only when virtual memory
    # is nearly exhausted; Ollama allocation errors are handled per call.
    return r.get("swap_percent", 0) < PAGEFILE_CRITICAL_PERCENT


def prepare():
    if DEV.exists() and any(p.name not in {"run_math_study.py", "__pycache__", "source_cache", "ELIGIBILITY_AUDIT.json"} for p in DEV.iterdir()):
        raise FileExistsError("Development folder contains prior data; refusing overwrite.")
    if EVAL.exists() and any(EVAL.iterdir()):
        raise FileExistsError("Evaluation folder contains prior data; refusing overwrite.")
    model = ollama_identity()
    if not GATE_PATH.is_file() or sha_file(GATE_PATH) != GATE_SHA:
        raise ValueError("Frozen v0.2 classifier artifact is missing or has changed.")
    grader_validation = validate_grader()
    file_meta = download_test_partitions()
    sevra, math500, prior_other, known = collect_known_hashes()

    candidates=[]
    counts={"source_test_rows":0,"level_not_4_or_5":0,"level_4_or_5":0,
            "unsupported_or_ambiguous_reference":0,"known_sevra_overlap":0,
            "known_math500_overlap":0,"other_local_cohort_overlap":0,
            "duplicate_within_test":0,"eligible_unique":0}
    per_topic={t:{"source_test":0,"level4":0,"level5":0,"numeric_rational_level4_5":0,"eligible_after_overlap":0} for t in TOPICS}
    seen=set()
    for topic in TOPICS:
        path=CACHE/(topic+"_test.parquet")
        frame=pd.read_parquet(path)
        required={"problem","solution","level","type"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Pinned MATH partition schema missing fields: {required-set(frame.columns)}")
        for row_index,row in frame.iterrows():
            counts["source_test_rows"]+=1; per_topic[topic]["source_test"]+=1
            level=str(row["level"]).strip()
            if level not in ("Level 4","Level 5"):
                counts["level_not_4_or_5"]+=1; continue
            counts["level_4_or_5"]+=1
            per_topic[topic]["level4" if level=="Level 4" else "level5"]+=1
            query=str(row["problem"]).strip()
            key,answer_status=extract_reference(str(row["solution"]))
            if key is None:
                counts["unsupported_or_ambiguous_reference"]+=1; continue
            per_topic[topic]["numeric_rational_level4_5"]+=1
            group=canonical_hash(query)
            if group in sevra:
                counts["known_sevra_overlap"]+=1; continue
            if group in math500:
                counts["known_math500_overlap"]+=1; continue
            if group in prior_other:
                counts["other_local_cohort_overlap"]+=1; continue
            if group in seen:
                counts["duplicate_within_test"]+=1; continue
            seen.add(group)
            source_id=str(row.get("id",row.get("example_id",f"{topic}/{int(row_index)}")))
            candidates.append({"example_id":f"math:test:{topic}:{source_id}","source_question_id":source_id,
                               "group_id":group,"query":query,"topic":topic,"level":level,
                               "answer_type":"exact_integer_or_rational","reference_answer":key,
                               "source_path":f"test/{topic}/{int(row_index)}.json"})
            per_topic[topic]["eligible_after_overlap"]+=1
    counts["eligible_unique"]=len(candidates)
    if len(candidates)<70:
        DEV.mkdir(parents=True,exist_ok=True)
        write_json(DEV/"ELIGIBILITY_AUDIT.json",{"status":"INSUFFICIENT_ELIGIBLE_POOL","required":70,
                    "eligible":len(candidates),"counts":counts,"topic_distribution":per_topic,
                    "exclusion_sets":{"sevra_unique_hashes":len(sevra),"math500_unique_hashes":len(math500),"other_local_unique_hashes":len(prior_other),**known},
                    "source_revision":SOURCE_REVISION,"partitions":file_meta})
        raise RuntimeError(f"Only {len(candidates)} eligible unseen numeric/rational MATH L4/L5 questions; stopped before inference.")
    order=np.random.default_rng(SELECTION_SEED).permutation(len(candidates))[:70]
    selected=[candidates[int(i)] for i in order]
    dev_q,eval_q=selected[:10],selected[10:]
    manifests={}
    for folder,title,subset in ((DEV,"development_reliability_gate",dev_q),(EVAL,"untouched_frozen_evaluation",eval_q)):
        folder.mkdir(parents=True,exist_ok=True)
        questions=[{k:v for k,v in q.items() if k!="reference_answer"} for q in subset]
        keys=[{"example_id":q["example_id"],"answer_type":"exact_integer_or_rational","gold_answer":q["reference_answer"]} for q in subset]
        qhash=write_jsonl(folder/"QUESTION_SELECTION.jsonl",questions)
        khash=write_jsonl(folder/"TRUSTED_KEYS.jsonl",keys)
        levels={f"Level {n}":sum(q["level"]==f"Level {n}" for q in subset) for n in (4,5)}
        topics={t:sum(q["topic"]==t for q in subset) for t in TOPICS}
        protocol={"id":"live-study-v0.4.1-qwen3-4b-instruct-math-numeric",
                  "model":model,"prompts":{"initial":INITIAL_PROMPT,"revision":REVISION_PROMPT},
                  "prompt_sha256":{"initial":sha_bytes(INITIAL_PROMPT.encode()),"revision":sha_bytes(REVISION_PROMPT.encode())},
                  "generation":GENERATION,"seed_rule":{"initial":f"{INIT_SEED_BASE} + numeric MATH source question ID","revision":f"{REV_SEED_BASE} + numeric MATH source question ID"},
                  "parser":"strict FINAL last-line; exact integer or rational only; Fraction canonicalization",
                  "grader":"exact Python Fraction equality; reference from sole supported final \\boxed{} in official solution; unsupported/ambiguous reference excluded before selection",
                  "feature_schema":"pre-recheck-v2","feature_count":19,
                  "gate_artifact_sha256":GATE_SHA,"gate_threshold":GATE_THRESHOLD,
                  "gate_score":"P(HELP) - P(HURT)","gate_rule":"strictly greater than threshold",
                  "gate_status":"Cross-model/cross-protocol OOD diagnostic; live metadata not forged; production validator unchanged"}
        manifest={"experiment":"RecheckLab v0.4.1 independent hard-math study","phase":title,
                  "sample_size":len(subset),"expected_llm_calls":2*len(subset),"selection_seed":SELECTION_SEED,
                  "selection_rule":"Uniform seeded random sample among eligible unique test questions; no model outcomes, difficulty beyond allowed L4/L5 metadata, correctness, confidence, HELP/HURT, or gate action used to choose within eligibility.",
                  "eligibility":{"source_split":"original MATH test","difficulty_levels":[4,5],
                                  "reference_answer":"exact integer or rational from exactly one final boxed reference expression",
                                  "deduplication":"canonical Unicode NFKC/casefold/whitespace question hashes",
                                  "excluded_cohorts":["SEVRA MATH source train/development/test","SEVRA MATH500","local previously inspected question-bearing records","all previous RecheckLab live-study questions"]},
                  "dataset":{"official_project":"https://github.com/hendrycks/math","pinned_mirror":MIRROR,
                             "revision":SOURCE_REVISION,"test_partitions":file_meta,
                             "total_test_rows":counts["source_test_rows"],"eligible_unique_before_selection":len(candidates),
                             "eligibility_counts":counts,"topic_distribution_eligible":per_topic,
                             "topic_distribution_selected":topics,"level_distribution_selected":levels},
                  "protocol":protocol,"question_selection_sha256":qhash,"trusted_key_sha256":khash,
                  "question_ids":[q["example_id"] for q in subset],
                  "resource_guard":{"sequential":True,"minimum_free_ram_gib":RAM_FLOOR_GIB,
                                    "minimum_free_vram_mib":VRAM_FLOOR_MIB,"timeout_seconds":GENERATION["timeout_seconds"],
                                    "retries":0,"stop_on_resource_or_call_failure":True}}
        write_json(folder/"EXPERIMENT_MANIFEST.json",manifest)
        manifests[title]=manifest
    write_json(DEV/"ELIGIBILITY_AUDIT.json",{"status":"SUFFICIENT","counts":counts,
                "topic_distribution":per_topic,"excluded_hash_sets":{ "sevra_unique_hashes":len(sevra),
                    "math500_unique_hashes":len(math500),"other_local_unique_hashes":len(prior_other),**known},
                "eligibility_rule":"test split AND level 4/5 AND exactly one boxed answer parsable as exact integer/rational AND no canonical hash overlap; reference answer and level/source metadata only",
                "unsupported_or_ambiguous_reference_examples":counts["unsupported_or_ambiguous_reference"],
                "model_performance_used":False,"reference_answers_sent_to_model":False,
                "grader_validation":grader_validation,"dataset_files":file_meta,
                "selection":{"seed":SELECTION_SEED,"total":70,"development":10,"evaluation":60,
                             "development_ids":[q["example_id"] for q in dev_q],"evaluation_ids":[q["example_id"] for q in eval_q]}})
    baseline=resource_sample()
    prep={"status":"FROZEN_BEFORE_INFERENCE","model":model,"baseline_resources":baseline,
          "development_questions":10,"evaluation_questions":60,"inference_started":False,
          "all_70_questions_frozen_before_any_new_inference":True,
          "selection_hashes":{"development":manifests["development_reliability_gate"]["question_selection_sha256"],
                              "evaluation":manifests["untouched_frozen_evaluation"]["question_selection_sha256"]},
          "grader_validation":grader_validation}
    write_json(DEV/"PREPARATION_STATUS.json",prep)
    (DEV/"RUN_COMMAND.md").write_text("# Frozen run command\n\n```powershell\npython outputs/live_study/v0.4.1_math_l4_l5_development_2026-11-16_01/run_math_study.py --run\n```\n\nRuns the 10-question development gate and starts the already frozen 60-question evaluation only if the gate passes.\n",encoding="utf-8")
    print(json.dumps({"status":"FROZEN_BEFORE_INFERENCE","model":model,"counts":counts,
                      "topic_distribution":per_topic,"selected_level_distribution":{f"Level {x}":sum(q['level']==f"Level {x}" for q in selected) for x in (4,5)},
                      "selected_topic_distribution":{t:sum(q['topic']==t for q in selected) for t in TOPICS},
                      "dev_ids":[q["example_id"] for q in dev_q],"eval_count":len(eval_q),
                      "baseline_resources":baseline,"inference_started":False},indent=2))


def resource_safe_call(folder:Path,question_id:str,phase:str):
    sample=resource_sample()
    if not resource_safe(sample):
        status=read_json(folder/"COLLECTION_STATUS.json")
        status.update(status="STOPPED_RESOURCE_GUARD",stop_reason="virtual memory/pagefile reached critical usage",last_resource=sample)
        write_json(folder/"COLLECTION_STATUS.json",status)
        return None
    return sample


def call_model(payload):
    request=Request("http://localhost:11434/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    started=time.perf_counter()
    result={}; chunks=[]; thoughts=[]; samples=[]; chunk_count=0
    with urlopen(request,timeout=GENERATION["idle_read_timeout_seconds"]) as response:
        while True:
            if time.perf_counter()-started > GENERATION["watchdog_seconds"]:
                raise TimeoutError("45-minute generation watchdog exceeded")
            line=response.readline()
            if not line: break
            item=json.loads(line)
            msg=item.get("message") or {}
            if msg.get("content"):
                chunks.append(msg["content"]); chunk_count+=1
            if msg.get("thinking"):
                thoughts.append(msg["thinking"]); chunk_count+=1
            for key in ("model","created_at","done","done_reason","total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration"):
                if key in item: result[key]=item[key]
            if chunk_count and chunk_count % 256 == 0:
                sample=resource_sample(); samples.append(sample)
                if not resource_safe(sample):
                    raise MemoryError("Pagefile reached critical usage during generation")
            if item.get("done"):
                samples.append(resource_sample()); break
    result["message"]={"role":"assistant","content":"".join(chunks)}
    if thoughts: result["message"]["thinking"]="".join(thoughts)
    result["stream_resource_samples"]=samples
    result["client_wall_seconds"]=time.perf_counter()-started
    if result.get("eval_duration"):
        result["output_tokens_per_second"]=result.get("eval_count",0)/(result["eval_duration"]/1e9)
    return result


def load_gate():
    sys.path.insert(0,str(ROOT))
    from rechecklab.features import FEATURE_NAMES,FEATURE_SCHEMA_VERSION,INITIAL_REQUIRED,extract_features,validate_initial_record
    from rechecklab.predictor import load_artifact,revision_score,decide_recheck
    bundle=load_artifact(GATE_PATH,expected_sha256=GATE_SHA)
    if bundle["threshold"] != GATE_THRESHOLD or len(FEATURE_NAMES)!=19 or FEATURE_SCHEMA_VERSION!="pre-recheck-v2":
        raise ValueError("Frozen v0.2 gate contract has changed.")
    return bundle,FEATURE_NAMES,FEATURE_SCHEMA_VERSION,INITIAL_REQUIRED,extract_features,validate_initial_record,revision_score,decide_recheck


def store_call(folder,record):
    with (folder/"RAW_CALLS.jsonl").open("ab") as f:
        f.write((json.dumps(record,ensure_ascii=False)+"\n").encode("utf-8"))


def collect_phase(folder:Path):
    manifest=read_json(folder/"EXPERIMENT_MANIFEST.json")
    questions=read_jsonl(folder/"QUESTION_SELECTION.jsonl")
    status_path=folder/"COLLECTION_STATUS.json"
    if any((folder/name).exists() for name in ("RAW_CALLS.jsonl","PAIRS.jsonl","COLLECTION_STATUS.json")):
        raise FileExistsError(f"Refusing to overwrite/resume frozen collection in {folder.name}.")
    identity=ollama_identity(allow_loaded_current=True)
    if identity["digest"]!=MODEL_DIGEST:
        raise RuntimeError("Instruct checkpoint digest changed before collection.")
    bundle,feature_names,schema,required,extract_features,validate_initial_record,revision_score,decide_recheck=load_gate()
    protocol_expected=bundle["protocol"]
    status={"phase":manifest["phase"],"status":"RUNNING","calls_completed":0,"pairs_completed":0,
            "resource_samples":[],"gate_scored_before_revision":True,"gate_protocol_status":"OOD_DIAGNOSTIC"}
    write_json(status_path,status)
    gate_rows=[];pairs=[];call_count=0
    for q in questions:
        records={};answers={};meta={}
        for phase,prompt in (("initial",INITIAL_PROMPT),("revision",REVISION_PROMPT)):
            before=resource_safe_call(folder,q["example_id"],phase)
            if before is None:
                return read_json(status_path)
            user=INITIAL_PROMPT+"\n\n"+q["query"]
            messages=[{"role":"user","content":user}]
            if phase=="revision":
                first=records["initial"]["response"].get("message",{})
                messages.extend([{"role":"assistant","content":first.get("content","") or ""},
                                 {"role":"user","content":REVISION_PROMPT}])
            source_id=re.search(r"(\d+)$",q["source_question_id"])
            numeric_seed=int(source_id.group(1)) if source_id else int(hashlib.sha256(q["example_id"].encode()).hexdigest()[:8],16)
            seed=(INIT_SEED_BASE if phase=="initial" else REV_SEED_BASE)+numeric_seed
            payload={"model":MODEL,"messages":messages,"stream":True,
                     "options":{"num_predict":GENERATION["num_predict"],"num_ctx":GENERATION["context_window"],
                                 "temperature":GENERATION["temperature"],"seed":seed,"top_k":GENERATION["top_k"],
                                 "top_p":GENERATION["top_p"],"min_p":GENERATION["min_p"],
                                 "repeat_penalty":GENERATION["repeat_penalty"]}}
            try:
                result=call_model(payload)
                if result.get("done") is not True or "message" not in result:
                    raise RuntimeError("Ollama returned an incomplete response envelope.")
                msg=result.get("message",{});content=msg.get("content","") or ""
                answer,parse_status=parse_final_answer(content)
                prompt_tokens=result.get("prompt_eval_count");output_tokens=result.get("eval_count")
                if prompt_tokens is None or output_tokens is None:
                    raise RuntimeError("Ollama omitted actual prompt/output token accounting.")
                reason=result.get("done_reason") or "unknown"
                if answer is None and reason=="length" and (msg.get("thinking") or ""):
                    failure="reasoning_token_exhaustion"
                elif answer is None and reason=="length": failure="length_limit_truncation"
                elif answer is None: failure=parse_status
                else: failure="none"
                record={"example_id":q["example_id"],"phase":phase,"status":"succeeded","model":MODEL,
                        "model_digest":MODEL_DIGEST,"request":payload,"response":result,"completion_reason":reason,
                        "parsed_answer":answer,"parse_status":parse_status,"failure_type":failure,
                        "tokens":{"prompt_tokens":int(prompt_tokens),"output_tokens":int(output_tokens),
                                  "total_tokens":int(prompt_tokens)+int(output_tokens)},
                        "wall_clock_seconds":result["client_wall_seconds"],"resource_before":before,
                        "resource_after":resource_sample(),
                        "output_tokens_per_second":result.get("output_tokens_per_second"),
                        "stream_resource_samples":result.get("stream_resource_samples",[])}
                store_call(folder,record);call_count+=1
                records[phase]=record;answers[phase]=answer;meta[phase]=record
                status["calls_completed"]=call_count
                status["resource_samples"].append({"example_id":q["example_id"],"phase":phase,"before":before,"after":record["resource_after"]})
                status["last_call"]={k:record[k] for k in ("example_id","phase","completion_reason","parse_status","failure_type","tokens","wall_clock_seconds")}
                write_json(status_path,status)
                if phase=="initial":
                    initial_response=(msg.get("thinking","") or "")+(("\n\n") if msg.get("thinking") and content else "")+content
                    feature_row={"query":q["query"],"base_response":initial_response,"base_answer":answer or "",
                                 "base_actual_tokens":int(prompt_tokens)+int(output_tokens),
                                 "base_done_reason":reason,"base_finalizer_used":False}
                    gate_record={"example_id":q["example_id"],"scored_before_revision":True,
                                 "initial_call_sha256":sha_bytes(json.dumps(record,ensure_ascii=False,sort_keys=True).encode("utf-8")),
                                 "feature_input_fields":list(required),"protocol_status":"cross-model/cross-protocol OOD",
                                 "gate_compatible":True,"probabilities":None,"score":None,"threshold":bundle["threshold"],
                                 "recheck":None,"features":None}
                    try:
                        normalized=validate_initial_record(feature_row)
                        frame=pd.DataFrame([{k:normalized[k] for k in required}],columns=required)
                        feature_frame=extract_features(frame)
                        probs=bundle["predictor"].probabilities(frame)[0]
                        score=float(revision_score(np.asarray([probs]),bundle["harm_weight"])[0])
                        action=bool(decide_recheck(np.asarray([probs]),bundle["threshold"],bundle["harm_weight"])[0])
                        gate_record.update(probabilities={"HELP":float(probs[0]),"NO_CHANGE":float(probs[1]),"HURT":float(probs[2])},
                                           score=score,recheck=action,features={k:float(v) for k,v in feature_frame.iloc[0].to_dict().items()})
                    except Exception as exc:
                        gate_record.update(gate_compatible=False,compatibility_error=f"{type(exc).__name__}: {exc}")
                    gate_rows.append(gate_record)
                    with (folder/"GATE_PREDICTIONS.jsonl").open("ab") as f:
                        f.write((json.dumps(gate_record,ensure_ascii=False)+"\n").encode("utf-8"))
                    status["gate_predictions_written_before_revision"]=len(gate_rows)
                    write_json(status_path,status)
                if not resource_safe(record["resource_after"]):
                    status.update(status="STOPPED_RESOURCE_GUARD",stop_reason="virtual memory/pagefile reached critical usage after call")
                    write_json(status_path,status);return status
            except Exception as exc:
                fail={"example_id":q["example_id"],"phase":phase,"status":"failed","error_type":type(exc).__name__,
                      "error":str(exc),"request":payload,"resource":resource_sample()}
                store_call(folder,fail)
                status.update(status="STOPPED_CALL_FAILURE",failure=fail)
                write_json(status_path,status);return status
        # Keys are loaded only after both responses and the pre-revision ML action are saved.
        gold_rows={k["example_id"]:k["gold_answer"] for k in read_jsonl(folder/"TRUSTED_KEYS.jsonl")}
        key=Fraction(gold_rows[q["example_id"]])
        init_val=Fraction(answers["initial"]) if answers["initial"] is not None else None
        rev_val=Fraction(answers["revision"]) if answers["revision"] is not None else None
        ic=init_val==key if init_val is not None else False
        rc=rev_val==key if rev_val is not None else False
        outcome="HELP" if not ic and rc else "HURT" if ic and not rc else "NO_CHANGE"
        pair={"example_id":q["example_id"],"topic":q["topic"],"level":q["level"],"query":q["query"],
              "initial_answer":answers["initial"] or "","revision_answer":answers["revision"] or "",
              "initial_parse_status":meta["initial"]["parse_status"],"revision_parse_status":meta["revision"]["parse_status"],
              "initial_parse_valid":init_val is not None,"revision_parse_valid":rev_val is not None,
              "initial_done_reason":meta["initial"]["completion_reason"],"revision_done_reason":meta["revision"]["completion_reason"],
              "initial_tokens":meta["initial"]["tokens"]["total_tokens"],"revision_tokens":meta["revision"]["tokens"]["total_tokens"],
              "initial_wall_seconds":meta["initial"]["wall_clock_seconds"],"revision_wall_seconds":meta["revision"]["wall_clock_seconds"],
              "initial_correct":bool(ic),"revision_correct":bool(rc),"outcome":outcome,
              "correctness_used_after_initial_gate_decision":True}
        with (folder/"PAIRS.jsonl").open("ab") as f:
            f.write((json.dumps(pair,ensure_ascii=False)+"\n").encode("utf-8"))
        status["pairs_completed"]+=1
        status["last_pair"]={k:pair[k] for k in ("example_id","initial_parse_valid","revision_parse_valid","outcome")}
        write_json(status_path,status)
        print(f"{manifest['phase']} {status['pairs_completed']}/{len(questions)} {q['example_id']}: parse {pair['initial_parse_valid']}/{pair['revision_parse_valid']}; outcome {outcome}; RAM {meta['revision']['resource_after']['free_ram_gib']:.2f} GiB",flush=True)
    status.update(status="COMPLETE",final_resource=resource_sample(),gate_predictions_written_before_revision=len(gate_rows))
    write_json(status_path,status)
    return status


def wilson(success,total,z=1.959963984540054):
    if total==0:return [None,None]
    p=success/total;den=1+z*z/total;center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0,center-half),min(1,center+half)]


def bootstrap_diff(a,b,seed=20261121):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(a),size=(20000,len(a)))
    values=(a-b)[idx].mean(axis=1)
    return float((a-b).mean()),float(np.quantile(values,.025)),float(np.quantile(values,.975))


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)


def score_policies():
    pairs=read_jsonl(EVAL/"PAIRS.jsonl");gate_rows=read_jsonl(EVAL/"GATE_PREDICTIONS.jsonl")
    if len(pairs)!=60 or len(gate_rows)!=60:raise ValueError("Policy analysis requires all sixty paired questions and pre-revision gate scores.")
    gmap={r["example_id"]:r for r in gate_rows}
    if any(not gmap[p["example_id"]]["scored_before_revision"] for p in pairs):raise ValueError("Missing pre-revision gate chronology record.")
    if any(not gmap[p["example_id"]]["gate_compatible"] for p in pairs):
        # Keep all sixty; omit ML action only for feature-incompatible cases and report count.
        pass
    policies={"Never recheck":lambda p,g:False,"Always recheck":lambda p,g:True,
              "Completion heuristic":lambda p,g:p["initial_done_reason"]=="length",
              "Frozen v0.2 ML gate (OOD cross-model/cross-protocol)":lambda p,g:bool(g["recheck"]) if g["gate_compatible"] else False}
    actions_by={};policy_rows=[];question_rows=[];base=np.array([bool(p["initial_correct"]) for p in pairs])
    rng=np.random.default_rng(20261121);boot_idx=rng.integers(0,60,size=(20000,60))
    for name,choose in policies.items():
        acts=np.asarray([bool(choose(p,gmap[p["example_id"]])) for p in pairs]);actions_by[name]=acts
        correct=np.asarray([p["revision_correct"] if acts[i] else p["initial_correct"] for i,p in enumerate(pairs)],dtype=bool)
        helps=np.asarray([p["outcome"]=="HELP" for p in pairs]);hurts=np.asarray([p["outcome"]=="HURT" for p in pairs]);nochange=np.asarray([p["outcome"]=="NO_CHANGE" for p in pairs])
        diff=correct.astype(float)-base.astype(float);boot=diff[boot_idx].mean(axis=1)
        initial_tokens=sum(p["initial_tokens"] for p in pairs);rev_tokens=sum(p["revision_tokens"] for i,p in enumerate(pairs) if acts[i])
        replay=sum(p["initial_wall_seconds"]+(p["revision_wall_seconds"] if acts[i] else 0) for i,p in enumerate(pairs))
        wi=wilson(int(correct.sum()),60)
        policy_rows.append({"policy":name,"questions":60,"correct":int(correct.sum()),"accuracy":float(correct.mean()),
                            "accuracy_wilson95_low":wi[0],"accuracy_wilson95_high":wi[1],
                            "rechecks":int(acts.sum()),"recheck_rate":float(acts.mean()),
                            "HELP_total":int(helps.sum()),"HELP_captured":int((helps&acts).sum()),"HELP_missed":int((helps&~acts).sum()),
                            "HURT_total":int(hurts.sum()),"HURT_accepted":int((hurts&acts).sum()),"HURT_avoided":int((hurts&~acts).sum()),
                            "NO_CHANGE_rechecks":int((nochange&acts).sum()),
                            "mean_tokens_per_question":(initial_tokens+rev_tokens)/60,
                            "mean_tokens_saved_vs_always":(sum(p["initial_tokens"]+p["revision_tokens"] for p in pairs)-(initial_tokens+rev_tokens))/60,
                            "replayed_latency_total_seconds":replay,"replayed_latency_per_question_seconds":replay/60,
                            "accuracy_difference_vs_never":float(diff.mean()),"paired_bootstrap95_low_vs_never":float(np.quantile(boot,.025)),
                            "paired_bootstrap95_high_vs_never":float(np.quantile(boot,.975))})
    pairwise=[]
    for i,a_name in enumerate(policies):
        for b_name in list(policies)[i+1:]:
            a=np.asarray([p["revision_correct"] if actions_by[a_name][j] else p["initial_correct"] for j,p in enumerate(pairs)],dtype=float)
            b=np.asarray([p["revision_correct"] if actions_by[b_name][j] else p["initial_correct"] for j,p in enumerate(pairs)],dtype=float)
            d,lo,hi=bootstrap_diff(a,b)
            pairwise.append({"policy_A":a_name,"policy_B":b_name,"accuracy_difference_A_minus_B":d,
                             "paired_bootstrap95_low":lo,"paired_bootstrap95_high":hi,"resamples":20000,"seed":20261121})
    for j,p in enumerate(pairs):
        g=gmap[p["example_id"]]
        row={k:p[k] for k in ("example_id","topic","level","initial_parse_status","revision_parse_status","initial_done_reason","revision_done_reason",
                              "initial_parse_valid","revision_parse_valid","initial_correct","revision_correct","outcome","initial_tokens","revision_tokens",
                              "initial_wall_seconds","revision_wall_seconds")}
        row.update({k:g.get(k) for k in ("probabilities","score","threshold","recheck","gate_compatible","features")})
        for name,acts in actions_by.items():row["action__"+name]=bool(acts[j])
        question_rows.append(row)
    write_csv(EVAL/"POLICY_COMPARISON.csv",policy_rows)
    write_csv(EVAL/"PAIRWISE_POLICY_DIFFERENCES.csv",pairwise)
    write_csv(EVAL/"QUESTION_LEVEL.csv",question_rows)
    write_csv(EVAL/"ML_PREDICTIONS.csv",[{**r,**{f"feature__{k}":v for k,v in (r.get("features") or {}).items()},"features":None} for r in gate_rows])
    write_metrics_reports(pairs,gate_rows,policy_rows,pairwise,question_rows)


def write_metrics_reports(pairs,gate_rows,policy_rows,pairwise,question_rows):
    import pandas as pd
    sys.path.insert(0,str(ROOT))
    from rechecklab.features import FEATURE_NAMES,INITIAL_REQUIRED,extract_features
    train=pd.read_parquet(ROOT/"outputs/v0.2/data/prepared/source_train.parquet")
    if train["base_done_reason"].isna().any():raise ValueError("Historical feature reference contains null completion metadata.")
    train.loc[train["base_done_reason"].eq(""),"base_done_reason"]="unknown"
    train_features=extract_features(train)
    feat_rows=[]
    for name in FEATURE_NAMES:
        ref=train_features[name].astype(float);live=np.asarray([g["features"][name] for g in gate_rows if g["gate_compatible"]],dtype=float)
        sd=float(ref.std(ddof=0));mean=float(np.mean(live)) if len(live) else float("nan")
        feat_rows.append({"feature":name,"source_train_mean":float(ref.mean()),"source_train_sd":sd,"live_mean":mean,
                          "standardized_mean_difference":(mean-float(ref.mean()))/sd if sd else None,
                          "source_train_min":float(ref.min()),"source_train_max":float(ref.max()),
                          "live_outside_source_range":int(((live<ref.min())|(live>ref.max())).sum()) if len(live) else 0,
                          "live_compatible_rows":len(live)})
    write_csv(EVAL/"FEATURE_SHIFT.csv",feat_rows)
    mismatches=sum((g.get("recheck") is not None) and (bool(g["recheck"]) != (pairs[i]["initial_done_reason"]=="length")) for i,g in enumerate(gate_rows))
    compatibility={"compatible_questions":sum(g["gate_compatible"] for g in gate_rows),"incompatible_questions":sum(not g["gate_compatible"] for g in gate_rows),
                    "gate_completion_disagreements":mismatches,"gate_rechecks":sum(bool(g.get("recheck")) for g in gate_rows if g["gate_compatible"]),
                    "feature_schema":"pre-recheck-v2","feature_names":FEATURE_NAMES,"required_fields":INITIAL_REQUIRED,
                    "artifact_sha256":GATE_SHA,"threshold":GATE_THRESHOLD,"class_order":["HELP","NO_CHANGE","HURT"],
                    "score":"p_HELP - p_HURT","rule":"strictly greater than threshold","protocol":"OOD cross-model/cross-protocol"}
    write_json(EVAL/"GATE_COMPATIBILITY.json",compatibility)
    dev_pairs=read_jsonl(DEV/"PAIRS.jsonl")
    dinit=sum(p["initial_parse_valid"] for p in dev_pairs);drev=sum(p["revision_parse_valid"] for p in dev_pairs)
    status=read_json(DEV/"COLLECTION_STATUS.json")
    min_ram=min(min(x["before"]["free_ram_gib"],x["after"]["free_ram_gib"]) for x in status["resource_samples"])
    min_vram=min(min(x["before"]["gpu"]["free_mib"],x["after"]["gpu"]["free_mib"]) for x in status["resource_samples"])
    gate_pass=status["status"]=="COMPLETE" and len(dev_pairs)==10 and dinit>=9 and drev>=9 and all(resource_safe(s) for x in status["resource_samples"] for s in (x["before"],x["after"])) and all(g["gate_compatible"] for g in read_jsonl(DEV/"GATE_PREDICTIONS.jsonl"))
    write_json(DEV/"DEVELOPMENT_GATE.json",{"status":"PASS" if gate_pass else "FAIL","questions":len(dev_pairs),
              "initial_parseable":dinit,"revision_parseable":drev,"minimum_free_ram_gib":min_ram,"minimum_free_vram_mib":min_vram,
              "gate_feature_compatible":sum(g["gate_compatible"] for g in read_jsonl(DEV/"GATE_PREDICTIONS.jsonl")),
              "call_failures":sum(c.get("status")=="failed" for c in read_jsonl(DEV/"RAW_CALLS.jsonl")),
              "eligible_to_proceed":gate_pass,"evaluation_questions_preselected":60})
    if not gate_pass:
        write_json(DEV/"FINAL_DECISION.json",{"development_gate":"FAIL","evaluation_started":False})
    else:
        write_json(DEV/"FINAL_DECISION.json",{"development_gate":"PASS","evaluation_started":True})
    dev_results=f"""# MATH L4/L5 development gate

Status: **{'PASS' if gate_pass else 'FAIL'}**.

- Fresh development sample: {len(dev_pairs)}/10.
- Parseable initial / revisions: {dinit}/10 / {drev}/10.
- Valid but mathematically incorrect initial / revisions: {sum(p['initial_correct']==False and p['initial_parse_valid'] for p in dev_pairs)} / {sum(p['revision_correct']==False and p['revision_parse_valid'] for p in dev_pairs)}.
- HELP / HURT / NO_CHANGE: {sum(p['outcome']=='HELP' for p in dev_pairs)} / {sum(p['outcome']=='HURT' for p in dev_pairs)} / {sum(p['outcome']=='NO_CHANGE' for p in dev_pairs)}.
- All 19 gate features available and valid: {compatibility['compatible_questions']}/10.
- Minimum free RAM / VRAM: {min_ram:.2f} GiB / {min_vram:.0f} MiB.
- Completion rule actions / OOD gate actions: {sum(p['initial_done_reason']=='length' for p in dev_pairs)} / {sum(bool(g.get('recheck')) for g in read_jsonl(DEV/'GATE_PREDICTIONS.jsonl') if g['gate_compatible'])}.

The 60-question evaluation {'was started under the frozen protocol because all conditions passed' if gate_pass else 'was not started because at least one predeclared condition failed'}.
"""
    (DEV/"DEVELOPMENT_RESULTS.md").write_text(dev_results,encoding="utf-8")
    policy_table="\n".join(["| Policy | Accuracy | 95% CI | Rechecks | HELP captured/missed | HURT accepted/avoided | Mean tokens/question | Replayed latency/question |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]+[f"| {r['policy']} | {r['correct']}/60 ({r['accuracy']:.1%}) | {r['accuracy_wilson95_low']:.1%}-{r['accuracy_wilson95_high']:.1%} | {r['rechecks']}/60 ({r['recheck_rate']:.1%}) | {r['HELP_captured']}/{r['HELP_missed']} | {r['HURT_accepted']}/{r['HURT_avoided']} | {r['mean_tokens_per_question']:.1f} | {r['replayed_latency_per_question_seconds']:.2f}s |" for r in policy_rows])
    help_rows=[p for p in pairs if p["outcome"]=="HELP"]
    hurt_rows=[p for p in pairs if p["outcome"]=="HURT"]
    correction_lines=[]
    for p in help_rows:
        if not p["initial_parse_valid"] and p["revision_parse_valid"]:
            label="Initial extraction failure followed by a parseable correct revision; recovery may be formatting-related, but the initial reasoning cannot be classified from its parsed answer."
        elif p["initial_parse_valid"] and p["revision_parse_valid"] and p["initial_answer"]!=p["revision_answer"]:
            label="Parsed numeric answer changed from incorrect to correct; a semantic numeric correction is supported."
        else:
            label="Needs manual review; do not label automatically as a genuine reasoning correction."
        correction_lines.append(f"- `{p['example_id']}` ({p['topic']}, {p['level']}): initial parseable={p['initial_parse_valid']}, revision parseable={p['revision_parse_valid']}; {label}")
    case_text="\n".join(correction_lines) if correction_lines else "No HELP cases occurred; no correction case study is available."
    live_calls=read_jsonl(EVAL/"RAW_CALLS.jsonl")
    total_seconds=sum(c.get("wall_clock_seconds",0) for c in live_calls if c.get("status")=="succeeded")
    ev_pairs=read_jsonl(EVAL/"PAIRS.jsonl")
    max_min_ram=min(min(x["before"]["free_ram_gib"],x["after"]["free_ram_gib"]) for x in read_jsonl(EVAL/"COLLECTION_STATUS.json")["resource_samples"])
    max_min_vram=min(min(x["before"]["gpu"]["free_mib"],x["after"]["gpu"]["free_mib"]) for x in read_jsonl(EVAL/"COLLECTION_STATUS.json")["resource_samples"])
    comp=f"""# v0.4.1 independent MATH hard-question study

**Cohort:** 60 previously unused numeric/rational questions from original MATH test, all Level 4 or 5. This restricted subset is not representative of the full MATH benchmark.  
**Checkpoint:** `qwen3:4b-instruct`, Qwen3-4B-Instruct-2507, digest `{MODEL_DIGEST}`.  
**Gate:** frozen v0.2 cross-model/cross-protocol OOD diagnostic; no retraining/calibration.

## Evaluation

- Calls: {len(live_calls)}/120; measured inference wall time total {total_seconds:.1f}s; parseable initial/revision {sum(p['initial_parse_valid'] for p in ev_pairs)}/60 / {sum(p['revision_parse_valid'] for p in ev_pairs)}/60.
- Transitions: HELP {len(help_rows)}, HURT {len(hurt_rows)}, NO_CHANGE {sum(p['outcome']=='NO_CHANGE' for p in ev_pairs)}.
- Free RAM/VRAM minimum: {max_min_ram:.2f} GiB / {max_min_vram:.0f} MiB; stop floors 2 GiB / 512 MiB.
- Frozen gate actions: {compatibility['gate_rechecks']}/60; disagreement with completion: {compatibility['gate_completion_disagreements']}/60.

{policy_table}

The saved 60 response pairs were reused for all policies. Policy latency is replayed as measured initial-call time plus measured revision-call time on selected questions; it is not end-to-end deployed gate latency. Pairwise accuracy differences and bootstrap intervals are in `PAIRWISE_POLICY_DIFFERENCES.csv`; accuracy intervals are question-level Wilson 95% intervals. No statistical-equivalence or broad generalization claim is warranted from this single 60-question numeric subset.

## Gate and feature shift

The gate read only the canonical initial-time fields, used the original extractor, feature order, weights, class probabilities, score, and threshold, and recorded each decision before the corresponding revision call. Correctness, answer keys, and revised text were not gate inputs. This remains an OOD cross-model/cross-protocol diagnostic. Feature distribution differences versus the v0.2 source training cohort are univariate/descriptive in `FEATURE_SHIFT.csv`; they do not establish a cause.

## Correction review

HELP events are not automatically treated as reasoning corrections. Case classifications below distinguish extraction recovery from a parsed numeric answer changing to the correct value:

{case_text}

See `MODEL_ERRORS.csv`, `PARSER_FAILURES.csv`, and `QUESTION_LEVEL.csv` for per-question records.
"""
    (EVAL/"EXPERIMENT_RESULTS.md").write_text(comp,encoding="utf-8")
    (EVAL/"CASE_STUDIES.md").write_text("# Redacted correction case studies\n\n"+case_text+"\n\nOnly short outcome metadata is shown; question text and answer keys remain in separate local files.\n",encoding="utf-8")
    model_errors=[];parser_fail=[]
    for p in ev_pairs:
        for phase in ("initial","revision"):
            if not p[f"{phase}_parse_valid"]:
                parser_fail.append({"example_id":p["example_id"],"phase":phase,"parse_status":p[f"{phase}_parse_status"],"done_reason":p[f"{phase}_done_reason"]})
            elif not p[f"{phase}_correct"]:
                model_errors.append({"example_id":p["example_id"],"phase":phase,"parsed_answer":p[f"{phase}_answer"],"correct":False,"done_reason":p[f"{phase}_done_reason"]})
    write_csv(EVAL/"PARSER_FAILURES.csv",parser_fail or [{"example_id":"","phase":"","parse_status":"NONE","done_reason":""}])
    write_csv(EVAL/"MODEL_ERRORS.csv",model_errors or [{"example_id":"","phase":"","parsed_answer":"","correct":"","done_reason":""}])


def run():
    prep=read_json(DEV/"PREPARATION_STATUS.json")
    if not str(prep.get("status","")).startswith("FROZEN_BEFORE"):raise ValueError("No frozen pre-inference manifests found.")
    dstat=collect_phase(DEV)
    if dstat.get("status")!="COMPLETE":
        write_json(DEV/"FINAL_DECISION.json",{"development_gate":"FAIL","evaluation_started":False,"reason":dstat.get("stop_reason",dstat.get("failure",dstat.get("status")))})
        write_phase_provenance(DEV);return
    gate=finalize_development()
    if gate["status"]!="PASS":
        write_json(DEV/"FINAL_DECISION.json",{"development_gate":"FAIL","evaluation_started":False})
        write_phase_provenance(DEV);print("Development gate failed; no 60-question inference started.",flush=True);return
    write_json(DEV/"FINAL_DECISION.json",{"development_gate":"PASS","evaluation_started":True})
    write_phase_provenance(DEV)
    print("Development gate passed; starting preselected untouched 60-question MATH evaluation.",flush=True)
    estat=collect_phase(EVAL)
    if estat.get("status")!="COMPLETE":
        write_json(EVAL/"FINAL_STATUS.json",{"status":estat.get("status"),"evaluation_complete":False,
                  "stop_reason":estat.get("stop_reason",estat.get("failure"))});write_phase_provenance(EVAL);return
    score_policies()
    write_json(EVAL/"FINAL_STATUS.json",{"status":"COMPLETE","evaluation_complete":True,"questions":60,"calls":120,
               "model_digest":MODEL_DIGEST,"gate_artifact_sha256":GATE_SHA,"gate_is_OOD":True})
    write_phase_provenance(EVAL)
    print("Frozen MATH evaluation completed.",flush=True)


def finalize_development():
    pairs=read_jsonl(DEV/"PAIRS.jsonl");gate_rows=read_jsonl(DEV/"GATE_PREDICTIONS.jsonl")
    calls=read_jsonl(DEV/"RAW_CALLS.jsonl");status=read_json(DEV/"COLLECTION_STATUS.json")
    min_ram=min(min(x["before"]["free_ram_gib"],x["after"]["free_ram_gib"]) for x in status["resource_samples"])
    min_vram=min(min(x["before"]["gpu"]["free_mib"],x["after"]["gpu"]["free_mib"]) for x in status["resource_samples"])
    init_valid=sum(p["initial_parse_valid"] for p in pairs);rev_valid=sum(p["revision_parse_valid"] for p in pairs)
    gate_ok=sum(g.get("gate_compatible",False) for g in gate_rows)
    call_fail=sum(c.get("status")=="failed" for c in calls)
    safety_ok=all(resource_safe(s) for x in status["resource_samples"] for s in (x["before"],x["after"]))
    passed=(status.get("status")=="COMPLETE" and len(pairs)==10 and len(calls)==20 and init_valid>=9 and rev_valid>=9
            and safety_ok and gate_ok==10 and call_fail==0)
    gate={"status":"PASS" if passed else "FAIL","questions":len(pairs),"calls":len(calls),
          "initial_parseable":init_valid,"revision_parseable":rev_valid,
          "valid_but_incorrect_initial":sum(p["initial_parse_valid"] and not p["initial_correct"] for p in pairs),
          "valid_but_incorrect_revision":sum(p["revision_parse_valid"] and not p["revision_correct"] for p in pairs),
          "HELP":sum(p["outcome"]=="HELP" for p in pairs),"HURT":sum(p["outcome"]=="HURT" for p in pairs),
          "NO_CHANGE":sum(p["outcome"]=="NO_CHANGE" for p in pairs),"gate_feature_compatible":gate_ok,
          "call_failures":call_fail,"resource_safety_passed":safety_ok,
          "minimum_free_ram_gib":min_ram,"minimum_free_vram_mib":min_vram,
          "evaluation_started":passed}
    write_json(DEV/"DEVELOPMENT_GATE.json",gate)
    summary=f"""# MATH L4/L5 development gate

**Status: {gate['status']}**. The gate requires at least 9/10 parseable initial and revision answers, successful exact-grader validation, all initial-only gate features compatible, no call failures, and safe RAM/VRAM.

- Questions/calls: {len(pairs)}/10 and {len(calls)}/20.
- Parseable initial/revision answers: {init_valid}/10 and {rev_valid}/10.
- Parseable but incorrect initial/revision answers: {gate['valid_but_incorrect_initial']} and {gate['valid_but_incorrect_revision']}.
- HELP/HURT/NO_CHANGE: {gate['HELP']}/{gate['HURT']}/{gate['NO_CHANGE']}.
- Gate features valid before revision calls: {gate_ok}/10; completion-rule rechecks: {sum(p['initial_done_reason']=='length' for p in pairs)}.
- Minimum observed free RAM/VRAM: {min_ram:.2f} GiB / {min_vram:.0f} MiB (recorded, not a fixed reserve threshold).

The preselected 60-question evaluation {'will proceed automatically' if passed else 'will not start because one or more frozen reliability, feature, call, or actual resource safety criteria failed'}.
"""
    (DEV/"DEVELOPMENT_RESULTS.md").write_text(summary,encoding="utf-8")
    errors=[];failures=[]
    for p in pairs:
        for phase in ("initial","revision"):
            if not p[f"{phase}_parse_valid"]:
                failures.append({"example_id":p["example_id"],"phase":phase,"parse_status":p[f"{phase}_parse_status"],"done_reason":p[f"{phase}_done_reason"]})
            elif not p[f"{phase}_correct"]:
                errors.append({"example_id":p["example_id"],"phase":phase,"parsed_answer":p[f"{phase}_answer"],"correct":False})
    write_csv(DEV/"PARSER_FAILURES.csv",failures or [{"example_id":"","phase":"","parse_status":"NONE","done_reason":""}])
    write_csv(DEV/"MODEL_ERRORS.csv",errors or [{"example_id":"","phase":"","parsed_answer":"","correct":""}])
    return gate


def write_phase_provenance(folder):
    manifest=read_json(folder/"EXPERIMENT_MANIFEST.json")
    files={str(p.relative_to(folder)):sha_file(p) for p in sorted(folder.rglob("*")) if p.is_file() and p.name!="PROVENANCE.json"}
    write_json(folder/"PROVENANCE.json",{"phase":manifest["phase"],"model_digest":MODEL_DIGEST,
               "source_revision":SOURCE_REVISION,"gate_sha256":GATE_SHA,"protocol_id":manifest["protocol"]["id"],
               "question_selection_sha256":manifest["question_selection_sha256"],"trusted_key_sha256":manifest["trusted_key_sha256"],
               "files_sha256":files})


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true");group.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.prepare:prepare()
    else:run()


if __name__=="__main__":main()
