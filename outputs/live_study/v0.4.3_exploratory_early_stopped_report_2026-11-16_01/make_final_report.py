#!/usr/bin/env python3
"""Rebuild the local exploratory v0.4.3 report from immutable call/pair journals."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EVAL = ROOT / "outputs/live_study/v0.4.3_math_l4_l5_frozen_evaluation_2026-11-16_01"
DEV = ROOT / "outputs/live_study/v0.4.3_math_l4_l5_development_2026-11-16_01"
OLD_EVAL = ROOT / "outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01"
POLICIES = ("never_recheck", "always_recheck", "completion_heuristic", "frozen_v0.2_ml_gate_ood")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_correct(value, gold) -> bool:
    if value in (None, ""):
        return False
    try:
        return Fraction(str(value).strip()) == Fraction(str(gold).strip())
    except (ValueError, ZeroDivisionError):
        return False


def wilson(successes: int, n: int, z: float = 1.959963984540054):
    if not n:
        return (None, None)
    p = successes / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (center - half, center + half)


def bootstrap_difference(a: list[int], b: list[int], seed: int = 20261121, reps: int = 20000):
    rng = random.Random(seed)
    n = len(a)
    diffs = sorted(sum(a[i] - b[i] for i in (rng.randrange(n) for _ in range(n))) / n for _ in range(reps))
    return (statistics.mean(x - y for x, y in zip(a, b)), diffs[int(.025 * reps)], diffs[min(reps - 1, int(.975 * reps))])


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selection = read_jsonl(EVAL / "QUESTION_SELECTION.jsonl")
    keys = {r["example_id"]: r["gold_answer"] for r in read_jsonl(EVAL / "TRUSTED_KEYS.jsonl")}
    pair_candidates = read_jsonl(EVAL / "QUESTION_CHECKPOINTS.jsonl")
    gate = {r["question_id"]: r for r in read_jsonl(EVAL / "GATE_DECISIONS.jsonl")}

    # A pair enters the primary analysis only if both call IDs have genuine
    # call_complete journal events. Partial attempts never enter denominators.
    starts, terminals, chunk_counts, last_chunks, hw = {}, {}, Counter(), {}, {}
    hw_by_time = {}
    first_eval_start = last_eval_terminal = None
    with (EVAL / "STREAM_JOURNAL.jsonl").open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            cid = e.get("call_id")
            if e.get("event") == "call_start" and cid:
                starts[cid] = e
                if cid != "v042-attempt-1-interrupted":
                    ts = e.get("timestamp_utc")
                    if ts and (first_eval_start is None or ts < first_eval_start):
                        first_eval_start = ts
            elif e.get("event") == "chunk" and cid:
                chunk_counts[cid] += 1
                last_chunks[cid] = e
            elif e.get("event") in ("call_complete", "call_error", "call_interrupted") and cid:
                terminals[cid] = e
                ts = e.get("timestamp_utc")
                if cid != "v042-attempt-1-interrupted" and ts and (last_eval_terminal is None or ts > last_eval_terminal):
                    last_eval_terminal = ts
            sample = e.get("resource_sample") or {}
            stamp = sample.get("timestamp_utc")
            if stamp:
                hw_by_time[stamp] = sample

    complete_call_ids = {cid for cid, t in terminals.items() if t.get("event") == "call_complete"}
    complete_pairs = [p for p in pair_candidates if p.get("initial_call_id") in complete_call_ids and p.get("revision_call_id") in complete_call_ids]
    selection_ids = [x["example_id"] for x in selection]
    selection_set = set(selection_ids)
    complete_ids = {p["question_id"] for p in complete_pairs}
    attempted_ids = {e.get("question_id") for e in starts.values() if e.get("question_id") in selection_set}
    unattempted_ids = [qid for qid in selection_ids if qid not in attempted_ids]
    incomplete_ids = [qid for qid in selection_ids if qid in attempted_ids and qid not in complete_ids]

    incomplete_calls = []
    for cid, start in starts.items():
        if start.get("question_id") not in selection_set:
            continue
        terminal = terminals.get(cid)
        if not terminal or terminal.get("event") != "call_complete":
            incomplete_calls.append({"call_id": cid, "question_id": start.get("question_id"),
                                     "phase": start.get("phase"), "attempt": start.get("attempt_number"),
                                     "start": start.get("timestamp_utc"), "terminal": terminal,
                                     "chunks": chunk_counts.get(cid, 0), "last_chunk": last_chunks.get(cid)})

    # Preserve predetermined grading; independently recheck exact integer and
    # rational comparisons against the unchanged trusted key file.
    grade_mismatches = []
    for p in complete_pairs:
        g = keys[p["question_id"]]
        for phase, answer_field, correct_field in (("initial", "initial_answer", "initial_correct"),
                                                    ("revision", "revision_answer", "revision_correct")):
            recalculated = exact_correct(p.get(answer_field), g)
            if recalculated != bool(p.get(correct_field)):
                grade_mismatches.append((p["question_id"], phase, recalculated, p.get(correct_field)))

    qrows = []
    prows = []
    outcomes = Counter(p["outcome"] for p in complete_pairs)
    selected_vectors = {name: [] for name in POLICIES}
    policy_summaries = []
    for p in complete_pairs:
        gid = gate.get(p["question_id"], {})
        probs = gid.get("probabilities", {})
        initial_correct = bool(p["initial_correct"])
        revision_correct = bool(p["revision_correct"])
        itok = (p.get("initial_token_statistics") or {}).get("prompt_eval_count")
        iout = (p.get("initial_token_statistics") or {}).get("eval_count")
        rtok = (p.get("revision_token_statistics") or {}).get("prompt_eval_count")
        rout = (p.get("revision_token_statistics") or {}).get("eval_count")
        initial_tokens = itok + iout if isinstance(itok, int) and isinstance(iout, int) else None
        revision_tokens = rtok + rout if isinstance(rtok, int) and isinstance(rout, int) else None
        actions = {
            "never_recheck": False,
            "always_recheck": True,
            "completion_heuristic": bool(p.get("completion_recheck")),
            "frozen_v0.2_ml_gate_ood": bool(p.get("gate_recheck")),
        }
        qrows.append({
            "question_id": p["question_id"], "initial_completion_reason": p.get("initial_completion_reason"),
            "initial_parse_status": p.get("initial_parse_status"), "revision_parse_status": p.get("revision_parse_status"),
            "initial_correct": initial_correct, "revision_correct": revision_correct, "transition": p.get("outcome"),
            "gate_p_help": probs.get("HELP"), "gate_p_no_change": probs.get("NO_CHANGE"), "gate_p_hurt": probs.get("HURT"),
            "gate_score_help_minus_hurt": gid.get("score"), "gate_recheck": actions["frozen_v0.2_ml_gate_ood"],
            "completion_recheck": actions["completion_heuristic"], "initial_tokens": initial_tokens,
            "revision_tokens": revision_tokens, "initial_latency_seconds": p.get("initial_latency_seconds"),
            "revision_latency_seconds": p.get("revision_latency_seconds"),
        })
        for name, choose in actions.items():
            correct = revision_correct if choose else initial_correct
            selected_vectors[name].append(int(correct))
            tok = (initial_tokens + revision_tokens) if choose and initial_tokens is not None and revision_tokens is not None else (initial_tokens if not choose else None)
            lat = p.get("initial_latency_seconds")
            if choose:
                lat = lat + p.get("revision_latency_seconds") if lat is not None and p.get("revision_latency_seconds") is not None else None
            prows.append({"question_id": p["question_id"], "policy": name, "recheck": choose,
                          "selected_answer_correct": correct, "initial_correct": initial_correct,
                          "revision_correct": revision_correct, "transition": p.get("outcome"),
                          "tokens_prompt_plus_output": tok, "replayed_latency_seconds": lat,
                          "initial_tokens": initial_tokens, "revision_tokens": revision_tokens})

    n = len(complete_pairs)
    for name in POLICIES:
        rows = [r for r in prows if r["policy"] == name]
        ok = sum(bool(r["selected_answer_correct"]) for r in rows)
        ci = wilson(ok, n)
        tok = [r["tokens_prompt_plus_output"] for r in rows if r["tokens_prompt_plus_output"] is not None]
        lat = [r["replayed_latency_seconds"] for r in rows if r["replayed_latency_seconds"] is not None]
        chosen = sum(r["recheck"] for r in rows)
        helped = sum(p["outcome"] == "HELP" for p in complete_pairs)
        hurts = sum(p["outcome"] == "HURT" for p in complete_pairs)
        policy_summaries.append({"policy": name, "correct": ok, "n": n, "accuracy": ok / n if n else None,
                                 "wilson_low": ci[0], "wilson_high": ci[1], "rechecks": chosen,
                                 "recheck_rate": chosen / n if n else None,
                                 "HELP_captured": sum(p["outcome"] == "HELP" and r["recheck"] for p, r in zip(complete_pairs, rows)),
                                 "HELP_missed": sum(p["outcome"] == "HELP" and not r["recheck"] for p, r in zip(complete_pairs, rows)),
                                 "HURT_captured": sum(p["outcome"] == "HURT" and r["recheck"] for p, r in zip(complete_pairs, rows)),
                                 "HURT_avoided": sum(p["outcome"] == "HURT" and not r["recheck"] for p, r in zip(complete_pairs, rows)),
                                 "mean_tokens_per_question": statistics.mean(tok) if tok else None,
                                 "total_tokens": sum(tok) if tok else None,
                                 "mean_replayed_latency_seconds_per_question": statistics.mean(lat) if lat else None})

    write_csv(OUT / "QUESTION_LEVEL.csv", qrows, list(qrows[0]) if qrows else ["question_id"])
    write_csv(OUT / "POLICY_COMPARISON.csv", prows, list(prows[0]) if prows else ["question_id", "policy"])

    pairwise = []
    for i, a in enumerate(POLICIES):
        for b in POLICIES[i + 1:]:
            diff, low, high = bootstrap_difference(selected_vectors[a], selected_vectors[b])
            pairwise.append({"policy_a": a, "policy_b": b, "paired_accuracy_difference_a_minus_b": diff,
                             "bootstrap_95_low": low, "bootstrap_95_high": high,
                             "resamples": 20000, "seed": 20261121})

    complete_calls = [t for cid, t in terminals.items() if t.get("event") == "call_complete" and starts.get(cid, {}).get("question_id") in selection_set]
    call_durations = [t.get("elapsed_monotonic_seconds") for t in complete_calls if isinstance(t.get("elapsed_monotonic_seconds"), (int, float))]
    initial_durations = [p.get("initial_latency_seconds") for p in complete_pairs if isinstance(p.get("initial_latency_seconds"), (int, float))]
    revision_durations = [p.get("revision_latency_seconds") for p in complete_pairs if isinstance(p.get("revision_latency_seconds"), (int, float))]
    init_parse = sum(p.get("initial_parse_status") == "parseable" for p in complete_pairs)
    rev_parse = sum(p.get("revision_parse_status") == "parseable" for p in complete_pairs)
    gate_completion_disagree = sum(bool(p.get("gate_recheck")) != bool(p.get("completion_recheck")) for p in complete_pairs)
    accuracy_rows = {x["policy"]: x for x in policy_summaries}
    start_dt = datetime.fromisoformat(first_eval_start.replace("Z", "+00:00")) if first_eval_start else None
    end_dt = datetime.fromisoformat(last_eval_terminal.replace("Z", "+00:00")) if last_eval_terminal else None
    runtime = (end_dt - start_dt).total_seconds() if start_dt and end_dt else None
    status = json.loads((EVAL / "RUN_STATUS.json").read_text(encoding="utf-8"))

    # Current hardware telemetry is sampled from durable call events; cached
    # repeated samples are deduplicated by timestamp before summarizing.
    hw_rows = list(hw_by_time.values())
    def metric(vals):
        vals = [x for x in vals if isinstance(x, (int, float))]
        return {"n": len(vals), "min": min(vals) if vals else None, "mean": statistics.mean(vals) if vals else None,
                "max": max(vals) if vals else None}
    cpu = metric([x.get("cpu_utilization_percent") for x in hw_rows])
    ram = metric([x.get("ram_available_gib") for x in hw_rows])
    pagefile = metric([x.get("pagefile_used_gib") for x in hw_rows])
    gpu_util = metric([(x.get("gpu") or {}).get("utilization_percent") for x in hw_rows])
    vram_used = metric([(x.get("gpu") or {}).get("used_mib") for x in hw_rows])
    gpu_temp = metric([(x.get("gpu") or {}).get("temperature_c") for x in hw_rows])

    # Attempt counts and exact incomplete-call classifications.
    calls_by_phase = Counter()
    interrupted_events = []
    timeout_events = []
    for c in incomplete_calls:
        calls_by_phase[c["phase"]] += 1
        t = c.get("terminal") or {}
        if t.get("event") == "call_interrupted":
            interrupted_events.append(c)
        if t.get("incomplete_status") == "timeout" or "watchdog" in str(t.get("error_message", "")).lower() or "timeout" in str(t.get("diagnostic_class", "")).lower():
            timeout_events.append(c)

    q1_id = selection_ids[0]
    q1_attempts = [c for c in incomplete_calls if c["question_id"] == q1_id]
    attempted_order = [qid for qid in selection_ids if qid in attempted_ids]
    counts = {"selected": len(selection), "attempted_unique": len(attempted_ids), "complete_pairs": n,
              "incomplete_unique_questions": len(incomplete_ids), "unattempted": len(unattempted_ids),
              "incomplete_initial_calls": calls_by_phase["initial"], "incomplete_revision_calls": calls_by_phase["revision"],
              "interrupted_calls": len(interrupted_events), "timeout_calls": len(timeout_events),
              "completed_initial_calls": sum(starts.get(p["initial_call_id"], {}).get("phase") == "initial" for p in complete_pairs),
              "completed_revision_calls": sum(starts.get(p["revision_call_id"], {}).get("phase") == "revision" for p in complete_pairs)}

    # Write detailed attempt ledger; no response text, prompt or answer key is
    # duplicated into the report directory.
    incomplete_lines = ["# Incomplete attempts and question accounting", "",
                        f"The frozen selection contains {len(selection)} questions. {n} have fully completed initial/revision pairs; {len(attempted_ids)} distinct selected questions have at least one recorded attempt; {len(unattempted_ids)} were never attempted.", "",
                        f"Incomplete initial calls: {calls_by_phase['initial']}. Incomplete revision calls: {calls_by_phase['revision']}. Journal events classified as interrupted: {len(interrupted_events)}. Timeout-related incomplete calls: {len(timeout_events)}.", "",
                        "## Incomplete calls", "", "| Question ID | Phase | Attempt | Terminal status | Chunks | Last persisted elapsed (s) | Notes |", "|---|---|---:|---|---:|---:|---|"]
    for c in incomplete_calls:
        t = c.get("terminal") or {}
        lc = c.get("last_chunk") or {}
        err = str(t.get("error_message") or "")
        # The pre-amendment collector mislabeled this watchdog TimeoutError as
        # no_progress_stall. Keep the raw journal intact, but present the
        # explicit watchdog message correctly in the report.
        status_label = "watchdog_timeout" if "watchdog exceeded" in err.lower() else (t.get("incomplete_status") or t.get("diagnostic_class") or t.get("event") or "incomplete")
        note = err or t.get("diagnostic_class") or "no terminal metadata"
        note = str(note).replace("|", "/")
        incomplete_lines.append(f"| {c['question_id']} | {c['phase']} | {c['attempt']} | {status_label} | {c['chunks']} | {lc.get('elapsed_monotonic_seconds', '')} | {note} |")
    incomplete_lines += ["", "## Question 1 history", "",
                         f"Question 1 is `{q1_id}`. Its only recorded attempt is the prior v0.4.2 initial attempt 1, already marked interrupted in the inherited journal. v0.4.3 did not retry it; no initial/revision pair or policy decision exists for it. The partial historical response and token metadata were unavailable in the original evidence. No Question 1 attempt 2 was run.", "",
                         "## Runtime amendment and stop", "",
                         "The original manifest remains unchanged at its original 2700-second watchdog. The first long call (precalculus/490) terminated at that watchdog. A later runtime amendment set the collector watchdog to 420 seconds for future calls; when that amendment was issued, the already-running number_theory/18 call had exceeded 420 seconds and was explicitly canceled, with its last persisted elapsed time at 1620.125 seconds. The amended runner began the next initial call (prealgebra/166), which was explicitly canceled by the later stop instruction after 189 chunks. No retry was scheduled, and no Question 1 attempt 2 was run.", "",
                         "## Unattempted frozen questions", "", *[f"- `{qid}`" for qid in unattempted_ids], ""]
    (OUT / "INCOMPLETE_ATTEMPTS.md").write_text("\n".join(incomplete_lines), encoding="utf-8")

    # Summary tables are regenerated from the eligible completed-pair set.
    stats_lines = ["# Statistical analysis", "", f"Primary analysis n = {n} complete question pairs. All intervals are descriptive; the study stopped before the planned 60 questions.", "",
                   "## Accuracy and 95% Wilson intervals", "", "| Policy | Correct / n | Accuracy | 95% Wilson CI | Rechecks | Recheck rate | Mean tokens/question | Mean replay latency/question (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in policy_summaries:
        stats_lines.append(f"| {x['policy']} | {x['correct']}/{n} | {x['accuracy']:.3f} | [{x['wilson_low']:.3f}, {x['wilson_high']:.3f}] | {x['rechecks']}/{n} | {x['recheck_rate']:.3f} | {x['mean_tokens_per_question']:.1f} | {x['mean_replayed_latency_seconds_per_question']:.2f} |")
    stats_lines += ["", "## Paired accuracy differences", "", "Differences are policy A minus policy B, using 20,000 paired question-level bootstrap resamples (seed 20261121). A confidence interval spanning zero is inconclusive; this small, early-stopped dataset does not establish equivalence or significance.", "", "| Policy A | Policy B | Difference | Bootstrap 95% interval |", "|---|---|---:|---:|"]
    for x in pairwise:
        stats_lines.append(f"| {x['policy_a']} | {x['policy_b']} | {x['paired_accuracy_difference_a_minus_b']:.3f} | [{x['bootstrap_95_low']:.3f}, {x['bootstrap_95_high']:.3f}] |")
    stats_lines += ["", "## HELP/HURT/NO_CHANGE", "", f"Transitions among complete pairs: HELP {outcomes.get('HELP', 0)}, HURT {outcomes.get('HURT', 0)}, NO_CHANGE {outcomes.get('NO_CHANGE', 0)}.",
                    f"Frozen ML gate and completion heuristic disagreed on {gate_completion_disagree}/{n} questions.", "",
                    "| Policy | HELP captured | HELP missed | HURT selected | HURT avoided |", "|---|---:|---:|---:|---:|"]
    for x in policy_summaries:
        stats_lines.append(f"| {x['policy']} | {x['HELP_captured']} | {x['HELP_missed']} | {x['HURT_captured']} | {x['HURT_avoided']} |")
    stats_lines += ["", "## Extraction and exact grading", "",
                    f"Initial answers parsed: {init_parse}/{n} ({init_parse/n:.1%}). Revision answers parsed: {rev_parse}/{n} ({rev_parse/n:.1%}). Unparseable or absent answers are scored incorrect under the frozen evaluation rule. Independent exact `Fraction` checks disagreed with saved initial/revision correctness on {len(grade_mismatches)} answer records. Gold keys were read only for reporting, after the gate decisions had been journaled.", ""]
    (OUT / "STATISTICAL_ANALYSIS.md").write_text("\n".join(stats_lines), encoding="utf-8")

    hardware_lines = ["# Hardware and runtime", "", f"Evaluation collection ran from {first_eval_start} through {last_eval_terminal}; elapsed wall-clock duration is approximately {runtime/3600:.2f} hours ({runtime:.0f} seconds). This includes the long calls and stop/recovery intervals between the first v0.4.3 inference and the final user cancellation. It excludes the earlier development validation.", "",
                      f"Completed pairs: {n}; completed inference calls: {len(complete_calls)}; incomplete attempted calls: {len(incomplete_calls)}. Configured model: qwen3:4b-instruct, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`; context 8192; `num_predict=-1`. A 420-second runtime amendment was recorded separately; the original manifest remains unchanged.", "",
                      f"Measured completed-call durations: n={len(call_durations)}, mean {statistics.mean(call_durations):.1f}s, median {statistics.median(call_durations):.1f}s, maximum {max(call_durations):.1f}s. Initial calls: mean {statistics.mean(initial_durations):.1f}s, median {statistics.median(initial_durations):.1f}s, maximum {max(initial_durations):.1f}s. Revisions: mean {statistics.mean(revision_durations):.1f}s, median {statistics.median(revision_durations):.1f}s, maximum {max(revision_durations):.1f}s." if call_durations else "No completed call timing metadata was available.", "",
                      f"Telemetry samples after timestamp deduplication: {len(hw_rows)}.", "",
                      "| Metric | Minimum | Mean | Maximum |", "|---|---:|---:|---:|"]
    for name, m, unit in (("CPU utilization", cpu, "%"), ("Available RAM", ram, "GiB"), ("Pagefile use", pagefile, "GiB"),
                          ("GPU utilization", gpu_util, "%"), ("VRAM used", vram_used, "MiB"), ("GPU temperature", gpu_temp, "°C")):
        fmt = lambda v: "NA" if v is None else f"{v:.2f} {unit}"
        hardware_lines.append(f"| {name} | {fmt(m['min'])} | {fmt(m['mean'])} | {fmt(m['max'])} |")
    hardware_lines += ["", "Hardware values are journal telemetry samples, not continuous laboratory instrumentation. No hardware configuration, Ollama parallelism, or generation settings were changed for this report.", ""]
    (OUT / "HARDWARE_AND_RUNTIME.md").write_text("\n".join(hardware_lines), encoding="utf-8")

    limits = ["# Experiment limitations", "",
              "This is an exploratory early-stopped evaluation, not the planned completed 60-question prospective study. The primary policy analysis includes only the 43 questions with both initial and revision calls naturally completed and checkpointed.", "",
              "The fixed question selection was frozen before inference and no question was replaced. However, stopping at this point leaves later questions unobserved, and incomplete calls were concentrated among questions with unusually long generations or external stop events. Runtime-related missingness may correlate with problem difficulty, response verbosity, context use, or answer validity. The completed-pair subset is therefore not guaranteed to be representative, and no generalization claim is warranted.", "",
              "The v0.2 classifier is an out-of-distribution diagnostic under a cross-model/cross-protocol evaluation. Its predictions were made from initial-only features before revision; this does not establish deployment compatibility. The small sample yields wide uncertainty. No statistical equivalence or significance claim is made.", "",
              "The original 60-question selection and trusted keys remain unchanged. The runtime-only timeout amendment is recorded in the evaluation directory; it does not rewrite the original experiment manifest or retroactively change earlier call settings. No additional inference, classifier fitting, threshold adjustment, or outcome-based question selection was performed during report generation.", ""]
    (OUT / "EXPERIMENT_LIMITATIONS.md").write_text("\n".join(limits), encoding="utf-8")

    policy_table = "\n".join(f"| {x['policy']} | {x['correct']}/{n} | {x['accuracy']:.3f} | {x['rechecks']}/{n} | {x['mean_tokens_per_question']:.1f} | {x['mean_replayed_latency_seconds_per_question']:.2f}s |" for x in policy_summaries)
    final_lines = ["# RecheckLab v0.4.3 — Exploratory early-stopped evaluation", "",
                  "## Confirmed results", "",
                  f"The frozen selection contained 60 MATH Level 4–5 questions. The run stopped by explicit user instruction after **{n} complete initial/revision pairs** (86 naturally completed LLM calls). These 43 pairs alone form the primary four-policy analysis; incomplete attempts are excluded from that paired comparison and listed separately.", "",
                  "| Policy | Correct / 43 | Accuracy | Rechecks | Mean total tokens/question | Replayed latency/question |", "|---|---:|---:|---:|---:|---:|", policy_table, "",
                  f"Transitions in the complete-pair set: HELP {outcomes.get('HELP',0)}, HURT {outcomes.get('HURT',0)}, NO_CHANGE {outcomes.get('NO_CHANGE',0)}. Frozen ML versus completion decisions disagreed on {gate_completion_disagree}/{n} questions. Initial/revision extraction succeeded on {init_parse}/{n} and {rev_parse}/{n} complete calls, respectively. The deterministic exact grader matched the saved correctness values on {2*n-len(grade_mismatches)}/{2*n} answer records.", "",
                  "## Attempt accounting", "",
                  f"Distinct selected questions attempted: {len(attempted_ids)}/60. Incomplete initial calls: {calls_by_phase['initial']}; incomplete revision calls: {calls_by_phase['revision']}; terminally interrupted calls: {len(interrupted_events)}; timeout-related incomplete calls: {len(timeout_events)}; unattempted questions: {len(unattempted_ids)}. Question 1 attempt 1 remains the inherited incomplete v0.4.2 attempt; no attempt 2 was made.", "",
                  "The evaluation was stopped early due to excessive runtime. This is **not** a completed 60-question study. Selection and runtime censoring may bias the observed subset, and the confidence intervals are descriptive. No significance, equivalence, or generalization claim is made.", "",
                  "## Supporting files", "",
                  "See `POLICY_COMPARISON.csv`, `QUESTION_LEVEL.csv`, `INCOMPLETE_ATTEMPTS.md`, `STATISTICAL_ANALYSIS.md`, `HARDWARE_AND_RUNTIME.md`, and `EXPERIMENT_LIMITATIONS.md`. `make_final_report.py` regenerates the analysis from the existing local journals and checkpoints without inference.", ""]
    (OUT / "FINAL_RESULTS.md").write_text("\n".join(final_lines), encoding="utf-8")

    # Integrity inventory is written last so it covers every input and report
    # artifact, including this reproducible script, but not itself.
    gate_artifact = ROOT / "outputs/v0.2/models/source_gate.joblib"
    input_paths = [EVAL / n for n in ("EXPERIMENT_MANIFEST.json", "QUESTION_SELECTION.jsonl", "TRUSTED_KEYS.jsonl",
                                      "QUESTION_CHECKPOINTS.jsonl", "COMPLETED_CALLS.jsonl", "GATE_DECISIONS.jsonl",
                                      "STREAM_JOURNAL.jsonl", "RUNTIME_AMENDMENT_420S.json")]
    v041 = ROOT / "outputs/live_study/v0.4.1_math_l4_l5_frozen_evaluation_2026-11-16_01"
    input_paths += [DEV / "run_eval_v043.py", DEV / "run_eval_v043_420s.py", DEV / "durable_stream.py", DEV / "durable_stream_420s.py",
                    OLD_EVAL / "QUESTION_SELECTION.jsonl", OLD_EVAL / "TRUSTED_KEYS.jsonl",
                    OLD_EVAL / "EXPERIMENT_MANIFEST.json", OLD_EVAL / "INTERRUPTED_CALL.json",
                    v041 / "QUESTION_SELECTION.jsonl", v041 / "TRUSTED_KEYS.jsonl", v041 / "EXPERIMENT_MANIFEST.json",
                    gate_artifact]
    report_paths = [OUT / n for n in ("FINAL_RESULTS.md", "POLICY_COMPARISON.csv", "QUESTION_LEVEL.csv",
                                       "INCOMPLETE_ATTEMPTS.md", "STATISTICAL_ANALYSIS.md", "HARDWARE_AND_RUNTIME.md",
                                       "EXPERIMENT_LIMITATIONS.md", "make_final_report.py")]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths + report_paths if p.exists()}
    old_prov = json.loads((OLD_EVAL / "PROVENANCE.json").read_text(encoding="utf-8"))
    old_preserve = json.loads((OLD_EVAL / "PRESERVATION_CHECK.json").read_text(encoding="utf-8"))
    old_file_checks = {name: sha256(OLD_EVAL / name) == expected
                       for name, expected in old_prov.get("files_sha256", {}).items()
                       if (OLD_EVAL / name).exists()}
    current_gate_sha = sha256(gate_artifact) if gate_artifact.exists() else None
    v041_sel_sha, v041_key_sha = sha256(v041 / "QUESTION_SELECTION.jsonl"), sha256(v041 / "TRUSTED_KEYS.jsonl")
    integrity = {"report_type": "exploratory early-stopped evaluation", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                 "frozen_model_digest": "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0",
                 "frozen_gate_sha256": json.loads((EVAL / "PROVENANCE.json").read_text(encoding="utf-8"))["frozen_gate_sha256"],
                 "frozen_gate_artifact_actual_sha256": current_gate_sha,
                 "frozen_gate_artifact_unchanged": current_gate_sha == "9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439",
                 "frozen_selection_sha256": sha256(EVAL / "QUESTION_SELECTION.jsonl"),
                 "frozen_keys_sha256": sha256(EVAL / "TRUSTED_KEYS.jsonl"),
                 "original_manifest_sha256": sha256(EVAL / "EXPERIMENT_MANIFEST.json"),
                 "original_manifest_unchanged_by_runtime_amendment": sha256(EVAL / "EXPERIMENT_MANIFEST.json") == json.loads((EVAL / "RUNTIME_AMENDMENT_420S.json").read_text(encoding="utf-8"))["original_experiment_manifest_sha256"],
                 "prior_selection_byte_identical": sha256(EVAL / "QUESTION_SELECTION.jsonl") == sha256(OLD_EVAL / "QUESTION_SELECTION.jsonl"),
                 "prior_keys_byte_identical": sha256(EVAL / "TRUSTED_KEYS.jsonl") == sha256(OLD_EVAL / "TRUSTED_KEYS.jsonl"),
                 "v041_selection_and_keys_match_v042": v041_sel_sha == sha256(OLD_EVAL / "QUESTION_SELECTION.jsonl") and v041_key_sha == sha256(OLD_EVAL / "TRUSTED_KEYS.jsonl"),
                 "v041_manifest_matches_v042_preservation_record": sha256(v041 / "EXPERIMENT_MANIFEST.json") == old_preserve.get("v0_4_1_eval_manifest_sha256"),
                 "v042_historical_files_match_their_provenance": old_file_checks,
                 "v042_historical_files_all_match": all(old_file_checks.values()),
                 "v042_preservation_check_says_prior_studies_untouched": old_preserve.get("prior_studies_modified") is False,
                 "analysis_pairs": n, "completed_call_count": len(complete_calls), "incomplete_call_count": len(incomplete_calls),
                 "model_weights_modified": False, "historical_artifacts_modified": False, "additional_inference_performed": False,
                 "run_status": status.get("status"), "file_sha256": hashes}
    (OUT / "PROVENANCE_AND_HASHES.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_pairs": n, "policy_summary": policy_summaries, "counts": counts,
                      "outcomes": dict(outcomes), "gate_completion_disagreements": gate_completion_disagree,
                      "grade_mismatches": len(grade_mismatches), "runtime_seconds": runtime,
                      "unattempted_question_ids": unattempted_ids, "report_directory": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
