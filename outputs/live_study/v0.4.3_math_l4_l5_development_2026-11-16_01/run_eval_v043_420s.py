"""Resumable v0.4.3 evaluation driver. Not run automatically."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, pathlib, re, socket, statistics, subprocess, sys, time, urllib.error, urllib.request
from fractions import Fraction

DEV=pathlib.Path(__file__).resolve().parent; ROOT=DEV.parents[2]
EVAL=ROOT/"outputs/live_study/v0.4.3_math_l4_l5_frozen_evaluation_2026-11-16_01"
sys.path.insert(0,str(DEV))
from durable_stream_420s import DurableStreamCollector,JsonlJournal,reconstruct_calls,next_attempt,utc_now
sys.path.insert(0,str(ROOT))
import run_math_study_reference as ref

MODEL=ref.MODEL; DIGEST=ref.MODEL_DIGEST; GATE_SHA=ref.GATE_SHA
FIRST_QID="math:test:intermediate_algebra:intermediate_algebra/823"
OLD_STOP=ROOT/"outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01/INTERRUPTED_CALL.json"
JOURNAL=EVAL/"STREAM_JOURNAL.jsonl"; CHECKPOINTS=EVAL/"COMPLETED_CALLS.jsonl"; PAIRS=EVAL/"QUESTION_CHECKPOINTS.jsonl"; GATELOG=EVAL/"GATE_DECISIONS.jsonl"
INITIAL=ref.INITIAL_PROMPT; REVISION=ref.REVISION_PROMPT
class ResourceSampler:
    """Cache OS/GPU telemetry for five seconds to avoid a subprocess per token."""
    def __init__(self): self.last=0;self.value={}
    def __call__(self):
        now=time.monotonic()
        if now-self.last<5:return self.value
        vm=ref.psutil.virtual_memory();sw=ref.psutil.swap_memory()
        gpu={}
        try:
            line=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu","--format=csv,noheader,nounits"],text=True,timeout=5).strip().splitlines()[0]
            name,total,used,free,util,temp=[x.strip() for x in line.split(",",5)]
            gpu={"name":name,"total_mib":int(total),"used_mib":int(used),"free_mib":int(free),"utilization_percent":int(util),"temperature_c":int(temp)}
        except Exception as exc:gpu={"error":type(exc).__name__}
        try: ps=subprocess.check_output(["ollama","ps"],text=True,timeout=5)
        except Exception as exc:ps=f"ERROR:{type(exc).__name__}"
        self.value={"timestamp_utc":utc_now(),"ram_available_gib":round(vm.available/1024**3,3),"ram_used_percent":vm.percent,
                    "pagefile_percent":sw.percent,"pagefile_used_gib":round(sw.used/1024**3,3),"cpu_utilization_percent":ref.psutil.cpu_percent(interval=None),"gpu":gpu,"ollama_ps":ps}
        self.last=now;return self.value

def read_rows(path):
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def append_jsonl(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("ab",buffering=0) as f:f.write((json.dumps(obj,ensure_ascii=False)+"\n").encode("utf-8"));f.flush();os.fsync(f.fileno())
def sha_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def actual_token_total(meta):
    p=meta.get("prompt_eval_count") if meta else None;o=meta.get("eval_count") if meta else None
    return int(p)+int(o) if isinstance(p,(int,float)) and isinstance(o,(int,float)) else None
def events():return JsonlJournal.read_events(JOURNAL)
def state_map():return {x["call_id"]:x for x in reconstruct_calls(JOURNAL)}
def ensure_prior_attempt():
    JOURNAL.parent.mkdir(parents=True,exist_ok=True)
    ev=events()
    if any(x.get("call_id")=="v042-attempt-1-interrupted" for x in ev):return
    prior=json.loads(OLD_STOP.read_text(encoding="utf-8"))
    append_jsonl(JOURNAL,{"event":"call_start","call_id":"v042-attempt-1-interrupted","question_id":prior["question_id"],"phase":"initial","attempt_number":1,
        "timestamp_utc":"2026-11-16T12:23:23Z","elapsed_monotonic_seconds":0,"request":prior["request"],"request_sha256":prior["request_sha256"],"legacy_source":"v0.4.2 INTERRUPTED_CALL.json","status_note":"Interrupted client attempt; no raw chunks persisted in v0.4.2"})
    append_jsonl(JOURNAL,{"event":"call_interrupted","call_id":"v042-attempt-1-interrupted","question_id":prior["question_id"],"phase":"initial","attempt_number":1,
        "timestamp_utc":"2026-11-16T12:38:23Z","elapsed_monotonic_seconds":900,"diagnostic_class":"legacy_interruption_without_chunk_journal",
        "error_type":"PriorVersionClientInterruption","error_message":"v0.4.2 client disconnected; the response body and any partial stream were not persisted",
        "response_metadata":None,"raw_chunks_available":False,"prior_record_sha256":sha_file(OLD_STOP)})

def completed_for(qid,phase):
    ev=events();states=state_map();starts=[x for x in ev if x.get("event")=="call_start" and x.get("question_id")==qid and x.get("phase")==phase]
    usable=[states[s["call_id"]] for s in starts if states.get(s["call_id"],{}).get("status")=="complete"]
    if not usable:return None
    state=max(usable,key=lambda x:x["attempt_number"]); checkpoints=[x for x in read_rows(CHECKPOINTS) if x.get("call_id")==state["call_id"]]
    if checkpoints:
        row=checkpoints[-1];row["content"]=state["content"];row["thinking"]=state["thinking"];return row
    parsed,status=ref.parse_final_answer(state["content"]);meta=state.get("token_statistics") or {}
    return {**state,"parsed_answer":parsed,"parse_status":status,"token_statistics":meta,
            "tokens":{"prompt_tokens":meta.get("prompt_eval_count"),"output_tokens":meta.get("eval_count")}}
def failed_for(qid,phase):
    return [x for x in state_map().values() if x["question_id"]==qid and x.get("status")!="complete" and any(e.get("call_id")==x["call_id"] and e.get("phase")==phase for e in events())]

def make_payload(q,phase,attempt,initial_text=""):
    sid=re.search(r"(\d+)$",q["source_question_id"]);base=int(sid.group(1)) if sid else int(hashlib.sha256(q["example_id"].encode()).hexdigest()[:8],16)
    seed=(ref.INIT_SEED_BASE if phase=="initial" else ref.REV_SEED_BASE)+base
    msgs=[{"role":"user","content":INITIAL+"\n\n"+q["query"]}]
    if phase=="revision":msgs.extend([{"role":"assistant","content":initial_text},{"role":"user","content":REVISION}])
    return {"model":MODEL,"messages":msgs,"stream":True,"options":{"num_predict":-1,"num_ctx":8192,"temperature":0.0,"seed":seed,"top_k":1,"top_p":1.0,"min_p":0.0,"repeat_penalty":1.0}}

def call_one(q,phase,attempt,initial_text="",timeout_seconds=420):
    payload=make_payload(q,phase,attempt,initial_text)
    call_id=f"{q['example_id']}|{phase}|attempt-{attempt}"
    existing=state_map().get(call_id)
    if existing:
        if existing["status"]=="complete":return existing
        raise RuntimeError(f"Call {call_id} already exists as {existing['status']}; explicit retry required")
    sampler=ResourceSampler();collector=DurableStreamCollector(JOURNAL,heartbeat=sampler,watchdog_seconds=timeout_seconds)
    request=urllib.request.Request("http://localhost:11434/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    start_mono=time.monotonic();start_utc=utc_now()
    call_metadata={"model":MODEL,"model_digest":DIGEST,"protocol_id":"live-study-v0.4.3-qwen3-4b-instruct-math-numeric"}
    collector.begin(question_id=q["example_id"],phase=phase,attempt_number=attempt,call_id=call_id,request=payload,call_metadata=call_metadata,started_monotonic=start_mono,started_utc=start_utc)
    try:
        with urllib.request.urlopen(request,timeout=120) as response:
            state=collector.consume(response,question_id=q["example_id"],phase=phase,attempt_number=attempt,call_id=call_id,request=payload,call_metadata=call_metadata,started_monotonic=start_mono,started_utc=start_utc)
    except urllib.error.HTTPError as exc:
        append_jsonl(JOURNAL,{"event":"call_error","call_id":call_id,"question_id":q["example_id"],"phase":phase,"attempt_number":attempt,"timestamp_utc":utc_now(),"diagnostic_class":"server_error","http_status":exc.code,"error_message":str(exc)})
        raise
    except (urllib.error.URLError,socket.timeout,TimeoutError) as exc:
        last=state_map().get(call_id,{}).get("last_event") or {}
        if last.get("event") not in ("call_error","call_interrupted","call_complete"):
            append_jsonl(JOURNAL,{"event":"call_error","call_id":call_id,"question_id":q["example_id"],"phase":phase,"attempt_number":attempt,"timestamp_utc":utc_now(),"diagnostic_class":"server_unavailable_or_no_progress","error_type":type(exc).__name__,"error_message":str(exc)})
        raise
    if state["status"]!="complete":raise RuntimeError(f"Ollama stream incomplete: {state['status']}")
    content=state["content"]; parsed,status=ref.parse_final_answer(content)
    meta=state.get("token_statistics") or {}
    record={"call_id":call_id,"question_id":q["example_id"],"phase":phase,"attempt_number":attempt,
        "status":"complete","completion_reason":state.get("completion_reason"),"parsed_answer":parsed,"parse_status":status,
        "tokens":{"prompt_tokens":meta.get("prompt_eval_count"),"output_tokens":meta.get("eval_count"),
                  "total_tokens":actual_token_total(meta)},
        "token_statistics":meta,"wall_clock_seconds":state.get("wall_seconds"),"first_chunk_seconds":state.get("first_chunk_seconds"),
        "response":{"message":{"role":"assistant","content":content,"thinking":state["thinking"]},"done":True,
                    **{k:v for k,v in meta.items()}},"journal_status":"durable_chunks_and_completion",
        "journal_sha256":sha_file(JOURNAL)}
    append_jsonl(CHECKPOINTS,record)
    return {**state,"parsed_answer":parsed,"parse_status":status,"checkpoint":record}

def frozen_gate_decision(q,result):
    gate_rows=read_rows(GATELOG)
    if any(x["question_id"]==q["example_id"] for x in gate_rows):return next(x for x in gate_rows if x["question_id"]==q["example_id"])
    bundle,names,schema,required,extract,validate,score_fn,decide=ref.load_gate()
    meta=result.get("token_statistics") or {}
    if actual_token_total(meta) is None or result.get("completion_reason") is None:
        raise ValueError("Frozen gate requires actual prompt/output token counts and completion reason; missing metadata will not be fabricated.")
    feature_response=(result.get("thinking") or "")+("\n\n" if result.get("thinking") and result.get("content") else "")+(result.get("content") or "")
    row={"query":q["query"],"base_response":feature_response,"base_answer":result.get("parsed_answer") or "",
         "base_actual_tokens":actual_token_total(meta),
         "base_done_reason":result.get("completion_reason") or "unknown","base_finalizer_used":False}
    norm=validate(row);frame=ref.pd.DataFrame([{k:norm[k] for k in required}],columns=required);feat=extract(frame);probs=bundle["predictor"].probabilities(frame)[0]
    score=float(score_fn(ref.np.asarray([probs]),bundle["harm_weight"])[0]);action=bool(decide(ref.np.asarray([probs]),bundle["threshold"],bundle["harm_weight"])[0])
    d={"question_id":q["example_id"],"call_id":result["call_id"],"attempt_number":result["attempt_number"],"scored_before_revision":True,
       "protocol_status":"cross-model/cross-protocol OOD","gate_compatible":True,"feature_schema":"pre-recheck-v2","features":feat.iloc[0].to_dict(),
       "probabilities":{"HELP":float(probs[0]),"NO_CHANGE":float(probs[1]),"HURT":float(probs[2])},"score":score,
       "threshold":bundle["threshold"],"recheck":action,"artifact_sha256":GATE_SHA}
    append_jsonl(GATELOG,d);return d

def saved_pair(qid):
    return next((x for x in read_rows(PAIRS) if x["question_id"]==qid),None)

def freeze_prior_attempt_record():
    legacy=json.loads(OLD_STOP.read_text(encoding="utf-8"));return legacy

def init_output():
    source=ROOT/"outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01"
    if (EVAL/"EXPERIMENT_MANIFEST.json").exists():
        existing=json.loads((EVAL/"EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
        if existing.get("experiment")!="RecheckLab v0.4.3 durable streaming and resumable MATH evaluation":
            raise FileExistsError("Existing v0.4.3 manifest differs; refusing to overwrite.")
        return
    allowed={"QUESTION_SELECTION.jsonl","TRUSTED_KEYS.jsonl"}
    unexpected={p.name for p in EVAL.iterdir() if p.is_file()}-allowed
    if unexpected:raise FileExistsError(f"v0.4.3 output directory contains unexpected files: {sorted(unexpected)}")
    source_manifest=json.loads((source/"EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest=dict(source_manifest);manifest["experiment"]="RecheckLab v0.4.3 durable streaming and resumable MATH evaluation"
    manifest["phase"]="untouched_frozen_evaluation_with_explicit_prior_attempt"
    manifest["protocol"]={**manifest["protocol"],"id":"live-study-v0.4.3-qwen3-4b-instruct-math-numeric",
        "instrumentation":"durable stream journal; generation and scientific protocol unchanged",
        "num_predict":-1,"num_ctx":8192,"watchdog_seconds":2700,"idle_stream_timeout_seconds":120}
    manifest["selection_provenance"]={"copied_verbatim_from":"v0.4.2 frozen evaluation","selection_sha256":sha_file(EVAL/"QUESTION_SELECTION.jsonl"),
        "keys_sha256":sha_file(EVAL/"TRUSTED_KEYS.jsonl"),"original_first_question_attempt":freeze_prior_attempt_record(),
        "first_question_retry_policy":"attempt 1 stays incomplete; any new v0.4.3 call is explicitly attempt 2 and reported separately"}
    manifest["resumption"]={"completed_calls_reused":True,"incomplete_attempts_never_overwritten":True,
        "retry_requires_flag":"--retry-incomplete","run_untouched_questions":True,"attempts_are_not_collapsed":True}
    (EVAL/"EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")

def audit_selection():
    old=ROOT/"outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01"
    assert sha_file(EVAL/"QUESTION_SELECTION.jsonl")==sha_file(old/"QUESTION_SELECTION.jsonl")
    assert sha_file(EVAL/"TRUSTED_KEYS.jsonl")==sha_file(old/"TRUSTED_KEYS.jsonl")
    assert len(read_rows(EVAL/"QUESTION_SELECTION.jsonl"))==60
    assert sha_file(ref.GATE_PATH)==GATE_SHA

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--initialize",action="store_true");ap.add_argument("--run",action="store_true");ap.add_argument("--retry-incomplete",action="store_true");ap.add_argument("--retry-question-id");ap.add_argument("--call-timeout-seconds",type=int,default=420);args=ap.parse_args()
    if args.initialize:
        audit_selection();init_output();ensure_prior_attempt()
        print("Initialized v0.4.3 journal. No inference performed.");return
    if not args.run:ap.error("Choose --initialize or --run")
    if args.call_timeout_seconds != 420: ap.error("This amended runner is frozen at 420 seconds per individual call.")
    manifest_path=EVAL/"EXPERIMENT_MANIFEST.json"
    if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")).get("experiment")!="RecheckLab v0.4.3 durable streaming and resumable MATH evaluation":
        raise RuntimeError("Run --initialize first to freeze and verify the v0.4.3 protocol.")
    audit_selection();ensure_prior_attempt()
    runtime=ref.ollama_identity(allow_loaded_current=True)
    if runtime["digest"]!=DIGEST:raise RuntimeError("Installed Ollama model digest differs from the frozen protocol.")
    questions=read_rows(EVAL/"QUESTION_SELECTION.jsonl")
    status={"status":"RUNNING","started_utc":utc_now(),"completed_questions":len(read_rows(PAIRS)),
            "completed_calls":sum(x["status"]=="complete" for x in state_map().values()),"retry_incomplete_enabled":args.retry_incomplete,"retry_question_id":args.retry_question_id,"runtime_timeout_seconds":args.call_timeout_seconds,"runtime_amendment":"RUNTIME_AMENDMENT_420S.json"}
    (EVAL/"RUN_STATUS.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    pending=[]
    for q in questions:
        qid=q["example_id"]
        if saved_pair(qid):continue
        init=completed_for(qid,"initial"); rev=completed_for(qid,"revision")
        # Existing partial attempts are never overwritten. A fresh attempt is
        # launched only under explicit --retry-incomplete authorization.
        if not init:
            prior_bad=failed_for(qid,"initial")
            if prior_bad and not (args.retry_incomplete or args.retry_question_id==qid):
                pending.append({"question_id":qid,"phase":"initial","reason":"incomplete_attempt_requires_explicit_retry"});continue
            attempt=next_attempt(events(),qid,"initial")
            if qid==FIRST_QID and attempt<2:attempt=2
            try:init=call_one(q,"initial",attempt,timeout_seconds=args.call_timeout_seconds)
            except Exception as exc:
                pending.append({"question_id":qid,"phase":"initial","reason":type(exc).__name__});continue
            status["completed_calls"]+=1
        decision=frozen_gate_decision(q,init)
        if not rev:
            prior_bad=failed_for(qid,"revision")
            if prior_bad and not (args.retry_incomplete or args.retry_question_id==qid):
                pending.append({"question_id":qid,"phase":"revision","reason":"incomplete_attempt_requires_explicit_retry"});continue
            attempt=next_attempt(events(),qid,"revision")
            try:rev=call_one(q,"revision",attempt,init["content"],timeout_seconds=args.call_timeout_seconds)
            except Exception as exc:
                pending.append({"question_id":qid,"phase":"revision","reason":type(exc).__name__});continue
            status["completed_calls"]+=1
        # Gold keys are opened only after the initial gate decision and revision
        # call have both been durably checkpointed.
        keys={x["example_id"]:x["gold_answer"] for x in read_rows(EVAL/"TRUSTED_KEYS.jsonl")}
        gold=Fraction(keys[qid]);ia=init.get("parsed_answer");ra=rev.get("parsed_answer")
        ic=Fraction(ia)==gold if ia is not None else False;rc=Fraction(ra)==gold if ra is not None else False
        outcome="HELP" if not ic and rc else "HURT" if ic and not rc else "NO_CHANGE"
        pair={"question_id":qid,"initial_call_id":init["call_id"],"initial_attempt":init["attempt_number"],
              "revision_call_id":rev["call_id"],"revision_attempt":rev["attempt_number"],"initial_answer":ia,"revision_answer":ra,
              "initial_parse_status":init.get("parse_status"),"revision_parse_status":rev.get("parse_status"),
              "initial_completion_reason":init.get("completion_reason"),"revision_completion_reason":rev.get("completion_reason"),
              "initial_token_statistics":init.get("token_statistics"),"revision_token_statistics":rev.get("token_statistics"),
              "initial_latency_seconds":init.get("wall_seconds"),"revision_latency_seconds":rev.get("wall_seconds"),
              "initial_correct":ic,"revision_correct":rc,"outcome":outcome,"gate_recheck":decision["recheck"],
              "completion_recheck":init.get("completion_reason")=="length"}
        append_jsonl(PAIRS,pair);status["completed_questions"]+=1
        status["last_checkpoint"]={"question_id":qid,"pair_saved":True,"attempts":[pair["initial_attempt"],pair["revision_attempt"]]}
        status["updated_utc"]=utc_now();(EVAL/"RUN_STATUS.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
        print(f"checkpoint {status['completed_questions']}/60 {qid} HELP/HURT/NO_CHANGE={outcome}",flush=True)
    final={**status,"status":"COMPLETE" if not pending and len(read_rows(PAIRS))==60 else "INCOMPLETE_RESUMABLE",
           "completed_question_pairs":len(read_rows(PAIRS)),"pending":pending,"finished_utc":utc_now()}
    (EVAL/"RUN_STATUS.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
    if final["status"]=="COMPLETE":write_policy_report()

def write_policy_report():
    pairs=read_rows(PAIRS);gate={x["question_id"]:x for x in read_rows(GATELOG)}
    keys={x["example_id"]:x for x in read_rows(EVAL/"QUESTION_SELECTION.jsonl")}
    policies={"never":lambda p:False,"always":lambda p:True,"completion":lambda p:p["completion_recheck"],"frozen_ml_gate_ood":lambda p:gate[p["question_id"]]["recheck"]}
    rows=[]
    for p in pairs:
        for name,fn in policies.items():
            choose=fn(p);ans=p["revision_answer"] if choose else p["initial_answer"]
            gold=Fraction(next(x["gold_answer"] for x in read_rows(EVAL/"TRUSTED_KEYS.jsonl") if x["example_id"]==p["question_id"]))
            calls=[p["initial_token_statistics"]]+([p["revision_token_statistics"]] if choose else [])
            tokens=[actual_token_total(x) for x in calls]
            latency=p["initial_latency_seconds"]+(p["revision_latency_seconds"] if choose else 0)
            rows.append({"question_id":p["question_id"],"policy":name,"recheck":choose,"correct":Fraction(ans)==gold if ans is not None else False,
                         "outcome":p["outcome"],"prompt_plus_output_tokens":sum(tokens) if all(x is not None for x in tokens) else None,
                         "replayed_latency_seconds":latency,"initial_parse_status":p["initial_parse_status"],"revision_parse_status":p["revision_parse_status"],
                         "initial_completion_reason":p["initial_completion_reason"],"gate_recheck":p["gate_recheck"],"completion_recheck":p["completion_recheck"]})
    with (EVAL/"POLICY_COMPARISON.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    summaries=[]; vectors={}
    for name in policies:
        rr=[r for r in rows if r["policy"]==name];n=len(rr);correct=sum(r["correct"] for r in rr)
        vectors[name]=[int(r["correct"]) for r in rr]
        toks=[r["prompt_plus_output_tokens"] for r in rr if r["prompt_plus_output_tokens"] is not None]
        lats=[r["replayed_latency_seconds"] for r in rr if r["replayed_latency_seconds"] is not None]
        summaries.append({"policy":name,"correct":correct,"n":n,"accuracy":correct/n if n else None,"wilson_95":ref.wilson(correct,n),
                          "rechecks":sum(r["recheck"] for r in rr),"mean_tokens_per_question":statistics.mean(toks) if toks else None,
                          "mean_replayed_latency_seconds_per_question":statistics.mean(lats) if lats else None,
                          "HELP_captured":sum(r["outcome"]=="HELP" and r["recheck"] for r in rr),"HELP_missed":sum(r["outcome"]=="HELP" and not r["recheck"] for r in rr),
                          "HURT_captured":sum(r["outcome"]=="HURT" and r["recheck"] for r in rr),"HURT_avoided":sum(r["outcome"]=="HURT" and not r["recheck"] for r in rr)})
    (EVAL/"POLICY_SUMMARY.json").write_text(json.dumps({"summaries":summaries,"completed_pairs":len(pairs),"attempts_retained_separately":True},indent=2)+"\n",encoding="utf-8")
    names=list(policies);diffs=[]
    for i,left in enumerate(names):
        for right in names[i+1:]:
            avg,lo,hi=ref.bootstrap_diff(vectors[left],vectors[right])
            diffs.append({"policy_a":left,"policy_b":right,"paired_accuracy_difference":avg,"bootstrap_95_low":lo,"bootstrap_95_high":hi,"resamples":20000,"seed":20261121})
    with (EVAL/"PAIRWISE_POLICY_DIFFERENCES.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=diffs[0].keys());w.writeheader();w.writerows(diffs)

if __name__=="__main__":main()
