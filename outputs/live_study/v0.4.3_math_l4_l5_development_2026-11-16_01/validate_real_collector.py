"""One real dev-only Ollama call through v0.4.3 durable streaming."""
import hashlib,json,pathlib,sys,time,urllib.request,subprocess
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE));from durable_stream import DurableStreamCollector,JsonlJournal,reconstruct_calls,utc_now
sys.path.insert(0,str(ROOT));import run_math_study_reference as ref
EVAL=ROOT/"outputs/live_study/v0.4.3_math_l4_l5_frozen_evaluation_2026-11-16_01"
JOURNAL=HERE/"REAL_CALL_VALIDATION_STREAM.jsonl";RESULT=HERE/"REAL_CALL_VALIDATION.json"
MODEL="qwen3:4b-instruct";DIGEST=ref.MODEL_DIGEST

def main():
    if RESULT.exists() or JOURNAL.exists():raise FileExistsError("Validation artifacts already exist; refusing to overwrite.")
    # Confirm frozen evaluation state before issuing a development-only call.
    assert hashlib.sha256((EVAL/"QUESTION_SELECTION.jsonl").read_bytes()).hexdigest()==hashlib.sha256((ROOT/"outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01/QUESTION_SELECTION.jsonl").read_bytes()).hexdigest()
    q=json.loads((HERE/"QUESTION_SELECTION.jsonl").read_text(encoding="utf-8").splitlines()[0])
    runtime=ref.ollama_identity(allow_loaded_current=True)
    if runtime["digest"]!=DIGEST:raise RuntimeError("Installed model digest mismatch")
    payload={"model":MODEL,"messages":[{"role":"user","content":ref.INITIAL_PROMPT+"\n\n"+q["query"]}],"stream":True,
      "options":{"num_predict":-1,"num_ctx":8192,"temperature":0.0,"seed":ref.INIT_SEED_BASE+int(q["source_question_id"].split("/")[-1]),
      "top_k":1,"top_p":1.0,"min_p":0.0,"repeat_penalty":1.0}}
    call_id="v043-dev-real-validation|"+q["example_id"]+"|attempt-1";t0=time.monotonic();ts=utc_now()
    req=urllib.request.Request("http://localhost:11434/api/chat",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    collector=DurableStreamCollector(JOURNAL,watchdog_seconds=2700,diagnostic_warning_interval_seconds=60)
    collector.begin(question_id=q["example_id"],phase="development_validation",attempt_number=1,call_id=call_id,request=payload,
        call_metadata={"model":MODEL,"model_digest":DIGEST,"protocol_id":"live-study-v0.4.3-qwen3-4b-instruct-math-numeric"},started_monotonic=t0,started_utc=ts)
    with urllib.request.urlopen(req,timeout=120) as response:
        result=collector.consume(response,question_id=q["example_id"],phase="development_validation",attempt_number=1,call_id=call_id,
            request=payload,call_metadata={"model":MODEL,"model_digest":DIGEST},started_monotonic=t0,started_utc=ts)
    restored=reconstruct_calls(JOURNAL)[0]
    parsed,status=ref.parse_final_answer(restored["content"])
    events=JsonlJournal.read_events(JOURNAL);chunks=[e for e in events if e["event"]=="chunk"]
    metadata=restored["token_statistics"] or {}
    valid=(restored["status"]=="complete" and restored["content"]==result["content"] and restored["thinking"]==result["thinking"]
           and restored["completion_reason"]==result["completion_reason"] and metadata.get("prompt_eval_count") is not None
           and metadata.get("eval_count") is not None and parsed is not None and result["completion_reason"]=="stop" and len(chunks)>0)
    out={"status":"PASS" if valid else "FAIL","phase":"development_only_real_inference","question_id":q["example_id"],"call_id":call_id,"attempt_number":1,
       "model":runtime,"protocol":{"num_ctx":8192,"num_predict":-1,"temperature":0.0,"stream":True,"watchdog_seconds":2700},
       "completion_reason":restored["completion_reason"],"parse_status":status,"parsed_answer":parsed,
       "prompt_tokens":metadata.get("prompt_eval_count"),"output_tokens":metadata.get("eval_count"),
       "wall_seconds":result["wall_seconds"],"first_chunk_seconds":result["first_chunk_seconds"],
       "chunks_durably_written":len(chunks),"content_chars":len(restored["content"]),"thinking_chars":len(restored["thinking"]),
       "reconstruction_exact":restored["content"]==result["content"] and restored["thinking"]==result["thinking"],
       "completion_metadata_exact":restored["completion_reason"]==result["completion_reason"] and restored["token_statistics"]==result["token_statistics"],
       "chunk_diagnostics_present":all("diagnostics" in x and "resource_sample" in x for x in chunks),
       "stream_journal_sha256":hashlib.sha256(JOURNAL.read_bytes()).hexdigest(),"evaluation_questions_touched":False,
       "started_utc":ts,"finished_utc":utc_now()}
    RESULT.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
    if not valid:raise SystemExit(2)

if __name__=="__main__":main()
