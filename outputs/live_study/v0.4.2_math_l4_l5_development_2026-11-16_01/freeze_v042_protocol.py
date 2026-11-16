"""Freeze v0.4.2 over the unchanged v0.4.1 development/evaluation selections."""
import hashlib, importlib.util, json, pathlib, urllib.request

DEV=pathlib.Path(__file__).resolve().parent
ROOT=DEV.parents[2]
EVAL=ROOT/"outputs/live_study/v0.4.2_math_l4_l5_frozen_evaluation_2026-11-16_01"
OLDDEV=ROOT/"outputs/live_study/v0.4.1_math_l4_l5_development_2026-11-16_01"
OLDEVAL=ROOT/"outputs/live_study/v0.4.1_math_l4_l5_frozen_evaluation_2026-11-16_01"
MODEL="qwen3:4b-instruct"; DIGEST="0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
GATE="9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439"
INITIAL=("Solve the problem carefully. Provide enough concise reasoning to make the result checkable. "
         "End with exactly one final line in this form: FINAL: <integer or exact fraction>. "
         "Use an exact fraction instead of a decimal when needed. Do not omit the final line.")
REVISION=("Review your previous solution against the problem and correct any mathematical error. "
          "Give a concise explanation of your check. End with exactly one final line in this form: "
          "FINAL: <integer or exact fraction>. Use an exact fraction instead of a decimal when needed. "
          "Do not omit the final line.")
GENERATION={"num_predict":-1,"num_ctx":8192,"temperature":0.0,"top_k":1,"top_p":1.0,"min_p":0.0,
            "repeat_penalty":1.0,"think":"omitted","stop":"model defaults; no stop override",
            "stream":True,"watchdog_seconds":2700,"idle_read_timeout_seconds":120,"retries":0}
def sha(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):return sha(p.read_bytes())
def readj(p):return json.loads(p.read_text(encoding="utf-8"))
def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def main():
    if any((DEV/n).exists() for n in ("RAW_CALLS.jsonl","PAIRS.jsonl","COLLECTION_STATUS.json")):
        raise FileExistsError("v0.4.2 development collection already exists; refusing to rewrite protocol.")
    if any((EVAL/n).exists() for n in ("RAW_CALLS.jsonl","PAIRS.jsonl","COLLECTION_STATUS.json","FINAL_STATUS.json")):
        raise FileExistsError("v0.4.2 evaluation contains inference results; refusing to overwrite.")
    for old,new in ((OLDDEV,DEV),(OLDEVAL,EVAL)):
        for name in ("QUESTION_SELECTION.jsonl","TRUSTED_KEYS.jsonl"):
            if file_sha(old/name)!=file_sha(new/name): raise ValueError(f"Copied frozen file differs: {old.name}/{name}")
    old_eval_files=sorted(p.name for p in OLDEVAL.iterdir() if p.is_file())
    if set(old_eval_files)!={"EXPERIMENT_MANIFEST.json","QUESTION_SELECTION.jsonl","TRUSTED_KEYS.jsonl"}:
        raise RuntimeError("Prior v0.4.1 evaluation is no longer demonstrably untouched.")
    old_dev_ids={x["example_id"] for x in rows(OLDDEV/"QUESTION_SELECTION.jsonl")}
    eval_ids={x["example_id"] for x in rows(EVAL/"QUESTION_SELECTION.jsonl")}
    if len(old_dev_ids)!=10 or len(eval_ids)!=60 or old_dev_ids & eval_ids:
        raise RuntimeError("Frozen development/evaluation selection is not 10/60 disjoint.")
    # The benchmark must have all four calls complete, all naturally stopped,
    # and parsed successfully. No answer keys are loaded here.
    spec=importlib.util.spec_from_file_location("runner",DEV/"run_math_study.py")
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    bench=rows(DEV/"CONTEXT_BENCHMARK_RAW.jsonl")
    summary={}
    for ctx in (8192,16384):
        part=[r for r in bench if r["context"]==ctx]
        parse=[]; tps=[]; secs=[]
        for r in part:
            response=r["response"]; _,status=runner.parse_final_answer(response["message"]["content"])
            parse.append(status=="parseable"); tps.append(response.get("eval_count",0)/(response.get("eval_duration",1)/1e9));secs.append(response["wall_seconds"])
            if response.get("done_reason")!="stop": raise RuntimeError("Context benchmark did not end naturally in all calls.")
        if len(part)!=4: raise RuntimeError(f"Context {ctx} does not have all four benchmark calls.")
        summary[str(ctx)]={"calls":len(part),"parseable":sum(parse),"natural_stop":True,
                           "mean_tokens_per_second":round(sum(tps)/len(tps),3),"mean_wall_seconds":round(sum(secs)/len(secs),3),
                           "minimum_free_ram_gib":round(min(x["response"]["stream_resource_samples"][-1]["ram_available_gib"] for x in part),3),
                           "maximum_gpu_used_mib":max(x["response"]["stream_resource_samples"][-1]["gpu"]["used_mib"] for x in part)}
    if summary["8192"]["parseable"]<4 or summary["8192"]["mean_tokens_per_second"]<=summary["16384"]["mean_tokens_per_second"]:
        raise RuntimeError("8K was not selected by the predeclared reliability/throughput/resource rule.")
    model_req=urllib.request.Request("http://localhost:11434/api/tags")
    with urllib.request.urlopen(model_req,timeout=10) as r: models=json.load(r).get("models",[])
    found=next((x for x in models if x.get("name")==MODEL),None)
    if not found or found.get("digest")!=DIGEST: raise RuntimeError("Installed model digest changed.")
    req=urllib.request.Request("http://localhost:11434/api/show",data=json.dumps({"model":MODEL,"verbose":True}).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=15) as r: show=json.load(r)
    identity={"name":MODEL,"digest":found["digest"],"size_bytes":found.get("size"),"family":show.get("details",{}).get("family"),
              "quantization":show.get("details",{}).get("quantization_level"),"fine_tune":show.get("model_info",{}).get("general.finetune"),
              "version":show.get("model_info",{}).get("general.version"),"architecture_context_length":show.get("model_info",{}).get("qwen3.context_length"),"ollama":"0.40.16"}
    for folder,phase,old in ((DEV,"development_reliability_gate",OLDDEV),(EVAL,"untouched_frozen_evaluation",OLDEVAL)):
        old_manifest=readj(old/"EXPERIMENT_MANIFEST.json")
        protocol=old_manifest["protocol"]
        protocol.update({"id":"live-study-v0.4.2-qwen3-4b-instruct-math-numeric","model":identity,
            "prompts":{"initial":INITIAL,"revision":REVISION},
            "prompt_sha256":{"initial":sha(INITIAL.encode()),"revision":sha(REVISION.encode())},
            "generation":GENERATION,"frozen_v0_2_gate_artifact_sha256":GATE,
            "gate_status":"Cross-model/cross-protocol OOD diagnostic; original production validator unchanged"})
        selection_sha=file_sha(folder/"QUESTION_SELECTION.jsonl"); keys_sha=file_sha(folder/"TRUSTED_KEYS.jsonl")
        manifest={**old_manifest,"experiment":"RecheckLab v0.4.2 independent hard-math evaluation",
                  "phase":phase,"expected_llm_calls":20 if phase.startswith("development") else 120,
                  "protocol":protocol,"question_selection_sha256":selection_sha,"trusted_key_sha256":keys_sha,
                  "selection_provenance":{"original_frozen_version":"v0.4.1","source_manifest_sha256":file_sha(old/"EXPERIMENT_MANIFEST.json"),
                      "source_question_selection_sha256":file_sha(old/"QUESTION_SELECTION.jsonl"),
                      "source_trusted_keys_sha256":file_sha(old/"TRUSTED_KEYS.jsonl"),
                      "selection_reused_verbatim":True,"evaluation_previously_uninferred":phase=="untouched_frozen_evaluation"}}
        manifest["resource_guard"]={"sequential_calls":True,"no_free_RAM_or_VRAM_reserve_cutoff":True,
            "stop_if_pagefile_percent_reaches":98.0,"Ollama_allocation_failures_stop_run":True,
            "whole_call_watchdog_seconds":2700,"idle_stream_read_timeout_seconds":120,"retries":0}
        write(folder/"EXPERIMENT_MANIFEST.json",manifest)
    bench_doc={"context_benchmark":summary,"selection_rule":"Use the smallest tested context that achieved full parse success and higher mean throughput without resource stress; correctness was not used.","selected_context":8192,
               "ollama_ps_observed":{"8192":"45% CPU / 55% GPU; 4.2 GB total runner allocation","16384":"56% CPU / 44% GPU; 5.4 GB total runner allocation"},
               "gpu_observed":"RTX 3050 Laptop GPU, 4 GiB; no OOM; peak temperature in context benchmark 70 C"}
    write(DEV/"CONTEXT_BENCHMARK_SUMMARY.json",bench_doc)
    write(DEV/"PREPARATION_STATUS.json",{"status":"FROZEN_BEFORE_V0.4.2_INFERENCE","model":identity,"dev_question_count":10,"eval_question_count":60,
        "development_question_selection_sha256":file_sha(DEV/"QUESTION_SELECTION.jsonl"),"evaluation_question_selection_sha256":file_sha(EVAL/"QUESTION_SELECTION.jsonl"),
        "development_trusted_keys_sha256":file_sha(DEV/"TRUSTED_KEYS.jsonl"),"evaluation_trusted_keys_sha256":file_sha(EVAL/"TRUSTED_KEYS.jsonl"),
        "frozen_gate_sha256":GATE,"prior_evaluation_untouched":True,"prior_evaluation_files":old_eval_files,
        "context_benchmark_calls":len(bench),"evaluation_inference_started":False,"final_protocol":GENERATION})
    (DEV/"PROTOCOL_SELECTION.md").write_text(
        "# v0.4.2 protocol selection\n\n"
        "The four-call 8K/16K development benchmark used the first two question IDs in the already frozen v0.4.1 development selection, with no answer keys loaded. All four calls at each context stopped naturally and parsed. Mean generation speed was higher at 8K, free RAM was higher, and `ollama ps` showed a smaller runner allocation; 8K is therefore frozen for the 10-question rerun. Selection did not use correctness.\n\n"
        "`num_predict=-1` is sent explicitly and verified empirically: all eight context-benchmark calls returned `done_reason=stop`, including outputs above 512 tokens. The model's own stop behavior is used; no stop-string override is sent. `num_ctx=8192` is far below the model-reported architecture context of 262,144.\n\n"
        "The final prompts and settings are in `EXPERIMENT_MANIFEST.json`. The 60 evaluation question selection and keys were copied byte-for-byte from v0.4.1 and remain untouched; only the new v0.4.2 protocol manifest was created before evaluation inference. Development extraction must reach 9/10 initial and revision parseability before the evaluation run can begin.\n",encoding="utf-8")
    print(json.dumps({"status":"FROZEN_BEFORE_V0.4.2_INFERENCE","context_benchmark":summary,"model":identity,"evaluation_untouched":True},indent=2))
if __name__=="__main__":main()
