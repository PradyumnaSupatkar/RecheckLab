"""Development-only uncapped Ollama context benchmark; never reads gold keys."""
import hashlib, json, pathlib, subprocess, time, urllib.request
import psutil

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = "qwen3:4b-instruct"
DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
INITIAL = ("Solve the problem carefully. Provide enough concise reasoning to make the result checkable. "
           "End with exactly one final line in this form: FINAL: <integer or exact fraction>. "
           "Use an exact fraction instead of a decimal when needed. Do not omit the final line.")
REVISION = ("Review your previous solution against the problem and correct any mathematical error. "
            "Give a concise explanation of your check. End with exactly one final line in this form: "
            "FINAL: <integer or exact fraction>. Use an exact fraction instead of a decimal when needed. "
            "Do not omit the final line.")
CONTEXTS = [8192, 16384]
WATCHDOG_SECONDS = 2700
IDLE_SECONDS = 120

def sha(b): return hashlib.sha256(b).hexdigest()
def resources():
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    try:
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().splitlines()[0]
        n,t,u,f,g,temp=[v.strip() for v in line.split(",")]
        gpu={"name":n,"total_mib":int(t),"used_mib":int(u),"free_mib":int(f),"utilization_percent":int(g),"temperature_c":int(temp)}
    except Exception as e: gpu={"error":type(e).__name__}
    try: ps=subprocess.check_output(["ollama","ps"],text=True,timeout=5)
    except Exception as e: ps=f"ERROR:{type(e).__name__}"
    return {"unix":time.time(),"ram_total_gib":round(vm.total/1024**3,3),"ram_available_gib":round(vm.available/1024**3,3),"ram_percent":vm.percent,"pagefile_percent":sw.percent,"pagefile_used_gib":round(sw.used/1024**3,3),"cpu_utilization_percent":psutil.cpu_percent(interval=0.1),"gpu":gpu,"ollama_ps":ps}

def stream_call(payload):
    raw=json.dumps(payload).encode()
    req=urllib.request.Request("http://localhost:11434/api/chat",data=raw,headers={"Content-Type":"application/json"})
    start=time.perf_counter(); chunks=[]; thinking=[]; final={}; last=time.monotonic(); samples=[]
    with urllib.request.urlopen(req,timeout=IDLE_SECONDS) as resp:
        while True:
            if time.perf_counter()-start>WATCHDOG_SECONDS: raise TimeoutError("45-minute request watchdog exceeded")
            line=resp.readline()
            if not line: break
            last=time.monotonic(); obj=json.loads(line)
            msg=obj.get("message") or {}
            if msg.get("content"): chunks.append(msg["content"])
            if msg.get("thinking"): thinking.append(msg["thinking"])
            for k in ("model","created_at","done","done_reason","total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration"):
                if k in obj: final[k]=obj[k]
            if obj.get("done"):
                samples.append(resources()); break
            if time.monotonic()-last>WATCHDOG_SECONDS: raise TimeoutError("No output progress")
            if len(chunks)+len(thinking) and (len(chunks)+len(thinking))%256==0: samples.append(resources())
    final["message"]={"role":"assistant","content":"".join(chunks)}
    if thinking: final["message"]["thinking"]="".join(thinking)
    final["wall_seconds"]=time.perf_counter()-start
    final["stream_resource_samples"]=samples or [resources()]
    return final

def main():
    qs=[json.loads(x) for x in (HERE/"QUESTION_SELECTION.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()][:2]
    manifest={"experiment":"v0.4.2 development context benchmark","purpose":"compare 8K and 16K context using uncapped num_predict on two already frozen development questions; no evaluation questions or keys read","question_ids":[q["example_id"] for q in qs],"contexts":CONTEXTS,"num_predict":-1,"timeout_seconds":WATCHDOG_SECONDS,"idle_read_timeout_seconds":IDLE_SECONDS,"initial_prompt":INITIAL,"revision_prompt":REVISION,"prompt_sha256":{"initial":sha(INITIAL.encode()),"revision":sha(REVISION.encode())},"request_streaming":True,"stop_override":"omitted; use model natural stop tokens","no_gold_answers_loaded":True,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    manifest_path=HERE/"CONTEXT_BENCHMARK_MANIFEST.json"
    raw_path=HERE/"CONTEXT_BENCHMARK_RAW.jsonl"
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding="utf-8"))
        for k in ("question_ids","contexts","num_predict","initial_prompt","revision_prompt","prompt_sha256"):
            if old.get(k)!=manifest.get(k): raise RuntimeError("Frozen context benchmark manifest differs; refusing to overwrite.")
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    raw_path.touch(exist_ok=True)
    done={}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            old=json.loads(line); done[(old["context"],old["example_id"],old["phase"])]=old
    for ctx in CONTEXTS:
        for q in qs:
            init_payload={"model":MODEL,"stream":True,"messages":[{"role":"user","content":INITIAL+"\n\n"+q["query"]}],"options":{"num_predict":-1,"num_ctx":ctx,"temperature":0,"seed":88001+int(q["source_question_id"].split("/")[-1]),"top_k":1,"top_p":1.0,"min_p":0.0,"repeat_penalty":1.0}}
            ikey=(ctx,q["example_id"],"initial")
            if ikey in done: a=done[ikey]["response"]
            else:
                before=resources(); a=stream_call(init_payload)
                row={"context":ctx,"example_id":q["example_id"],"phase":"initial","payload":init_payload,"before":before,"response":a}
                with raw_path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
                done[ikey]=row
            content=a["message"]["content"]
            print(json.dumps({"context":ctx,"question":q["example_id"],"phase":"initial","done_reason":a.get("done_reason"),"eval_count":a.get("eval_count"),"seconds":round(a["wall_seconds"],2),"tokens_per_second":round(a.get("eval_count",0)/(a.get("eval_duration",1)/1e9),2) if a.get("eval_duration") else None,"content_chars":len(content),"free_ram_gib":a["stream_resource_samples"][-1].get("ram_available_gib"),"gpu":a["stream_resource_samples"][-1].get("gpu")}),flush=True)
            rev_payload={"model":MODEL,"stream":True,"messages":[{"role":"user","content":INITIAL+"\n\n"+q["query"]},{"role":"assistant","content":content},{"role":"user","content":REVISION}],"options":{"num_predict":-1,"num_ctx":ctx,"temperature":0,"seed":99001+int(q["source_question_id"].split("/")[-1]),"top_k":1,"top_p":1.0,"min_p":0.0,"repeat_penalty":1.0}}
            rkey=(ctx,q["example_id"],"revision")
            if rkey in done: b=done[rkey]["response"]
            else:
                before=resources(); b=stream_call(rev_payload)
                row={"context":ctx,"example_id":q["example_id"],"phase":"revision","payload":rev_payload,"before":before,"response":b}
                with raw_path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
                done[rkey]=row
            print(json.dumps({"context":ctx,"question":q["example_id"],"phase":"revision","done_reason":b.get("done_reason"),"eval_count":b.get("eval_count"),"seconds":round(b["wall_seconds"],2),"tokens_per_second":round(b.get("eval_count",0)/(b.get("eval_duration",1)/1e9),2) if b.get("eval_duration") else None,"content_chars":len(b["message"]["content"]),"free_ram_gib":b["stream_resource_samples"][-1].get("ram_available_gib"),"gpu":b["stream_resource_samples"][-1].get("gpu")}),flush=True)

if __name__=="__main__": main()
