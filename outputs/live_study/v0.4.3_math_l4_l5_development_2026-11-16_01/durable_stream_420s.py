"""Durable Ollama stream journal and restart-safe call reconstruction."""
from __future__ import annotations
import hashlib, json, os, re, time, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

def utc_now(): return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
def canonical_json(obj): return json.dumps(obj,ensure_ascii=False,separators=(",",":"))

class JsonlJournal:
    """Append-only JSONL journal. Each event is flushed and fsync'd before return."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        # Validate existing complete lines before appending; preserve any corrupt tail.
        if self.path.exists(): self.read_events(self.path)
    @staticmethod
    def read_events(path):
        if not Path(path).exists(): return []
        rows=[]
        with Path(path).open("rb") as f:
            for n,line in enumerate(f,1):
                if not line.endswith(b"\n"):
                    raise ValueError(f"Incomplete JSONL tail at line {n}; preserve and repair explicitly")
                try: rows.append(json.loads(line))
                except Exception as exc: raise ValueError(f"Invalid JSONL at line {n}") from exc
        return rows
    def append(self,event):
        data=(canonical_json(event)+"\n").encode("utf-8")
        with self.path.open("ab",buffering=0) as f:
            f.write(data);f.flush();os.fsync(f.fileno())

def call_state(events,call_id):
    relevant=[e for e in events if e.get("call_id")==call_id]
    if not relevant: return None
    start=next((e for e in relevant if e.get("event")=="call_start"),None)
    chunks=[e for e in relevant if e.get("event")=="chunk"]
    terminal=next((e for e in reversed(relevant) if e.get("event") in ("call_complete","call_interrupted","call_error")),None)
    content="".join((e.get("raw_chunk") or {}).get("message",{}).get("content","") or "" for e in chunks)
    thinking="".join((e.get("raw_chunk") or {}).get("message",{}).get("thinking","") or "" for e in chunks)
    stats=(terminal.get("response_metadata") or None) if terminal and terminal.get("event")=="call_complete" else None
    first_chunk=next((e.get("elapsed_monotonic_seconds") for e in chunks if e.get("chunk_index")==1),None)
    return {"call_id":call_id,"question_id":(start or {}).get("question_id"),"attempt_number":(start or {}).get("attempt_number"),
            "status":"complete" if terminal and terminal.get("event")=="call_complete" else "incomplete" if terminal else "in_progress",
            "content":content,"thinking":thinking,"chunk_count":len(chunks),"content_chars":len(content),
            "thinking_chars":len(thinking),"last_elapsed_seconds":chunks[-1].get("elapsed_monotonic_seconds") if chunks else 0.0,
            "first_chunk_seconds":first_chunk,"wall_seconds":terminal.get("elapsed_monotonic_seconds") if terminal else (chunks[-1].get("elapsed_monotonic_seconds") if chunks else 0.0),
            "total_call_duration_seconds":terminal.get("elapsed_monotonic_seconds") if terminal else None,
            "completion_reason":(stats or {}).get("done_reason"),"token_statistics":stats,
            "last_event":terminal}

def next_attempt(events,question_id,phase):
    attempts=[e.get("attempt_number",0) for e in events if e.get("event")=="call_start" and e.get("question_id")==question_id and e.get("phase")==phase]
    return max(attempts,default=0)+1

def find_repetition(text,window_chars=1200):
    """Diagnostic only: return repeated long lines / n-grams; never terminates generation."""
    tail=text[-window_chars:]
    lines=[re.sub(r"\s+"," ",x.strip().lower()) for x in tail.splitlines() if len(x.strip())>=24]
    repeated_lines=sorted({x for x in lines if lines.count(x)>=3})
    words=re.findall(r"\w+|[^\w\s]",tail.lower())
    repeats=[]
    for n in (8,12,16):
        grams=[tuple(words[i:i+n]) for i in range(max(0,len(words)-3*n),len(words)-n+1)]
        repeated=next((g for g,count in Counter(grams).items() if count>=3),None)
        if repeated: repeats.append({"n_tokens":n,"phrase":" ".join(repeated)[:200]})
    return {"repeated_lines":repeated_lines[:5],"repeated_ngrams":repeats,"warning":bool(repeated_lines or repeats)}

class DurableStreamCollector:
    def __init__(self,journal_path,heartbeat=None,warning_callback=None,watchdog_seconds=2700,diagnostic_warning_interval_seconds=60):
        self.journal=JsonlJournal(journal_path);self.heartbeat=heartbeat or (lambda: {})
        self.warning_callback=warning_callback or (lambda _:None)
        self.watchdog_seconds=watchdog_seconds;self.diagnostic_warning_interval_seconds=diagnostic_warning_interval_seconds
    def begin(self,*,question_id,phase,attempt_number,call_id=None,request=None,call_metadata=None,started_monotonic=None,started_utc=None):
        call_id=call_id or str(uuid.uuid4());t0=started_monotonic if started_monotonic is not None else time.monotonic();started_utc=started_utc or utc_now()
        events=self.journal.read_events(self.journal.path)
        if call_state(events,call_id) is not None: raise ValueError(f"call_id already exists: {call_id}")
        self.journal.append({"event":"call_start","call_id":call_id,"question_id":question_id,"phase":phase,
            "attempt_number":attempt_number,"timestamp_utc":started_utc,"elapsed_monotonic_seconds":0.0,
            "request":request,"request_sha256":hashlib.sha256(canonical_json(request or {}).encode()).hexdigest(),
            "call_metadata":call_metadata or {},"resource_sample":self.heartbeat()})
        return call_id,t0,started_utc
    def consume(self,lines,*,question_id,phase,attempt_number,call_id=None,request=None,call_metadata=None,started_monotonic=None,started_utc=None):
        call_id=call_id or str(uuid.uuid4()); t0=started_monotonic if started_monotonic is not None else time.monotonic();started_utc=started_utc or utc_now()
        state=call_state(self.journal.read_events(self.journal.path),call_id)
        if state is None:self.begin(question_id=question_id,phase=phase,attempt_number=attempt_number,call_id=call_id,request=request,call_metadata=call_metadata,started_monotonic=t0,started_utc=started_utc)
        elif state["status"]!="in_progress":raise ValueError(f"call_id already terminal: {call_id}")
        elif state["chunk_count"]>0:raise ValueError(f"cannot attach a new stream to partial call {call_id}; create an explicit new attempt")
        text="";think="";first=None;last=None;chunks=0;last_warning_bucket=0;logged_repetitions=set()
        try:
            for raw_line in lines:
                elapsed=time.monotonic()-t0;chunk_utc=utc_now()
                if elapsed>self.watchdog_seconds:raise TimeoutError(f"{self.watchdog_seconds}s call watchdog exceeded")
                if not raw_line: continue
                if isinstance(raw_line,bytes): raw_line=raw_line.decode("utf-8")
                try: chunk=json.loads(raw_line)
                except json.JSONDecodeError:
                    self.journal.append({"event":"malformed_stream_chunk","call_id":call_id,"question_id":question_id,"phase":phase,
                        "attempt_number":attempt_number,"timestamp_utc":chunk_utc,"elapsed_monotonic_seconds":elapsed,
                        "raw_chunk_line":raw_line,"chunk_index":chunks+1,"diagnostic_class":"malformed_server_stream_json"})
                    raise
                msg=chunk.get("message") or {};c=msg.get("content") or "";th=msg.get("thinking") or ""
                if first is None: first=elapsed
                gap=None if last is None else elapsed-last
                last=elapsed;text+=c;think+=th;chunks+=1
                repeated=find_repetition(text+think)
                self.journal.append({"event":"chunk","call_id":call_id,"question_id":question_id,"phase":phase,
                    "attempt_number":attempt_number,"timestamp_utc":chunk_utc,"elapsed_monotonic_seconds":elapsed,
                    "chunk_index":chunks,"raw_chunk_line":raw_line,"raw_chunk":chunk,"accumulated":{"content_chars":len(text),"thinking_chars":len(think),
                    "content_sha256":hashlib.sha256(text.encode()).hexdigest(),"thinking_sha256":hashlib.sha256(think.encode()).hexdigest()},
                    "diagnostics":{"time_to_first_chunk_seconds":first,"time_since_last_chunk_seconds":gap,
                                   "total_call_seconds":elapsed,"repetition":repeated},"resource_sample":self.heartbeat()})
                if repeated["warning"]:
                    signature=hashlib.sha256(json.dumps(repeated,sort_keys=True).encode()).hexdigest()
                    if signature not in logged_repetitions:
                        warning={"call_id":call_id,"question_id":question_id,"diagnostic":"repetitive_output_warning","details":repeated}
                        self.journal.append({"event":"diagnostic_warning","timestamp_utc":utc_now(),"elapsed_monotonic_seconds":elapsed,**warning})
                        self.warning_callback(warning);logged_repetitions.add(signature)
                bucket=int(elapsed//self.diagnostic_warning_interval_seconds)
                if bucket>last_warning_bucket:
                    last_warning_bucket=bucket
                    warning={"call_id":call_id,"question_id":question_id,"diagnostic":"long_active_generation_warning","elapsed_seconds":elapsed,"chunks":chunks,"content_chars":len(text)+len(think)}
                    self.journal.append({"event":"diagnostic_warning","timestamp_utc":utc_now(),"elapsed_monotonic_seconds":elapsed,**warning})
                    self.warning_callback(warning)
                if chunk.get("done"):
                    meta={k:chunk[k] for k in ("done_reason","total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration") if k in chunk}
                    self.journal.append({"event":"call_complete","call_id":call_id,"question_id":question_id,"phase":phase,
                        "attempt_number":attempt_number,"timestamp_utc":utc_now(),"elapsed_monotonic_seconds":time.monotonic()-t0,
                        "response_metadata":meta,"resource_sample":self.heartbeat()})
                    break
            else:
                raise ConnectionError("stream ended before Ollama completion metadata")
        except BaseException as exc:
            kind="call_interrupted" if isinstance(exc,(KeyboardInterrupt,SystemExit)) else "call_error"
            watchdog_timeout=isinstance(exc,TimeoutError) and "watchdog exceeded" in str(exc)
            diagnostic=("watchdog_timeout" if watchdog_timeout else
                        "no_progress_stall" if isinstance(exc,TimeoutError) else
                        "client_disconnected" if isinstance(exc,ConnectionError) else
                        "server_error" if getattr(exc,"code",None) is not None else
                        "malformed_server_stream_json" if isinstance(exc,json.JSONDecodeError) else
                        "stream_or_client_error")
            self.journal.append({"event":kind,"diagnostic_class":diagnostic,
                "call_id":call_id,"question_id":question_id,"phase":phase,"attempt_number":attempt_number,
                "timestamp_utc":utc_now(),"elapsed_monotonic_seconds":time.monotonic()-t0,
                "total_call_duration_seconds":time.monotonic()-t0,
                "error_type":type(exc).__name__,"error_message":str(exc),"first_chunk_seconds":first,
                "last_chunk_elapsed_seconds":last,"chunk_count":chunks,"content_chars":len(text),"thinking_chars":len(think),
                **({"incomplete_status":"timeout","timeout_limit_seconds":self.watchdog_seconds} if watchdog_timeout else {}),
                "resource_sample":self.heartbeat()})
            raise
        result=call_state(self.journal.read_events(self.journal.path),call_id)
        result["first_chunk_seconds"]=first;result["wall_seconds"]=time.monotonic()-t0
        return result

def reconstruct_calls(journal_path):
    events=JsonlJournal.read_events(journal_path)
    ids=list(dict.fromkeys(e["call_id"] for e in events if "call_id" in e))
    return [call_state(events,cid) for cid in ids]
