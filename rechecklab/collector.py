"""Collect actual initial/revised LLM responses with Ollama; resumable JSONL."""
from __future__ import annotations

import json
import math
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data import collection_protocol, question_hash, transition_label

INITIAL_INSTRUCTION = "Solve the question. Give a brief explanation, then finish with FINAL: followed by your answer."
RECHECK_INSTRUCTION = ("Recheck your previous answer against the question. Keep it if justified; change it only if you identify an error. "
                       "Give a brief explanation, then finish with FINAL: followed by your answer.")


class OllamaResponseError(ValueError):
    """An invalid returned call, retained only as unvalidated failure evidence."""

    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


def parse_answer(text, answer_type):
    # Require an explicit answer marker; do not guess the last arithmetic intermediate.
    marked=re.findall(r"(?:FINAL(?:\s+ANSWER)?|Answer)\s*:\s*([^\n]+)",str(text),re.I)
    candidate=marked[-1].strip() if marked else str(text).strip()
    candidate=candidate.strip(" *`.$")
    if answer_type=="boolean":
        value=candidate.lower().strip(" .!")
        return "yes" if value in ("yes","true") else "no" if value in ("no","false") else None
    if answer_type=="choice":
        value=candidate.upper().strip(" .)")
        return value if len(value)==1 and value in "ABCDE" else None
    if answer_type=="number":
        candidate=candidate.replace(",","").strip().rstrip(".")
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?",candidate):
            return None
        try:
            return str(Decimal(candidate).normalize())
        except InvalidOperation:
            return None
    raise ValueError("answer_type must be number, boolean, or choice")


def ollama_chat(model, messages, host, budget, seed, temperature, thinking, *, timeout=600, context_window=None, decoding_options=None):
    payload=dict(model=model,messages=messages,stream=False,think=thinking,
                 options=dict(num_predict=budget,seed=seed,temperature=temperature))
    if context_window is not None:
        if isinstance(context_window, bool) or not isinstance(context_window, int) or context_window < 1:
            raise ValueError("context_window must be a positive integer when supplied.")
        payload["options"]["num_ctx"] = context_window
    if decoding_options is not None:
        if not isinstance(decoding_options, dict) or set(decoding_options) - {"top_k", "top_p", "min_p", "repeat_penalty", "stop"}:
            raise ValueError("Unsupported explicit decoding options.")
        payload["options"].update(decoding_options)
    request=Request(host.rstrip("/")+"/api/chat",data=json.dumps(payload).encode(),
                    headers={"Content-Type":"application/json"})
    started=time.perf_counter()
    try:
        with urlopen(request,timeout=timeout) as response:
            result=json.load(response)
    except HTTPError as exc:
        # Ollama includes the runner failure in its JSON body. Read a bounded
        # amount so a generic HTTP 500 does not hide memory/context diagnostics.
        body = exc.read(4097)
        truncated = len(body) > 4096
        text = body[:4096].decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(text)
            detail = payload.get("error") if isinstance(payload, dict) else text
            detail = detail if isinstance(detail, str) and detail.strip() else text
        except (ValueError, TypeError):
            detail = text
        detail = detail or str(exc.reason)
        if truncated:
            detail += " [server error body truncated]"
        raise URLError(f"Ollama HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict) or result.get("done") is not True or "message" not in result:
        raise OllamaResponseError("Ollama returned an incomplete or invalid response.", result)
    result["wall_seconds"]=time.perf_counter()-started
    return result


def response_text(result):
    message=result["message"]
    thinking=message.get("thinking","")
    content=message.get("content","")
    return (thinking+"\n\n"+content).strip() if thinking else content


def actual_tokens(result):
    if "prompt_eval_count" not in result or "eval_count" not in result:
        raise ValueError("Ollama did not provide actual token counts. Refusing to estimate them.")
    counts = []
    for key in ("prompt_eval_count", "eval_count"):
        value = result[key]
        if isinstance(value, bool):
            raise ValueError(f"Ollama {key} must be a finite nonnegative integer.")
        try:
            count = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Ollama {key} must be a finite nonnegative integer.") from exc
        if not math.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError(f"Ollama {key} must be a finite nonnegative integer.")
        counts.append(int(count))
    return sum(counts)


def collect_pairs(input_path, output_path, model="qwen3:1.7b", host="http://localhost:11434", budget=768,
                  seed=42, temperature=0.0, thinking=False, limit=None, *, context_window=None):
    output=Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    metadata_path=output.with_suffix(".metadata.json")
    config=dict(model=model,host=host,budget=budget,seed=seed,temperature=temperature,thinking=thinking,
                initial_instruction=INITIAL_INSTRUCTION,recheck_instruction=RECHECK_INSTRUCTION,protocol_version="0.1")
    if context_window is not None:
        config.update(context_window=context_window, protocol_version="0.2")
    collection_protocol(config)  # Validate before writing metadata or making calls.
    chat_options = {} if context_window is None else {"context_window": context_window}
    saved_config=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    saved_parameters={k:v for k,v in saved_config.items() if k!="model_digest"} if saved_config else None
    if saved_parameters is not None and saved_parameters!=config:
        raise ValueError("Generation settings differ from this saved run. Use a new output filename.")
    existing={}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row=json.loads(line)
                existing[str(row["example_id"])]=row
        if not metadata_path.exists():
            raise ValueError("Existing output has no protocol metadata; use a new output filename.")
    questions=[json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({str(q["example_id"]) for q in questions})!=len(questions):
        raise ValueError("Input contains duplicate example IDs.")
    if limit:
        questions=questions[:limit]
    for question in questions:
        if parse_answer(question["gold_answer"],question["answer_type"]) is None:
            raise ValueError(f"Cannot normalize the answer key for {question['example_id']}")
        if str(question["example_id"]) in existing:
            if question_hash(question["query"])!=question_hash(existing[str(question["example_id"])]["query"]):
                raise ValueError("A resumed example ID now refers to a different question.")
    # One preflight avoids repeated failed inference calls when the server/model is unavailable.
    try:
        with urlopen(host.rstrip("/")+"/api/tags",timeout=10) as response:
            installed={m["name"]:m.get("digest","") for m in json.load(response).get("models",[])}
    except (URLError,TimeoutError) as exc:
        raise RuntimeError("Local Ollama is unavailable. Install/start Ollama before collecting new responses.") from exc
    if model not in installed:
        raise RuntimeError(f"Model {model} is not installed. Run: ollama pull {model}")
    config["model_digest"]=installed[model]
    if saved_config is not None and saved_config.get("model_digest")!=config["model_digest"]:
        raise ValueError("The installed model digest changed. Use a new output filename to keep runs separate.")
    metadata_path.write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    completed=0
    for index, question in enumerate(questions,1):
        if str(question["example_id"]) in existing:
            print(f"[{index}/{len(questions)}] already saved",flush=True)
            continue
        user=INITIAL_INSTRUCTION+"\n\n"+question["query"]
        messages=[dict(role="user",content=user)]
        # Neither generation call receives the benchmark answer key.
        base=ollama_chat(model,messages,host,budget,seed,temperature,thinking,**chat_options)
        previous=base["message"].get("content","") or response_text(base)
        revised=ollama_chat(model,messages+[dict(role="assistant",content=previous),
            dict(role="user",content=RECHECK_INSTRUCTION)],host,budget,seed+1,temperature,thinking,**chat_options)
        base_answer=parse_answer(base["message"].get("content",""),question["answer_type"])
        final_answer=parse_answer(revised["message"].get("content",""),question["answer_type"])
        gold=parse_answer(question["gold_answer"],question["answer_type"])
        if gold is None:
            raise ValueError(f"Cannot normalize the answer key for {question['example_id']}")
        base_correct=base_answer is not None and base_answer==gold
        final_correct=final_answer is not None and final_answer==gold
        row=dict(example_id=question["example_id"],dataset=question["dataset"],query=question["query"],
            gold_answer=gold,answer_type=question["answer_type"],model=model,rollout=0,seed=seed,
            base_response=response_text(base),base_answer=base_answer or "",base_correct=base_correct,
            action="recheck",intervention_response=response_text(revised),final_answer=final_answer or "",
            final_correct=final_correct,label=transition_label(base_correct,final_correct),
            base_actual_tokens=actual_tokens(base),action_actual_tokens=actual_tokens(revised),
            base_done_reason=base.get("done_reason",""),intervention_done_reason=revised.get("done_reason",""),
            base_finalizer_used=False,base_parse_valid=base_answer is not None,final_parse_valid=final_answer is not None,
            base_wall_seconds=base["wall_seconds"],action_wall_seconds=revised["wall_seconds"],
            model_digest=config["model_digest"],
            provenance="New local Ollama generation; fixed initial answer followed by one explicit self-recheck")
        with output.open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(row,ensure_ascii=False)+"\n")
        completed+=1
        print(f"[{index}/{len(questions)}] {row['label']} | {row['base_actual_tokens']} + {row['action_actual_tokens']} tokens",flush=True)
    return completed
