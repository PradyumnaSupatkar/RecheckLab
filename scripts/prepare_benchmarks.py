"""Prepare a small real GSM8K/BoolQ question file for local Ollama collection."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["gsm8k","boolq"], required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    if args.limit < 1 or args.offset < 0:
        parser.error("Limit must be positive and offset nonnegative.")
    dataset, config, split = ("openai/gsm8k","main","train") if args.task == "gsm8k" else ("google/boolq","default","train")
    records=[]
    for start in range(args.offset,args.offset+args.limit,100):
        params=dict(dataset=dataset,config=config,split=split,offset=start,length=min(100,args.offset+args.limit-start))
        with urlopen("https://datasets-server.huggingface.co/rows?"+urlencode(params),timeout=90) as response:
            payload=json.load(response)
        for entry in payload["rows"]:
            row=entry["row"]
            if args.task == "gsm8k":
                query=row["question"]
                gold=row["answer"].split("####")[-1].strip()
                kind="number"
            else:
                query=f"Passage:\n{row['passage']}\n\nQuestion: {row['question']}\nAnswer yes or no using the passage."
                gold="yes" if row["answer"] else "no"
                kind="boolean"
            records.append(dict(example_id=f"{args.task}:train:{entry['row_idx']}",dataset=args.task,
                query=query,gold_answer=gold,answer_type=kind,source_dataset=dataset,source_split=split))
    output=args.output or ROOT/"data"/(args.task+"_questions.jsonl")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    print(f"Prepared {len(records)} benchmark questions in {output}")
    print("Answer keys are used only by offline scoring; they are not included in prompts.")


if __name__=="__main__":
    main()
