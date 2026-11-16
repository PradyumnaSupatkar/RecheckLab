import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rechecklab.collector import collect_pairs


def main():
    parser=argparse.ArgumentParser(description="Collect real paired responses from a local Ollama model.")
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--model",default="qwen3:1.7b")
    parser.add_argument("--host",default="http://localhost:11434")
    parser.add_argument("--budget",default=768,type=int)
    parser.add_argument("--context-window",default=4096,type=int,
                        help="Explicit Ollama context limit; recorded in the generation protocol (default: 4096).")
    parser.add_argument("--seed",default=42,type=int)
    parser.add_argument("--temperature",default=0.0,type=float)
    parser.add_argument("--thinking",action="store_true")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.budget < 32 or args.context_window < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Budget must be >=32; context window and limit must be positive.")
    try:
        count=collect_pairs(args.input,args.output,args.model,args.host,args.budget,args.seed,args.temperature,args.thinking,args.limit,
                            context_window=args.context_window)
        print(f"Saved {count} new paired outcomes.")
    except (RuntimeError,ValueError,OSError) as exc:
        parser.exit(1,f"Collection stopped: {exc}\nCompleted pairs remain saved; rerun with the same settings to resume.\n")


if __name__=="__main__":
    main()
