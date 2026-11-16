"""Inspect a real checkpoint, smoke a candidate, then freeze and reuse exact calls."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rechecklab.study_collection import build_pair, journal_state, read_jsonl, validate_manifest, write_json
from rechecklab.study_protocol import create_protocol, freeze_protocol, load_protocol, protocol_hash, verify_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("outputs/live_study/v0.3.0"))
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--candidate", action="store_true", help="Inspect actual backend and save a provisional protocol before smoke.")
    args = parser.parse_args()
    root = args.study_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.candidate:
        protocol = create_protocol(args.host, args.model)
        candidate = root / "candidate_protocol.json"
        if candidate.exists() and json.loads(candidate.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Candidate changed; create a new versioned study rather than mixing smoke calls.")
        write_json(candidate, protocol)
        print(f"Candidate: {candidate}\nProtocol hash: {protocol_hash(protocol)}\nRun --smoke 4 before freezing.")
        return
    protocol = json.loads((root / "candidate_protocol.json").read_text(encoding="utf-8"))
    verify_backend(protocol, args.host)
    smoke = root / "smoke"
    smoke_protocol, digest = load_protocol(smoke)
    smoke_status = json.loads((smoke / "collection_status.json").read_text(encoding="utf-8"))
    pairs = read_jsonl(smoke / "pairs.jsonl")
    if smoke_protocol != protocol or smoke_status.get("smoke_status") != "COMPLETE" or not pairs:
        raise ValueError("Candidate smoke did not finish under the final exact protocol.")
    if any(p["partition"] not in ("source_train", "target_adaptation_pool") for p in pairs):
        raise ValueError("Smoke contains held-out questions.")
    if not any(p["base_parse_valid"] and p["final_parse_valid"] for p in pairs):
        raise ValueError("Smoke has no pair with both parsed answers; resolve parsing/budget problems in a new protocol version.")
    questions = read_jsonl(root / "questions.jsonl")
    keys = validate_manifest(questions, read_jsonl(root / "trusted_keys.jsonl"))
    successes, failures, pending = journal_state(read_jsonl(smoke / "calls.jsonl"), questions, digest, protocol)
    if failures or pending:
        raise ValueError("Smoke journal contains failed or uncertain calls; resolve before protocol promotion.")
    for pair in pairs:
        identity = pair["example_id"]
        question = next(q for q in questions if q["example_id"] == identity)
        if not all((identity, phase) in successes for phase in ("initial", "recheck")):
            raise ValueError("Smoke pair lacks two real validated call responses.")
        rebuilt = build_pair(question, keys[identity], successes[(identity, "initial")], successes[(identity, "recheck")], protocol, digest)
        if any(pair.get(key) != value for key, value in rebuilt.items() if key != "timestamp_utc"):
            raise ValueError("Smoke pair differs from actual responses/metadata or trusted offline grading.")
    freeze_protocol(root, protocol)
    for name in ("calls.jsonl", "pairs.jsonl"):
        destination = root / name
        if not destination.exists():
            shutil.copyfile(smoke / name, destination)
    write_json(root / "smoke_reuse.json", {"protocol_hash": digest, "reused_pairs": len(pairs),
               "reason": "exact frozen protocol and original training/adaptation partitions; no test smoke"})
    print(f"Frozen {root / 'protocol.json'}\nProtocol hash: {digest}\nReused {len(pairs)} genuine smoke pairs.")


if __name__ == "__main__":
    main()
