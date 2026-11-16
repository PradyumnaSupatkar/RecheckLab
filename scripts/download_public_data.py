"""Restore the version-pinned public SEVRA data, preserving its dataset card."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "10411137fcaa02b3bd549a2285b43ffe315417f6"
DATASET = "Sajib-006/sevra-recovery-outcomes"


def main():
    folder = ROOT / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    files = [("README.md","DATASET_CARD.md"), ("release_manifest.json","release_manifest.json")]
    files += [(f"data/{split}-00000-of-00001.parquet",f"{split}.parquet") for split in ("train","math500","gsm8k","commonsenseqa")]
    hashes = {}
    for remote, local in files:
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{remote}"
        with urlopen(url, timeout=120) as response:
            content = response.read()
        (folder/local).write_bytes(content)
        hashes[local] = hashlib.sha256(content).hexdigest()
        print(f"Saved {local}: {len(content):,} bytes")
    (folder/"download_metadata.json").write_text(json.dumps({"dataset_id":DATASET,"revision":REVISION,
        "source":f"https://huggingface.co/datasets/{DATASET}","sha256":hashes},indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
