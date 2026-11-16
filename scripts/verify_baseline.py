"""Record and verify checksums of the untouched v0.1 baseline and raw data."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'BASELINE_v0.1_SHA256.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def baseline_files():
    for directory in ('data/raw', 'data/prepared', 'models', 'reports'):
        yield from sorted((ROOT / directory).rglob('*'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    if args.record:
        if MANIFEST.exists():
            raise SystemExit('Baseline manifest already exists; refusing to replace it.')
        files = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in baseline_files() if p.is_file()}
        MANIFEST.write_text(json.dumps({'baseline': 'Original shipped v0.1; never overwritten by v0.2 runs', 'sha256': files}, indent=2) + '\n', encoding='utf-8')
        print('Recorded', len(files), 'baseline files.')
    else:
        hashes = json.loads(MANIFEST.read_text(encoding='utf-8'))['sha256']
        bad = [name for name, expected in hashes.items() if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
        if bad:
            raise SystemExit('Baseline changed: ' + ', '.join(bad))
        print('Verified unchanged:', len(hashes), 'v0.1/raw files.')

if __name__ == '__main__':
    main()
