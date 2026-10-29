from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.records import timestamp, write_json

URL = 'https://huggingface.co/deepinv/Restormer/resolve/aa2667c79d766b67da2c1489c7ce5f0423bae798/deraining.pth?download=true'
EXPECTED = '225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681'


if __name__ == '__main__':
    target = Path('weights/deraining.pth')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED:
        print('Existing checkpoint matches the pinned published SHA-256')
    else:
        partial = target.with_suffix('.part')
        subprocess.run(['curl.exe', '-L', '--fail', '--retry', '2', '--max-time', '1200', '-C', '-',
                        URL, '-o', str(partial)], check=True)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != EXPECTED:
            raise RuntimeError('Checkpoint hash mismatch; refusing to install')
        partial.replace(target)
        print('Checkpoint downloaded and SHA-256 verified')
    write_json(target.parent / 'provenance.json', {'at': timestamp(), 'url': URL, 'sha256': EXPECTED,
        'mirror_revision': 'aa2667c79d766b67da2c1489c7ce5f0423bae798',
        'original_drive_equivalence': 'Not independently established while original is quota blocked'})
