from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inputs import ValidationError
from src.methods import Identity, Median, Method
from src.records import sha256
from src.restormer_adapter import RestormerAdapter
from src.runner import StudyRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay saved input/reference pixels using their frozen protocol')
    parser.add_argument('run', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/deraining.pth'))
    parser.add_argument('--source', type=Path, default=Path('third_party/restormer'))
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    args = parser.parse_args()
    metadata = json.loads((args.run / 'metadata.json').read_text(encoding='utf-8'))
    frozen = json.loads((args.run / 'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    if config['protocol'] != 'rgb_full_frame_png_v1' or config['resizing'] != 'none':
        raise ValidationError('Replay supports only the declared frozen full-frame PNG protocol')
    methods: list[Method] = []
    for name in metadata['methods']:
        if name == 'restormer':
            if sha256(args.checkpoint) != config['methods'][name]['checkpoint_sha256']:
                raise ValidationError('Replay checkpoint differs from saved checkpoint hash')
            methods.append(RestormerAdapter(args.source, args.checkpoint, config['methods'][name]['device']))
        elif name in {'identity', 'median3'}:
            methods.append(Identity() if name == 'identity' else Median())
        else:
            raise ValidationError(f'Unsupported replay method: {name}')
    rows = []
    for row in frozen:
        inp = (args.run / 'assets' / row['sample_id'] / 'input.png').resolve()
        ref = (inp.parent / 'reference.png') if row['reference_path'] else None
        rows.append({**row, 'input_path': str(inp), 'reference_path': str(ref) if ref else '',
                     'input_sha256': sha256(inp), 'reference_sha256': sha256(ref) if ref else '',
                     'source': f'Replay saved PNG pixels of {metadata["run_id"]}; original source: {row["source"]}'})
    path = Path('.runtime') / f'replay-{metadata["run_id"]}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(StudyRunner(methods).execute(path, args.runs, f'Replay of {metadata["run_id"]}: '
                                     f'{metadata["evidence_label"]}').resolve())
