from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inputs import ValidationError, read_image
from src.records import sha256


def freeze(inputs: Path, references: Path | None, output: Path, kind: str, source: str,
           split: str, reference_status: str) -> None:
    files = sorted(path for path in inputs.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not files:
        raise ValidationError('No PNG/JPEG inputs found')
    if output.exists():
        raise ValidationError('Refusing to overwrite frozen manifest; choose a new file')
    rows = []
    for index, path in enumerate(files):
        reference = path if kind == 'clean' else (references / path.name if references else None)
        image = read_image(path)
        if reference and (not reference.is_file() or read_image(reference).shape != image.shape):
            raise ValidationError(f'Exact filename pairing failed for {path.name}; supply verified matching names')
        rows.append({'sample_id': f'{kind}-{index:04d}', 'input_path': str(path.resolve()),
                     'reference_path': str(reference.resolve()) if reference else '',
                     'split': split, 'source': source, 'scene_id': '', 'kind': kind,
                     'input_sha256': sha256(path), 'reference_sha256': sha256(reference) if reference else '',
                     'width': image.shape[1], 'height': image.shape[0], 'reference_status': reference_status})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Frozen {len(rows)} samples: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freeze explicit exact-filename image pairs before scoring')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['paired', 'clean', 'real', 'synthetic'], required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--reference-status', choices=['benchmark', 'aligned_user', 'self_clean', 'none',
                                                      'synthetic'], required=True)
    args = parser.parse_args()
    freeze(args.inputs, args.references, args.output, args.kind, args.source, args.split, args.reference_status)
