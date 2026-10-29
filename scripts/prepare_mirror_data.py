from __future__ import annotations

import csv
import hashlib
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inputs import ValidationError, load_manifest, read_image
from src.records import sha256, timestamp, write_json

REVISION = 'c681112a79de207e5b0e810ce9fb4b30a037903a'


def verified_extract(archive: Path, destination: Path, expected: str) -> None:
    if sha256(archive) != expected:
        raise ValidationError(f'Archive checksum mismatch: {archive}')
    if destination.exists():
        raise ValidationError(f'Refusing to overwrite extracted data: {destination}')
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        seen: set[str] = set()
        if sum(member.file_size for member in bundle.infolist()) > 1_000_000_000:
            raise ValidationError('Archive exceeds extraction size limit')
        for member in bundle.infolist():
            resolved = (root / member.filename).resolve()
            if not resolved.is_relative_to(root) or '\\' in member.filename or ':' in member.filename:
                raise ValidationError('Unsafe archive path')
            key = member.filename.casefold()
            if key in seen or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValidationError('Duplicate archive path or symbolic link')
            seen.add(key)
        if bundle.testzip() is not None:
            raise ValidationError('Archive CRC failure')
        destination.mkdir(parents=True)
        bundle.extractall(destination)


def numbered_files(directory: Path, prefix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(prefix + r'-(\d+)\.png', path.name)
        if match is None or int(match[1]) in result:
            raise ValidationError(f'Unrecognized or duplicate numeric sample name: {path}')
        result[int(match[1])] = path
    return result


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    load_manifest(path)


def main() -> None:
    Path('docs').mkdir(parents=True, exist_ok=True)
    rain_manifest = Path('manifests/rain100l_mirror_test.csv')
    real_manifest = Path('manifests/practical_mirror_qualitative.csv')
    if rain_manifest.exists() or real_manifest.exists():
        raise ValidationError('Mirror manifests already frozen; reuse them rather than overwrite')
    root = Path('data/mirror_20261029')
    verified_extract(Path('data/Rain100L_mirror.zip'), root / 'paired',
                     '4beeb9788effde191f7738fe0704ca59b97dc1d74f8be7d039c7b6e6b916bbc3')
    verified_extract(Path('data/Practical_mirror.zip'), root / 'real',
                     'a9062098d65e4c86171da6370c3a8667367976d82958042fe3bfa89ead1fb2be')
    inputs = numbered_files(root / 'paired/Rain100L/image', 'rain')
    references = numbered_files(root / 'paired/Rain100L/gt', 'norain')
    if set(inputs) != set(range(1, 101)) or inputs.keys() != references.keys():
        raise ValidationError('Expected exactly the numeric IDs 1..100 in both paired folders')
    rows: list[dict[str, Any]] = []
    dimensions: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    for number in sorted(inputs):
        inp, ref = inputs[number], references[number]
        image, clean = read_image(inp), read_image(ref)
        if image.shape != clean.shape:
            raise ValidationError(f'Geometry mismatch at sample {number}')
        scene = hashlib.sha256(clean.tobytes()).hexdigest()[:20]
        scenes[scene] += 1
        h, w = image.shape[:2]
        dimensions[f'{w}x{h}'] += 1
        rows.append({'sample_id': f'rain100l-{number:03d}', 'input_path': str(inp.resolve()),
                     'reference_path': str(ref.resolve()), 'split': 'test', 'kind': 'paired',
                     'scene_id': scene, 'source': f'Rain100L-labeled AaronCIH/PIR mirror {REVISION}; '
                     'original archive byte equivalence unverified; docs/mirror_data_provenance.json',
                     'input_sha256': sha256(inp), 'reference_sha256': sha256(ref), 'width': w, 'height': h,
                     'reference_status': 'benchmark'})
    write_manifest(rain_manifest, rows)
    real_rows = []
    candidates = sorted((root / 'real/Practical/image').glob('*.jpg'))
    for index, path in enumerate(candidates[:5]):
        image = read_image(path)
        real_rows.append({'sample_id': f'practical-{index:02d}', 'input_path': str(path.resolve()),
                          'reference_path': '', 'split': 'qualitative', 'kind': 'real',
                          'scene_id': path.stem, 'source': f'Practical-labeled AaronCIH/PIR mirror {REVISION}; '
                          'no aligned clean reference; local qualitative use', 'input_sha256': sha256(path),
                          'reference_sha256': '', 'width': image.shape[1], 'height': image.shape[0],
                          'reference_status': 'none'})
    write_manifest(real_manifest, real_rows)
    write_json(Path('docs/mirror_data_provenance.json'), {
        'frozen_at': timestamp(), 'mirror': 'https://huggingface.co/datasets/AaronCIH/PIR',
        'revision': REVISION, 'paired_count': len(rows), 'paired_dimensions': dict(dimensions),
        'unique_reference_pixel_hash_groups': len(scenes), 'repeated_groups': sum(n > 1 for n in scenes.values()),
        'pairing': 'Explicit numeric suffix mapping rain-NNN.png to norain-NNN.png; exact IDs 1..100',
        'selection': 'All 100 paired entries; first five lexicographically sorted Practical jpgs before outputs',
        'qualitative_available': len(candidates), 'qualitative_selected': len(real_rows),
        'resizing': 'none; native decoded geometry', 'selected_before_scores': True,
        'archive_sha256': {'Rain100L.zip': sha256(Path('data/Rain100L_mirror.zip')),
                           'Practical.zip': sha256(Path('data/Practical_mirror.zip'))},
        'limitations': 'Mirror integrity verified against its published LFS pointers. Original Restormer/author '
        'archives unavailable; exact original equivalence not independently established. No explicit dataset '
        'redistribution license found. Keep image-containing runs local; share measurements and attribution only. '
        'Rain100L itself is a synthetic-rain benchmark, not real-rain ground truth. Scene IDs group identical '
        'reference pixels; different views of one scene may remain unidentified.'})
    print(f'Frozen {len(rows)} native paired samples and {len(real_rows)} unreferenced qualitative samples')
    print(f'Paired dimensions: {dict(dimensions)}; unique reference groups: {len(scenes)}')


if __name__ == '__main__':
    main()
