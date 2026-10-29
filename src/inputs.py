from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from src.records import sha256

RGBImage = NDArray[np.uint8]


class ValidationError(ValueError):
    pass


def read_image(path: Path) -> RGBImage:
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValidationError(f'{path.name}: exceeds 20 MB limit')
    try:
        with Image.open(path) as image:
            if image.format not in {'PNG', 'JPEG'}:
                raise ValidationError('Only PNG and JPEG still images are supported')
            if getattr(image, 'n_frames', 1) != 1:
                raise ValidationError('Animated images are unsupported')
            if image.width * image.height > 20_000_000:
                raise ValidationError('Image exceeds 20 megapixels')
            if image.mode not in {'RGB', 'L'} or 'transparency' in image.info:
                raise ValidationError(f'Unsupported color mode/transparency: {image.mode}; use opaque RGB')
            oriented = ImageOps.exif_transpose(image)
            if min(oriented.size) < 8:
                raise ValidationError('Both image dimensions must be at least 8 pixels')
            return np.asarray(oriented.convert('RGB'), dtype=np.uint8).copy()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValidationError(f'Cannot decode {path.name}: {error}') from error


def save_image(path: Path, array: RGBImage) -> None:
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValidationError('Output must be an H x W x 3 uint8 RGB array')
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format='PNG')


@dataclass(frozen=True)
class Sample:
    sample_id: str
    input_path: str
    reference_path: str
    split: str
    source: str
    scene_id: str
    kind: str
    input_sha256: str
    reference_sha256: str
    width: int
    height: int
    reference_status: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path) -> list[Sample]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValidationError('Manifest contains no samples')
    samples: list[Sample] = []
    ids: set[str] = set()
    scenes: dict[str, str] = {}
    for row in rows:
        required = {'sample_id', 'input_path', 'reference_path', 'split', 'source', 'scene_id', 'kind',
                    'input_sha256', 'reference_sha256', 'width', 'height', 'reference_status'}
        if required - row.keys():
            raise ValidationError(f'Missing manifest columns: {sorted(required - row.keys())}')
        sid = row['sample_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in ids:
            raise ValidationError(f'Invalid or duplicate sample ID: {sid}')
        ids.add(sid)
        if row['kind'] not in {'paired', 'clean', 'real', 'synthetic'}:
            raise ValidationError(f'{sid}: unknown sample kind')
        if not row['source'] or not row['split']:
            raise ValidationError(f'{sid}: source and split are required')
        scene = row['scene_id']
        if scene and scenes.setdefault(scene, row['split']) != row['split']:
            raise ValidationError(f'{sid}: scene occurs in multiple splits')
        inp = (path.parent / row['input_path']).resolve()
        ref = (path.parent / row['reference_path']).resolve() if row['reference_path'] else None
        image = read_image(inp)
        if sha256(inp) != row['input_sha256']:
            raise ValidationError(f'{sid}: input hash mismatch')
        if (image.shape[1], image.shape[0]) != (int(row['width']), int(row['height'])):
            raise ValidationError(f'{sid}: manifest dimensions disagree with decoded image')
        status = row['reference_status']
        if ref:
            if status not in {'benchmark', 'aligned_user', 'self_clean', 'synthetic'}:
                raise ValidationError(f'{sid}: reference alignment/provenance not acknowledged')
            if sha256(ref) != row['reference_sha256']:
                raise ValidationError(f'{sid}: reference hash mismatch')
            if read_image(ref).shape != image.shape:
                raise ValidationError(f'{sid}: mismatched reference geometry')
        elif status != 'none' or row['kind'] in {'paired', 'clean', 'synthetic'}:
            raise ValidationError(f'{sid}: missing required reference')
        if row['kind'] == 'clean' and (not ref or not np.array_equal(image, read_image(ref))):
            raise ValidationError(f'{sid}: clean control reference must equal original input')
        samples.append(Sample(sid, str(inp), str(ref) if ref else '', row['split'], row['source'], scene,
                              row['kind'], row['input_sha256'], row['reference_sha256'],
                              int(row['width']), int(row['height']), status))
    return sorted(samples, key=lambda sample: sample.sample_id)
