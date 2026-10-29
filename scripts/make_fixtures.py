from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.freeze_manifest import freeze
from src.inputs import save_image


def main() -> None:
    Path('manifests').mkdir(parents=True, exist_ok=True)
    if Path('manifests/synthetic.csv').exists():
        raise ValueError('Synthetic manifest already frozen; reuse it or choose a new fixture dataset')
    root = Path('data/synthetic')
    for folder in ['input', 'reference']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for index in range(5):
        y, x = np.mgrid[:64 + index, :80 + index]
        clean = np.stack([(x * 3) % 256, (y * 4) % 256, ((x + y) * 2) % 256], axis=-1).astype(np.uint8)
        rainy = clean.copy()
        mask = rng.random(clean.shape[:2]) < 0.08
        rainy[mask] = 235
        save_image(root / 'input' / f'fixture-{index}.png', rainy)
        save_image(root / 'reference' / f'fixture-{index}.png', clean)
    freeze(root / 'input', root / 'reference', Path('manifests/synthetic.csv'), 'synthetic',
           'Project-generated deterministic software fixtures; NOT real photographs or benchmark data',
           'development', 'synthetic')


if __name__ == '__main__':
    main()
