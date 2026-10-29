from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inputs import load_manifest, read_image
from src.methods import Identity, Median, Method
from src.records import timestamp, write_json
from src.restormer_adapter import RestormerAdapter
from src.timing import benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--learned', action='store_true')
    args = parser.parse_args()
    methods: list[Method] = [Identity(), Median()]
    if args.learned:
        methods.append(RestormerAdapter(Path('vendor/Restormer'), Path('weights/deraining.pth')))
    cases = load_manifest(args.manifest)[:5]
    write_json(args.output, {'at': timestamp(), 'selection': 'first five sorted manifest samples',
                             'results': [benchmark(method, read_image(Path(sample.input_path)))
                                         for sample in cases for method in methods]})


if __name__ == '__main__':
    main()
