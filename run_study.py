from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.methods import Identity, Median, Method
from src.restormer_adapter import RestormerAdapter
from src.runner import StudyRunner

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a frozen WeatherRestore manifest')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--methods', nargs='+', choices=['identity', 'median3', 'restormer'],
                        default=['identity', 'median3', 'restormer'])
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/deraining.pth'))
    parser.add_argument('--source', type=Path, default=Path('third_party/restormer'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--label', required=True, help='Describe scope, dataset and limitations honestly')
    args = parser.parse_args()
    methods: list[Method] = []
    for name in args.methods:
        if name == 'restormer':
            methods.append(RestormerAdapter(args.source, args.checkpoint, args.device))
        else:
            methods.append(Identity() if name == 'identity' else Median())
    run = StudyRunner(methods).execute(args.manifest, args.runs, args.label)
    print(run.resolve())
    if json.loads((run / 'metadata.json').read_text(encoding='utf-8'))['status'] != 'complete':
        LOGGER.error('Study is partial; inspect saved errors and coverage at %s', run)
        raise SystemExit(2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.exception('Study stopped')
        raise SystemExit(1) from error
