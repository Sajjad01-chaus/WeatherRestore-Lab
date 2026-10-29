from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.inputs import RGBImage
from src.methods import Method


def benchmark(method: Method, image: RGBImage, warmups: int = 10, repeats: int = 30) -> dict[str, Any]:
    for _ in range(warmups):
        method.restore(image)
        method.synchronize()
    measurements = []
    for _ in range(repeats):
        method.synchronize()
        start = time.perf_counter()
        method.restore(image)
        method.synchronize()
        measurements.append((time.perf_counter() - start) * 1000)
    return {'method': method.name, 'width': image.shape[1], 'height': image.shape[0],
            'warmups': warmups, 'repeats': repeats, 'median_ms': float(np.median(measurements)),
            'p95_ms': float(np.percentile(measurements, 95)), 'measurements_ms': measurements,
            'boundary': 'decoded input to returned uint8 output; includes device transfers and quantization; '
                        'excludes decode, save, metrics and rendering', 'method_metadata': method.metadata()}
