from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from skimage.metrics import structural_similarity

from src.inputs import RGBImage, ValidationError

PROTOCOL = 'rgb_full_frame_png_v1'
ssim_score = cast(Callable[..., float], structural_similarity)


def measure(output: RGBImage, reference: RGBImage | None) -> dict[str, Any]:
    if reference is None:
        return {'psnr_db': None, 'psnr_exact_match': False, 'ssim': None, 'mse': None,
                'mean_absolute_change': None, 'max_absolute_change': None,
                'metric_reason': 'no_aligned_reference'}
    if reference.shape != output.shape or min(output.shape[:2]) < 7:
        raise ValidationError('Metrics require equal RGB geometry and dimensions >= 7')
    a = output.astype(np.float64) / 255.0
    b = reference.astype(np.float64) / 255.0
    difference = np.abs(a - b)
    mse = float(np.mean((a - b) ** 2))
    exact = mse == 0.0
    return {'psnr_db': None if exact else 10.0 * math.log10(1.0 / mse),
            'psnr_exact_match': exact, 'mse': mse,
            'ssim': float(ssim_score(a, b, data_range=1.0, channel_axis=-1,
                                                win_size=7, gaussian_weights=False,
                                                use_sample_covariance=True)),
            'mean_absolute_change': float(difference.mean()),
            'max_absolute_change': float(difference.max()), 'metric_reason': None}
