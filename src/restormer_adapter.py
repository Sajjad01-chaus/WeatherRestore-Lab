from __future__ import annotations

import importlib.util
import pickle
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from filelock import FileLock, Timeout

from src.inputs import RGBImage
from src.methods import RestorationError
from src.records import revision, sha256


class RestormerAdapter:
    name = 'restormer'

    def __init__(self, source: Path, checkpoint: Path, device: str = 'cuda') -> None:
        started = time.perf_counter()
        lock_path = Path(__file__).resolve().parents[1] / '.runtime/gpu.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.gpu_lock = FileLock(lock_path, timeout=0)
        if not checkpoint.is_file():
            raise RestorationError(f'Missing official deraining checkpoint: {checkpoint}')
        architecture = source / 'restormer_arch.py'
        if not architecture.is_file():
            architecture = source / 'basicsr/models/archs/restormer_arch.py'
        if not architecture.is_file():
            raise RestorationError(f'Missing official architecture: {architecture}')
        try:
            import torch
            torch.manual_seed(42)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if device == 'cuda' and not torch.cuda.is_available():
                raise RestorationError('CUDA unavailable; use saved-run viewer or explicitly select CPU')
            spec = importlib.util.spec_from_file_location('weatherrestore_upstream', architecture)
            if spec is None or spec.loader is None:
                raise RestorationError('Cannot import upstream architecture')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.torch = torch
            self.device = device
            self.model: Any = module.Restormer(inp_channels=3, out_channels=3, dim=48,
                                               num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                                               heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                                               bias=False, LayerNorm_type='WithBias',
                                               dual_pixel_task=False)
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.model.load_state_dict(weights['params'], strict=True)
            self.model.to(device=device, dtype=torch.float32).eval()
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            self.synchronize()
        except (ImportError, OSError, RuntimeError, KeyError, ValueError, pickle.UnpicklingError) as error:
            raise RestorationError(f'Restormer initialization failed: {error}') from error
        revision_file = source / 'SOURCE_REVISION'
        source_revision = revision_file.read_text(encoding='utf-8').strip() if revision_file.is_file() \
            else revision(source)
        self.details = {'source_revision': source_revision, 'architecture_sha256': sha256(architecture),
                        'checkpoint_sha256': sha256(checkpoint), 'checkpoint': str(checkpoint.resolve()),
                        'device': device, 'precision': 'fp32', 'padding': 'reflect to multiple of 8',
                        'seed': 42, 'tf32': False,
                        'startup_ms': (time.perf_counter() - started) * 1000,
                        'training_data': 'Rain13K per upstream configuration; overlap not independently verified',
                        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None}

    def restore(self, image: RGBImage) -> RGBImage:
        try:
            with self.gpu_lock:
                return self._restore(image)
        except Timeout as error:
            raise RestorationError('Another project inference is active; wait for it to finish') from error

    def _restore(self, image: RGBImage) -> RGBImage:
        torch = self.torch
        h, w = image.shape[:2]
        try:
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
            with torch.inference_mode():
                tensor = tensor.unsqueeze(0).to(self.device)
                tensor = torch.nn.functional.pad(tensor, (0, (-w) % 8, 0, (-h) % 8), mode='reflect')
                output = self.model(tensor)[:, :, :h, :w].clamp(0, 1)
                array = output[0].permute(1, 2, 0).cpu().numpy()
            if not np.isfinite(array).all():
                raise RestorationError('Restormer returned nonfinite pixels')
            return cast(RGBImage, np.rint(array * 255).astype(np.uint8))
        except RuntimeError as error:
            raise RestorationError(f'Restormer inference failed at {w}x{h}: {error}') from error

    def synchronize(self) -> None:
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def metadata(self) -> dict[str, Any]:
        return {**self.details, 'peak_allocated_gpu_bytes':
                self.torch.cuda.max_memory_allocated() if self.device == 'cuda' else None}
