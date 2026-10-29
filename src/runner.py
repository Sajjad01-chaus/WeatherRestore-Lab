from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.inputs import Sample, ValidationError, load_manifest, read_image, save_image
from src.methods import Method, RestorationError
from src.metrics import PROTOCOL, measure
from src.records import environment, event, sha256, timestamp, write_json
from src.reporting import export


class StudyRunner:
    def __init__(self, methods: Sequence[Method]) -> None:
        self.methods = list(methods)
        names = [method.name for method in methods]
        if not names or len(set(names)) != len(names) or 'identity' not in names:
            raise ValidationError('Select unique methods including identity')
        self.methods.sort(key=lambda method: method.name != 'identity')

    def execute(self, manifest: Path, root: Path, label: str) -> Path:
        samples = load_manifest(manifest)
        run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        run = root / run_id
        run.mkdir(parents=True, exist_ok=False)
        config = {'protocol': PROTOCOL, 'methods': {m.name: m.metadata() for m in self.methods},
                  'quantization': 'numpy rint (nearest, ties-to-even), clamp [0,1], uint8 RGB PNG',
                  'orientation': 'EXIF transpose before RGB conversion; no ICC transform',
                  'precision': 'fp32 learned inference; float64 metrics', 'batch_size': 1,
                  'resizing': 'none', 'evidence_label': label,
                  'inspection_thresholds': {'psnr_decrease_db': 0.1, 'ssim_decrease': 0.001}}
        write_json(run / 'config.json', config)
        write_json(run / 'manifest.json', [sample.record() for sample in samples])
        metadata: dict[str, Any] = {'run_id': run_id, 'started_at': timestamp(), 'status': 'running',
                                    'protocol': PROTOCOL, 'evidence_label': label,
                                    'methods': [m.name for m in self.methods],
                                    'manifest_sha256': sha256(manifest),
                                    'frozen_manifest_sha256': sha256(run / 'manifest.json'),
                                    'config_sha256': sha256(run / 'config.json'),
                                    'expected_records': len(samples) * len(self.methods),
                                    'environment': environment(), 'exclusions': []}
        write_json(run / 'source_hashes.json', {str(path): sha256(path) for path in
                   sorted([*Path('src').glob('*.py'), Path('run_study.py'), Path('app.py')])})
        write_json(run / 'metadata.json', metadata)
        records: list[dict[str, Any]] = []
        event(run / 'events.jsonl', 'run_started', run_id=run_id)
        try:
            for sample in samples:
                self._sample(sample, run, records)
                write_json(run / 'records.json', records)
                export(run, metadata, records)
            metadata['status'] = 'partial' if any(r['status'] != 'success' for r in records) else 'complete'
        except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
            metadata['status'] = 'partial'
            metadata['fatal_error'] = f'{type(error).__name__}: {error}'
            event(run / 'events.jsonl', 'run_aborted', error=metadata['fatal_error'])
            raise
        finally:
            metadata['finished_at'] = timestamp()
            metadata['actual_records'] = len(records)
            metadata['method_metadata'] = {m.name: m.metadata() for m in self.methods}
            write_json(run / 'records.json', records)
            try:
                export(run, metadata, records)
            except (OSError, ValueError) as error:
                metadata['status'] = 'partial'
                metadata['export_error'] = f'{type(error).__name__}: {error}'
                write_json(run / 'metadata.json', metadata)
                raise
            write_json(run / 'metadata.json', metadata)
            event(run / 'events.jsonl', 'run_finalized', status=metadata['status'], records=len(records))
        return run

    def _sample(self, sample: Sample, run: Path, records: list[dict[str, Any]]) -> None:
        if sha256(Path(sample.input_path)) != sample.input_sha256 or (sample.reference_path and
                sha256(Path(sample.reference_path)) != sample.reference_sha256):
            raise ValidationError(f'{sample.sample_id}: source changed after manifest validation')
        image = read_image(Path(sample.input_path))
        reference = read_image(Path(sample.reference_path)) if sample.reference_path else None
        input_asset = f'assets/{sample.sample_id}/input.png'
        reference_asset = f'assets/{sample.sample_id}/reference.png' if reference is not None else ''
        save_image(run / input_asset, image)
        if reference is not None:
            save_image(run / reference_asset, reference)
        identity: dict[str, Any] | None = None
        for method in self.methods:
            row: dict[str, Any] = {'sample_id': sample.sample_id, 'scene_id': sample.scene_id,
                                    'kind': sample.kind, 'split': sample.split, 'source': sample.source,
                                    'run_id': run.name, 'method': method.name, 'status': 'running',
                                    'input_sha256': sample.input_sha256,
                                    'reference_sha256': sample.reference_sha256,
                                    'checkpoint_sha256': method.metadata().get('checkpoint_sha256'),
                                    'width': sample.width, 'height': sample.height,
                                    'reference_status': sample.reference_status,
                                    'input_asset': input_asset, 'reference_asset': reference_asset,
                                    'output_path': '', 'error': None, 'delta_psnr_db': None,
                                    'delta_ssim': None, 'inspection_flag': False,
                                    'metric_disagreement': False, 'precision': 'fp32' if
                                    method.name == 'restormer' else 'uint8', 'resize_policy': 'none'}
            try:
                method.synchronize()
                application_start = time.perf_counter()
                inference_start = time.perf_counter()
                output = method.restore(image.copy())
                method.synchronize()
                row['inference_ms'] = (time.perf_counter() - inference_start) * 1000
                if output.shape != image.shape:
                    raise ValidationError('Restoration changed image geometry')
                output_path = f'outputs/{method.name}/{sample.sample_id}.png'
                save_image(run / output_path, output)
                row['application_processing_ms'] = (time.perf_counter() - application_start) * 1000
                decoded = read_image(run / output_path)
                row.update(measure(decoded, reference))
                row['output_path'] = output_path
                row['output_sha256'] = sha256(run / output_path)
                row['processed_width'], row['processed_height'] = decoded.shape[1], decoded.shape[0]
                row['input_mean_absolute_change'] = float(np.abs(decoded.astype(float) - image).mean() / 255)
                row['input_max_absolute_change'] = float(np.abs(decoded.astype(float) - image).max() / 255)
                if method.name == 'identity':
                    identity = row
                elif identity is not None:
                    if row['psnr_db'] is not None and identity['psnr_db'] is not None:
                        row['delta_psnr_db'] = row['psnr_db'] - identity['psnr_db']
                    if row['ssim'] is not None and identity['ssim'] is not None:
                        row['delta_ssim'] = row['ssim'] - identity['ssim']
                    dp, ds = row['delta_psnr_db'], row['delta_ssim']
                    row['inspection_flag'] = bool((dp is not None and dp < -0.1) or
                                                  (ds is not None and ds < -0.001))
                    row['metric_disagreement'] = bool(dp is not None and ds is not None and dp * ds < 0)
                row['status'] = 'success'
            except (RestorationError, ValidationError, OSError, ValueError) as error:
                row['status'] = 'failed'
                row['error'] = f'{type(error).__name__}: {error}'
                row['metric_reason'] = 'method_failed'
            records.append(row)
            event(run / 'events.jsonl', 'method_finished', **row)
            write_json(run / 'records.json', records)
