from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from scripts.freeze_manifest import freeze
from src.inputs import RGBImage, ValidationError, load_manifest, read_image, save_image
from src.methods import Identity, Median, RestorationError
from src.metrics import measure
from src.records import write_json
from src.reporting import summarize
from src.runner import StudyRunner


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    inputs, refs = tmp_path / 'input', tmp_path / 'reference'
    inputs.mkdir()
    refs.mkdir()
    for index in range(2):
        clean = np.full((17, 19, 3), 60 + index, dtype=np.uint8)
        rainy = clean.copy()
        rainy[8, 9] = 240
        save_image(inputs / f'{index}.png', rainy)
        save_image(refs / f'{index}.png', clean)
    output = tmp_path / 'manifest.csv'
    freeze(inputs, refs, output, 'synthetic', 'generated validation fixtures', 'dev', 'synthetic')
    return output


def test_identity_and_saved_representation(tmp_path: Path) -> None:
    image = np.random.default_rng(4).integers(0, 256, (17, 23, 3), dtype=np.uint8)
    output = Identity().restore(image)
    save_image(tmp_path / 'output.png', output)
    decoded = read_image(tmp_path / 'output.png')
    assert np.array_equal(image, decoded)
    result = measure(decoded, image)
    assert result['mse'] == 0 and result['psnr_db'] is None and result['psnr_exact_match']
    assert result['ssim'] == pytest.approx(1)
    write_json(tmp_path / 'exact.json', result)
    assert 'Infinity' not in (tmp_path / 'exact.json').read_text()


def test_psnr_independent_formula_and_channels() -> None:
    reference = np.zeros((9, 9, 3), dtype=np.uint8)
    output = reference.copy()
    output[:, :, 0] = 51
    result = measure(output, reference)
    mse = 0.2 ** 2 / 3
    assert result['mse'] == pytest.approx(mse)
    assert result['psnr_db'] == pytest.approx(-10 * math.log10(mse))
    assert result['max_absolute_change'] == pytest.approx(.2)
    assert measure(output, None)['metric_reason'] == 'no_aligned_reference'
    with pytest.raises(ValidationError):
        measure(output, reference[:8])


def test_median_is_channelwise_and_reflects() -> None:
    image = np.full((9, 9, 3), [10, 70, 220], dtype=np.uint8)
    image[4, 4] = [255, 255, 255]
    output = Median().restore(image)
    assert np.array_equal(output[4, 4], [10, 70, 220])
    assert np.array_equal(output[0, 0], [10, 70, 220])


def test_invalid_intake(tmp_path: Path) -> None:
    for mode, size in [('RGBA', (10, 10)), ('RGB', (3, 3))]:
        path = tmp_path / f'{mode}.png'
        Image.new(mode, size).save(path)
        with pytest.raises(ValidationError):
            read_image(path)
    broken = tmp_path / 'broken.png'
    broken.write_bytes(b'not an image')
    with pytest.raises(ValidationError):
        read_image(broken)


def test_manifest_rejects_tampering_and_duplicates(manifest: Path) -> None:
    samples = load_manifest(manifest)
    assert len(samples) == 2
    path = Path(samples[0].input_path)
    save_image(path, np.zeros((17, 19, 3), dtype=np.uint8))
    with pytest.raises(ValidationError, match='hash mismatch'):
        load_manifest(manifest)


class FailingMedian(Median):
    def restore(self, image: RGBImage) -> RGBImage:
        raise RestorationError('injected isolated method failure')


def test_partial_run_and_equal_denominators(manifest: Path, tmp_path: Path) -> None:
    run = StudyRunner([Identity(), FailingMedian()]).execute(manifest, tmp_path / 'runs', 'TEST ONLY')
    metadata = json.loads((run / 'metadata.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    assert metadata['status'] == 'partial' and metadata['actual_records'] == 4
    assert summary['common_sample_ids'] == []
    assert summary['by_kind']['synthetic']['identity']['common_successful'] == 0
    assert summary['by_kind']['synthetic']['median3']['failed'] == 2
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    assert sum(e['event'] == 'method_finished' for e in events) == 4


def test_complete_run_scores_match_saved_files(manifest: Path, tmp_path: Path) -> None:
    run = StudyRunner([Identity(), Median()]).execute(manifest, tmp_path / 'runs', 'TEST ONLY')
    records = json.loads((run / 'records.json').read_text())
    assert json.loads((run / 'metadata.json').read_text())['status'] == 'complete'
    for row in records:
        scores = measure(read_image(run / row['output_path']), read_image(run / row['reference_asset']))
        assert scores['psnr_db'] == row['psnr_db']
        assert scores['ssim'] == row['ssim']
    assert len(summarize(records, ['identity', 'median3'])['common_sample_ids']) == 2


def test_mismatched_and_missing_reference(manifest: Path) -> None:
    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows[0]['reference_path'] = ''
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValidationError, match='missing required reference'):
        load_manifest(manifest)


def test_json_rejects_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_json(tmp_path / 'bad.json', {'score': float('inf')})


def test_missing_checkpoint_is_explicit(tmp_path: Path) -> None:
    from src.restormer_adapter import RestormerAdapter
    with pytest.raises(RestorationError, match='Missing official'):
        RestormerAdapter(tmp_path, tmp_path / 'missing.pth')


def test_viewer_opens_without_loading_model() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py').run(timeout=20)
    assert not app.exception
    assert app.title[0].value == 'WeatherRestore Lab'


def test_scene_bootstrap_preserves_grouping() -> None:
    from src.reporting import bootstrap
    rows: list[dict[str, Any]] = [{'scene_id': f'scene-{i}', 'sample_id': str(i),
                                  'delta_psnr_db': float(i)} for i in range(6)]
    interval = bootstrap(rows, 'delta_psnr_db')
    assert interval is not None and interval['unit'] == 'scene'
    assert interval['low'] <= 2.5 <= interval['high']


def test_manifest_duplicate_ids_are_rejected(manifest: Path) -> None:
    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows[1]['sample_id'] = rows[0]['sample_id']
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValidationError, match='duplicate sample ID'):
        load_manifest(manifest)


def test_unreferenced_real_image_never_gets_reference_scores(tmp_path: Path) -> None:
    inputs = tmp_path / 'real'
    inputs.mkdir()
    save_image(inputs / 'photo.png', np.full((13, 15, 3), 80, dtype=np.uint8))
    manifest_path = tmp_path / 'real.csv'
    freeze(inputs, None, manifest_path, 'real', 'generated API-contract fixture', 'dev', 'none')
    run = StudyRunner([Identity(), Median()]).execute(manifest_path, tmp_path / 'runs', 'TEST ONLY')
    records = json.loads((run / 'records.json').read_text())
    assert all(row['psnr_db'] is None and row['ssim'] is None and
               row['metric_reason'] == 'no_aligned_reference' for row in records)


def test_report_export_failure_never_marks_complete(manifest: Path, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_export(*args: Any, **kwargs: Any) -> None:
        raise OSError('injected disk/export failure')
    monkeypatch.setattr('src.runner.export', fail_export)
    with pytest.raises(OSError, match='export failure'):
        StudyRunner([Identity(), Median()]).execute(manifest, tmp_path / 'runs', 'TEST ONLY')
    metadata_file = next((tmp_path / 'runs').glob('*/metadata.json'))
    metadata = json.loads(metadata_file.read_text())
    assert metadata['status'] == 'partial'
    assert 'export_error' in metadata
    assert (metadata_file.parent / 'records.json').is_file()
