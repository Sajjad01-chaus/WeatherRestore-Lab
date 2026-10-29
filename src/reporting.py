from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from src.records import write_json


def bootstrap(rows: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(metric) is not None]
    grouped: dict[str, list[float]] = {}
    unknown = any(not row.get('scene_id') for row in valid)
    for row in valid:
        grouped.setdefault(row.get('scene_id') or row['sample_id'], []).append(row[metric])
    if len(grouped) < 5:
        return None
    groups = list(grouped.values())
    rng = np.random.default_rng(42)
    estimates = [float(np.mean([value for index in rng.integers(0, len(groups), len(groups))
                                for value in groups[int(index)]])) for _ in range(2000)]
    return {'low': float(np.percentile(estimates, 2.5)), 'high': float(np.percentile(estimates, 97.5)),
            'resamples': 2000, 'seed': 42, 'unit': 'image (potentially optimistic)' if unknown else 'scene',
            'independent_units': len(groups)}


def summarize(records: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    successes = {method: {row['sample_id'] for row in records
                          if row['method'] == method and row['status'] == 'success'} for method in methods}
    common = set.intersection(*(successes[method] for method in methods)) if methods else set()
    sections: dict[str, Any] = {}
    for kind in sorted({row['kind'] for row in records}):
        values: dict[str, Any] = {}
        for method in methods:
            attempted = [row for row in records if row['method'] == method and row['kind'] == kind]
            selected = [row for row in attempted if row['sample_id'] in common and row['status'] == 'success']
            psnr = [row['psnr_db'] for row in selected if row.get('psnr_db') is not None]
            ssim = [row['ssim'] for row in selected if row.get('ssim') is not None]
            changes = [row['mean_absolute_change'] for row in selected
                       if row.get('mean_absolute_change') is not None]
            values[method] = {'attempted': len(attempted), 'successful': len(successes[method] &
                              {row['sample_id'] for row in attempted}),
                              'failed': sum(row['status'] == 'failed' for row in attempted),
                              'common_successful': len(selected), 'excluded': 0,
                              'mean_finite_psnr_db': statistics.mean(psnr) if psnr else None,
                              'median_finite_psnr_db': statistics.median(psnr) if psnr else None,
                              'finite_psnr_count': len(psnr),
                              'exact_match_count': sum(bool(row.get('psnr_exact_match')) for row in selected),
                              'mean_ssim': statistics.mean(ssim) if ssim else None,
                              'mean_absolute_change': statistics.mean(changes) if changes else None,
                              'inspection_flags': sum(bool(row.get('inspection_flag')) for row in selected),
                              'psnr_change_95ci': bootstrap(selected, 'delta_psnr_db'),
                              'ssim_change_95ci': bootstrap(selected, 'delta_ssim')}
        sections[kind] = values
    gallery: list[dict[str, Any]] = []
    for method in methods:
        if method == 'identity':
            continue
        rainy = sorted([row for row in records if row['method'] == method and
                        row['sample_id'] in common and row['kind'] in {'paired', 'synthetic'} and
                        row.get('delta_psnr_db') is not None], key=lambda row: row['delta_psnr_db'])
        choices = [('largest_negative', rainy[0] if rainy and rainy[0]['delta_psnr_db'] < 0 else None),
                   ('median_finite_change', rainy[len(rainy) // 2] if rainy else None),
                   ('largest_positive', rainy[-1] if rainy and rainy[-1]['delta_psnr_db'] > 0 else None),
                   ('metric_disagreement', next((row for row in rainy if row.get('metric_disagreement')), None))]
        clean = [row for row in records if row['method'] == method and row['kind'] == 'clean' and
                 row['sample_id'] in common and row.get('mean_absolute_change') is not None]
        choices.append(('largest_clean_change', max(clean, key=lambda row: row['mean_absolute_change'])
                        if clean else None))
        for reason, row in choices:
            gallery.append({'method': method, 'selection': reason,
                            'sample_id': row['sample_id'] if row else None,
                            'output_path': row['output_path'] if row else None})
    return {'common_sample_ids': sorted(common), 'by_kind': sections, 'gallery': gallery,
            'aggregation': 'Arithmetic mean of finite per-image PSNR; exact matches counted separately',
            'interval_limit': 'Descriptive only; omitted with fewer than five scene/image units'}


def export(run: Path, metadata: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(records, metadata['methods'])
    write_json(run / 'summary.json', summary)
    keys = sorted({key for record in records for key in record})
    temporary = run / 'per_image.csv.tmp'
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(run / 'per_image.csv')
    escape = html.escape
    rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(row.get(key, "")))}</td>' for key in
                   ['sample_id', 'kind', 'method', 'status', 'psnr_db', 'psnr_exact_match', 'ssim',
                    'delta_psnr_db', 'inspection_flag', 'error']) + '</tr>' for row in records)
    gallery = ''.join(f'<figure><figcaption>{escape(item["method"])} · {escape(item["selection"])} · '
                      f'{escape(str(item["sample_id"] or "category absent"))}</figcaption>' +
                      (f'<img loading="lazy" src="{escape(item["output_path"], quote=True)}">'
                       if item['output_path'] else '') + '</figure>' for item in summary['gallery'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>WeatherRestore evidence</title>
<style>body{{font:15px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#20252c}}
table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #ddd;padding:7px}}
img{{max-width:480px;width:100%}}pre{{white-space:pre-wrap}}figure{{display:inline-block;vertical-align:top}}</style>
<h1>WeatherRestore Lab</h1><p>Run {escape(metadata['run_id'])} · {escape(metadata['status'])}</p>
<p>{escape(metadata['evidence_label'])}</p><p>Protocol: {escape(metadata['protocol'])}. Saved 8-bit RGB PNG,
full frame, no border crop or test-time resizing. Higher PSNR/SSIM means closer reference agreement,
not guaranteed perceptual correctness. Exact PSNR matches are null with a separate flag.</p>
<p>Model and datasets belong to their credited authors. Restormer: Zamir et al., CVPR 2022.
This is not a reproduction of the authors' MATLAB benchmark scores, a new architecture, or an all-weather system.</p>
<h2>Coverage and common-set summaries</h2><pre>{escape(json.dumps(summary['by_kind'], indent=2))}</pre>
<p>{escape(summary['aggregation'])}. {escape(summary['interval_limit'])}.</p>
<h2>Per-image evidence</h2><p>Blank metrics mean unavailable. See metric_reason and errors in CSV.</p>
<div style="overflow:auto"><table><thead><tr><th>Sample</th><th>Kind</th><th>Method</th><th>Status</th>
<th>PSNR dB</th><th>Exact match</th><th>SSIM</th><th>PSNR change dB</th><th>Review flag</th><th>Error</th>
</tr></thead><tbody>{rows}</tbody></table></div><h2>Rule-selected examples</h2>{gallery}
<p>Selection is disclosed; adverse examples are retained. Review flags are project thresholds,
not validated perceptual harm boundaries. Use the viewer for aligned crops and labeled differences.</p>
<h2>Provenance</h2><p><a href="metadata.json">Metadata</a> · <a href="manifest.json">Frozen manifest</a> ·
<a href="config.json">Configuration</a> · <a href="per_image.csv">Measurements CSV</a> ·
<a href="events.jsonl">Execution log</a></p></html>'''
    (run / 'report.html').write_text(page, encoding='utf-8')
    return summary
