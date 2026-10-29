from __future__ import annotations

import json
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.inputs import ValidationError, read_image
from src.methods import Identity, Median, RestorationError
from src.metrics import measure

ROOT = Path(__file__).resolve().parent
SOURCES = '''
**Model:** [Restormer, CVPR 2022](https://github.com/swz30/Restormer), by Zamir and coauthors.
The credited architecture and MIT notice are included in `third_party/restormer`.
Pretrained weights are downloaded separately and checked against a pinned SHA-256.

**Protocol:** `rgb_full_frame_png_v1` evaluates saved 8-bit RGB PNGs at full frame,
with no border crop or test-time resize. SSIM uses an explicit 7-pixel window and data range 1.
This differs from the authors' MATLAB evaluation. Reference metrics require an aligned clean reference.

**Scope:** rain removal with pretrained weights, identity and a fixed median baseline.
The reported 100-pair experiment used a pinned Rain100L research mirror; equivalence to the unavailable
original archive is unverified. Datasets, checkpoints, local reports and personal files are not distributed.
No new architecture, real-time performance or all-weather capability is claimed.
'''


def run_title(path: Path) -> str:
    try:
        metadata = json.loads(path.read_text(encoding='utf-8'))
        label = str(metadata.get('evidence_label', ''))
        if 'Rain100L mirror' in label:
            name = 'Rain100L mirror - 100 pairs'
        elif 'Practical mirror' in label:
            name = 'Real rainy photographs - qualitative'
        elif 'clean scenes' in label or 'clean controls' in label:
            name = 'Clean photograph controls'
        elif 'SOFTWARE VALIDATION' in label:
            name = 'Software test fixtures'
        else:
            name = label.split(':')[0][:60]
        return f'{name} | {metadata.get("status", "unknown")} | {path.parent.name}'
    except (OSError, ValueError):
        return f'Unreadable run | {path.parent.name}'


def compare(images: dict[str, Any]) -> None:
    if not images:
        return
    first = next(iter(images.values()))
    h, w = first.shape[:2]
    crop = st.checkbox('Inspect a shared crop', key='crop-' + str(w) + '-' + str(h))
    if crop:
        x = st.slider('Crop left (pixels)', 0, w - 8, 0) if w > 8 else 0
        y = st.slider('Crop top (pixels)', 0, h - 8, 0) if h > 8 else 0
        size = st.slider('Crop side (pixels)', 8, min(w - x, h - y), min(128, w - x, h - y)) \
            if min(w - x, h - y) > 8 else 8
        st.caption(f'Identical crop for every image: x={x}, y={y}, width={size}, height={size}. '
                   'Cropping affects inspection only; saved full-frame scores are unchanged.')
        images = {name: image[y:y + size, x:x + size] for name, image in images.items()}
    zoom = st.select_slider('Synchronized zoom', options=[1, 2, 4], value=1)
    columns = st.columns(min(len(images), 3))
    for index, (name, image) in enumerate(images.items()):
        with columns[index % len(columns)]:
            st.markdown(f'**{name}**')
            shown = Image.fromarray(image)
            if zoom > 1:
                shown = shown.resize((shown.width * zoom, shown.height * zoom), Image.Resampling.NEAREST)
            st.image(shown, width='content')
    outputs = [name for name in images if name not in {'Input', 'Reference'}]
    if outputs:
        selected = st.selectbox('Difference output', outputs)
        against = 'Reference' if 'Reference' in images else 'Input'
        gain = st.select_slider('Difference display amplification', options=[1, 2, 4, 8], value=4)
        delta = np.abs(images[selected].astype(float) - images[against].astype(float))
        st.image(np.clip(delta * gain, 0, 255).astype(np.uint8), caption=
                 f'Absolute RGB difference to {against.lower()} × {gain}; clipped at 255. '
                 'Diagnostic display, not a confidence map.', width='content')


def saved_view() -> None:
    st.subheader('Saved studies')
    directories = sorted((ROOT / 'runs').glob('*/metadata.json'), reverse=True)
    if not directories:
        st.info('No saved study yet. Run the batch command in README to generate an evidence package.')
        return
    release_path = ROOT / 'docs/release_evidence.json'
    preferred = json.loads(release_path.read_text(encoding='utf-8')).get('preferred_run', '') \
        if release_path.is_file() else ''
    preferred_index = next((index for index, path in enumerate(directories)
                            if str(path.parent.relative_to(ROOT)).replace('\\', '/') == preferred), 0)
    selected = st.selectbox('Run', directories, index=preferred_index, format_func=run_title)
    run = selected.parent
    try:
        metadata = json.loads(selected.read_text(encoding='utf-8'))
        records = json.loads((run / 'records.json').read_text(encoding='utf-8'))
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        if metadata['status'] == 'complete':
            st.success('Saved run · execution complete')
        else:
            st.warning(f'Saved run · {metadata["status"]}; inspect coverage and errors')
        st.write(metadata['evidence_label'])
        st.caption('Execution completion does not establish study scope or scientific success.')
        st.write('Coverage and common successful samples')
        for kind, methods in summary['by_kind'].items():
            st.markdown(f'**{kind}**')
            st.dataframe(pd.DataFrame(methods).T, width='stretch')
        frame = pd.DataFrame(records)
        if frame.empty:
            st.warning('This run has no completed method records. Inspect the run error and event log.')
            st.json(metadata)
            return
        finite_changes = frame.dropna(subset=['delta_psnr_db'])
        if not finite_changes.empty:
            with st.expander('Distribution of per-image fidelity changes'):
                st.caption('PSNR change to identity (dB), for successful finite comparisons only. '
                           'Coverage and exact matches remain in the summary above.')
                counts = {}
                edges = np.histogram_bin_edges(finite_changes['delta_psnr_db'], bins=10)
                for name, group in finite_changes.groupby('method'):
                    counts[str(name)] = np.histogram(group['delta_psnr_db'], bins=edges)[0]
                bins = [f'{left:.2f} to {right:.2f}' for left, right in pairwise(edges)]
                st.bar_chart(pd.DataFrame(counts, index=bins), x_label='PSNR change interval (dB)',
                             y_label='Number of images')
        with st.expander('All per-image measurements and failures'):
            st.dataframe(frame, width='stretch', hide_index=True)
        for file, mime in [('per_image.csv', 'text/csv'), ('report.html', 'text/html'),
                           ('metadata.json', 'application/json')]:
            st.download_button(f'Download {file}', (run / file).read_bytes(), file_name=file, mime=mime)
        successful = [row for row in records if row['status'] == 'success']
        ids = sorted({row['sample_id'] for row in records})
        if ids:
            gallery = [item for item in summary['gallery'] if item['sample_id'] is not None]
            recommendation = st.selectbox('Suggested review case', [None, *range(len(gallery))],
                                           format_func=lambda index: 'Choose any sample below' if index is None
                                           else f'{gallery[index]["method"]}: {gallery[index]["selection"]} '
                                           f'({gallery[index]["sample_id"]})')
            default = ids.index(gallery[recommendation]['sample_id']) if recommendation is not None else 0
            sid = st.selectbox('Inspect sample', ids, index=default, key=f'sample-{run.name}-{recommendation}')
            sample_rows = [row for row in successful if row['sample_id'] == sid]
            if sample_rows:
                base = sample_rows[0]
                images = {'Input': read_image(run / base['input_asset'])}
                if base.get('reference_asset'):
                    images['Reference'] = read_image(run / base['reference_asset'])
                images.update({row['method']: read_image(run / row['output_path']) for row in sample_rows})
                compare(images)
            else:
                st.error('All methods failed for this sample; inspect the error records above.')
        with st.expander('Research details and selection rule'):
            st.json(metadata)
            st.json(summary['gallery'])
            st.caption('Rainy review flags: PSNR decrease > 0.1 dB or SSIM decrease > 0.001. '
                       'These are project inspection thresholds, not proven harm boundaries.')
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Cannot load saved evidence: {type(error).__name__}: {error}')


@st.cache_resource
def learned_model(device: str) -> Any:
    from src.restormer_adapter import RestormerAdapter
    return RestormerAdapter(ROOT / 'third_party/restormer', ROOT / 'weights/deraining.pth', device)


def live_view() -> None:
    st.subheader('Compare a photograph')
    st.caption('Live local preview · no image leaves this computer. Full-frame metrics require an aligned reference.')
    uploaded = st.file_uploader('Input photograph (opaque PNG/JPEG; ≤20 MB, ≤20 MP)', type=['png', 'jpg', 'jpeg'])
    reference = st.file_uploader('Optional clean reference', type=['png', 'jpg', 'jpeg'])
    aligned = st.checkbox('I confirm the reference depicts the same scene with aligned geometry')
    options = ['identity', 'median3']
    if (ROOT / 'weights/deraining.pth').is_file():
        options.append('restormer')
    else:
        st.warning('Restormer unavailable: weights/deraining.pth is missing. Baselines remain available.')
    methods = st.multiselect('Methods', options, default=options[:2])
    profile = st.radio('Input profile', ['Native resolution', 'Preview: long edge ≤512 pixels'])
    device = st.selectbox('Learned-model device', ['cuda', 'cpu'])
    if st.button('Run comparison', disabled=uploaded is None or not methods) and uploaded is not None:
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'input.png'
                path.write_bytes(uploaded.getvalue())
                image = read_image(path)
                ref = None
                if reference is not None and aligned:
                    path.write_bytes(reference.getvalue())
                    ref = read_image(path)
                    if ref.shape != image.shape:
                        raise ValidationError('Reference dimensions disagree; use an aligned reference')
                elif reference is not None:
                    st.warning('Reference alignment not acknowledged; reference metrics disabled.')
                if profile.startswith('Preview'):
                    size = (max(8, round(image.shape[1] * min(1, 512 / max(image.shape[:2])))),
                            max(8, round(image.shape[0] * min(1, 512 / max(image.shape[:2])))))
                    image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))
                    if ref is not None:
                        ref = np.asarray(Image.fromarray(ref).resize(size, Image.Resampling.LANCZOS))
                images = {'Input': image}
                if ref is not None:
                    images['Reference'] = ref
                scores = []
                with st.spinner('Processing selected methods'):
                    for name in methods:
                        method = learned_model(device) if name == 'restormer' else \
                            (Identity() if name == 'identity' else Median())
                        output = method.restore(image)
                        images[name] = output
                        scores.append({'method': name, **measure(output, ref)})
                st.session_state['live'] = (images, scores, profile)
        except (ValidationError, RestorationError, OSError, RuntimeError, ValueError) as error:
            st.session_state.pop('live', None)
            st.error(f'Comparison stopped: {type(error).__name__}: {error}. '
                     'For GPU memory errors, try the separate preview profile or browse a saved run.')
    if 'live' in st.session_state:
        images, scores, saved_profile = st.session_state['live']
        st.info(f'Live result from last Run comparison · {saved_profile}')
        st.dataframe(pd.DataFrame(scores), hide_index=True)
        st.caption('PSNR is in dB; SSIM is unitless. Exact matches have a separate flag. '
                   'Scores here describe the displayed uint8 output; preview scores are not benchmark scores.')
        compare(images)


def main() -> None:
    st.set_page_config(page_title='WeatherRestore Lab', page_icon=None, layout='wide',
                       initial_sidebar_state='collapsed')
    st.title('WeatherRestore Lab')
    st.write('Inspect rain removal. Measure fidelity. Keep the failures visible.')
    page = st.radio('Workspace', ['Saved studies', 'Compare photograph', 'Sources'], horizontal=True)
    if page == 'Saved studies':
        saved_view()
    elif page == 'Compare photograph':
        live_view()
    else:
        st.subheader('Sources and protocol')
        st.markdown(SOURCES)
        st.caption('The implemented learned baseline is Restormer (CVPR 2022); '
                   'no detection pipeline or all-weather claim is included.')


if __name__ == '__main__':
    main()
