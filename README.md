# WeatherRestore Lab

Local rain-removal evaluation with **pretrained Restormer**, unchanged input and a fixed 3x3 median baseline. Includes a Streamlit comparison UI, batch experiments, aligned crops, difference views, PSNR/SSIM, structured failures and HTML/CSV exports.

## Setup

Tested on Windows with Python 3.10 and an NVIDIA RTX 4050. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python scripts/fetch_checkpoint.py
streamlit run app.py
```

PyTorch must match your hardware; use its official installer for other platforms or CPU-only execution. The checkpoint helper uses Windows curl and verifies a pinned mirror's SHA-256. The credited architecture is included under `third_party/restormer`. Exact versions from the experiment environment are recorded in `requirements.lock`.

## Batch evaluation

Place your permitted PNG/JPEG images in local folders. Rainy and reference filenames must match, with aligned geometry:

```powershell
python scripts/freeze_manifest.py --inputs data/rain --references data/clean --output manifests/test.csv --kind paired --source "Your dataset attribution" --reference-status benchmark
python run_study.py --manifest manifests/test.csv --label "Your dataset and experiment scope"
```

Use `--methods identity median3` for a baseline-only run, or `--device cpu` to explicitly select CPU inference. Reference-free photographs use `--kind real --reference-status none` without `--references`; they never receive PSNR/SSIM. Saved runs appear automatically in the UI and remain browsable without loading the model.

For software-only smoke fixtures, run `python scripts/make_fixtures.py`, then evaluate `manifests/synthetic.csv` with an explicit software-validation label. These generated fixtures are not benchmark evidence. `scripts/benchmark.py` measures repeated latency separately; `scripts/replay_run.py` replays saved input pixels into a new run.

## Validation and measured scope

```powershell
python -m pytest -q
python -m ruff check src scripts tests app.py run_study.py
```

The recorded native-resolution experiment evaluated 100 Rain100L pairs from a pinned research mirror: mean RGB PSNR was **25.52 dB** for identity, **24.39 dB** for median and **37.58 dB** for Restormer. All 300 method-image records completed. Ten clean controls and five unreferenced real photographs were also evaluated. The test suite has 18 passing tests; the latest Windows mypy rerun was blocked by Application Control, so a fresh typing pass is not claimed.

Protocol `rgb_full_frame_png_v1` evaluates saved 8-bit RGB PNGs without border crop or test-time resizing. SSIM parameters and environment are stored per run. Comparisons use common successful samples and report coverage. Exact PSNR matches use a separate flag instead of invalid JSON infinity. Clean-image changes and unfavorable cases remain visible.

The Rain100L mirror checksum was verified; byte equivalence to the unavailable original distribution is unverified. These RGB scores do not reproduce the authors' MATLAB protocol. This project does not introduce or train Restormer, establish real-world reconstruction accuracy, or claim real-time/all-weather performance.

## Attribution

Restormer: Zamir et al., CVPR 2022, [official repository](https://github.com/swz30/Restormer). Source revision `68dc6ac472db26f16361150cb7a96a1bc87da93f`; MIT notice retained in `third_party/restormer/LICENSE.md`. Checkpoint mirror: [DeepInverse](https://huggingface.co/deepinv/Restormer). Evaluation data mirror: [AaronCIH/PIR](https://huggingface.co/datasets/AaronCIH/PIR), revision `c681112a79de207e5b0e810ce9fb4b30a037903a`.

Datasets, weights, research guides, logs, generated reports, local runs and personal files are intentionally excluded from this repository. Respect dataset/checkpoint terms before redistribution. See `THIRD_PARTY_NOTICES.md` for model attribution.
