# AI vs Real Image Classification

CIFAKE binary classifier (CNN + ResNet-18). All commands from project root.

## Setup

```sh
uv sync --project final
```

## Data

```sh
uv run --project final python scripts/1_get_original_data.py
```

## Pipeline (per model: `cnn`, `resnet_scratch`)

```sh
# hyper-parameter sweep
uv run --project final python -m final.hp_sweep --model cnn

# train (resume with --resume; ablate with --no-normalize)
uv run --project final python -m final.train --model cnn

# test-set metrics + plots
uv run --project final python -m final.evaluate --model cnn
```

## Aggregate plots

```sh
uv run --project final python -m final.hp_sweep_combined   # lr sweep overlay
uv run --project final python -m final.roc_combined        # ROC overlay
```

## Data analysis

```sh
uv run --project final python -m final.sample_grid    # raw + normalized
uv run --project final python -m final.pixel_hist     # per-channel histograms
uv run --project final python -m final.fft_spectrum   # 2D FFT signatures
```

## Cross-generator evaluation (Defactify)

```sh
uv run --project final python scripts/2_external_eval.py
```

## Outputs

Scratch (gitignored): `final/results/<model>/` -- full run state (`best.pth`, `last.pth`, `history.json`, `metrics.json`, all PNGs).

Committed (curated): `persist/`
- `persist/checkpoints/<model>/` -- `best.pth`, `history.json`, `metrics.json`
- `persist/images/<model>/` -- per-model plots
- `persist/images/` -- aggregate plots (sample grids, FFT, histograms, ROC overlay, sweep overlay)
