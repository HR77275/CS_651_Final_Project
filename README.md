# Benchmarking Lion Against Standard Optimizers Across Multiple Tasks

A clean, reproducible deep learning benchmark for comparing optimizers on **semantic segmentation** using **DeepLabV3-ResNet50** and **Pascal VOC 2012**.

This repository was built as part of a graduate-level empirical study on optimizer behavior across tasks. The current phase focuses on a vision benchmark and compares **SGD**, **Adam**, **AdamW**, and **Lion** under a fixed training budget while tracking both model quality and systems efficiency.

## Why this project matters

Most optimizer comparisons are presented only through final accuracy. This project goes further by measuring:

- segmentation quality through `mIoU`, loss, and pixel accuracy
- convergence behavior across epochs
- optimizer update cost through per-step timing
- memory footprint through optimizer state tracking
- reproducibility through YAML-driven experiments and saved histories

The broader project roadmap extends this benchmark to **ADE20K** and **AG News** so the final study covers both vision and NLP workloads.

## Current benchmark

- **Task:** Semantic segmentation
- **Dataset:** Pascal VOC 2012
- **Model:** DeepLabV3-ResNet50
- **Resolution:** `256 x 256`
- **Batch size:** `8`
- **GPU used:** NVIDIA RTX 3060 (6 GB)
- **Comparison budget:** first `25` epochs for each optimizer

## Key results

Under the common 25-epoch comparison budget:

- **SGD** achieved the strongest overall performance with best validation `mIoU = 0.6564`
- **AdamW** was the strongest adaptive baseline with best validation `mIoU = 0.6421`
- **Adam** remained competitive but trailed SGD and AdamW
- **Lion** used lower optimizer-state memory than Adam/AdamW, but underperformed in both accuracy and optimizer-step efficiency in this setup

These results make Pascal VOC a strong first-stage benchmark for a larger optimizer study across datasets and tasks.

## Repository structure

```text
651_project/
|-- configs/                 # YAML experiment configs
|-- docs/                    # Report source and report-ready figures
|-- outputs/                 # Training runs, checkpoints, and comparison plots
|-- scripts/                 # Training, sanity checks, and plotting entry points
|-- src/
|   |-- data/                # Pascal VOC dataset and transforms
|   |-- models/              # Model factory for segmentation architectures
|   `-- training/            # Step logic, epoch metrics, and optimizer utilities
|-- README.md
`-- requirements.txt
```

## Main features

- Pascal VOC dataloader with paired image-mask transforms
- DeepLabV3-ResNet50 model factory
- YAML-configured training pipeline with `tqdm` progress bars
- Epoch-level tracking for:
  - loss
  - `mIoU`
  - pixel accuracy
  - throughput
  - forward/backward/optimizer step time
  - optimizer-state memory
- Multi-optimizer experiment setup
- Plot generation for side-by-side comparison
- LaTeX report assets for submission-ready writeups

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using the same Windows-based environment from WSL, the commands used in this project looked like:

```bash
/mnt/c/Users/HIMANSHU/Desktop/langchain/practice/myenv/Scripts/python.exe -m pip install -r requirements.txt
```

## Dataset setup

Download Pascal VOC 2012 through the dataloader utility:

```bash
python scripts/check_voc_dataloader.py --download
```

By default, the dataset is stored under:

```text
data/VOCdevkit/VOC2012
```

## Training

The default SGD configuration is:

- learning rate: `0.01`
- momentum: `0.9`
- weight decay: `1e-4`

Start training with:

```bash
python scripts/train.py --config configs/deeplabv3_resnet50_voc.yaml
```

Run the other optimizers with:

```bash
python scripts/train.py --config configs/deeplabv3_resnet50_voc_adam_50ep.yaml
python scripts/train.py --config configs/deeplabv3_resnet50_voc_adamw_50ep.yaml
python scripts/train.py --config configs/deeplabv3_resnet50_voc_lion_50ep.yaml
```

If you are launching from WSL with the Windows interpreter used during development:

```bash
/mnt/c/Users/HIMANSHU/Desktop/langchain/practice/myenv/Scripts/python.exe scripts/train.py --config configs/deeplabv3_resnet50_voc.yaml
```

## Outputs

Each run writes results to its own folder under `outputs/`, including:

- `history.jsonl` with per-epoch metrics
- `best.pt` for the best validation checkpoint
- `last.pt` for the latest checkpoint

Current experiment folders:

- `outputs/deeplabv3_resnet50_voc_sgd_50ep`
- `outputs/deeplabv3_resnet50_voc_adam_50ep`
- `outputs/deeplabv3_resnet50_voc_adamw_50ep`
- `outputs/deeplabv3_resnet50_voc_lion_50ep`

## Plot generation

Generate comparison plots with:

```bash
python scripts/plot_optimizer_comparison.py --max-epoch 25 --output-dir outputs/comparison_plots_25ep
```

The generated plots are stored in:

- `outputs/comparison_plots_25ep`
- `docs/report_figures` for report-ready renamed copies

## Sanity-check scripts

The repository includes small utilities to verify individual pieces before running full experiments:

- `scripts/check_voc_dataloader.py`
- `scripts/check_model.py`
- `scripts/check_train_step.py`
- `scripts/check_epoch_metrics.py`

These are useful for debugging dataset, model, and metric issues before launching long runs.

## Report assets

The LaTeX writeup and report figure assets are stored in:

- `docs/optimizer_report.tex`
- `docs/report_figures/`

## Future work

The next phase of the project extends the benchmark to:

- **ADE20K** for a harder semantic segmentation benchmark
- **AG News** for a text classification benchmark

That extension is meant to test whether optimizer behavior remains consistent across datasets, modalities, and training regimes.

## Author

Developed by **Ananya Jha, Himanshu Ranjan, Mayank, and Sethupathy Raghunathan Venkatraman**.

