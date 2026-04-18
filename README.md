# CS8813 Amphibian Pose Estimation

This repository supports comparative animal pose estimation experiments in the CS8813 context, with emphasis on robust evaluation across biologically different targets (Fly32 and Shrimps) using multiple toolchains.

Frameworks used:
- DeepLabCut (DLC)
- SLEAP
- YOLO pose workflows

## Objective

The primary objective is to benchmark and operationalize pose-estimation workflows under controlled conditions, then transfer those workflows to challenging species-specific data.

In practice, this repository is used to:
- prepare train/validation/test splits,
- convert annotations between SLEAP and DLC formats,
- train and run inference in each framework,
- compare outputs with consistent metrics,
- keep reproducible artifacts in a project-centered structure.

## Methodological Direction

Inspired by the CS_8813 report style, the workflow follows a simple experimental logic:
- fixed or documented dataset splits,
- independent training/inference per framework,
- unified error reporting,
- explicit tracking of assumptions and holdout usage.

The current canonical working area is consolidated under `Projects/`.

## Repository Organization

- `Projects/`
  - canonical workspace by target project (`Fly32`, `Shrimps`)
  - framework subfolders (`DLC`, `SLEAP`, `YOLO` when present)
- `DLC/`
  - DLC project roots, training artifacts, and env resources
- `SLEAP/`
  - SLEAP packages, models, predictions, and configs
- `YOLO_TRAIN/`
  - YOLO training scripts and checkpoints
- `tools/`
  - copied utility scripts for conversion, splitting, extraction, and metric evaluation
- `inspect_packages.py`
  - small SLEAP package inspection helper kept inside the publishable tree
- `dataset/`
  - legacy or auxiliary locations (not the canonical shrimp source of truth)

## Canonical Splits (Shrimps)

Current shrimp split policy:
- Train: `Projects/Shrimps/SLEAP/propershrimp/shrimp_train.pkg.slp`
- Validation: `Projects/Shrimps/SLEAP/propershrimp/shrimp_val.pkg.slp`
- Test (real holdout): `Projects/Shrimps/SLEAP/shrimp_last150.pkg.slp`

## Core Utilities

- `Projects/tools/RecycleInferedToGroundTruth.py`
  - converts inferred SLEAP labels into GT-style packages
- `Projects/tools/split_sleap_pkg_chronological.py`
  - chronological split for SLEAP package files
- `Projects/tools/export_sleap_pkg_to_dlc.py`
  - general SLEAP-to-DLC conversion
- `Projects/tools/export_sleap_shrimp_to_dlc.py`
  - shrimp-focused SLEAP-to-DLC conversion helper
- `Projects/tools/CompareMetrics.py`
  - metric comparison across methods
- `Projects/tools/extract_frames_from_video.py`
  - frame extraction from source videos
- `Projects/tools/make_mp4_from_images.py`
  - video reconstruction from image sequences

## Reproducibility and Environment

Primary Python environment:
- `DLC/.venv` (Python 3.11)

Windows PowerShell example:

```powershell
& "DLC/.venv/Scripts/python.exe" Projects/tools/CompareMetrics.py
```

## Standard Workflow

1. Prepare or verify SLEAP labels and split policy.
2. Export to DLC format when cross-framework comparison is needed.
3. Train/infer in the selected framework (DLC, SLEAP, or YOLO).
4. Evaluate with common metrics and store lightweight outputs.
5. Keep `Projects/` as the canonical path for active experiments.

## Version Control Policy

`Projects/.gitignore` is configured to avoid pushing heavy raw artifacts (videos, images, raw label packages, model folders) while preserving lightweight scientific outputs (CSV, JSON, YAML, TXT, PY, NPZ, CKPT).

## Reference

Project report used as style reference:
- `Projects/CS_8813.pdf`
