# Shrimp evaluation

This folder compares the shrimp holdout labels against the DLC and SLEAP predictions and writes metrics plus a violin plot of L2 pixel error.

Inputs used by the default script:

- Ground truth: `labeled-data/three_big_shrimps_video_two_holdout/CollectedData_Lolo.csv`
- DLC predictions: `labeled-data/three_big_shrimps_video_two_holdout/three_big_shrimps_video_two_last1000DLC_mobnet_100_ShrimpsApr12shuffle1_10000_el.csv`
- SLEAP predictions: `SLEAP/Shrimps/CorrectedishSLEAPAnalysisToBeUsedAsGroundTruth/shrimp_last1000.pkg.slp`

Run from the repository root with:

```bash
python DLC/Shrimps-Lolo-2026-04-12/evaluate/evaluate_shrimp_holdout.py
```

Outputs are written to:

- `evaluate/DLC/`
- `evaluate/SLEAP/`
- `evaluate/shrimp_l2_pixel_error_violin.png`