from pathlib import Path
import shutil

import pandas as pd

PROJECT = Path(r"DLC/Shrimps-Lolo-2026-04-12")
SRC_DIR = PROJECT / "labeled-data" / "three_big_shrimps_video_two"
HOLDOUT_DIR = PROJECT / "labeled-data" / "three_big_shrimps_video_two_holdout"

TRAIN_CSV = SRC_DIR / "CollectedData_Lolo.csv"
TRAIN_H5 = SRC_DIR / "CollectedData_Lolo.h5"

HOLDOUT_CSV = HOLDOUT_DIR / "CollectedData_Lolo.csv"
HOLDOUT_H5 = HOLDOUT_DIR / "CollectedData_Lolo.h5"

BACKUP_DIR = PROJECT / "holdout_backup"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Backup originals once
for src in [TRAIN_CSV, TRAIN_H5]:
    dst = BACKUP_DIR / src.name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

# Load full labels (sorted in current row order)
df = pd.read_csv(TRAIN_CSV, header=[0, 1, 2, 3], index_col=0)

if len(df) < 1000:
    raise RuntimeError(f"Need at least 1000 rows, found {len(df)}")

train_df = df.iloc[:-1000].copy()
holdout_df = df.iloc[-1000:].copy()

# Repoint holdout image paths to holdout folder
old_prefix = "labeled-data/three_big_shrimps_video_two/"
new_prefix = "labeled-data/three_big_shrimps_video_two_holdout/"

new_index = []
for idx in holdout_df.index:
    if not idx.startswith(old_prefix):
        raise RuntimeError(f"Unexpected index format: {idx}")
    new_index.append(idx.replace(old_prefix, new_prefix, 1))
holdout_df.index = new_index

# Write train files back (last 1000 removed)
train_df.to_csv(TRAIN_CSV)
train_df.to_hdf(TRAIN_H5, key="df_with_missing", format="table")

# Write holdout files
HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
holdout_df.to_csv(HOLDOUT_CSV)
holdout_df.to_hdf(HOLDOUT_H5, key="df_with_missing", format="table")

# Copy holdout images
copied = 0
for idx in holdout_df.index:
    img_name = Path(idx).name
    src_img = SRC_DIR / img_name
    dst_img = HOLDOUT_DIR / img_name
    if not src_img.exists():
        raise RuntimeError(f"Missing source image: {src_img}")
    shutil.copy2(src_img, dst_img)
    copied += 1

print("Split complete")
print(f"Train rows: {len(train_df)}")
print(f"Holdout rows: {len(holdout_df)}")
print(f"Copied holdout images: {copied}")
print(f"Train CSV: {TRAIN_CSV}")
print(f"Holdout CSV: {HOLDOUT_CSV}")
print(f"Backup folder: {BACKUP_DIR}")
