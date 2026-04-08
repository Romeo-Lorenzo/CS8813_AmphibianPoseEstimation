#!/usr/bin/env python3
"""
Copy images with frog/toad observations to a new folder preserving directory structure.
Filters for:
  - Non-empty Qte1 (column 6, "something in picture")
  - Frog or toad species
"""

import csv
import os
import shutil
from pathlib import Path

# Frog/toad species to keep
FROG_TOAD_SPECIES = {
    'Crapaud_commun',
    'Grenouille_indetermine',
    'Grenouille_rousse',
    'Grenouille_verte',
    'Rainette_verte',
    'Anoure_indetermine'
}

# Parent directory where script is located
SCRIPT_DIR = Path(__file__).parent

# Source and output paths
SOURCE_CSV = SCRIPT_DIR / 'fixed_Batrachoduc_Boucq_2025_T1_a_moitie_T8.csv'
DATA_DIR = '/mnt/cerema/2025/data'
OUTPUT_DIR = SCRIPT_DIR / 'data_filtered_frogs'
RENAMED_DIR = SCRIPT_DIR / 'data_filtered_frogs_renamed'
RENAMED_MAP_CSV = SCRIPT_DIR / 'renamed_images_map.csv'

def is_frog_toad_observation(row):
    """Check if row has frog/toad and non-empty Qte1."""
    qte1 = (row.get('Qte1') or '').strip()
    espece1 = (row.get('Espece1') or '').strip()
    
    # Must have Qte1 (something observed) and be frog/toad
    return qte1 and espece1 in FROG_TOAD_SPECIES

def main():
    # Read CSV and collect image paths to copy
    images_to_copy = []
    
    if not os.path.exists(SOURCE_CSV):
        print(f"Error: {SOURCE_CSV} not found")
        return
    
    print(f"Reading {SOURCE_CSV}...")
    with open(SOURCE_CSV, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_frog_toad_observation(row):
                rel_path = (row.get('RelativePath') or '').strip()
                file_name = (row.get('File') or '').strip()
                if rel_path and file_name:
                    # Normalize Windows-style paths to Unix-style.
                    rel_path = rel_path.replace('\\', '/')
                    images_to_copy.append(f"{rel_path}/{file_name}")

    # Remove duplicates while preserving CSV order.
    images_to_copy = list(dict.fromkeys(images_to_copy))
    
    print(f"Found {len(images_to_copy)} images with frog/toad observations")
    
    if not images_to_copy:
        print("No images found matching criteria")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENAMED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {OUTPUT_DIR}")
    print(f"Created renamed output directory: {RENAMED_DIR}")
    print("Starting copy...")
    
    # Copy images preserving structure
    copied = 0
    failed = 0
    missing_files = []
    renamed_rows = []
    renamed_index = 0
    total = len(images_to_copy)

    for idx, rel_path in enumerate(images_to_copy, start=1):
        src = Path(DATA_DIR) / rel_path
        dst = OUTPUT_DIR / rel_path
        
        if not src.exists():
            print(f"  WARNING: {src} not found")
            missing_files.append(str(src))
            failed += 1
            continue
        if not src.is_file():
            print(f"  WARNING: {src} is not a file")
            missing_files.append(str(src))
            failed += 1
            continue
        
        # Create destination directory
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy individual file only
        try:
            shutil.copy2(src, dst)

            # Also export a flat renamed copy for downstream processing.
            renamed_name = f"img{renamed_index:04d}.png"
            renamed_dst = RENAMED_DIR / renamed_name
            shutil.copy2(src, renamed_dst)
            renamed_rows.append((str(src), renamed_name))
            renamed_index += 1

            copied += 1
        except Exception as e:
            print(f"  ERROR copying {src}: {e}")
            failed += 1

        if idx % 25 == 0 or idx == total:
            print(f"Progress: {idx}/{total} processed ({copied} copied, {failed} failed)", flush=True)
    
    print(f"\nDone: {copied} copied, {failed} failed")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Renamed output folder: {RENAMED_DIR}")

    # Write source-to-renamed mapping for traceability.
    with open(RENAMED_MAP_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_path', 'renamed_file'])
        writer.writerows(renamed_rows)
    print(f"Renamed mapping CSV: {RENAMED_MAP_CSV}")

    # Save missing paths to a report file in CS8813.
    if missing_files:
        missing_report = SCRIPT_DIR / 'missing_files.txt'
        with open(missing_report, 'w', encoding='utf-8') as f:
            f.write('\n'.join(missing_files) + '\n')
        print(f"Missing files report: {missing_report}")

if __name__ == '__main__':
    main()