#!/usr/bin/env python3
"""
Split a SLEAP .pkg.slp/.slp file into chronological train/val/test packages.

Default split: 80/10/10
Output names: <base_name>_train.pkg.slp, <base_name>_val.pkg.slp, <base_name>_test.pkg.slp
"""

import argparse
from pathlib import Path
from typing import Any, Callable, cast

import sleap_io as sio
from sleap_io.model.labels import Labels


def _build_subset(labels: Labels, subset_frames: list) -> Labels:
    """Create a Labels object with shared metadata and a subset of frames."""
    return Labels(
        labeled_frames=subset_frames,
        videos=labels.videos,
        skeletons=labels.skeletons,
        tracks=labels.tracks,
        suggestions=labels.suggestions,
        sessions=labels.sessions,
        provenance=labels.provenance,
        rois=labels.rois,
        masks=labels.masks,
    )


def split_chronological(
    input_path: Path,
    output_dir: Path,
    base_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[Path, Path, Path]:
    slp_loader = cast(Callable[[str], Any], getattr(sio, "load_slp", None))
    if not callable(slp_loader):
        raise RuntimeError("sleap_io.load_slp is not available in this environment.")

    labels: Labels = slp_loader(str(input_path))

    if not labels.labeled_frames:
        raise RuntimeError("Input file has no labeled frames.")

    # Chronological split by (video, frame_idx)
    video_order = {video: i for i, video in enumerate(labels.videos)}
    sorted_frames = sorted(
        labels.labeled_frames,
        key=lambda lf: (video_order.get(lf.video, 0), lf.frame_idx),
    )

    n = len(sorted_frames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # Ensure non-empty splits for reasonably sized datasets.
    if n >= 3:
        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

    train_frames = sorted_frames[:n_train]
    val_frames = sorted_frames[n_train:n_train + n_val]
    test_frames = sorted_frames[n_train + n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / f"{base_name}_train.pkg.slp"
    val_path = output_dir / f"{base_name}_val.pkg.slp"
    test_path = output_dir / f"{base_name}_test.pkg.slp"

    _build_subset(labels, train_frames).save(str(train_path))
    _build_subset(labels, val_frames).save(str(val_path))
    _build_subset(labels, test_frames).save(str(test_path))

    print(f"Input: {input_path}")
    print(f"Total frames: {n}")
    print(f"Train frames: {len(train_frames)} -> {train_path}")
    print(f"Val frames:   {len(val_frames)} -> {val_path}")
    print(f"Test frames:  {len(test_frames)} -> {test_path}")

    return train_path, val_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split SLEAP package into chronological train/val/test files."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .slp/.pkg.slp path")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--base-name", type=str, default="shrimp", help="Output filename prefix")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"({args.train_ratio}, {args.val_ratio}, {args.test_ratio})"
        )

    split_chronological(
        input_path=args.input,
        output_dir=args.out_dir,
        base_name=args.base_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
