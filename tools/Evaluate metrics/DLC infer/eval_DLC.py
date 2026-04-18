import argparse
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sleap_io as sio


def load_dlc_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)


def get_keypoints(df: pd.DataFrame) -> list[str]:
    lvl1 = list(dict.fromkeys(df.columns.get_level_values(1)))
    coords = set(df.columns.get_level_values(2))
    if not {"x", "y"}.issubset(coords):
        raise ValueError(f"CSV is missing x/y coordinates: {coords}")
    return lvl1


def extract_xy(df: pd.DataFrame, keypoint: str) -> np.ndarray:
    col_x = [c for c in df.columns if c[1] == keypoint and c[2] == "x"]
    col_y = [c for c in df.columns if c[1] == keypoint and c[2] == "y"]
    if not col_x or not col_y:
        raise KeyError(f"Missing x/y columns for keypoint: {keypoint}")

    x = pd.to_numeric(df[col_x[0]], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[col_y[0]], errors="coerce").to_numpy(dtype=float)
    return np.column_stack([x, y])


def extract_likelihood(df: pd.DataFrame, keypoint: str) -> np.ndarray | None:
    col_l = [c for c in df.columns if c[1] == keypoint and c[2] == "likelihood"]
    if not col_l:
        return None
    return pd.to_numeric(df[col_l[0]], errors="coerce").to_numpy(dtype=float)


def get_frame_indices_from_slp(slp_path: Path) -> list[int]:
    slp_loader = cast(Callable[[str], Any], getattr(sio, "load_slp", None))
    if not callable(slp_loader):
        raise RuntimeError("sleap_io.load_slp is not available in this environment.")

    labels = slp_loader(str(slp_path))
    return [int(lf.frame_idx) for lf in labels.labeled_frames]


def align_inference_to_gt(
    gt_df: pd.DataFrame,
    inf_df: pd.DataFrame,
    slp_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if slp_path is not None and slp_path.exists():
        frame_indices = get_frame_indices_from_slp(slp_path)
        if len(frame_indices) != len(gt_df):
            raise ValueError(
                f"test.pkg.slp frame count ({len(frame_indices)}) does not match "
                f"ground-truth rows ({len(gt_df)})."
            )

        inf_aligned = inf_df.copy()
        inf_aligned.index = pd.to_numeric(inf_aligned.index, errors="coerce")
        missing = [idx for idx in frame_indices if idx not in set(inf_aligned.index.dropna())]
        if missing:
            raise KeyError(
                f"Inference CSV missing {len(missing)} frame indices from SLP mapping. "
                f"First missing: {missing[:5]}"
            )

        inf_subset = inf_aligned.loc[frame_indices]
        inf_subset.index = gt_df.index
        return gt_df.copy(), inf_subset

    n = min(len(gt_df), len(inf_df))
    gt_subset = gt_df.iloc[:n].copy()
    inf_subset = inf_df.iloc[:n].copy()
    inf_subset.index = gt_subset.index
    return gt_subset, inf_subset


def compute_l2_per_keypoint(
    gt_df: pd.DataFrame,
    inf_df: pd.DataFrame,
    min_likelihood: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_kps = set(get_keypoints(gt_df))
    inf_kps = set(get_keypoints(inf_df))
    common_kps = sorted(gt_kps.intersection(inf_kps))
    if not common_kps:
        raise ValueError("No common keypoints between GT and inference CSV.")

    rows = []
    dist_rows = []
    for kp in common_kps:
        gt_xy = extract_xy(gt_df, kp)
        inf_xy = extract_xy(inf_df, kp)
        likelihood = extract_likelihood(inf_df, kp)

        valid = np.isfinite(gt_xy).all(axis=1) & np.isfinite(inf_xy).all(axis=1)
        if likelihood is not None:
            valid &= np.isfinite(likelihood) & (likelihood >= min_likelihood)

        if not np.any(valid):
            rows.append({
                "keypoint": kp,
                "mean_l2": np.nan,
                "median_l2": np.nan,
                "std_l2": np.nan,
                "n_samples": 0,
            })
            continue

        d = np.linalg.norm(inf_xy[valid] - gt_xy[valid], axis=1)
        for val in d:
            dist_rows.append({"keypoint": kp, "l2": float(val)})
        rows.append({
            "keypoint": kp,
            "mean_l2": float(np.mean(d)),
            "median_l2": float(np.median(d)),
            "std_l2": float(np.std(d)),
            "n_samples": int(d.size),
        })

    result = pd.DataFrame(rows).sort_values("mean_l2", ascending=False)
    dist_df = pd.DataFrame(dist_rows)
    return result, dist_df


def save_plot(metrics_df: pd.DataFrame, output_png: Path, title: str) -> None:
    plot_df = metrics_df.dropna(subset=["mean_l2"]).copy()
    if plot_df.empty:
        raise RuntimeError("No valid metrics to plot.")

    plt.figure(figsize=(14, 6))
    bars = plt.bar(plot_df["keypoint"], plot_df["mean_l2"])
    plt.xticks(rotation=65, ha="right")
    plt.ylabel("Mean L2 Error (pixels)")
    plt.title(title)
    plt.tight_layout()

    for bar, n in zip(bars, plot_df["n_samples"]):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y, str(int(n)), ha="center", va="bottom", fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def save_violin_plot(dist_df: pd.DataFrame, output_png: Path, title: str) -> None:
    if dist_df.empty:
        raise RuntimeError("No valid L2 distances to plot.")

    ordered_kps = (
        dist_df.groupby("keypoint")["l2"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    data = [dist_df.loc[dist_df["keypoint"] == kp, "l2"].to_numpy() for kp in ordered_kps]

    plt.figure(figsize=(14, 6))
    vp = plt.violinplot(data, showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_alpha(0.55)

    plt.xticks(np.arange(1, len(ordered_kps) + 1), ordered_kps, rotation=65, ha="right")
    plt.ylabel("L2 Error (pixels)")
    plt.title(title)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DLC inference against GT with per-keypoint L2 metrics and plot."
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=Path("dataset/fly32/DLC/results to evaluate/fly32_test_groundtruth_from_test_pkg.csv"),
        help="Ground-truth DLC CSV path.",
    )
    parser.add_argument(
        "--inference-csv",
        type=Path,
        default=Path("dataset/fly32/DLC/results to evaluate/fly32_test_inference_dlc_mobnet100_shuffle1_iter100000.csv"),
        help="Inference DLC CSV path.",
    )
    parser.add_argument(
        "--test-slp",
        type=Path,
        default=Path("dataset/fly32/test.pkg.slp"),
        help="SLEAP test pkg used to map GT rows to original frame indices.",
    )
    parser.add_argument(
        "--min-likelihood",
        type=float,
        default=0.0,
        help="Minimum DLC likelihood to include inference points.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/Evaluate metrics/output"),
        help="Output folder for metrics CSV and plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {args.ground_truth_csv}")
    if not args.inference_csv.exists():
        raise FileNotFoundError(f"Inference CSV not found: {args.inference_csv}")

    gt_df = load_dlc_csv(args.ground_truth_csv)
    inf_df = load_dlc_csv(args.inference_csv)

    slp_path = args.test_slp if args.test_slp.exists() else None
    gt_aligned, inf_aligned = align_inference_to_gt(gt_df, inf_df, slp_path)

    metrics_df, dist_df = compute_l2_per_keypoint(
        gt_df=gt_aligned,
        inf_df=inf_aligned,
        min_likelihood=args.min_likelihood,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.out_dir / "l2_per_keypoint.csv"
    metrics_png = args.out_dir / "l2_per_keypoint.png"
    violin_png = args.out_dir / "l2_per_keypoint_violin.png"
    dist_csv = args.out_dir / "l2_per_keypoint_distances.csv"

    metrics_df.to_csv(metrics_csv, index=False)
    dist_df.to_csv(dist_csv, index=False)
    save_plot(metrics_df, metrics_png, "DLC L2 Error by Keypoint (Fly32 Test)")
    save_violin_plot(dist_df, violin_png, "DLC L2 Error Distribution by Keypoint (Fly32 Test)")

    print("Evaluation complete.")
    print(f"Ground truth: {args.ground_truth_csv}")
    print(f"Inference: {args.inference_csv}")
    print(f"Aligned rows: {len(gt_aligned)}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Distances CSV: {dist_csv}")
    print(f"Metrics plot: {metrics_png}")
    print(f"Violin plot: {violin_png}")


if __name__ == "__main__":
    main()
