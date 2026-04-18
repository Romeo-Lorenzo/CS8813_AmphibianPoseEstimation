import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = {"frame_idx", "node", "x", "y"}


def load_analysis_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def reduce_to_one_point_per_frame_node(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    if "instance_score" in out.columns:
        out["instance_score"] = pd.to_numeric(out["instance_score"], errors="coerce").fillna(-np.inf)
        out = out.sort_values(["frame_idx", "node", "instance_score"], ascending=[True, True, False])
        out = out.drop_duplicates(subset=["frame_idx", "node"], keep="first")
    else:
        out = out.drop_duplicates(subset=["frame_idx", "node"], keep="first")

    out = out[["frame_idx", "node", "x", "y"]]
    return out


def compute_l2_distribution(gt_df: pd.DataFrame, inf_df: pd.DataFrame) -> pd.DataFrame:
    gt = reduce_to_one_point_per_frame_node(gt_df).rename(columns={"x": "x_gt", "y": "y_gt"})
    inf = reduce_to_one_point_per_frame_node(inf_df).rename(columns={"x": "x_inf", "y": "y_inf"})

    merged = gt.merge(inf, on=["frame_idx", "node"], how="inner")
    merged = merged.dropna(subset=["x_gt", "y_gt", "x_inf", "y_inf"]).copy()
    merged["l2"] = np.sqrt((merged["x_inf"] - merged["x_gt"]) ** 2 + (merged["y_inf"] - merged["y_gt"]) ** 2)
    return merged


def summarize_by_keypoint(dist_df: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        dist_df.groupby("node")["l2"]
        .agg(mean_l2="mean", median_l2="median", std_l2="std", n_samples="count")
        .reset_index()
        .sort_values("mean_l2", ascending=False)
    )
    return metrics


def save_violin(dist_df: pd.DataFrame, out_png: Path) -> None:
    if dist_df.empty:
        raise RuntimeError("No overlapping frame/node points to plot.")

    order = (
        dist_df.groupby("node")["l2"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    data = [dist_df.loc[dist_df["node"] == node, "l2"].to_numpy() for node in order]

    plt.figure(figsize=(14, 6))
    vp = plt.violinplot(data, showmeans=True, showmedians=True)
    for body in vp["bodies"]:
        body.set_alpha(0.55)

    plt.xticks(np.arange(1, len(order) + 1), order, rotation=65, ha="right")
    plt.ylabel("L2 Error (pixels)")
    plt.title("SLEAP Fly32 Test: L2 Distribution by Keypoint")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SLEAP inference vs GT with per-keypoint L2 violin plot.")
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=Path("dataset/fly32/SLEAP/results to evaluate/fly32_test_groundtruth.analysis.csv"),
        help="Ground-truth SLEAP analysis CSV.",
    )
    parser.add_argument(
        "--inference-csv",
        type=Path,
        default=Path("dataset/fly32/SLEAP/results to evaluate/fly32_test_inference_complete.analysis.csv"),
        help="Inference SLEAP analysis CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tools/Evaluate metrics/output_sleap"),
        help="Output directory for metrics and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {args.ground_truth_csv}")
    if not args.inference_csv.exists():
        raise FileNotFoundError(f"Inference CSV not found: {args.inference_csv}")

    gt_df = load_analysis_csv(args.ground_truth_csv)
    inf_df = load_analysis_csv(args.inference_csv)

    dist_df = compute_l2_distribution(gt_df, inf_df)
    metrics_df = summarize_by_keypoint(dist_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dist_csv = args.out_dir / "sleap_l2_distances_per_point.csv"
    metrics_csv = args.out_dir / "sleap_l2_per_keypoint.csv"
    violin_png = args.out_dir / "sleap_l2_per_keypoint_violin.png"

    dist_df.to_csv(dist_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    save_violin(dist_df, violin_png)

    print("SLEAP evaluation complete.")
    print(f"Ground truth: {args.ground_truth_csv}")
    print(f"Inference: {args.inference_csv}")
    print(f"Matched points: {len(dist_df)}")
    print(f"Distances CSV: {dist_csv}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Violin plot: {violin_png}")


if __name__ == "__main__":
    main()
