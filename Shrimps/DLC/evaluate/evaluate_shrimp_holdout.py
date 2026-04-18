from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = PROJECT_ROOT / "labeled-data" / "three_big_shrimps_video_two_holdout"
GT_CSV = HOLDOUT_DIR / "CollectedData_Lolo.csv"
DLC_CSV = HOLDOUT_DIR / "three_big_shrimps_video_two_last1000DLC_mobnet_100_ShrimpsApr12shuffle1_10000_el.csv"
SLEAP_PKG = (
    PROJECT_ROOT.parent.parent
    / "SLEAP"
    / "Shrimps"
    / "CorrectedishSLEAPAnalysisToBeUsedAsGroundTruth"
    / "shrimp_last1000.pkg.slp"
)
OUTPUT_DIR = PROJECT_ROOT / "evaluate"
DLC_OUT_DIR = OUTPUT_DIR / "DLC"
SLEAP_OUT_DIR = OUTPUT_DIR / "SLEAP"
VIOLIN_PLOT = OUTPUT_DIR / "shrimp_l2_pixel_error_violin.png"
COMPARISON_JSON = OUTPUT_DIR / "shrimp_l2_pixel_error_comparison.json"
COMPARISON_CSV = OUTPUT_DIR / "shrimp_l2_pixel_error_summary.csv"

COMPARE_FRAMES = 1000
COMPARE_INDIVIDUALS = 3
BODYPARTS = ["head", "mid1", "mid2", "mid3", "tail"]


@dataclass
class MethodMetrics:
    method: str
    comparable_frames: int
    comparable_points: int
    mean_pixel_error: float
    median_pixel_error: float
    rmse_pixel_error: float
    p95_pixel_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate shrimp DLC and SLEAP holdout predictions.")
    parser.add_argument("--compare-frames", type=int, default=COMPARE_FRAMES, help="Number of holdout frames to compare from the tail of the dataset.")
    return parser.parse_args()


def _finite_xy(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _pixel_distance(px: float, py: float, gx: float, gy: float) -> float:
    return float(math.hypot(px - gx, py - gy))


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)


def select_gt_individuals(gt_slice: pd.DataFrame, max_count: int) -> List[str]:
    counts: List[Tuple[int, str]] = []
    for individual in gt_slice.columns.get_level_values(1).unique():
        subframe = gt_slice.xs(individual, level=1, axis=1)
        count = int(subframe.notna().any(axis=1).sum())
        if count > 0:
            counts.append((count, str(individual)))
    counts.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in counts[:max_count]]


def extract_gt_entities(gt_slice: pd.DataFrame, individuals: Sequence[str]) -> List[Dict[str, Tuple[float, float]]]:
    entities: List[Dict[str, Tuple[float, float]]] = []
    for individual in individuals:
        entity: Dict[str, Tuple[float, float]] = {}
        for bodypart in BODYPARTS:
            x = gt_slice.iloc[0][(gt_slice.columns[0][0], individual, bodypart, "x")]
            y = gt_slice.iloc[0][(gt_slice.columns[0][0], individual, bodypart, "y")]
            if _finite_xy(x) and _finite_xy(y):
                entity[bodypart] = (float(x), float(y))
        entities.append(entity)
    return entities


def extract_dlc_entities(pred_slice: pd.DataFrame, individuals: Sequence[str]) -> List[Dict[str, Tuple[float, float]]]:
    scorer = pred_slice.columns.get_level_values(0)[0]
    entities: List[Dict[str, Tuple[float, float]]] = []
    for individual in individuals:
        entity: Dict[str, Tuple[float, float]] = {}
        for bodypart in BODYPARTS:
            x = pred_slice.iloc[0][(scorer, individual, bodypart, "x")]
            y = pred_slice.iloc[0][(scorer, individual, bodypart, "y")]
            if _finite_xy(x) and _finite_xy(y):
                entity[bodypart] = (float(x), float(y))
        entities.append(entity)
    return entities


def extract_sleap_entities(labeled_frame: object, bodyparts: Sequence[str]) -> List[Dict[str, Tuple[float, float]]]:
    predicted = [inst for inst in getattr(labeled_frame, "instances", []) if type(inst).__name__ == "PredictedInstance"]
    predicted.sort(key=lambda inst: float(getattr(inst, "score", 0.0) or 0.0), reverse=True)
    entities: List[Dict[str, Tuple[float, float]]] = []
    for inst in predicted:
        points = np.asarray(inst.numpy(), dtype=float)
        entity: Dict[str, Tuple[float, float]] = {}
        for bodypart, xy in zip(bodyparts, points):
            x = float(xy[0])
            y = float(xy[1])
            if math.isfinite(x) and math.isfinite(y):
                entity[bodypart] = (x, y)
        entities.append(entity)
    return entities


def best_match_errors(
    gt_entities: Sequence[Dict[str, Tuple[float, float]]],
    pred_entities: Sequence[Dict[str, Tuple[float, float]]],
    bodyparts: Sequence[str],
) -> Tuple[List[float], Dict[str, List[float]]]:
    if not gt_entities or not pred_entities:
        return [], {bodypart: [] for bodypart in bodyparts}

    pair_count = min(len(gt_entities), len(pred_entities))
    gt_subset = list(gt_entities[:pair_count])
    pred_subset = list(pred_entities[:pair_count])

    cost = np.full((pair_count, pair_count), np.nan, dtype=float)
    for pi, pred_entity in enumerate(pred_subset):
        for gi, gt_entity in enumerate(gt_subset):
            distances: List[float] = []
            for bodypart in bodyparts:
                if bodypart not in pred_entity or bodypart not in gt_entity:
                    continue
                px, py = pred_entity[bodypart]
                gx, gy = gt_entity[bodypart]
                if math.isfinite(px) and math.isfinite(py) and math.isfinite(gx) and math.isfinite(gy):
                    distances.append(_pixel_distance(px, py, gx, gy))
            if distances:
                cost[pi, gi] = float(np.mean(distances))

    best_perm: Tuple[int, ...] | None = None
    best_score: Tuple[int, float, float] | None = None
    for perm in permutations(range(pair_count)):
        finite = [cost[pi, gi] for pi, gi in enumerate(perm) if np.isfinite(cost[pi, gi])]
        if not finite:
            continue
        score = (len(finite), -float(np.mean(finite)), -float(np.sum(finite)))
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm

    if best_perm is None:
        return [], {bodypart: [] for bodypart in bodyparts}

    point_errors: List[float] = []
    bodypart_errors: Dict[str, List[float]] = {bodypart: [] for bodypart in bodyparts}
    for pi, gi in enumerate(best_perm):
        pred_entity = pred_subset[pi]
        gt_entity = gt_subset[gi]
        for bodypart in bodyparts:
            if bodypart not in pred_entity or bodypart not in gt_entity:
                continue
            px, py = pred_entity[bodypart]
            gx, gy = gt_entity[bodypart]
            if math.isfinite(px) and math.isfinite(py) and math.isfinite(gx) and math.isfinite(gy):
                distance = _pixel_distance(px, py, gx, gy)
                point_errors.append(distance)
                bodypart_errors[bodypart].append(distance)

    return point_errors, bodypart_errors


def evaluate_dlc(gt: pd.DataFrame, pred: pd.DataFrame, compare_frames: int) -> Tuple[MethodMetrics, Dict[str, List[float]], List[float], pd.DataFrame]:
    gt_slice = gt.tail(compare_frames).reset_index(drop=True)
    pred_slice = pred.tail(compare_frames).reset_index(drop=True)

    gt_individuals = select_gt_individuals(gt_slice, COMPARE_INDIVIDUALS)
    pred_individuals = [str(name) for name in pred_slice.columns.get_level_values(1).unique()]
    pred_individuals = pred_individuals[:COMPARE_INDIVIDUALS]

    frame_rows: List[Dict[str, object]] = []
    all_errors: List[float] = []
    bodypart_errors: Dict[str, List[float]] = {bodypart: [] for bodypart in BODYPARTS}

    for frame_idx in range(min(len(gt_slice), len(pred_slice))):
        gt_frame = gt_slice.iloc[[frame_idx]]
        pred_frame = pred_slice.iloc[[frame_idx]]
        gt_entities = extract_gt_entities(gt_frame, gt_individuals)
        pred_entities = extract_dlc_entities(pred_frame, pred_individuals)
        point_errors, body_errors = best_match_errors(gt_entities, pred_entities, BODYPARTS)
        all_errors.extend(point_errors)
        for bodypart, values in body_errors.items():
            bodypart_errors[bodypart].extend(values)
        frame_rows.append(
            {
                "frame_index": frame_idx,
                "frame_mean_pixel_error": float(np.mean(point_errors)) if point_errors else math.nan,
                "frame_point_count": len(point_errors),
                "gt_individuals": ",".join(gt_individuals),
                "pred_individuals": ",".join(pred_individuals),
            }
        )

    metrics = MethodMetrics(
        method="DLC",
        comparable_frames=sum(1 for row in frame_rows if math.isfinite(row["frame_mean_pixel_error"])),
        comparable_points=len(all_errors),
        mean_pixel_error=float(np.mean(all_errors)) if all_errors else math.nan,
        median_pixel_error=float(np.median(all_errors)) if all_errors else math.nan,
        rmse_pixel_error=float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else math.nan,
        p95_pixel_error=_percentile(all_errors, 0.95),
    )
    return metrics, bodypart_errors, all_errors, pd.DataFrame(frame_rows)


def evaluate_sleap(gt: pd.DataFrame, pkg_path: Path, compare_frames: int) -> Tuple[MethodMetrics, Dict[str, List[float]], List[float], pd.DataFrame]:
    try:
        import sleap_io as sio
    except ImportError as exc:
        raise RuntimeError("SLEAP evaluation requires sleap_io in the active Python environment.") from exc

    labels = sio.load_slp(str(pkg_path))
    labeled_frames = list(labels.labeled_frames)[-compare_frames:]
    gt_slice = gt.tail(len(labeled_frames)).reset_index(drop=True)

    frame_count = min(len(gt_slice), len(labeled_frames))
    gt_slice = gt_slice.iloc[:frame_count].reset_index(drop=True)
    labeled_frames = labeled_frames[:frame_count]

    gt_individuals = select_gt_individuals(gt_slice, COMPARE_INDIVIDUALS)
    bodyparts = [str(node.name).lower() for node in labels.skeletons[0].nodes]
    bodyparts = [bp for bp in bodyparts if bp in BODYPARTS] or BODYPARTS

    frame_rows: List[Dict[str, object]] = []
    all_errors: List[float] = []
    bodypart_errors: Dict[str, List[float]] = {bodypart: [] for bodypart in BODYPARTS}

    for frame_idx, labeled_frame in enumerate(labeled_frames):
        gt_frame = gt_slice.iloc[[frame_idx]]
        gt_entities = extract_gt_entities(gt_frame, gt_individuals)
        pred_entities = extract_sleap_entities(labeled_frame, bodyparts)
        point_errors, body_errors = best_match_errors(gt_entities, pred_entities, BODYPARTS)
        all_errors.extend(point_errors)
        for bodypart, values in body_errors.items():
            bodypart_errors[bodypart].extend(values)
        frame_rows.append(
            {
                "frame_index": frame_idx,
                "source_frame_idx": int(getattr(labeled_frame, "frame_idx", frame_idx)),
                "frame_mean_pixel_error": float(np.mean(point_errors)) if point_errors else math.nan,
                "frame_point_count": len(point_errors),
                "gt_individuals": ",".join(gt_individuals),
                "pred_instances": len(pred_entities),
            }
        )

    metrics = MethodMetrics(
        method="SLEAP",
        comparable_frames=sum(1 for row in frame_rows if math.isfinite(row["frame_mean_pixel_error"])),
        comparable_points=len(all_errors),
        mean_pixel_error=float(np.mean(all_errors)) if all_errors else math.nan,
        median_pixel_error=float(np.median(all_errors)) if all_errors else math.nan,
        rmse_pixel_error=float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else math.nan,
        p95_pixel_error=_percentile(all_errors, 0.95),
    )
    return metrics, bodypart_errors, all_errors, pd.DataFrame(frame_rows)


def write_method_outputs(out_dir: Path, metrics: MethodMetrics, framewise: pd.DataFrame, bodypart_errors: Dict[str, List[float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "metrics_summary.csv"
    framewise_path = out_dir / "framewise_errors.csv"
    bodypart_path = out_dir / "bodypart_errors.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "method",
            "comparable_frames",
            "comparable_points",
            "mean_pixel_error",
            "median_pixel_error",
            "rmse_pixel_error",
            "p95_pixel_error",
        ])
        writer.writerow([
            metrics.method,
            metrics.comparable_frames,
            metrics.comparable_points,
            f"{metrics.mean_pixel_error:.6f}",
            f"{metrics.median_pixel_error:.6f}",
            f"{metrics.rmse_pixel_error:.6f}",
            f"{metrics.p95_pixel_error:.6f}",
        ])

    framewise.to_csv(framewise_path, index=False)

    with bodypart_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["bodypart", "count", "mean_pixel_error", "median_pixel_error", "rmse_pixel_error", "p95_pixel_error"])
        for bodypart in BODYPARTS:
            values = bodypart_errors.get(bodypart, [])
            if values:
                writer.writerow([
                    bodypart,
                    len(values),
                    f"{float(np.mean(values)):.6f}",
                    f"{float(np.median(values)):.6f}",
                    f"{float(np.sqrt(np.mean(np.square(values)))):.6f}",
                    f"{_percentile(values, 0.95):.6f}",
                ])
            else:
                writer.writerow([bodypart, 0, "", "", "", ""])


def write_comparison_files(dlc: MethodMetrics, sleap: MethodMetrics) -> None:
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "method",
            "comparable_frames",
            "comparable_points",
            "mean_pixel_error",
            "median_pixel_error",
            "rmse_pixel_error",
            "p95_pixel_error",
        ])
        for metrics in (dlc, sleap):
            writer.writerow([
                metrics.method,
                metrics.comparable_frames,
                metrics.comparable_points,
                f"{metrics.mean_pixel_error:.6f}",
                f"{metrics.median_pixel_error:.6f}",
                f"{metrics.rmse_pixel_error:.6f}",
                f"{metrics.p95_pixel_error:.6f}",
            ])

    payload = {
        "lower_error_is_better": True,
        "metrics": {
            "DLC": dlc.__dict__,
            "SLEAP": sleap.__dict__,
        },
        "comparison_dlc_minus_sleap": {
            "mean_pixel_error": dlc.mean_pixel_error - sleap.mean_pixel_error,
            "median_pixel_error": dlc.median_pixel_error - sleap.median_pixel_error,
            "rmse_pixel_error": dlc.rmse_pixel_error - sleap.rmse_pixel_error,
            "p95_pixel_error": dlc.p95_pixel_error - sleap.p95_pixel_error,
        },
        "winner_by_mean_error": "DLC" if dlc.mean_pixel_error < sleap.mean_pixel_error else "SLEAP",
    }
    with COMPARISON_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_violin(dlc_errors: Sequence[float], sleap_errors: Sequence[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
    parts = ax.violinplot([dlc_errors, sleap_errors], positions=[1, 2], widths=0.75, showmeans=False, showmedians=True, showextrema=False)
    palette = ["#c75b12", "#2b8a8a"]
    for body, color in zip(parts["bodies"], palette):
        body.set_facecolor(color)
        body.set_edgecolor("#2b2b2b")
        body.set_alpha(0.78)

    ax.scatter([1, 2], [np.mean(dlc_errors), np.mean(sleap_errors)], color="#1f1f1f", s=28, zorder=3, label="mean")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["DLC", "SLEAP"])
    ax.set_ylabel("L2 pixel error")
    ax.set_title("Shrimp holdout pixel error")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(VIOLIN_PLOT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    for path in (GT_CSV, DLC_CSV, SLEAP_PKG):
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation input: {path}")

    gt = load_ground_truth(GT_CSV)
    dlc_metrics, dlc_bodyparts, dlc_errors, dlc_framewise = evaluate_dlc(gt, pd.read_csv(DLC_CSV, header=[0, 1, 2, 3], index_col=0), args.compare_frames)
    sleap_metrics, sleap_bodyparts, sleap_errors, sleap_framewise = evaluate_sleap(gt, SLEAP_PKG, args.compare_frames)

    write_method_outputs(DLC_OUT_DIR, dlc_metrics, dlc_framewise, dlc_bodyparts)
    write_method_outputs(SLEAP_OUT_DIR, sleap_metrics, sleap_framewise, sleap_bodyparts)
    write_comparison_files(dlc_metrics, sleap_metrics)
    plot_violin(dlc_errors, sleap_errors)

    print("Evaluation complete.")
    print(f"Ground truth: {GT_CSV}")
    print(f"DLC predictions: {DLC_CSV}")
    print(f"SLEAP predictions: {SLEAP_PKG}")
    print(f"DLC summary: {DLC_OUT_DIR / 'metrics_summary.csv'}")
    print(f"SLEAP summary: {SLEAP_OUT_DIR / 'metrics_summary.csv'}")
    print(f"Comparison CSV: {COMPARISON_CSV}")
    print(f"Comparison JSON: {COMPARISON_JSON}")
    print(f"Violin plot: {VIOLIN_PLOT}")
    print(f"DLC mean error: {dlc_metrics.mean_pixel_error:.4f} px")
    print(f"SLEAP mean error: {sleap_metrics.mean_pixel_error:.4f} px")


if __name__ == "__main__":
    main()