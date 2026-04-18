"""Export SLEAP labels to DeepLabCut multi-animal CollectedData format."""

from pathlib import Path

import numpy as np
import pandas as pd
from sleap_io import load_slp


def _xy_from_sleap_point(point) -> tuple[float, float]:
    """Handle SLEAP point objects and numpy.void tuples robustly."""
    if point is None:
        return (np.nan, np.nan)

    if hasattr(point, "x") and hasattr(point, "y"):
        return (float(point.x), float(point.y))

    try:
        coords = point[0]
        return (float(coords[0]), float(coords[1]))
    except Exception:
        return (np.nan, np.nan)


def export_sleap_to_dlc_multianimal(
    slp_path: str,
    output_dir: str,
    video_name: str,
    scorer: str,
    individuals: list[str],
):
    """Create DLC-compatible multi-animal CollectedData CSV/H5.

    Output format is 4-level columns:
    scorer / individual / bodyparts / coords(x,y)
    and index are relative image paths in labeled-data.
    """
    labels = load_slp(slp_path)
    bodyparts = list(labels.skeleton.node_names)
    labeled_frames = sorted(
        [lf for lf in labels.labeled_frames if len(lf.instances) > 0],
        key=lambda lf: lf.frame_idx,
    )

    col_tuples = []
    for ind in individuals:
        for bp in bodyparts:
            col_tuples.append((scorer, ind, bp, "x"))
            col_tuples.append((scorer, ind, bp, "y"))
    columns = pd.MultiIndex.from_tuples(
        col_tuples,
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    rows = []
    index = []
    for lf in labeled_frames:
        row = {c: np.nan for c in columns}
        instances = list(lf.instances)[: len(individuals)]
        for i, inst in enumerate(instances):
            ind = individuals[i]
            for bp_idx, bp in enumerate(bodyparts):
                if bp_idx < len(inst.points):
                    x, y = _xy_from_sleap_point(inst.points[bp_idx])
                    row[(scorer, ind, bp, "x")] = x
                    row[(scorer, ind, bp, "y")] = y

        img_name = f"{video_name}_{lf.frame_idx:06d}.png"
        rel_path = f"labeled-data/{video_name}/{img_name}"
        rows.append(row)
        index.append(rel_path)

    df = pd.DataFrame(rows, index=index, columns=columns)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"CollectedData_{scorer}.csv"
    h5_path = out_dir / f"CollectedData_{scorer}.h5"

    df.to_csv(csv_path)
    df.to_hdf(h5_path, key="df_with_missing", format="table")

    print("Export complete.")
    print(f"Frames: {len(df)}")
    print(f"Bodyparts: {bodyparts}")
    print(f"Individuals: {individuals}")
    print(f"CSV: {csv_path}")
    print(f"H5: {h5_path}")
    return df


if __name__ == "__main__":
    slp_path = r"SLEAP/Shrimps/CorrectedishSLEAPAnalysisToBeUsedAsGroundTruth/ApplyingTrainedModelThreeBigShrimp.pkg.slp"
    output_dir = r"DLC/Shrimps-Lolo-2026-04-12/labeled-data/three_big_shrimps_video_two"
    video_name = "three_big_shrimps_video_two"
    scorer = "Lolo"
    individuals = [
        "shrimp1",
        "shrimp2",
        "shrimp3",
        "shrimp4",
        "shrimp5",
        "shrimp6",
        "shrimp7",
    ]

    export_sleap_to_dlc_multianimal(
        slp_path=slp_path,
        output_dir=output_dir,
        video_name=video_name,
        scorer=scorer,
        individuals=individuals,
    )
