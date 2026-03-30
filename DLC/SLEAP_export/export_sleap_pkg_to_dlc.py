import argparse
import csv
import math
from pathlib import Path

import pandas as pd
from PIL import Image
import sleap_io as sio


def format_coord(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.10f}".rstrip("0").rstrip(".")


def pick_instance(labeled_frame):
    if labeled_frame.user_instances:
        return labeled_frame.user_instances[0]
    if labeled_frame.instances:
        return labeled_frame.instances[0]
    return None


def export_sleap_to_dlc(slp_path: Path, output_dir: Path, scorer: str, subset_name: str) -> None:
    labels = sio.load_slp(str(slp_path))
    if not labels.skeletons:
        raise RuntimeError("No skeletons found in the SLEAP package.")

    labeled_data_dir = output_dir / "labeled-data"
    subset_dir = labeled_data_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    csv_in_subset = subset_dir / f"CollectedData_{scorer}.csv"
    csv_in_labeled_data = labeled_data_dir / f"CollectedData_{scorer}.csv"
    h5_in_subset = subset_dir / f"CollectedData_{scorer}.h5"

    node_names = [node.name for node in labels.skeletons[0].nodes]
    header_scorer = ["scorer"] + [scorer for _ in node_names for __ in (0, 1)]
    header_bodyparts = ["bodyparts"] + [name for name in node_names for _ in (0, 1)]
    header_coords = ["coords"] + ["x", "y"] * len(node_names)

    rows = [header_scorer, header_bodyparts, header_coords]

    exported_count = 0
    skipped_count = 0

    for row_index, labeled_frame in enumerate(labels.labeled_frames):
        image_array = labeled_frame.image
        instance = pick_instance(labeled_frame)

        if image_array is None or instance is None:
            skipped_count += 1
            continue

        image_name = f"img{row_index:04d}.png"
        image_path = subset_dir / image_name

        image_array = image_array.squeeze()
        Image.fromarray(image_array).save(image_path)

        points = instance.numpy()
        row = [f"labeled-data/{subset_name}/{image_name}"]
        for x, y in points:
            row.extend([format_coord(x), format_coord(y)])
        rows.append(row)
        exported_count += 1

    with csv_in_subset.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)

    with csv_in_labeled_data.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)

    df = pd.read_csv(csv_in_subset, header=[0, 1, 2], index_col=0)
    df.to_hdf(h5_in_subset, key="df_with_missing", mode="w")

    print("Export complete.")
    print(f"Input SLP: {slp_path}")
    print(f"Output root: {output_dir}")
    print(f"Subset folder: {subset_dir}")
    print(f"Frames exported: {exported_count}")
    print(f"Frames skipped: {skipped_count}")
    print(f"CSV (subset): {csv_in_subset}")
    print(f"CSV (labeled-data): {csv_in_labeled_data}")
    print(f"H5 (subset): {h5_in_subset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a SLEAP .slp package into a DLC-style labeled-data layout "
            "with frames, CollectedData CSV, and H5."
        )
    )
    parser.add_argument(
        "--slp",
        type=Path,
        required=True,
        help="Path to the input SLEAP package (.slp).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("SLEAP_Exported_to_DLC"),
        help=(
            "Target folder where DLC-like structure will be created. "
            "Defaults to ./SLEAP_Exported_to_DLC"
        ),
    )
    parser.add_argument(
        "--scorer",
        type=str,
        required=True,
        help="Scorer name used in CollectedData_<scorer>.csv.",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        required=True,
        help="Subfolder name inside labeled-data (example: fly32_train).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    slp_path = args.slp.resolve()
    out_dir = args.out_dir.resolve()

    if not slp_path.exists():
        raise FileNotFoundError(f"SLP file does not exist: {slp_path}")

    export_sleap_to_dlc(
        slp_path=slp_path,
        output_dir=out_dir,
        scorer=args.scorer,
        subset_name=args.subset_name,
    )


if __name__ == "__main__":
    main()
