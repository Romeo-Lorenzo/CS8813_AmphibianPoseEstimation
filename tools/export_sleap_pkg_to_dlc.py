import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable, cast
import pandas as pd
from PIL import Image
import sleap_io as sio


###############################################################################
###############################################################################
###                                                                         ###
###                              HOW TO RUN                                 ###
###                                                                         ###
###  1) Open a terminal in this script's folder.                            ###
###  2) Run with defaults (from the section below):                         ###
###     python export_sleap_pkg_to_dlc.py                                   ###
###  3) Optional override example:                                           ###
###     python export_sleap_pkg_to_dlc.py --slp train.pkg.slp               ###
###         --scorer Lorenzo                                                 ###
###         --out-dir SLEAP_Exported_to_DLC                                 ###
###                                                                         ###
###  Output structure created:                                               ###
###  SLEAP_Exported_to_DLC/labeled-data/                                    ###
###    - img0000.png, img0001.png, ...                                      ###
###    - CollectedData_<scorer>.csv                                         ###
###    - CollectedData_<scorer>.h5                                          ###
###                                                                         ###
###############################################################################
###############################################################################


###############################################################################
###############################################################################
###                                                                         ###
###                    CHANGE YOUR DEFAULT INPUTS HERE                      ###
###                                                                         ###
###  Example values:                                                        ###
###  - "train.pkg.slp"                                                      ###
###  - "my_dataset.pkg.slp"                                                 ###
###  - "User"                                                            ###
###                                                                         ###
###  If you run without CLI flags, these defaults will be used:            ###
###  --slp, --scorer                                                         ###
###                                                                         ###
###############################################################################
###############################################################################
DEFAULT_SLEAP_PACKAGE_NAME = "../SLEAP/Shrimps/ApplyingTrainedModelThreeBigShrimp.slp"
DEFAULT_SCORER = "User"  # Change this to your name or a nickname if you like!
###############################################################################
###############################################################################
###                                                                         ###
###                    END OF DEFAULT INPUTS EDIT SECTION                   ###
###                                                                         ###
###############################################################################
###############################################################################


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


def export_sleap_to_dlc(slp_path: Path, output_dir: Path, scorer: str) -> None:
    slp_loader = cast(Callable[[str], Any], getattr(sio, "load_slp", None))
    if not callable(slp_loader):
        raise RuntimeError("sleap_io.load_slp is not available in this environment.")

    labels = slp_loader(str(slp_path))
    if not labels.skeletons:
        raise RuntimeError("No skeletons found in the SLEAP package.")

    labeled_data_dir = output_dir / "labeled-data"
    labeled_data_dir.mkdir(parents=True, exist_ok=True)

    csv_in_labeled_data = labeled_data_dir / f"CollectedData_{scorer}.csv"
    h5_in_labeled_data = labeled_data_dir / f"CollectedData_{scorer}.h5"

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
        image_path = labeled_data_dir / image_name

        image_array = image_array.squeeze()
        Image.fromarray(image_array).save(image_path)

        points = instance.numpy()
        row = [f"labeled-data/{image_name}"]
        for x, y in points:
            row.extend([format_coord(x), format_coord(y)])
        rows.append(row)
        exported_count += 1

    with csv_in_labeled_data.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)

    df = pd.read_csv(csv_in_labeled_data, header=[0, 1, 2], index_col=0)
    df.to_hdf(h5_in_labeled_data, key="df_with_missing", mode="w")

    print("Export complete.")
    print(f"Input SLP: {slp_path}")
    print(f"Output root: {output_dir}")
    print(f"Labeled-data folder: {labeled_data_dir}")
    print(f"Frames exported: {exported_count}")
    print(f"Frames skipped: {skipped_count}")
    print(f"CSV: {csv_in_labeled_data}")
    print(f"H5: {h5_in_labeled_data}")


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
        default=Path(DEFAULT_SLEAP_PACKAGE_NAME),
        help=(
            "Path to the input SLEAP package (.slp). "
            "Defaults to DEFAULT_SLEAP_PACKAGE_NAME in this script."
        ),
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
        default=DEFAULT_SCORER,
        help=(
            "Scorer name used in CollectedData_<scorer>.csv. "
            "Defaults to DEFAULT_SCORER in this script."
        ),
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
    )


if __name__ == "__main__":
    main()
