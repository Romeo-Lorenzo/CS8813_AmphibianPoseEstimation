from pathlib import Path
import csv
import math

import pandas as pd
from PIL import Image
import sleap_io as sio


SCORER = "Lorenzo"
PROJECT_DIR = Path(r"c:\Users\romeo\Desktop\CS8813\DLC\Fly32-Lorenzo-2026-03-16")
SLP_PATH = Path(__file__).with_name("train.pkg.slp")
IMAGE_DIR = PROJECT_DIR / "labeled-data" / "fly32_train"
CSV_IN_IMAGE_DIR = IMAGE_DIR / f"CollectedData_{SCORER}.csv"
CSV_IN_LABEL_DIR = PROJECT_DIR / "labeled-data" / f"CollectedData_{SCORER}.csv"
H5_PATH = IMAGE_DIR / f"CollectedData_{SCORER}.h5"


def _format_coord(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.10f}".rstrip("0").rstrip(".")


def main() -> None:
    labels = sio.load_slp(str(SLP_PATH))
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    node_names = [node.name for node in labels.skeletons[0].nodes]
    header_scorer = ["scorer"] + [SCORER for _ in node_names for __ in (0, 1)]
    header_bodyparts = ["bodyparts"] + [name for name in node_names for _ in (0, 1)]
    header_coords = ["coords"] + ["x", "y"] * len(node_names)

    rows = [header_scorer, header_bodyparts, header_coords]

    for row_index, labeled_frame in enumerate(labels.labeled_frames):
        image_array = labeled_frame.image
        if image_array is None:
            continue

        image_name = f"img{row_index:04d}.png"
        image_path = IMAGE_DIR / image_name

        image_array = image_array.squeeze()
        Image.fromarray(image_array).save(image_path)

        if labeled_frame.user_instances:
            instance = labeled_frame.user_instances[0]
        elif labeled_frame.instances:
            instance = labeled_frame.instances[0]
        else:
            continue

        points = instance.numpy()
        row = [f"labeled-data/fly32_train/{image_name}"]
        for x, y in points:
            row.extend([_format_coord(x), _format_coord(y)])
        rows.append(row)

    with CSV_IN_IMAGE_DIR.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)

    with CSV_IN_LABEL_DIR.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)

    df = pd.read_csv(CSV_IN_IMAGE_DIR, header=[0, 1, 2], index_col=0)
    df.to_hdf(H5_PATH, key="df_with_missing", mode="w")

    print(f"Exported {len(rows) - 3} labeled frames.")
    print(f"Wrote CSV: {CSV_IN_IMAGE_DIR}")
    print(f"Wrote CSV: {CSV_IN_LABEL_DIR}")
    print(f"Wrote H5:  {H5_PATH}")


if __name__ == "__main__":
    main()