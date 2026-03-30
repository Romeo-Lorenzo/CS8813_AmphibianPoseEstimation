import csv
from pathlib import Path

CSV_PATH = Path(r"c:\Users\romeo\Desktop\CS8813\DLC\Fly32-Lorenzo-2026-03-16\labeled-data\CollectedData_Lorenzo.csv")
SCORER = "Lorenzo"


def main() -> None:
    rows = list(csv.reader(CSV_PATH.open("r", newline="", encoding="utf-8")))
    if len(rows) < 4:
        raise RuntimeError("CSV does not contain expected DLC headers and data rows.")

    header_scorer, header_bodyparts, _ = rows[0], rows[1], rows[2]

    # Input currently stores triplets (x, y, likelihood) for each bodypart.
    bodyparts = [header_bodyparts[i] for i in range(1, len(header_bodyparts), 3)]

    new_header_scorer = ["scorer"] + [SCORER for _ in bodyparts for __ in (0, 1)]
    new_header_bodyparts = ["bodyparts"] + [bp for bp in bodyparts for _ in (0, 1)]
    new_header_coords = ["coords"] + ["x", "y"] * len(bodyparts)

    new_rows = [new_header_scorer, new_header_bodyparts, new_header_coords]
    for row in rows[3:]:
        new_row = [row[0]]
        for i in range(1, len(row), 3):
            new_row.extend([row[i], row[i + 1]])
        new_rows.append(new_row)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file, lineterminator="\n")
        writer.writerows(new_rows)

    print(f"Converted {len(new_rows) - 3} labeled rows.")
    print(f"Output columns: {len(new_rows[0])}")


if __name__ == "__main__":
    main()
