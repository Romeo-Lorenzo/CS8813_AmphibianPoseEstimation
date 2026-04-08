import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast


###############################################################################
###############################################################################
###                                                                         ###
###                              HOW TO RUN                                 ###
###                                                                         ###
###  This script compares two inference methods (SLEAP and DLC) against     ###
###  the same ground-truth DLC CSV and exports comparative metrics.          ###
###                                                                         ###
###  1) Open a terminal in this script's folder.                            ###
###  2) Run with defaults from the section below:                            ###
###     python SLEAPeval.py                                                  ###
###  3) Optional override example:                                           ###
###     python SLEAPeval.py ... --out-dir SLEAP_Exported_to_DLC/labeled-data###
###  4) Optional full override example:                                      ###
###     python SLEAPeval.py --gt-csv <ground_truth.csv>                      ###
###         --sleap-file <sleap_inference.slp|csv> --dlc-csv <dlc_inference.csv> ###
###                                                                         ###
###  Input formats:                                                          ###
###  - Ground truth: DLC CSV (3-row scorer/bodyparts/coords header)         ###
###  - SLEAP inference: .slp or DLC CSV                                      ###
###  - DLC inference: DLC CSV                                                ###
###                                                                         ###
###  Output files are written to the labeled-data folder:                   ###
###  - method_metrics_summary.csv                                            ###
###  - method_metrics_comparison.json                                        ###
###  - method_metrics_framewise.csv                                          ###
###                                                                         ###
###############################################################################
###############################################################################


###############################################################################
###############################################################################
###                                                                         ###
###                    CHANGE YOUR DEFAULT INPUTS HERE                      ###
###                                                                         ###
###  Edit these defaults so you can run directly in VS Code (Run button)   ###
###  without passing terminal arguments. Paths are relative to this script. ###
###                                                                         ###
###  Example values:                                                        ###
###  - "SLEAP_Exported_to_DLC/labeled-data/CollectedData_User.csv"         ###
###  - "test.pkg.slp.260330_105705.predictions.slp"                         ###
###  - "SLEAP_Exported_to_DLC/labeled-data/dlc_inference.csv"              ###
###  - "SLEAP_Exported_to_DLC/labeled-data"                                ###
###                                                                         ###
###  If you run without CLI flags, these defaults will be used:            ###
###  --gt-csv, --sleap-file, --dlc-csv, --out-dir                           ###
###                                                                         ###
###############################################################################
###############################################################################
DEFAULT_DLC_GT_CSV = "SLEAP_Exported_to_DLC/labeled-data/CollectedData_User.csv"
DEFAULT_SLEAP_FILE = "test.pkg.slp_test.pkg.slp.260330_105705.predictions.slp"
DEFAULT_DLC_CSV = "labeled_data_videoDLC_mobnet_100_Fly32Mar16shuffle1_100000.csv"
DEFAULT_OUT_DIR = "SLEAP_Exported_to_DLC/comparison_metrics"
###############################################################################
###############################################################################
###                                                                         ###
###                    END OF DEFAULT INPUTS EDIT SECTION                   ###
###                                                                         ###
###############################################################################
###############################################################################


FrameKey = str
Bodypart = str
XY = Tuple[float, float]
PoseTable = Dict[FrameKey, Dict[Bodypart, XY]]


@dataclass
class EvalResult:
	method_name: str
	comparable_frames: int
	comparable_points: int
	mean_pixel_error: float
	median_pixel_error: float
	p95_pixel_error: float


def parse_args() -> argparse.Namespace:
	script_dir = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate SLEAP and DLC inference CSV files against a DLC-formatted "
			"ground truth CSV and export comparative metrics."
		)
	)
	parser.add_argument(
		"--gt-csv",
		type=Path,
		default=script_dir / DEFAULT_DLC_GT_CSV,
		help="Path to DLC-formatted ground-truth CSV.",
	)
	parser.add_argument(
		"--sleap-file",
		"--sleap-csv",
		type=Path,
		dest="sleap_file",
		default=script_dir / DEFAULT_SLEAP_FILE,
		help="Path to SLEAP inference file (.slp or DLC-formatted CSV).",
	)
	parser.add_argument(
		"--dlc-csv",
		type=Path,
		default=script_dir / DEFAULT_DLC_CSV,
		help="Path to DLC-formatted DLC inference CSV.",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=script_dir / DEFAULT_OUT_DIR,
		help="Output folder for exported metric files (default: labeled-data folder).",
	)
	return parser.parse_args()


def _parse_float(value: str) -> float:
	value = value.strip()
	if value == "":
		return math.nan
	return float(value)


def _normalize_frame_key(raw_key: str) -> str:
	return Path(raw_key.replace("\\", "/")).name


def _frame_aliases_from_csv_key(raw_key: str) -> List[str]:
	raw = raw_key.strip()
	aliases: List[str] = []

	# Keep normalized raw key.
	aliases.append(_normalize_frame_key(raw))

	# DLC analysis CSVs often use integer frame indices (0, 1, 2, ...).
	if raw.isdigit():
		aliases.append(f"img{int(raw):04d}.png")

	# Keep unique aliases in order.
	seen = set()
	uniq: List[str] = []
	for alias in aliases:
		if alias not in seen:
			seen.add(alias)
			uniq.append(alias)
	return uniq


def _normalize_bodypart_key(raw_key: str) -> str:
	return raw_key.strip().lower()


def read_dlc_csv(csv_path: Path) -> PoseTable:
	with csv_path.open("r", newline="", encoding="utf-8") as handle:
		rows = list(csv.reader(handle))

	if len(rows) < 4:
		raise RuntimeError(f"File has fewer than 4 rows and is not a DLC CSV: {csv_path}")

	bodyparts_header = rows[1]
	coords_header = rows[2]
	data_rows = rows[3:]

	if not bodyparts_header or bodyparts_header[0].lower() != "bodyparts":
		raise RuntimeError(f"Missing DLC bodyparts header in: {csv_path}")
	if not coords_header or coords_header[0].lower() != "coords":
		raise RuntimeError(f"Missing DLC coords header in: {csv_path}")

	# Build mapping from CSV column pairs to bodypart names.
	col_pairs: List[Tuple[int, int, Bodypart]] = []
	col = 1
	while col + 1 < len(bodyparts_header):
		bodypart = _normalize_bodypart_key(bodyparts_header[col])
		cx = coords_header[col].strip().lower()
		cy = coords_header[col + 1].strip().lower()
		if cx == "x" and cy == "y" and bodypart != "":
			col_pairs.append((col, col + 1, bodypart))
		col += 2

	table: PoseTable = {}
	for row in data_rows:
		if not row:
			continue
		frame_raw = row[0].strip()
		if frame_raw == "":
			continue
		frame_points: Dict[Bodypart, XY] = {}
		for x_col, y_col, bp in col_pairs:
			if x_col >= len(row) or y_col >= len(row):
				continue
			x_val = _parse_float(row[x_col])
			y_val = _parse_float(row[y_col])
			frame_points[bp] = (x_val, y_val)
		for frame_key in _frame_aliases_from_csv_key(frame_raw):
			table[frame_key] = frame_points
	return table


def pick_instance(labeled_frame: Any) -> Any:
	if getattr(labeled_frame, "user_instances", None):
		return labeled_frame.user_instances[0]
	if getattr(labeled_frame, "instances", None):
		return labeled_frame.instances[0]
	return None


def _extract_image_name_from_labeled_frame(labeled_frame: Any) -> str:
	image_obj = getattr(labeled_frame, "image", None)
	if image_obj is not None:
		for attr in ("filename", "path", "name"):
			value = getattr(image_obj, attr, None)
			if isinstance(value, str) and value.strip() != "":
				return _normalize_frame_key(value)
		backend = getattr(image_obj, "backend", None)
		if backend is not None:
			for attr in ("filename", "path", "name"):
				value = getattr(backend, attr, None)
				if isinstance(value, str) and value.strip() != "":
					return _normalize_frame_key(value)

	frame_idx = getattr(labeled_frame, "frame_idx", None)
	if isinstance(frame_idx, int):
		return f"img{frame_idx:04d}.png"

	video_obj = getattr(labeled_frame, "video", None)
	if video_obj is not None:
		for attr in ("filename", "path", "name"):
			value = getattr(video_obj, attr, None)
			if isinstance(value, str) and value.strip() != "":
				return _normalize_frame_key(value)

	raise RuntimeError("Unable to derive frame/image filename from SLEAP labeled frame.")


def _extract_frame_aliases(labeled_frame: Any, order_idx: int) -> List[str]:
	aliases: List[str] = []

	# Alias 1: sequential export-style naming (matches export_sleap_pkg_to_dlc.py rows).
	aliases.append(f"img{order_idx:04d}.png")

	# Alias 2: frame_idx-based naming if available.
	frame_idx = getattr(labeled_frame, "frame_idx", None)
	if isinstance(frame_idx, int):
		aliases.append(f"img{frame_idx:04d}.png")

	# Alias 3: any filename/path discoverable in SLEAP metadata.
	try:
		aliases.append(_extract_image_name_from_labeled_frame(labeled_frame))
	except RuntimeError:
		pass

	# Keep unique aliases while preserving order.
	seen = set()
	unique_aliases: List[str] = []
	for alias in aliases:
		key = _normalize_frame_key(alias)
		if key not in seen:
			seen.add(key)
			unique_aliases.append(key)
	return unique_aliases


def read_sleap_slp(slp_path: Path) -> PoseTable:
	try:
		import sleap_io as sio
	except ImportError as exc:
		raise RuntimeError(
			"Reading .slp requires sleap_io. Install with: pip install sleap-io"
		) from exc

	slp_loader = cast(Any, getattr(sio, "load_slp", None))
	if not callable(slp_loader):
		raise RuntimeError("sleap_io.load_slp is unavailable in this environment.")

	labels = cast(Any, slp_loader(str(slp_path)))
	if not getattr(labels, "skeletons", None):
		raise RuntimeError(f"No skeleton found in SLEAP file: {slp_path}")

	node_names = [_normalize_bodypart_key(node.name) for node in labels.skeletons[0].nodes]
	table: PoseTable = {}

	for row_idx, labeled_frame in enumerate(labels.labeled_frames):
		instance = pick_instance(labeled_frame)
		if instance is None:
			continue
		points = instance.numpy()

		if len(points) != len(node_names):
			continue

		frame_points: Dict[Bodypart, XY] = {}
		for bp, xy in zip(node_names, points):
			x_val = float(xy[0])
			y_val = float(xy[1])
			frame_points[bp] = (x_val, y_val)

		for frame_alias in _extract_frame_aliases(labeled_frame, row_idx):
			table[frame_alias] = frame_points

	if not table:
		raise RuntimeError(f"No usable labeled frames found in SLEAP file: {slp_path}")

	return table


def read_prediction_table(pred_path: Path) -> PoseTable:
	ext = pred_path.suffix.lower()
	if ext == ".csv":
		return read_dlc_csv(pred_path)
	if ext == ".slp":
		return read_sleap_slp(pred_path)
	raise RuntimeError(f"Unsupported SLEAP prediction file type: {pred_path}")


def _percentile(sorted_vals: List[float], p: float) -> float:
	if not sorted_vals:
		return math.nan
	if len(sorted_vals) == 1:
		return sorted_vals[0]
	rank = (len(sorted_vals) - 1) * p
	lo = int(math.floor(rank))
	hi = int(math.ceil(rank))
	if lo == hi:
		return sorted_vals[lo]
	weight = rank - lo
	return sorted_vals[lo] * (1.0 - weight) + sorted_vals[hi] * weight


def _iter_common_distances(gt_table: PoseTable, pred_table: PoseTable) -> Iterable[Tuple[FrameKey, Bodypart, float]]:
	for frame in sorted(set(gt_table) & set(pred_table)):
		gt_points = gt_table[frame]
		pred_points = pred_table[frame]
		for bp in sorted(set(gt_points) & set(pred_points)):
			gt_x, gt_y = gt_points[bp]
			pr_x, pr_y = pred_points[bp]
			if not (
				math.isfinite(gt_x)
				and math.isfinite(gt_y)
				and math.isfinite(pr_x)
				and math.isfinite(pr_y)
			):
				continue
			dx = gt_x - pr_x
			dy = gt_y - pr_y
			yield frame, bp, math.sqrt(dx * dx + dy * dy)


def evaluate_method(method_name: str, gt_table: PoseTable, pred_table: PoseTable) -> Tuple[EvalResult, Dict[FrameKey, List[float]]]:
	by_frame: Dict[FrameKey, List[float]] = {}
	all_dists: List[float] = []

	for frame, _bp, dist in _iter_common_distances(gt_table, pred_table):
		by_frame.setdefault(frame, []).append(dist)
		all_dists.append(dist)

	if not all_dists:
		gt_frames = set(gt_table)
		pred_frames = set(pred_table)
		common_frames = sorted(gt_frames & pred_frames)
		common_bodyparts = set()
		if common_frames:
			f0 = common_frames[0]
			common_bodyparts = set(gt_table[f0]) & set(pred_table[f0])
		raise RuntimeError(
			f"No comparable points for {method_name}. "
			f"GT frames={len(gt_frames)}, Pred frames={len(pred_frames)}, "
			f"Common frames={len(common_frames)}, Common bodyparts in first common frame={len(common_bodyparts)}."
		)

	sorted_dists = sorted(all_dists)
	mean_dist = sum(all_dists) / len(all_dists)
	median_dist = _percentile(sorted_dists, 0.5)
	p95_dist = _percentile(sorted_dists, 0.95)

	result = EvalResult(
		method_name=method_name,
		comparable_frames=len(by_frame),
		comparable_points=len(all_dists),
		mean_pixel_error=mean_dist,
		median_pixel_error=median_dist,
		p95_pixel_error=p95_dist,
	)
	return result, by_frame


def write_summary_csv(out_path: Path, sleap_result: EvalResult, dlc_result: EvalResult) -> None:
	headers = [
		"method",
		"comparable_frames",
		"comparable_points",
		"mean_pixel_error",
		"median_pixel_error",
		"p95_pixel_error",
	]
	with out_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle, lineterminator="\n")
		writer.writerow(headers)
		for r in (sleap_result, dlc_result):
			writer.writerow(
				[
					r.method_name,
					r.comparable_frames,
					r.comparable_points,
					f"{r.mean_pixel_error:.6f}",
					f"{r.median_pixel_error:.6f}",
					f"{r.p95_pixel_error:.6f}",
				]
			)


def write_comparison_json(out_path: Path, sleap_result: EvalResult, dlc_result: EvalResult) -> None:
	delta_mean = dlc_result.mean_pixel_error - sleap_result.mean_pixel_error
	delta_median = dlc_result.median_pixel_error - sleap_result.median_pixel_error
	delta_p95 = dlc_result.p95_pixel_error - sleap_result.p95_pixel_error

	payload = {
		"lower_error_is_better": True,
		"metrics": {
			"SLEAP": sleap_result.__dict__,
			"DLC": dlc_result.__dict__,
		},
		"comparison_dlc_minus_sleap": {
			"mean_pixel_error": delta_mean,
			"median_pixel_error": delta_median,
			"p95_pixel_error": delta_p95,
		},
		"winner_by_mean_error": "SLEAP" if sleap_result.mean_pixel_error < dlc_result.mean_pixel_error else "DLC",
	}

	with out_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2)


def write_framewise_csv(out_path: Path, sleap_by_frame: Dict[FrameKey, List[float]], dlc_by_frame: Dict[FrameKey, List[float]]) -> None:
	frames = sorted(set(sleap_by_frame) | set(dlc_by_frame))
	with out_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle, lineterminator="\n")
		writer.writerow(
			[
				"frame",
				"sleap_mean_pixel_error",
				"dlc_mean_pixel_error",
				"dlc_minus_sleap",
			]
		)
		for frame in frames:
			sleap_vals = sleap_by_frame.get(frame, [])
			dlc_vals = dlc_by_frame.get(frame, [])
			sleap_mean = (sum(sleap_vals) / len(sleap_vals)) if sleap_vals else math.nan
			dlc_mean = (sum(dlc_vals) / len(dlc_vals)) if dlc_vals else math.nan
			delta = dlc_mean - sleap_mean if (math.isfinite(dlc_mean) and math.isfinite(sleap_mean)) else math.nan
			writer.writerow(
				[
					frame,
					"" if not math.isfinite(sleap_mean) else f"{sleap_mean:.6f}",
					"" if not math.isfinite(dlc_mean) else f"{dlc_mean:.6f}",
					"" if not math.isfinite(delta) else f"{delta:.6f}",
				]
			)


def main() -> None:
	args = parse_args()

	gt_csv = args.gt_csv.resolve()
	sleap_file = args.sleap_file.resolve()
	dlc_csv = args.dlc_csv.resolve()
	out_dir = args.out_dir.resolve()

	for p in (gt_csv, sleap_file, dlc_csv):
		if not p.exists():
			raise FileNotFoundError(f"Input file not found: {p}")

	out_dir.mkdir(parents=True, exist_ok=True)

	gt_table = read_dlc_csv(gt_csv)
	sleap_table = read_prediction_table(sleap_file)
	dlc_table = read_dlc_csv(dlc_csv)

	sleap_result, sleap_by_frame = evaluate_method("SLEAP", gt_table, sleap_table)
	dlc_result, dlc_by_frame = evaluate_method("DLC", gt_table, dlc_table)

	summary_csv = out_dir / "method_metrics_summary.csv"
	comparison_json = out_dir / "method_metrics_comparison.json"
	framewise_csv = out_dir / "method_metrics_framewise.csv"

	write_summary_csv(summary_csv, sleap_result, dlc_result)
	write_comparison_json(comparison_json, sleap_result, dlc_result)
	write_framewise_csv(framewise_csv, sleap_by_frame, dlc_by_frame)

	print("Evaluation complete.")
	print(f"Ground truth CSV: {gt_csv}")
	print(f"SLEAP inference file: {sleap_file}")
	print(f"DLC inference CSV: {dlc_csv}")
	print(f"Summary CSV: {summary_csv}")
	print(f"Comparison JSON: {comparison_json}")
	print(f"Framewise CSV: {framewise_csv}")
	print(f"SLEAP mean error: {sleap_result.mean_pixel_error:.4f} px")
	print(f"DLC mean error: {dlc_result.mean_pixel_error:.4f} px")


if __name__ == "__main__":
	main()