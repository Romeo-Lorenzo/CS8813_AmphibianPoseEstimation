import argparse
import re
from pathlib import Path
from typing import Any, cast


###############################################################################
###############################################################################
###                                                                         ###
###                              HOW TO RUN                                 ###
###                                                                         ###
###  1) Open a terminal in this script's folder.                            ###
###  2) Run with defaults (from the section below):                         ###
###     python make_mp4_from_images.py                                      ###
###  3) Optional override example:                                          ###
###     python make_mp4_from_images.py --images-dir SLEAP_Exported_to_DLC/  ###
###         labeled-data --output-video SLEAP_Exported_to_DLC/labeled-data/ ###
###         labeled_data_video.mp4 --fps 20                                 ###
###                                                                         ###
###  This script creates an MP4 from image frames named like img0000.png.   ###
###                                                                         ###
###############################################################################
###############################################################################


###############################################################################
###############################################################################
###                                                                         ###
###                    CHANGE YOUR DEFAULT INPUTS HERE                      ###
###                                                                         ###
###  Paths are relative to this script's folder.                            ###
###                                                                         ###
###  If you run without CLI flags, these defaults will be used:             ###
###  --images-dir, --output-video, --fps                                    ###
###                                                                         ###
###############################################################################
###############################################################################
DEFAULT_IMAGES_DIR = "SLEAP_Exported_to_DLC/labeled-data"
DEFAULT_OUTPUT_VIDEO = "SLEAP_Exported_to_DLC/labeled-data/labeled_data_video.mp4"
DEFAULT_FPS = 15
###############################################################################
###############################################################################
###                                                                         ###
###                    END OF DEFAULT INPUTS EDIT SECTION                   ###
###                                                                         ###
###############################################################################
###############################################################################


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create an MP4 video from exported frame images."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=script_dir / DEFAULT_IMAGES_DIR,
        help="Folder containing frame images (default: labeled-data folder).",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=script_dir / DEFAULT_OUTPUT_VIDEO,
        help="Path to output MP4 file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Frames per second for the output video.",
    )
    return parser.parse_args()


def _natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_frames(images_dir: Path):
    candidates = [p for p in images_dir.glob("img*.png") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No img*.png files found in: {images_dir}")
    return sorted(candidates, key=_natural_sort_key)


def write_video_with_cv2(frame_paths, output_video: Path, fps: float) -> None:
    import cv2

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first.shape[:2]
    # Some cv2 stubs miss VideoWriter_fourcc, so resolve it dynamically.
    fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
    if not callable(fourcc_func):
        raise RuntimeError("cv2.VideoWriter_fourcc is unavailable in this OpenCV build.")
    fourcc_raw = cast(Any, fourcc_func(*"mp4v"))
    fourcc = int(fourcc_raw)
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_video}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def write_video_with_imageio(frame_paths, output_video: Path, fps: float) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(str(output_video), fps=fps, codec="libx264", macro_block_size=None) as raw_writer:
        writer = cast(Any, raw_writer)
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def main() -> None:
    args = parse_args()

    images_dir = args.images_dir.resolve()
    output_video = args.output_video.resolve()

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(images_dir)

    writer_used = ""
    try:
        write_video_with_cv2(frames, output_video, args.fps)
        writer_used = "cv2"
    except Exception:
        write_video_with_imageio(frames, output_video, args.fps)
        writer_used = "imageio"

    print("Video export complete.")
    print(f"Frames folder: {images_dir}")
    print(f"Frames used: {len(frames)}")
    print(f"FPS: {args.fps}")
    print(f"Writer used: {writer_used}")
    print(f"Output video: {output_video}")


if __name__ == "__main__":
    main()
