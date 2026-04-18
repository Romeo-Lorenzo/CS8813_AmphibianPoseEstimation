"""
Extract frames from video file and save as PNG images.
"""

import cv2
from pathlib import Path
import sys


def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 1 = extract all frames)
    """
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path.name}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frame interval: {frame_interval}")
    print(f"Output directory: {output_dir}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            filename = f"{video_path.stem}_{frame_count:06d}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames saved: {saved_count}")
    
    return True


if __name__ == "__main__":
    video_path = r"DLC/Shrimps-Lolo-2026-04-12/videos/three_big_shrimps_video_two.avi"
    output_dir = r"DLC/Shrimps-Lolo-2026-04-12/labeled-data/three_big_shrimps_video_two"
    
    # Extract every frame (interval=1)
    extract_frames(video_path, output_dir, frame_interval=1)
