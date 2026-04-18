#!/usr/bin/env python3
"""
RecycleInferedToGroundTruth.py

Convert inferred labels to ground truth in a SLEAP package file (.pkg.slp),
while enforcing a maximum number of animals per frame.

For multi-animal pose estimation:
- User-labeled instances are prioritized
- Inferred instances fill up to the max_animals limit
- Frames exceeding max_animals keep only user labels + necessary inferred labels
- All resulting labels are marked as ground truth

Usage:
    python RecycleInferedToGroundTruth.py --input input.pkg.slp \
                                            --output output.pkg.slp \
                                            --num-animals 3
"""

import argparse
from pathlib import Path
from typing import Any, Callable, cast
import sleap_io as sio
from sleap_io.model.instance import Instance, PredictedInstance


def process_frame(labeled_frame, max_animals: int) -> None:
    """
    Process a single labeled frame to enforce max_animals constraint.
    
    Strategy:
    1. Separate user-labeled instances from inferred instances
    2. If total instances > max_animals:
       - Keep all user-labeled instances (they're authoritative)
       - Add inferred instances only if total doesn't exceed max_animals
    3. Mark all remaining instances as user instances (ground truth)
    
    Args:
        labeled_frame: The frame to process
        max_animals: Maximum number of animal instances allowed per frame
    """
    # Get all instances (both user and predicted).
    all_instances = list(labeled_frame.instances)
    user_instances = list(labeled_frame.user_instances)

    # Separate user-labeled from inferred.
    inferred_instances = [inst for inst in all_instances if inst not in user_instances]

    # User-requested policy:
    # - If frame has too many labels, keep only user labels.
    # - Otherwise, include inferred labels up to max_animals.
    if len(all_instances) > max_animals:
        selected_instances = user_instances[:max_animals]
    else:
        selected_instances = user_instances.copy()
        remaining_slots = max_animals - len(selected_instances)
        if remaining_slots > 0:
            selected_instances.extend(inferred_instances[:remaining_slots])

    # Convert selected predicted instances to regular Instance objects,
    # then write back through the writable `instances` field.
    final_instances = []
    for inst in selected_instances:
        if isinstance(inst, PredictedInstance):
            final_instances.append(
                Instance.from_numpy(
                    inst.numpy(),
                    skeleton=inst.skeleton,
                    track=inst.track,
                )
            )
        else:
            final_instances.append(inst)

    labeled_frame.instances = final_instances


def recycle_inferred_to_ground_truth(
    input_path: Path,
    output_path: Path,
    max_animals: int,
    verbose: bool = True
) -> None:
    """
    Load a SLEAP package, convert inferred labels to ground truth,
    and save as a new package.
    
    Args:
        input_path: Path to input .pkg.slp file
        output_path: Path to output .pkg.slp file
        max_animals: Maximum number of animal instances per frame
        verbose: Print progress information
    """
    # Load input labels
    if verbose:
        print(f"Loading {input_path}...")
    slp_loader = cast(Callable[[str], Any], getattr(sio, "load_slp", None))
    if not callable(slp_loader):
        raise RuntimeError("sleap_io.load_slp is not available in this environment.")

    labels = slp_loader(str(input_path))
    
    if verbose:
        print(f"Loaded {len(labels.labeled_frames)} frames")
        print(f"Max animals per frame: {max_animals}")
    
    # Process each frame
    frames_modified = 0
    frames_with_excess = 0
    
    for frame_idx, labeled_frame in enumerate(labels.labeled_frames):
        original_count = len(labeled_frame.instances)
        user_count = len(labeled_frame.user_instances)
        
        # Process the frame
        process_frame(labeled_frame, max_animals)
        
        final_count = len(labeled_frame.instances)
        
        # Track changes
        if final_count < original_count:
            frames_with_excess += 1
            if verbose and frames_with_excess <= 10:  # Show first 10 examples
                print(f"  Frame {frame_idx}: {original_count} → {final_count} instances "
                      f"(user={user_count}, excess removed)")
        
        if final_count != original_count or user_count != original_count:
            frames_modified += 1
    
    # Save output labels
    if verbose:
        print(f"\nSaving {output_path}...")
    labels.save(str(output_path))
    
    # Print summary
    if verbose:
        print("\n=== Summary ===")
        print(f"Total frames processed: {len(labels.labeled_frames)}")
        print(f"Frames modified: {frames_modified}")
        print(f"Frames with excess instances: {frames_with_excess}")
        print(f"All labels are now marked as ground truth.")
        print(f"Output saved: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input SLEAP package (.pkg.slp)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output SLEAP package (.pkg.slp)"
    )
    
    parser.add_argument(
        "--num-animals",
        type=int,
        default=3,
        help="Maximum number of animals per frame (default: 3)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Validate inputs
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if args.num_animals < 1:
        raise ValueError("num_animals must be >= 1")
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    recycle_inferred_to_ground_truth(
        input_path=args.input,
        output_path=args.output,
        max_animals=args.num_animals,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
