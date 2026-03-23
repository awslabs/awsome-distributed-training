#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Generate a synthetic video dataset for V-JEPA 2 benchmarking.

Creates short random-content video files and a CSV file compatible
with V-JEPA 2's VideoDataset format.

Usage:
    python generate_synthetic_dataset.py \
        --output_dir /fsx/<your_username>/vjepa2/datasets/synthetic \
        --num_videos 5000 \
        --num_frames 32 \
        --width 256 \
        --height 256 \
        --fps 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def generate_video(output_path, num_frames=32, width=256, height=256, fps=4, seed=0):
    """Generate a synthetic video with random content using ffmpeg."""
    duration = num_frames / fps
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
        "-vf",
        f"drawtext=text='frame %{{n}}':x=10:y=10:fontsize=20:fontcolor=white",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error for {output_path}: {result.stderr}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic video dataset for V-JEPA 2"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_videos",
        type=int,
        default=5000,
        help="Number of synthetic videos to generate",
    )
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=174,
        help="Number of label classes (174 matches SSv2)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "synthetic_train_paths.csv"
    success = 0
    fail = 0

    with open(csv_path, "w") as csv_file:
        for i in range(args.num_videos):
            video_path = video_dir / f"video_{i:06d}.mp4"
            label = i % args.num_classes

            if video_path.exists():
                csv_file.write(f"{video_path} {label}\n")
                success += 1
            else:
                ok = generate_video(
                    str(video_path),
                    num_frames=args.num_frames,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    seed=i,
                )
                if ok:
                    csv_file.write(f"{video_path} {label}\n")
                    success += 1
                else:
                    fail += 1

            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{args.num_videos} videos...")

    print(f"\nDone: {success} videos generated, {fail} failed")
    print(f"CSV: {csv_path}")
    print(f"Videos: {video_dir}")


if __name__ == "__main__":
    main()
