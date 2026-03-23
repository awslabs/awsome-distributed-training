#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Generate a synthetic image dataset for V-JEPA 2.1 image co-training.

V-JEPA 2.1 co-trains on images and videos simultaneously. Image ranks use
VideoDataset with dataset_fpcs=[1] and load .jpg/.png files via
torchvision.io.read_image(). This script generates random-content JPEG
images and a CSV file compatible with that format.

Usage:
    python generate_synthetic_images.py \
        --output_dir /fsx/<your_username>/vjepa2.1/datasets/synthetic_images \
        --num_images 50000 \
        --width 256 \
        --height 256

Note: We recommend generating at least 50,000 images for reliable benchmark
results. With fewer images (e.g. 5,000), the data loader must frequently
re-initialize workers between epochs, which inflates iteration times and
masks the true GPU throughput.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def generate_image(output_path, width=256, height=256, seed=0):
    """Generate a synthetic JPEG image with random content."""
    try:
        from PIL import Image
    except ImportError:
        # Fallback: use raw numpy + simple PPM -> JPEG via subprocess
        import subprocess

        rng = np.random.RandomState(seed)
        pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        ppm_path = output_path.replace(".jpg", ".ppm")
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode())
            f.write(pixels.tobytes())
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", ppm_path, output_path],
            capture_output=True,
            text=True,
        )
        if os.path.exists(ppm_path):
            os.remove(ppm_path)
        return result.returncode == 0

    rng = np.random.RandomState(seed)
    pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, "RGB")
    img.save(output_path, "JPEG", quality=85)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic image dataset for V-JEPA 2.1"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--num_images",
        type=int,
        default=50000,
        help="Number of synthetic images to generate (50k recommended for benchmarks)",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of label classes (1000 matches ImageNet)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "synthetic_image_paths.csv"
    success = 0
    fail = 0

    with open(csv_path, "w") as csv_file:
        for i in range(args.num_images):
            image_path = image_dir / f"img_{i:06d}.jpg"
            label = i % args.num_classes

            if image_path.exists():
                csv_file.write(f"{image_path} {label}\n")
                success += 1
            else:
                ok = generate_image(
                    str(image_path),
                    width=args.width,
                    height=args.height,
                    seed=i,
                )
                if ok:
                    csv_file.write(f"{image_path} {label}\n")
                    success += 1
                else:
                    fail += 1

            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{args.num_images} images...")

    print(f"\nDone: {success} images generated, {fail} failed")
    print(f"CSV: {csv_path}")
    print(f"Images: {image_dir}")


if __name__ == "__main__":
    main()
