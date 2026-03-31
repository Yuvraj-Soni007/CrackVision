"""
Synthetic Data Generator
=========================
Generates realistic synthetic crack images for training and demonstration
when real datasets are unavailable.
"""

import cv2
import numpy as np
import os
from typing import Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, ensure_dir, project_root, logger


def generate_crack_image(size: Tuple[int, int] = (512, 512),
                         n_cracks: int = 3,
                         severity: str = "moderate") -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic concrete wall image with realistic cracks.

    Returns (image, ground_truth_mask).
    """
    h, w = size

    # ── Background: realistic concrete / plaster texture ──
    bg = np.random.randint(160, 200, (h, w), dtype=np.uint8)
    # Add subtle texture noise
    noise = np.random.normal(0, 8, (h, w)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Smooth slightly
    bg = cv2.GaussianBlur(bg, (5, 5), 1.0)

    # Add concrete-like texture patches
    for _ in range(20):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(30, 80)
        shade = np.random.randint(-15, 15)
        cv2.circle(bg, (cx, cy), radius, int(np.clip(180 + shade, 0, 255)), -1)
    bg = cv2.GaussianBlur(bg, (7, 7), 2.0)

    mask = np.zeros((h, w), dtype=np.uint8)
    image = bg.copy()

    # ── Severity-dependent parameters ──
    params = {
        "none":     {"n": 0, "thickness": (1, 1), "length": (0, 0), "branches": 0},
        "minor":    {"n": n_cracks, "thickness": (1, 2), "length": (60, 120), "branches": 1},
        "moderate": {"n": n_cracks, "thickness": (2, 4), "length": (100, 250), "branches": 2},
        "severe":   {"n": n_cracks, "thickness": (3, 7), "length": (150, 350), "branches": 4},
        "critical": {"n": n_cracks, "thickness": (5, 12), "length": (200, 450), "branches": 6},
    }
    p = params.get(severity, params["moderate"])

    for _ in range(p["n"]):
        # Random start point
        sx = np.random.randint(w // 6, 5 * w // 6)
        sy = np.random.randint(h // 6, 5 * h // 6)
        thickness = np.random.randint(*p["thickness"])
        length = np.random.randint(*p["length"])

        # Draw main crack as a random walk
        _draw_crack_walk(image, mask, sx, sy, length, thickness, h, w)

        # Branches
        for _ in range(np.random.randint(0, p["branches"] + 1)):
            bx = sx + np.random.randint(-40, 40)
            by = sy + np.random.randint(-40, 40)
            bl = length // np.random.randint(2, 5)
            bt = max(1, thickness - 1)
            _draw_crack_walk(image, mask, bx, by, bl, bt, h, w)

    # ── Post-processing for realism ──
    # Slight darkening around cracks
    dilated = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    image[dilated > 0] = np.clip(image[dilated > 0].astype(np.int16) - 20, 0, 255).astype(np.uint8)

    # Global noise
    final_noise = np.random.normal(0, 3, (h, w)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + final_noise, 0, 255).astype(np.uint8)

    return image, mask


def _draw_crack_walk(image, mask, sx, sy, length, thickness, h, w):
    """Draw a single crack via random walk with controlled direction."""
    x, y = sx, sy
    angle = np.random.uniform(0, 2 * np.pi)
    step = 3

    points = [(x, y)]
    for _ in range(length // step):
        angle += np.random.normal(0, 0.3)  # gentle turns
        x += int(step * np.cos(angle))
        y += int(step * np.sin(angle))
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        points.append((x, y))

    # Draw on image (dark crack)
    for i in range(len(points) - 1):
        t = max(1, thickness + np.random.randint(-1, 2))
        shade = np.random.randint(30, 80)
        cv2.line(image, points[i], points[i + 1], int(shade), t)
        cv2.line(mask, points[i], points[i + 1], 255, t)


def generate_dataset(output_dir: str = None, n_per_class: int = 20):
    """Generate a full synthetic dataset with per-severity folders."""
    if output_dir is None:
        output_dir = str(project_root() / "data" / "synthetic")

    severities = ["none", "minor", "moderate", "severe", "critical"]

    for sev in severities:
        img_dir = ensure_dir(os.path.join(output_dir, sev, "images"))
        mask_dir = ensure_dir(os.path.join(output_dir, sev, "masks"))

        for i in range(n_per_class):
            n_cracks = 0 if sev == "none" else np.random.randint(1, 5)
            img, mask = generate_crack_image(n_cracks=n_cracks, severity=sev)

            cv2.imwrite(os.path.join(img_dir, f"{sev}_{i:03d}.png"), img)
            cv2.imwrite(os.path.join(mask_dir, f"{sev}_{i:03d}_mask.png"), mask)

        logger.info(f"Generated {n_per_class} {sev} samples → {img_dir}")

    logger.info(f"Synthetic dataset created: {len(severities) * n_per_class} total images")
    return output_dir


if __name__ == "__main__":
    generate_dataset(n_per_class=25)
    print("✅ Synthetic dataset generated successfully!")
