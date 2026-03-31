"""
Shape from Texture Module
===========================
Estimates surface orientation from texture gradient analysis.

CV Syllabus: Shape from X —
  shape-from-texture
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from ..utils import load_config, logger


class ShapeFromTexture:
    """Estimate surface tilt from texture gradient analysis."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        sa = cfg["shape_analysis"]
        self.gradient_method = sa["texture_gradient_method"]

    # ------------------------------------------------------------------
    # Texture gradient estimation
    # ------------------------------------------------------------------
    def texture_gradient(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the texture gradient (density change) across the image.

        Texture compression along a direction indicates surface tilt.
        """
        gray = self._gray(img).astype(np.float64)

        if self.gradient_method == "sobel":
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        else:
            gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

        # Local texture energy (windowed variance)
        energy = gx ** 2 + gy ** 2
        win = 15
        local_energy = cv2.blur(energy, (win, win))

        # Texture gradient = gradient of the energy field
        tex_gx = cv2.Sobel(local_energy, cv2.CV_64F, 1, 0, ksize=5)
        tex_gy = cv2.Sobel(local_energy, cv2.CV_64F, 0, 1, ksize=5)

        logger.debug("Texture gradient computed")
        return tex_gx, tex_gy

    # ------------------------------------------------------------------
    # Surface tilt estimation
    # ------------------------------------------------------------------
    def estimate_tilt(self, img: np.ndarray, block_size: int = 64) -> np.ndarray:
        """Estimate per-block surface tilt angle from texture gradient.

        Returns tilt_map of shape (H//block, W//block) in degrees.
        """
        tex_gx, tex_gy = self.texture_gradient(img)
        h, w = tex_gx.shape
        n_rows = h // block_size
        n_cols = w // block_size

        tilt_map = np.zeros((n_rows, n_cols), dtype=np.float64)
        for r in range(n_rows):
            for c in range(n_cols):
                y0, x0 = r * block_size, c * block_size
                block_gx = tex_gx[y0:y0 + block_size, x0:x0 + block_size]
                block_gy = tex_gy[y0:y0 + block_size, x0:x0 + block_size]

                mean_gx = np.mean(block_gx)
                mean_gy = np.mean(block_gy)
                tilt = np.arctan2(np.sqrt(mean_gx ** 2 + mean_gy ** 2), 1.0)
                tilt_map[r, c] = np.degrees(tilt)

        logger.info(f"Tilt map: {n_rows}×{n_cols} blocks, mean tilt={tilt_map.mean():.1f}°")
        return tilt_map

    # ------------------------------------------------------------------
    # Texture segmentation for crack isolation
    # ------------------------------------------------------------------
    def texture_segmentation(self, img: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """Segment the image based on local texture features.

        Uses local variance + gradient energy as feature, then K-means.
        """
        gray = self._gray(img).astype(np.float64)
        h, w = gray.shape

        # Feature 1: local variance
        win = 11
        mean = cv2.blur(gray, (win, win))
        sq_mean = cv2.blur(gray ** 2, (win, win))
        variance = sq_mean - mean ** 2
        variance = np.clip(variance, 0, None)

        # Feature 2: gradient energy
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy = cv2.blur(gx ** 2 + gy ** 2, (win, win))

        # Stack and normalize
        feat = np.stack([variance.flatten(), energy.flatten()], axis=1).astype(np.float32)
        feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(feat, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        segmented = labels.reshape(h, w).astype(np.uint8)

        # Map to 0-255 range
        segmented = (segmented * (255 // max(n_clusters - 1, 1))).astype(np.uint8)
        logger.info(f"Texture segmentation: {n_clusters} clusters")
        return segmented

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
