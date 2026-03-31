"""
Morphological Processing Module
=================================
Applies morphological operations to clean segmentation masks and
extract crack skeleton / topology.

CV Syllabus: Low-Level Processing — filtering, convolution
"""

import cv2
import numpy as np
from typing import Optional

from ..utils import load_config, logger


class MorphologicalProcessor:
    """Morphological operations for crack mask refinement."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        seg = cfg["segmentation"]
        self.kernel_size = seg["morph_kernel_size"]
        self.iterations = seg["morph_iterations"]

    def _kernel(self, shape: int = cv2.MORPH_ELLIPSE, size: Optional[int] = None) -> np.ndarray:
        s = size or self.kernel_size
        return cv2.getStructuringElement(shape, (s, s))

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------
    def erode(self, mask: np.ndarray, iterations: Optional[int] = None) -> np.ndarray:
        return cv2.erode(mask, self._kernel(), iterations=iterations or self.iterations)

    def dilate(self, mask: np.ndarray, iterations: Optional[int] = None) -> np.ndarray:
        return cv2.dilate(mask, self._kernel(), iterations=iterations or self.iterations)

    def opening(self, mask: np.ndarray) -> np.ndarray:
        """Remove small noise (erode then dilate)."""
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel(), iterations=self.iterations)

    def closing(self, mask: np.ndarray) -> np.ndarray:
        """Fill small gaps (dilate then erode)."""
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel(), iterations=self.iterations)

    def gradient(self, mask: np.ndarray) -> np.ndarray:
        """Morphological gradient (dilation – erosion) for boundaries."""
        return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, self._kernel())

    def top_hat(self, img: np.ndarray) -> np.ndarray:
        """Top-hat transform: highlights bright details on dark background."""
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, self._kernel(size=15))

    def black_hat(self, img: np.ndarray) -> np.ndarray:
        """Black-hat transform: highlights dark details (cracks) on lighter background."""
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, self._kernel(size=15))

    # ------------------------------------------------------------------
    # Skeletonisation
    # ------------------------------------------------------------------
    def skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Extract crack skeleton via iterative morphological thinning."""
        skel = np.zeros_like(mask)
        element = self._kernel(cv2.MORPH_CROSS, 3)
        temp = mask.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            sub = cv2.subtract(temp, opened)
            skel = cv2.bitwise_or(skel, sub)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        logger.debug("Skeleton extracted")
        return skel

    # ------------------------------------------------------------------
    # Crack topology analysis
    # ------------------------------------------------------------------
    def analyse_skeleton(self, skeleton: np.ndarray) -> dict:
        """Compute topological properties of the crack skeleton."""
        total_pixels = int(np.sum(skeleton > 0))

        # Find endpoints and branch points via 3×3 neighbourhood analysis
        endpoints = 0
        branches = 0
        h, w = skeleton.shape
        skel_bin = (skeleton > 0).astype(np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skel_bin[y, x] == 0:
                    continue
                n = int(np.sum(skel_bin[y - 1:y + 2, x - 1:x + 2])) - 1
                if n == 1:
                    endpoints += 1
                elif n >= 3:
                    branches += 1

        # Find contours for length estimation
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_length = sum(cv2.arcLength(c, False) for c in contours)

        result = {
            "skeleton_pixels": total_pixels,
            "estimated_length_px": round(total_length, 2),
            "endpoints": endpoints,
            "branch_points": branches,
            "num_segments": len(contours),
        }
        logger.info(f"Skeleton analysis: {result}")
        return result

    # ------------------------------------------------------------------
    # Full cleanup pipeline
    # ------------------------------------------------------------------
    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Default cleanup: close gaps → remove noise → thin."""
        cleaned = self.closing(mask)
        cleaned = self.opening(cleaned)
        logger.debug("Mask cleaned (close → open)")
        return cleaned
