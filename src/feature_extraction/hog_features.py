"""
HOG Feature Extraction Module
===============================
Implements Histogram of Oriented Gradients for crack region classification.

CV Syllabus: Feature Extraction —
  HOG (Histogram of Oriented Gradients)
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from ..utils import load_config, logger


class HOGExtractor:
    """HOG descriptor computation for image patches / full images."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        feat = cfg["feature_extraction"]
        self.orientations = feat["hog_orientations"]
        self.pix_per_cell = tuple(feat["hog_pixels_per_cell"])
        self.cells_per_block = tuple(feat["hog_cells_per_block"])
        self.block_norm = feat["hog_block_norm"]

    # ------------------------------------------------------------------
    # Full HOG descriptor via OpenCV
    # ------------------------------------------------------------------
    def compute(self, img: np.ndarray, cell_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Compute the HOG descriptor for an image.

        The image is resized to a multiple of cell_size for clean tiling.
        """
        gray = self._gray(img)
        ppc = cell_size or self.pix_per_cell
        cpb = self.cells_per_block

        # Ensure dimensions are multiples of cell * block
        bh = ppc[1] * cpb[1]
        bw = ppc[0] * cpb[0]
        h = (gray.shape[0] // bh) * bh
        w = (gray.shape[1] // bw) * bw
        gray = cv2.resize(gray, (w, h))

        win_size = (w, h)
        block_size = (bw, bh)
        block_stride = (ppc[0], ppc[1])
        cell_size_cv = (ppc[0], ppc[1])

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                                cell_size_cv, self.orientations)
        descriptor = hog.compute(gray)
        descriptor = descriptor.flatten()
        logger.debug(f"HOG descriptor length: {len(descriptor)}")
        return descriptor

    # ------------------------------------------------------------------
    # Custom HOG (pure NumPy — for educational clarity)
    # ------------------------------------------------------------------
    def compute_manual(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Manual HOG computation step-by-step.

        Returns (descriptor, hog_image) for visualisation.
        Educational implementation matching the syllabus.
        """
        gray = self._gray(img).astype(np.float64)
        rows, cols = gray.shape

        # 1. Gradient computation (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        direction = np.arctan2(gy, gx) * (180.0 / np.pi) % 180  # unsigned 0-180

        # 2. Cell histograms
        ppc_y, ppc_x = self.pix_per_cell
        n_cells_y = rows // ppc_y
        n_cells_x = cols // ppc_x
        n_bins = self.orientations
        bin_width = 180.0 / n_bins

        cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float64)

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0 = cy * ppc_y
                x0 = cx * ppc_x
                mag_cell = magnitude[y0:y0 + ppc_y, x0:x0 + ppc_x]
                dir_cell = direction[y0:y0 + ppc_y, x0:x0 + ppc_x]

                for b in range(n_bins):
                    lo = b * bin_width
                    hi = lo + bin_width
                    mask = ((dir_cell >= lo) & (dir_cell < hi))
                    cell_hists[cy, cx, b] = np.sum(mag_cell[mask])

        # 3. Block normalisation
        cpb_y, cpb_x = self.cells_per_block
        descriptors = []
        for by in range(n_cells_y - cpb_y + 1):
            for bx in range(n_cells_x - cpb_x + 1):
                block = cell_hists[by:by + cpb_y, bx:bx + cpb_x, :].flatten()
                norm = np.sqrt(np.sum(block ** 2) + 1e-6)
                block = block / norm
                # Clip for L2-Hys
                block = np.clip(block, 0, 0.2)
                norm2 = np.sqrt(np.sum(block ** 2) + 1e-6)
                block = block / norm2
                descriptors.append(block)

        descriptor = np.concatenate(descriptors) if descriptors else np.array([])

        # 4. HOG visualisation image
        hog_image = self._draw_hog(cell_hists, n_cells_y, n_cells_x, ppc_y, ppc_x, n_bins)

        logger.debug(f"Manual HOG: descriptor length={len(descriptor)}")
        return descriptor, hog_image

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_hog(cell_hists, n_cells_y, n_cells_x, ppc_y, ppc_x, n_bins) -> np.ndarray:
        """Render HOG cell histograms as a star pattern image."""
        vis = np.zeros((n_cells_y * ppc_y, n_cells_x * ppc_x), dtype=np.float64)
        bin_width = 180.0 / n_bins
        max_mag = cell_hists.max() + 1e-8
        radius = min(ppc_y, ppc_x) // 2 - 1

        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                center_y = cy * ppc_y + ppc_y // 2
                center_x = cx * ppc_x + ppc_x // 2
                for b in range(n_bins):
                    angle = (b * bin_width + bin_width / 2) * np.pi / 180.0
                    strength = cell_hists[cy, cx, b] / max_mag
                    dx = int(radius * strength * np.cos(angle))
                    dy = int(radius * strength * np.sin(angle))
                    cv2.line(vis,
                             (center_x - dx, center_y - dy),
                             (center_x + dx, center_y + dy),
                             strength, 1)

        vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return vis

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
