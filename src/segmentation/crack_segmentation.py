"""
Crack Segmentation Module
===========================
Segments crack regions from background using multiple approaches:
  • Adaptive thresholding + Otsu
  • GrabCut (graph-cut based)
  • Mean-Shift segmentation
  • Watershed segmentation
  • Connected component analysis

CV Syllabus: Feature Extraction & Image Segmentation —
  graph-cut, mean-shift, texture segmentation
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

from ..utils import load_config, logger


class CrackSegmenter:
    """Multi-method crack region segmentation."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        seg = cfg["segmentation"]
        self.adaptive_block = seg["adaptive_block_size"]
        self.adaptive_c = seg["adaptive_c"]
        self.min_area = seg["min_crack_area"]
        self.max_area = seg["max_crack_area"]
        self.grabcut_iters = seg["grabcut_iterations"]
        self.ms_sp = seg["meanshift_spatial_radius"]
        self.ms_sr = seg["meanshift_color_radius"]
        self.ms_level = seg["meanshift_max_level"]

    # ------------------------------------------------------------------
    # Adaptive Thresholding + Otsu
    # ------------------------------------------------------------------
    def threshold_segment(self, img: np.ndarray) -> np.ndarray:
        """Otsu + adaptive thresholding for crack segmentation."""
        gray = self._gray(img)
        # Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Adaptive
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         self.adaptive_block, self.adaptive_c)
        # Combine
        combined = cv2.bitwise_or(otsu, adaptive)
        logger.debug("Threshold segmentation complete")
        return combined

    # ------------------------------------------------------------------
    # GrabCut (graph-cut based interactive segmentation)
    # ------------------------------------------------------------------
    def grabcut_segment(self, img: np.ndarray, rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Graph-cut based segmentation using GrabCut.

        If no rect is provided, uses an automatic central region.
        """
        bgr = self._bgr(img)
        mask = np.zeros(bgr.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        if rect is None:
            h, w = bgr.shape[:2]
            margin = 10
            rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model,
                    self.grabcut_iters, cv2.GC_INIT_WITH_RECT)

        # Create binary mask (foreground + probable foreground)
        result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        logger.debug("GrabCut segmentation complete")
        return result_mask

    # ------------------------------------------------------------------
    # Mean-Shift Segmentation
    # ------------------------------------------------------------------
    def meanshift_segment(self, img: np.ndarray) -> np.ndarray:
        """Mean-shift filtering for colour-based segmentation."""
        bgr = self._bgr(img)
        shifted = cv2.pyrMeanShiftFiltering(bgr, self.ms_sp, self.ms_sr, maxLevel=self.ms_level)
        gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_shifted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        logger.debug("Mean-shift segmentation complete")
        return binary

    # ------------------------------------------------------------------
    # Watershed Segmentation
    # ------------------------------------------------------------------
    def watershed_segment(self, img: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
        """Marker-based watershed segmentation.

        Uses distance transform on the edge map to find markers.
        """
        bgr = self._bgr(img)
        # Distance transform for sure foreground
        dist = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Sure background via dilation
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(edge_map, kernel, iterations=3)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers
        n_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(bgr, markers)

        # Create boundary mask
        boundary_mask = np.zeros_like(edge_map)
        boundary_mask[markers == -1] = 255
        logger.debug(f"Watershed: {n_labels} regions found")
        return boundary_mask

    # ------------------------------------------------------------------
    # Connected Component Analysis
    # ------------------------------------------------------------------
    def connected_components(self, binary: np.ndarray) -> Tuple[int, np.ndarray, List[dict]]:
        """Find and filter connected components by area.

        Returns (count, label_map, list_of_stats_dicts).
        """
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        filtered_stats = []
        for i in range(1, n_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_area <= area <= self.max_area:
                filtered_stats.append({
                    "label": i,
                    "x": int(stats[i, cv2.CC_STAT_LEFT]),
                    "y": int(stats[i, cv2.CC_STAT_TOP]),
                    "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                    "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                    "area": int(area),
                    "cx": float(centroids[i, 0]),
                    "cy": float(centroids[i, 1]),
                })

        logger.info(f"Connected components: {len(filtered_stats)}/{n_labels - 1} pass area filter")
        return len(filtered_stats), labels, filtered_stats

    # ------------------------------------------------------------------
    # Full segmentation pipeline
    # ------------------------------------------------------------------
    def segment(self, img: np.ndarray, edge_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[dict]]:
        """Run the default segmentation pipeline.

        1. Threshold segmentation
        2. Morphological cleanup (external)
        3. Connected component analysis
        """
        seg_mask = self.threshold_segment(img)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        count, labels, stats = self.connected_components(seg_mask)
        logger.info(f"Segmentation pipeline: {count} crack regions detected")
        return seg_mask, stats

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _bgr(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
