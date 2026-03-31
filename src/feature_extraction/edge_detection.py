"""
Edge Detection Module
======================
Implements multiple edge detection algorithms for crack boundary extraction:
  • Canny edge detector
  • Laplacian of Gaussian (LoG)
  • Difference of Gaussians (DoG)
  • Sobel / Scharr gradient operators
  • Hough Line Transform for linear crack detection

CV Syllabus: Feature Extraction & Image Segmentation —
  Canny / LoG / DoG edges, Hough transform, Gaussian derivatives
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

from ..utils import load_config, logger


class EdgeDetector:
    """Multi-algorithm edge detection for crack analysis."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        feat = cfg["feature_extraction"]
        self.canny_low = feat["canny_low"]
        self.canny_high = feat["canny_high"]
        self.log_sigma = feat["log_sigma"]
        self.dog_sigma1 = feat["dog_sigma1"]
        self.dog_sigma2 = feat["dog_sigma2"]

    # ------------------------------------------------------------------
    # Canny
    # ------------------------------------------------------------------
    def canny(self, img: np.ndarray, low: Optional[int] = None, high: Optional[int] = None) -> np.ndarray:
        """Canny edge detection with automatic or manual thresholds."""
        gray = self._gray(img)
        lo = low or self.canny_low
        hi = high or self.canny_high
        edges = cv2.Canny(gray, lo, hi, apertureSize=3, L2gradient=True)
        logger.debug(f"Canny edges: low={lo}, high={hi}")
        return edges

    def auto_canny(self, img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """Automatic Canny using median-based threshold selection."""
        gray = self._gray(img)
        median_val = np.median(gray)
        lo = int(max(0, (1.0 - sigma) * median_val))
        hi = int(min(255, (1.0 + sigma) * median_val))
        return cv2.Canny(gray, lo, hi, apertureSize=3, L2gradient=True)

    # ------------------------------------------------------------------
    # Laplacian of Gaussian (LoG)
    # ------------------------------------------------------------------
    def laplacian_of_gaussian(self, img: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """LoG edge detection: Gaussian blur followed by Laplacian."""
        gray = self._gray(img).astype(np.float64)
        s = sigma or self.log_sigma
        ksize = int(6 * s + 1) | 1  # ensure odd
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), s)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        # Zero-crossing approximation
        log_abs = np.abs(log)
        log_norm = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(log_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.debug(f"LoG edges: sigma={s}")
        return binary

    # ------------------------------------------------------------------
    # Difference of Gaussians (DoG)
    # ------------------------------------------------------------------
    def difference_of_gaussians(self, img: np.ndarray,
                                 sigma1: Optional[float] = None,
                                 sigma2: Optional[float] = None) -> np.ndarray:
        """DoG edge detection: difference between two Gaussian-blurred images."""
        gray = self._gray(img).astype(np.float64)
        s1 = sigma1 or self.dog_sigma1
        s2 = sigma2 or self.dog_sigma2
        k1 = int(6 * s1 + 1) | 1
        k2 = int(6 * s2 + 1) | 1
        g1 = cv2.GaussianBlur(gray, (k1, k1), s1)
        g2 = cv2.GaussianBlur(gray, (k2, k2), s2)
        dog = g1 - g2
        dog_norm = cv2.normalize(np.abs(dog), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(dog_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.debug(f"DoG edges: sigma1={s1}, sigma2={s2}")
        return binary

    # ------------------------------------------------------------------
    # Sobel & Scharr gradients
    # ------------------------------------------------------------------
    def sobel_gradient(self, img: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradient magnitude and direction using Sobel."""
        gray = self._gray(img).astype(np.float64)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        direction = np.arctan2(gy, gx)
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag_norm, magnitude, direction

    def scharr_gradient(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scharr operator (more accurate than 3×3 Sobel)."""
        gray = self._gray(img).astype(np.float64)
        gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag_norm, np.arctan2(gy, gx)

    # ------------------------------------------------------------------
    # Hough Transform (crack-line detection)
    # ------------------------------------------------------------------
    def hough_lines(self, edge_img: np.ndarray, threshold: int = 80,
                    min_length: int = 50, max_gap: int = 10) -> List[np.ndarray]:
        """Probabilistic Hough Line Transform for linear crack detection."""
        lines = cv2.HoughLinesP(edge_img, rho=1, theta=np.pi / 180,
                                threshold=threshold, minLineLength=min_length, maxLineGap=max_gap)
        if lines is None:
            logger.debug("No Hough lines detected")
            return []
        logger.debug(f"Hough lines detected: {len(lines)}")
        return [l[0] for l in lines]

    def hough_lines_standard(self, edge_img: np.ndarray, threshold: int = 150) -> List[Tuple[float, float]]:
        """Standard Hough Transform returning (rho, theta) pairs."""
        lines = cv2.HoughLines(edge_img, 1, np.pi / 180, threshold)
        if lines is None:
            return []
        return [(l[0][0], l[0][1]) for l in lines]

    # ------------------------------------------------------------------
    # Combined edge map
    # ------------------------------------------------------------------
    def multi_scale_edges(self, img: np.ndarray) -> np.ndarray:
        """Combine Canny, LoG, and DoG for robust edge detection."""
        canny_edges = self.canny(img)
        log_edges = self.laplacian_of_gaussian(img)
        dog_edges = self.difference_of_gaussians(img)
        combined = np.maximum(np.maximum(canny_edges, log_edges), dog_edges)
        logger.info("Multi-scale edge map computed (Canny + LoG + DoG)")
        return combined

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
