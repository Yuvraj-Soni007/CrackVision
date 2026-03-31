"""
Severity Analyzer Module
==========================
Analyses detected crack regions and assigns severity scores based on:
  • Crack width (via distance transform)
  • Crack length (skeleton analysis)
  • Crack area density
  • Orientation distribution
  • Branching complexity

CV Syllabus: Pattern Analysis — classification, feature-based scoring
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple

from ..utils import load_config, logger


class SeverityAnalyzer:
    """Compute structural severity scores for detected crack regions."""

    SEVERITY_LEVELS = {
        0: ("none", "No crack detected"),
        1: ("minor", "Hairline crack, cosmetic only"),
        2: ("moderate", "Visible crack, monitor recommended"),
        3: ("severe", "Wide crack, repair needed soon"),
        4: ("critical", "Major structural crack, immediate action"),
    }

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        self.levels = cfg["classification"]["severity_levels"]

    # ------------------------------------------------------------------
    # Width estimation (distance transform)
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_width(crack_mask: np.ndarray) -> Dict:
        """Estimate crack width from the distance transform of the mask.

        The maximum distance to the nearest background pixel gives
        half the local width.
        """
        dist = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
        skeleton = SeverityAnalyzer._skeletonize(crack_mask)
        widths = dist[skeleton > 0] * 2.0  # diameter ≈ 2 × distance

        if len(widths) == 0:
            return {"mean_width": 0.0, "max_width": 0.0, "std_width": 0.0}
        return {
            "mean_width": float(np.mean(widths)),
            "max_width": float(np.max(widths)),
            "std_width": float(np.std(widths)),
        }

    # ------------------------------------------------------------------
    # Length estimation
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_length(crack_mask: np.ndarray) -> float:
        """Estimate total crack length from contour arc-lengths."""
        contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total = sum(cv2.arcLength(c, False) for c in contours)
        return float(total)

    # ------------------------------------------------------------------
    # Area density
    # ------------------------------------------------------------------
    @staticmethod
    def area_density(crack_mask: np.ndarray) -> float:
        """Ratio of crack pixels to total image area."""
        total = crack_mask.shape[0] * crack_mask.shape[1]
        crack_px = int(np.sum(crack_mask > 0))
        return crack_px / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Orientation distribution
    # ------------------------------------------------------------------
    @staticmethod
    def orientation_histogram(crack_mask: np.ndarray, n_bins: int = 18) -> np.ndarray:
        """Histogram of crack pixel orientations (gradient direction)."""
        gray = crack_mask.astype(np.float64)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(gy, gx) * (180.0 / np.pi) % 180  # 0-180
        mask = crack_mask > 0
        angles = angle[mask]
        hist, _ = np.histogram(angles, bins=n_bins, range=(0, 180))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        return hist

    # ------------------------------------------------------------------
    # Comprehensive severity scoring
    # ------------------------------------------------------------------
    def compute_severity(self, crack_mask: np.ndarray) -> Dict:
        """Compute a multi-factor severity score.

        Returns a dict with individual metrics and a final severity level.
        """
        width_info = self.estimate_width(crack_mask)
        length = self.estimate_length(crack_mask)
        density = self.area_density(crack_mask)
        orient = self.orientation_histogram(crack_mask)
        orient_entropy = float(-np.sum(orient * np.log2(orient + 1e-10)))

        # Composite score (weighted)
        score = (
            0.30 * min(width_info["max_width"] / 20.0, 1.0) +  # width factor
            0.25 * min(length / 500.0, 1.0) +                    # length factor
            0.25 * min(density / 0.05, 1.0) +                    # area factor
            0.20 * min(orient_entropy / 4.0, 1.0)                # complexity factor
        )

        # Map score to severity level
        if score < 0.10:
            level = 0
        elif score < 0.30:
            level = 1
        elif score < 0.55:
            level = 2
        elif score < 0.80:
            level = 3
        else:
            level = 4

        severity_name, severity_desc = self.SEVERITY_LEVELS[level]

        result = {
            **width_info,
            "total_length_px": round(length, 2),
            "area_density": round(density, 6),
            "orientation_entropy": round(orient_entropy, 4),
            "composite_score": round(score, 4),
            "severity_level": level,
            "severity_name": severity_name,
            "severity_description": severity_desc,
        }
        logger.info(f"Severity: {severity_name} (score={score:.4f})")
        return result

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------
    def analyse_regions(self, crack_mask: np.ndarray, regions: List[Dict]) -> List[Dict]:
        """Compute severity for each detected crack region independently."""
        results = []
        for r in regions:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            roi = crack_mask[y:y + h, x:x + w]
            sev = self.compute_severity(roi)
            sev["region"] = r
            results.append(sev)
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _skeletonize(mask: np.ndarray) -> np.ndarray:
        skel = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = mask.copy()
        while True:
            eroded = cv2.erode(temp, element)
            dilated = cv2.dilate(eroded, element)
            sub = cv2.subtract(temp, dilated)
            skel = cv2.bitwise_or(skel, sub)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skel
