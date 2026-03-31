"""
Histogram Processing Module
=============================
Implements histogram analysis, equalization, matching, and back-projection
for crack-region contrast analysis.

CV Syllabus: Digital Image Formation & Low-Level Processing —
  histogram processing, enhancement
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from ..utils import load_config, logger, ensure_dir, project_root


class HistogramProcessor:
    """Histogram analysis and manipulation utilities."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        self.results_dir = str(project_root() / cfg["paths"]["results_dir"])
        ensure_dir(self.results_dir)

    # ------------------------------------------------------------------
    # Histogram computation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
        """Compute normalised histogram of a grayscale image."""
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        return hist

    @staticmethod
    def compute_color_histogram(img: np.ndarray, bins: int = 64) -> dict:
        """Compute per-channel histograms for a colour image."""
        if len(img.shape) == 2:
            return {"gray": cv2.calcHist([img], [0], None, [bins], [0, 256]).flatten()}
        colors = ("b", "g", "r")
        hists = {}
        for i, c in enumerate(colors):
            hists[c] = cv2.calcHist([img], [i], None, [bins], [0, 256]).flatten()
        return hists

    # ------------------------------------------------------------------
    # Equalization
    # ------------------------------------------------------------------
    @staticmethod
    def equalize(img: np.ndarray) -> np.ndarray:
        """Global histogram equalization."""
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    # ------------------------------------------------------------------
    # Histogram matching (specification)
    # ------------------------------------------------------------------
    @staticmethod
    def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match the histogram of *source* to that of *reference*.

        Uses CDF-based mapping (histogram specification).
        """
        from skimage.exposure import match_histograms as _match
        matched = _match(source, reference, channel_axis=None if len(source.shape) == 2 else -1)
        return matched.astype(np.uint8)

    # ------------------------------------------------------------------
    # Histogram back-projection
    # ------------------------------------------------------------------
    @staticmethod
    def back_project(target: np.ndarray, roi_hist: np.ndarray) -> np.ndarray:
        """Histogram back-projection for region-of-interest highlighting."""
        hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV) if len(target.shape) == 3 else target
        back = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # Convolve with a circular disc
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(back, -1, disc, back)
        return back

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_stats(img: np.ndarray) -> dict:
        """Compute basic intensity statistics."""
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {
            "mean": float(np.mean(gray)),
            "std": float(np.std(gray)),
            "median": float(np.median(gray)),
            "min": int(np.min(gray)),
            "max": int(np.max(gray)),
            "skewness": float(_skewness(gray)),
            "kurtosis": float(_kurtosis(gray)),
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_histogram(self, img: np.ndarray, title: str = "Histogram", save_name: str | None = None):
        """Plot and optionally save histogram."""
        hist = self.compute_histogram(img)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(hist)), hist, width=1, color="steelblue", edgecolor="none")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Normalised frequency")
        ax.set_xlim([0, 256])
        plt.tight_layout()
        if save_name:
            fig.savefig(f"{self.results_dir}/{save_name}.png", dpi=150)
            logger.info(f"Histogram saved → {save_name}.png")
        plt.close(fig)
        return hist


# -----------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------

def _skewness(arr: np.ndarray) -> float:
    m = np.mean(arr)
    s = np.std(arr) + 1e-8
    return float(np.mean(((arr - m) / s) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    m = np.mean(arr)
    s = np.std(arr) + 1e-8
    return float(np.mean(((arr - m) / s) ** 4) - 3)
