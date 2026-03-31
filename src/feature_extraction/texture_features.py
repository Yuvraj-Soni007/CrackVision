"""
Texture Feature Extraction Module
===================================
Extracts texture descriptors for crack vs. non-crack region classification:
  • Gabor filter bank responses
  • Local Binary Patterns (LBP)
  • Gray-Level Co-occurrence Matrix (GLCM) features
  • Discrete Wavelet Transform (DWT) features

CV Syllabus: Feature Extraction & Image Segmentation —
  Gabor filters, DWT, texture segmentation
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

from ..utils import load_config, logger


class TextureExtractor:
    """Texture-based feature extraction for crack analysis."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        feat = cfg["feature_extraction"]
        self.gabor_ksize = feat["gabor_ksize"]
        self.gabor_sigmas = feat["gabor_sigmas"]
        self.gabor_thetas = feat["gabor_thetas"]
        self.gabor_lambdas = feat["gabor_lambdas"]
        self.gabor_gammas = feat["gabor_gammas"]

    # ------------------------------------------------------------------
    # Gabor Filter Bank
    # ------------------------------------------------------------------
    def build_gabor_bank(self) -> List[Tuple[np.ndarray, dict]]:
        """Create a bank of Gabor filters at multiple orientations/scales."""
        filters = []
        for sigma in self.gabor_sigmas:
            for theta in self.gabor_thetas:
                for lambd in self.gabor_lambdas:
                    for gamma in self.gabor_gammas:
                        kern = cv2.getGaborKernel(
                            (self.gabor_ksize, self.gabor_ksize),
                            sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F,
                        )
                        kern /= (1.5 * kern.sum() + 1e-8)
                        params = dict(sigma=sigma, theta=round(theta, 3),
                                      lambd=lambd, gamma=gamma)
                        filters.append((kern, params))
        logger.info(f"Gabor filter bank built: {len(filters)} kernels")
        return filters

    def gabor_features(self, img: np.ndarray) -> np.ndarray:
        """Extract mean and variance of each Gabor filter response → feature vector."""
        gray = self._gray(img).astype(np.float32)
        bank = self.build_gabor_bank()
        feats = []
        for kern, _ in bank:
            response = cv2.filter2D(gray, cv2.CV_32F, kern)
            feats.append(np.mean(response))
            feats.append(np.var(response))
        return np.array(feats, dtype=np.float32)

    def gabor_response_images(self, img: np.ndarray) -> List[np.ndarray]:
        """Return all Gabor response images (for visualisation)."""
        gray = self._gray(img).astype(np.float32)
        bank = self.build_gabor_bank()
        responses = []
        for kern, _ in bank:
            resp = cv2.filter2D(gray, cv2.CV_32F, kern)
            resp_norm = cv2.normalize(np.abs(resp), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            responses.append(resp_norm)
        return responses

    # ------------------------------------------------------------------
    # Local Binary Pattern (LBP) — Vectorised
    # ------------------------------------------------------------------
    @staticmethod
    def lbp(img: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern descriptor image (vectorised).

        Instead of triple-nested Python loops, this uses NumPy roll /
        shifted-slice operations for each neighbour direction, running
        orders of magnitude faster.
        """
        gray = TextureExtractor._gray(img).astype(np.int32)
        rows, cols = gray.shape
        lbp_img = np.zeros((rows, cols), dtype=np.uint16)

        # Pre-compute offsets for each of the n_points neighbours
        for k in range(n_points):
            angle = 2 * np.pi * k / n_points
            dy = -int(round(radius * np.sin(angle)))
            dx = int(round(radius * np.cos(angle)))

            # Shifted view of the image
            # Determine valid region
            r0 = max(0, -dy)
            r1 = min(rows, rows - dy)
            c0 = max(0, -dx)
            c1 = min(cols, cols - dx)

            centre = gray[r0:r1, c0:c1]
            neighbour = gray[r0 + dy:r1 + dy, c0 + dx:c1 + dx]

            bits = (neighbour >= centre).astype(np.uint16) << k
            lbp_img[r0:r1, c0:c1] += bits

        return lbp_img.astype(np.uint8)

    @staticmethod
    def lbp_histogram(lbp_img: np.ndarray, n_bins: int = 256) -> np.ndarray:
        """Compute a normalised histogram from an LBP image."""
        hist, _ = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        return hist

    # ------------------------------------------------------------------
    # GLCM (Gray-Level Co-occurrence Matrix) features
    # ------------------------------------------------------------------
    @staticmethod
    def glcm_features(img: np.ndarray, distances: list = None, angles: list = None,
                      levels: int = 256) -> dict:
        """Compute GLCM-based texture features: contrast, correlation,
        energy, homogeneity."""
        gray = TextureExtractor._gray(img)
        if distances is None:
            distances = [1, 3, 5]
        if angles is None:
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        # Quantise to fewer levels for speed
        q_levels = 64
        gray_q = (gray / (256 / q_levels)).astype(np.uint8)
        gray_q = np.clip(gray_q, 0, q_levels - 1)

        all_features = {}
        for d in distances:
            for a in angles:
                glcm = _compute_glcm(gray_q, d, a, q_levels)
                contrast = _glcm_contrast(glcm, q_levels)
                energy = _glcm_energy(glcm)
                homogeneity = _glcm_homogeneity(glcm, q_levels)
                key = f"d{d}_a{round(np.degrees(a))}"
                all_features[f"{key}_contrast"] = contrast
                all_features[f"{key}_energy"] = energy
                all_features[f"{key}_homog"] = homogeneity
        return all_features

    # ------------------------------------------------------------------
    # DWT (Discrete Wavelet Transform) features
    # ------------------------------------------------------------------
    @staticmethod
    def dwt_features(img: np.ndarray, levels: int = 3) -> np.ndarray:
        """Haar-wavelet decomposition via simple averaging/differencing.

        Returns feature vector of sub-band statistics.
        """
        gray = TextureExtractor._gray(img).astype(np.float64)
        features = []

        current = gray.copy()
        for _ in range(levels):
            rows, cols = current.shape
            rows = rows - rows % 2
            cols = cols - cols % 2
            current = current[:rows, :cols]

            # Row-wise transform
            low_rows = (current[:, 0::2] + current[:, 1::2]) / 2.0
            high_rows = (current[:, 0::2] - current[:, 1::2]) / 2.0

            # Column-wise on low
            LL = (low_rows[0::2, :] + low_rows[1::2, :]) / 2.0
            LH = (low_rows[0::2, :] - low_rows[1::2, :]) / 2.0

            # Column-wise on high
            HL = (high_rows[0::2, :] + high_rows[1::2, :]) / 2.0
            HH = (high_rows[0::2, :] - high_rows[1::2, :]) / 2.0

            for band_name, band in [("LH", LH), ("HL", HL), ("HH", HH)]:
                features.extend([
                    np.mean(np.abs(band)),
                    np.std(band),
                    np.max(np.abs(band)),
                    float(np.mean(band ** 2)),  # energy
                ])

            current = LL

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Combined texture feature vector
    # ------------------------------------------------------------------
    def extract(self, img: np.ndarray) -> np.ndarray:
        """Combine Gabor + LBP + DWT into a single feature vector."""
        gabor = self.gabor_features(img)
        lbp_img = self.lbp(img)
        lbp_hist = self.lbp_histogram(lbp_img)
        dwt = self.dwt_features(img)
        combined = np.concatenate([gabor, lbp_hist, dwt])
        logger.debug(f"Texture feature vector length: {len(combined)}")
        return combined

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


# -----------------------------------------------------------------------
# GLCM helpers — VECTORISED (no Python loops over pixels)
# -----------------------------------------------------------------------

def _compute_glcm(gray_q: np.ndarray, distance: int, angle: float, levels: int) -> np.ndarray:
    """Compute a single GLCM matrix using vectorised NumPy operations."""
    dy = int(round(-distance * np.sin(angle)))
    dx = int(round(distance * np.cos(angle)))
    rows, cols = gray_q.shape

    # Determine the valid slice for both source and offset
    r0 = max(0, -dy)
    r1 = min(rows, rows - dy)
    c0 = max(0, -dx)
    c1 = min(cols, cols - dx)

    src = gray_q[r0:r1, c0:c1].ravel()
    dst = gray_q[r0 + dy:r1 + dy, c0 + dx:c1 + dx].ravel()

    # Use np.add.at (or bincount on a flat index) for fast accumulation
    flat_idx = src.astype(np.int64) * levels + dst.astype(np.int64)
    glcm = np.bincount(flat_idx, minlength=levels * levels).astype(np.float64)
    glcm = glcm.reshape(levels, levels)
    s = glcm.sum()
    if s > 0:
        glcm /= s
    return glcm


def _glcm_contrast(glcm: np.ndarray, levels: int) -> float:
    idx = np.arange(levels)
    diff = (idx[:, None] - idx[None, :]) ** 2
    return float(np.sum(glcm * diff))


def _glcm_energy(glcm: np.ndarray) -> float:
    return float(np.sum(glcm ** 2))


def _glcm_homogeneity(glcm: np.ndarray, levels: int) -> float:
    idx = np.arange(levels)
    diff = np.abs(idx[:, None] - idx[None, :])
    return float(np.sum(glcm / (1 + diff)))
