"""
Image Enhancement Module
=========================
Applies various enhancement techniques to improve crack visibility:
  • CLAHE (Contrast-Limited Adaptive Histogram Equalization)
  • Bilateral filtering (edge-preserving denoising)
  • Gaussian blur / sharpening
  • Non-local means denoising
  • Fourier-domain filtering

CV Syllabus: Digital Image Formation & Low-Level Processing —
  convolution, filtering, Fourier transform, enhancement, restoration
"""

import cv2
import numpy as np
from typing import Optional

from ..utils import load_config, logger


class ImageEnhancer:
    """Collection of image enhancement routines for crack detection."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        pre = cfg["preprocessing"]
        self.clahe_clip = pre["clahe_clip_limit"]
        self.clahe_grid = tuple(pre["clahe_grid_size"])
        self.bilateral_d = pre["bilateral_d"]
        self.bilateral_sc = pre["bilateral_sigma_color"]
        self.bilateral_ss = pre["bilateral_sigma_space"]
        self.gauss_k = tuple(pre["gaussian_kernel"])
        self.gauss_sigma = pre["gaussian_sigma"]
        self.denoise_h = pre["denoise_strength"]

    # ------------------------------------------------------------------
    # CLAHE  (Contrast-Limited Adaptive Histogram Equalization)
    # ------------------------------------------------------------------
    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE for local contrast enhancement."""
        gray = self._ensure_gray(img)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        enhanced = clahe.apply(gray)
        logger.debug("CLAHE applied")
        return enhanced

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------
    def bilateral_filter(self, img: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filter."""
        return cv2.bilateralFilter(img, self.bilateral_d, self.bilateral_sc, self.bilateral_ss)

    def gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Standard Gaussian smoothing."""
        return cv2.GaussianBlur(img, self.gauss_k, self.gauss_sigma)

    def non_local_means(self, img: np.ndarray) -> np.ndarray:
        """Non-local Means denoising (strong but slow)."""
        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, None, self.denoise_h, 7, 21)
        return cv2.fastNlMeansDenoisingColored(img, None, self.denoise_h, self.denoise_h, 7, 21)

    # ------------------------------------------------------------------
    # Sharpening (Unsharp Mask)
    # ------------------------------------------------------------------
    def sharpen(self, img: np.ndarray, amount: float = 1.5) -> np.ndarray:
        """Unsharp-mask sharpening: original + amount*(original – blurred)."""
        blurred = self.gaussian_blur(img)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Fourier-Domain Filtering
    # ------------------------------------------------------------------
    def fourier_highpass(self, img: np.ndarray, cutoff: int = 30) -> np.ndarray:
        """High-pass filter in the Fourier domain to enhance edges/cracks.

        Demonstrates the Fourier transform from the syllabus.
        """
        gray = self._ensure_gray(img)
        # Optimal DFT size
        rows, cols = gray.shape
        m = cv2.getOptimalDFTSize(rows)
        n = cv2.getOptimalDFTSize(cols)
        padded = np.zeros((m, n), dtype=np.float32)
        padded[:rows, :cols] = gray.astype(np.float32)

        # DFT
        dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=[0, 1])

        # Create high-pass mask (ideal)
        cy, cx = m // 2, n // 2
        mask = np.ones((m, n, 2), dtype=np.float32)
        cv2.circle(mask, (cx, cy), cutoff, (0, 0), -1)  # block low freq

        # Apply mask
        filtered = dft_shift * mask
        f_ishift = np.fft.ifftshift(filtered, axes=[0, 1])
        result = cv2.idft(f_ishift)
        magnitude = cv2.magnitude(result[:, :, 0], result[:, :, 1])
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        logger.debug("Fourier high-pass filter applied")
        return magnitude[:rows, :cols].astype(np.uint8)

    def fourier_bandpass(self, img: np.ndarray, low_cut: int = 10, high_cut: int = 80) -> np.ndarray:
        """Band-pass filter in the Fourier domain."""
        gray = self._ensure_gray(img)
        rows, cols = gray.shape
        m = cv2.getOptimalDFTSize(rows)
        n = cv2.getOptimalDFTSize(cols)
        padded = np.zeros((m, n), dtype=np.float32)
        padded[:rows, :cols] = gray.astype(np.float32)

        dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=[0, 1])

        cy, cx = m // 2, n // 2
        mask = np.zeros((m, n, 2), dtype=np.float32)
        cv2.circle(mask, (cx, cy), high_cut, (1, 1), -1)
        # Remove low frequencies
        low_mask = np.ones((m, n, 2), dtype=np.float32)
        cv2.circle(low_mask, (cx, cy), low_cut, (0, 0), -1)
        mask *= low_mask

        filtered = dft_shift * mask
        f_ishift = np.fft.ifftshift(filtered, axes=[0, 1])
        result = cv2.idft(f_ishift)
        magnitude = cv2.magnitude(result[:, :, 0], result[:, :, 1])
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        return magnitude[:rows, :cols].astype(np.uint8)

    # ------------------------------------------------------------------
    # Full enhancement pipeline
    # ------------------------------------------------------------------
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """Run the default enhancement pipeline for crack detection."""
        gray = self._ensure_gray(img)
        denoised = self.bilateral_filter(gray)
        enhanced = self.apply_clahe(denoised)
        logger.info("Enhancement pipeline complete")
        return enhanced

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
