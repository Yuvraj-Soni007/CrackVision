"""
Surface Normal Estimation Module (Photometric Stereo)
======================================================
Estimates surface normals and albedo from multiple images captured
under different lighting directions — enables detection of surface
irregularities indicating sub-surface cracks.

CV Syllabus: Shape from X —
  Photometric stereo, surface normals, shading, reflectance maps, albedo estimation
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

from ..utils import load_config, logger


class SurfaceNormalEstimator:
    """Photometric stereo: estimate surface normals from multi-light images."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        sa = cfg["shape_analysis"]
        self.n_lights = sa["light_directions"]
        self.estimate_albedo = sa["albedo_estimation"]
        self.method = sa["surface_normal_method"]

    # ------------------------------------------------------------------
    # Simulate multi-light images (for demo when only 1 image exists)
    # ------------------------------------------------------------------
    @staticmethod
    def simulate_multi_light(img: np.ndarray, n_lights: int = 4) -> Tuple[List[np.ndarray], np.ndarray]:
        """Simulate images under different directional lights.

        Uses gradient-based shading simulation for demonstration.
        Returns (list_of_images, light_directions_matrix).
        """
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64) / 255.0
        h, w = gray.shape

        # Generate light directions (unit vectors in 3D)
        angles = np.linspace(0, 2 * np.pi, n_lights, endpoint=False)
        light_dirs = np.zeros((n_lights, 3))
        for i, a in enumerate(angles):
            light_dirs[i] = [np.cos(a) * 0.7, np.sin(a) * 0.7, 0.7]
            light_dirs[i] /= np.linalg.norm(light_dirs[i])

        # Create gradient-based normal proxy
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5) * 2.0
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5) * 2.0
        gz = np.ones_like(gray)
        norm_factor = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-8
        nx = gx / norm_factor
        ny = gy / norm_factor
        nz = gz / norm_factor

        images = []
        for ld in light_dirs:
            intensity = np.clip(nx * ld[0] + ny * ld[1] + nz * ld[2], 0, 1)
            # Multiply by base image for realism
            shaded = gray * (0.3 + 0.7 * intensity)
            shaded = (shaded * 255).astype(np.uint8)
            images.append(shaded)

        logger.info(f"Simulated {n_lights} multi-light images")
        return images, light_dirs

    # ------------------------------------------------------------------
    # Photometric Stereo — Least Squares
    # ------------------------------------------------------------------
    def photometric_stereo(self, images: List[np.ndarray],
                           light_dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute surface normals and albedo from photometric stereo.

        Parameters
        ----------
        images : list of N grayscale images (same size)
        light_dirs : (N, 3) matrix of unit light direction vectors

        Returns
        -------
        normals : (H, W, 3) surface normal map
        albedo  : (H, W) albedo (reflectance) map
        depth   : (H, W) integrated depth map
        """
        n = len(images)
        h, w = images[0].shape[:2]

        # Stack intensity observations: I(x,y) = ρ(x,y) · (n(x,y) · l)
        # I = L · g   where g = ρ·n
        I_matrix = np.zeros((n, h * w), dtype=np.float64)
        for i, img in enumerate(images):
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            I_matrix[i] = gray.astype(np.float64).flatten()

        L = light_dirs  # (N, 3)
        # Solve  L · g = I  →  g = (L^T L)^{-1} L^T I  for each pixel
        LtL_inv = np.linalg.pinv(L.T @ L)
        G = LtL_inv @ L.T @ I_matrix  # (3, H*W)

        # Albedo = ||g||
        albedo_flat = np.linalg.norm(G, axis=0)
        albedo = albedo_flat.reshape(h, w)

        # Normals = g / ||g||
        normals_flat = G / (albedo_flat[np.newaxis, :] + 1e-8)
        normals = normals_flat.T.reshape(h, w, 3)

        # Integrate normals to get depth
        depth = self._integrate_normals(normals)

        # Normalise for display
        albedo = cv2.normalize(albedo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        logger.info("Photometric stereo: normals, albedo, depth computed")
        return normals, albedo, depth

    # ------------------------------------------------------------------
    # Normal Map Visualisation (map nx,ny,nz → R,G,B)
    # ------------------------------------------------------------------
    @staticmethod
    def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
        """Convert a normal map to an RGB visualisation.

        Maps (nx,ny,nz) → (R,G,B) in [0,255].
        """
        vis = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
        return vis

    # ------------------------------------------------------------------
    # Surface irregularity detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_irregularities(normals: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Detect surface irregularities from normal map discontinuities.

        High-gradient normals indicate cracks / surface damage.
        """
        # Compute normal gradient magnitude
        nx = normals[:, :, 0]
        ny = normals[:, :, 1]

        gnx_x = cv2.Sobel(nx, cv2.CV_64F, 1, 0, ksize=3)
        gnx_y = cv2.Sobel(nx, cv2.CV_64F, 0, 1, ksize=3)
        gny_x = cv2.Sobel(ny, cv2.CV_64F, 1, 0, ksize=3)
        gny_y = cv2.Sobel(ny, cv2.CV_64F, 0, 1, ksize=3)

        grad_mag = np.sqrt(gnx_x ** 2 + gnx_y ** 2 + gny_x ** 2 + gny_y ** 2)
        grad_norm = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

        irregular_mask = (grad_norm > threshold).astype(np.uint8) * 255
        logger.debug(f"Irregularities detected: {np.sum(irregular_mask > 0)} pixels")
        return irregular_mask

    # ------------------------------------------------------------------
    # Depth integration (Frankot-Chellappa)
    # ------------------------------------------------------------------
    @staticmethod
    def _integrate_normals(normals: np.ndarray) -> np.ndarray:
        """Integrate surface normals to depth using Frankot-Chellappa method."""
        h, w, _ = normals.shape
        nz = normals[:, :, 2] + 1e-8
        p = -normals[:, :, 0] / nz  # dz/dx
        q = -normals[:, :, 1] / nz  # dz/dy

        # Fourier integration
        P = np.fft.fft2(p)
        Q = np.fft.fft2(q)

        u = np.fft.fftfreq(h).reshape(-1, 1) * 2 * np.pi
        v = np.fft.fftfreq(w).reshape(1, -1) * 2 * np.pi

        denom = u ** 2 + v ** 2
        denom[0, 0] = 1.0  # avoid division by zero

        Z = (1j * u * P + 1j * v * Q) / denom
        Z[0, 0] = 0

        depth = np.real(np.fft.ifft2(Z))
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth
