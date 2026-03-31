"""
Depth Analysis Module
=======================
Simulates stereo-vision depth estimation and homography-based
rectification for 3D crack profiling.

CV Syllabus: Depth Estimation & Multi-Camera Views —
  stereo vision, epipolar geometry, homography, rectification, RANSAC, DLT
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict

from ..utils import load_config, logger


class DepthAnalyzer:
    """Depth estimation and multi-view geometry for crack 3D profiling."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        da = cfg["depth_analysis"]
        self.num_disparities = da["stereo_num_disparities"]
        self.block_size = da["stereo_block_size"]
        self.min_disp = da["stereo_min_disparity"]
        self.baseline = da["baseline"]
        self.focal_length = da["focal_length"]

    # ------------------------------------------------------------------
    # Stereo Block Matching — Disparity Map
    # ------------------------------------------------------------------
    def compute_disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute a disparity map from a rectified stereo pair.

        Uses Semi-Global Block Matching (SGBM) for improved quality.
        """
        left_g = self._gray(left)
        right_g = self._gray(right)

        stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * self.block_size ** 2,
            P2=32 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disparity = stereo.compute(left_g, right_g).astype(np.float32) / 16.0
        logger.info("Stereo disparity map computed (SGBM)")
        return disparity

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Convert disparity to depth: Z = (f × B) / d."""
        depth = np.zeros_like(disparity, dtype=np.float64)
        valid = disparity > 0
        depth[valid] = (self.focal_length * self.baseline) / disparity[valid]
        logger.debug(f"Depth range: {depth[valid].min():.3f} – {depth[valid].max():.3f} m")
        return depth

    # ------------------------------------------------------------------
    # Simulated Stereo Pair (for demonstration when only 1 image is available)
    # ------------------------------------------------------------------
    @staticmethod
    def simulate_stereo(img: np.ndarray, shift_px: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Create a simulated stereo pair by horizontally shifting the image.

        Adds slight noise and a horizontal translation to mimic a
        second camera.  For demo/testing only.
        """
        h, w = img.shape[:2]
        M = np.float32([[1, 0, -shift_px], [0, 1, 0]])
        right = cv2.warpAffine(img, M, (w, h))
        noise = np.random.normal(0, 3, right.shape).astype(np.int16)
        right = np.clip(right.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        logger.debug(f"Simulated stereo pair: shift={shift_px}px")
        return img, right

    # ------------------------------------------------------------------
    # Homography Estimation (RANSAC + DLT)
    # ------------------------------------------------------------------
    def estimate_homography(self, src_pts: np.ndarray, dst_pts: np.ndarray,
                            method: str = "ransac") -> Tuple[np.ndarray, np.ndarray]:
        """Estimate a 3×3 homography matrix using RANSAC or DLT.

        Parameters
        ----------
        src_pts : (N, 2) array of source points
        dst_pts : (N, 2) array of destination points
        method : 'ransac' or 'dlt'
        """
        if method == "ransac":
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        else:
            # Direct Linear Transform (without RANSAC)
            H, mask = cv2.findHomography(src_pts, dst_pts, 0)
        logger.info(f"Homography estimated ({method}): inliers={mask.sum() if mask is not None else 'N/A'}")
        return H, mask

    def warp_perspective(self, img: np.ndarray, H: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Apply homography warp to an image."""
        if size is None:
            size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, H, size)

    # ------------------------------------------------------------------
    # Epipolar Geometry — Fundamental Matrix
    # ------------------------------------------------------------------
    @staticmethod
    def compute_fundamental(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fundamental matrix using RANSAC."""
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
        return F, mask

    @staticmethod
    def draw_epipolar_lines(img1: np.ndarray, img2: np.ndarray,
                            pts1: np.ndarray, pts2: np.ndarray,
                            F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Draw epipolar lines on both images."""
        img1c = img1.copy() if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2c = img2.copy() if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)

        h, w = img1c.shape[:2]
        for line, pt in zip(lines1, pts1):
            a, b, c = line
            x0, y0 = 0, int(-c / (b + 1e-8))
            x1, y1 = w, int(-(c + a * w) / (b + 1e-8))
            colour = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(img1c, (x0, y0), (x1, y1), colour, 1)
            cv2.circle(img1c, tuple(pt.astype(int)), 5, colour, -1)

        return img1c, img2c

    # ------------------------------------------------------------------
    # Stereo Rectification
    # ------------------------------------------------------------------
    @staticmethod
    def rectify_pair(left: np.ndarray, right: np.ndarray,
                     pts_l: np.ndarray, pts_r: np.ndarray,
                     F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stereo rectification using the fundamental matrix."""
        h, w = left.shape[:2]
        _, H1, H2 = cv2.stereoRectifyUncalibrated(pts_l, pts_r, F, (w, h))
        left_rect = cv2.warpPerspective(left, H1, (w, h))
        right_rect = cv2.warpPerspective(right, H2, (w, h))
        logger.info("Stereo pair rectified")
        return left_rect, right_rect

    # ------------------------------------------------------------------
    # Crack depth profiling
    # ------------------------------------------------------------------
    def crack_depth_profile(self, depth_map: np.ndarray, crack_mask: np.ndarray) -> Dict:
        """Extract depth statistics along detected crack regions."""
        crack_depths = depth_map[crack_mask > 0]
        valid = crack_depths[crack_depths > 0]
        if len(valid) == 0:
            return {"mean_depth": 0, "max_depth": 0, "min_depth": 0, "std_depth": 0}
        return {
            "mean_depth": float(np.mean(valid)),
            "max_depth": float(np.max(valid)),
            "min_depth": float(np.min(valid)),
            "std_depth": float(np.std(valid)),
            "n_points": int(len(valid)),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


