"""
Video Inspector / Motion Analysis Module
==========================================
Analyses video sequences for temporal crack monitoring:
  • Optical flow (Farnebäck / Lucas-Kanade)
  • Background subtraction (MOG2 / KNN)
  • Frame differencing for crack propagation detection
  • Motion estimation for structural vibration analysis

CV Syllabus: Pattern Analysis & Motion Analysis —
  optical flow, background subtraction, motion estimation
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple

from ..utils import load_config, logger


class VideoInspector:
    """Temporal analysis of crack regions using motion / change detection."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        ma = cfg["motion_analysis"]
        self.of_method = ma["optical_flow_method"]
        self.fb_pyr_scale = ma["farneback_pyr_scale"]
        self.fb_levels = ma["farneback_levels"]
        self.fb_winsize = ma["farneback_winsize"]
        self.fb_iterations = ma["farneback_iterations"]
        self.fb_poly_n = ma["farneback_poly_n"]
        self.fb_poly_sigma = ma["farneback_poly_sigma"]

        self.bg_type = ma["bg_subtractor"]
        self.bg_history = ma["bg_history"]
        self.bg_threshold = ma["bg_threshold"]
        self.bg_shadows = ma["bg_detect_shadows"]
        self.min_motion_area = ma["min_motion_area"]

    # ------------------------------------------------------------------
    # Optical Flow — Farnebäck (Dense)
    # ------------------------------------------------------------------
    def farneback_flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute dense optical flow using Farnebäck's method."""
        prev_g = self._gray(prev)
        curr_g = self._gray(curr)
        flow = cv2.calcOpticalFlowFarneback(
            prev_g, curr_g, None,
            self.fb_pyr_scale, self.fb_levels, self.fb_winsize,
            self.fb_iterations, self.fb_poly_n, self.fb_poly_sigma, 0
        )
        logger.debug("Farnebäck optical flow computed")
        return flow

    # ------------------------------------------------------------------
    # Optical Flow — Lucas-Kanade (Sparse)
    # ------------------------------------------------------------------
    def lucas_kanade_flow(self, prev: np.ndarray, curr: np.ndarray,
                          points: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute sparse optical flow at given points.

        If no points supplied, uses Shi-Tomasi good features.
        """
        prev_g = self._gray(prev)
        curr_g = self._gray(curr)

        if points is None:
            points = cv2.goodFeaturesToTrack(prev_g, maxCorners=200,
                                             qualityLevel=0.01, minDistance=10)

        if points is None or len(points) == 0:
            return np.array([]), np.array([]), np.array([])

        lk_params = dict(winSize=(21, 21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, points, None, **lk_params)
        status = status.flatten()
        good_old = points[status == 1]
        good_new = next_pts[status == 1]

        logger.debug(f"Lucas-Kanade: {len(good_old)} tracked points")
        return good_old, good_new, err

    # ------------------------------------------------------------------
    # Optical flow visualisation
    # ------------------------------------------------------------------
    @staticmethod
    def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
        """Convert a dense flow field to HSV colour representation."""
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ------------------------------------------------------------------
    # Background Subtraction
    # ------------------------------------------------------------------
    def create_bg_subtractor(self):
        """Create a background subtractor instance."""
        if self.bg_type == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self.bg_history, varThreshold=self.bg_threshold,
                detectShadows=self.bg_shadows
            )
        else:
            return cv2.createBackgroundSubtractorKNN(
                history=self.bg_history, dist2Threshold=self.bg_threshold,
                detectShadows=self.bg_shadows
            )

    def subtract_background(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply background subtraction to a sequence of frames."""
        subtractor = self.create_bg_subtractor()
        masks = []
        for frame in frames:
            fg_mask = subtractor.apply(frame)
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            masks.append(fg_mask)
        logger.info(f"Background subtraction: {len(masks)} frames processed")
        return masks

    # ------------------------------------------------------------------
    # Frame Differencing
    # ------------------------------------------------------------------
    @staticmethod
    def frame_difference(frame1: np.ndarray, frame2: np.ndarray, threshold: int = 30) -> np.ndarray:
        """Compute absolute difference between two frames → change mask."""
        g1 = frame1 if len(frame1.shape) == 2 else cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        g2 = frame2 if len(frame2.shape) == 2 else cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return mask

    # ------------------------------------------------------------------
    # Crack propagation detection (temporal)
    # ------------------------------------------------------------------
    def detect_propagation(self, frames: List[np.ndarray]) -> dict:
        """Detect if a crack is propagating over a sequence of frames.

        Analyses the growth of changed areas over time.
        """
        if len(frames) < 2:
            return {"propagation_detected": False}

        change_areas = []
        for i in range(1, len(frames)):
            diff = self.frame_difference(frames[i - 1], frames[i])
            area = int(np.sum(diff > 0))
            change_areas.append(area)

        # Linear regression on change areas
        x = np.arange(len(change_areas), dtype=np.float64)
        y = np.array(change_areas, dtype=np.float64)
        if len(x) > 1 and np.std(y) > 0:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        result = {
            "propagation_detected": slope > self.min_motion_area,
            "mean_change": float(np.mean(change_areas)),
            "max_change": float(np.max(change_areas)),
            "trend_slope": float(slope),
            "n_frames": len(frames),
        }
        logger.info(f"Propagation analysis: slope={slope:.2f}, detected={result['propagation_detected']}")
        return result

    # ------------------------------------------------------------------
    # Motion energy for structural vibration
    # ------------------------------------------------------------------
    def motion_energy(self, flow: np.ndarray) -> float:
        """Compute total motion energy from a dense flow field."""
        return float(np.sum(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2))

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
