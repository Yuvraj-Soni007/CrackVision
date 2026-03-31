"""
Keypoint Feature Extraction Module
=====================================
Implements keypoint detection and descriptor computation:
  • SIFT (Scale-Invariant Feature Transform)
  • ORB (as open-source SURF alternative)
  • Harris Corner Detector
  • Hessian-based blob detector
  • Image Pyramid for multi-scale analysis
  • Feature matching (BFMatcher / FLANN)

CV Syllabus: Feature Extraction & Image Segmentation —
  Harris / Hessian, SIFT, SURF, image pyramids
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List

from ..utils import load_config, logger


class KeypointExtractor:
    """Keypoint detection and descriptor extraction for crack analysis."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        feat = cfg["feature_extraction"]
        self.sift_n_features = feat["sift_n_features"]
        self.sift_n_octave_layers = feat["sift_n_octave_layers"]
        self.sift_contrast_threshold = feat["sift_contrast_threshold"]
        self.sift_edge_threshold = feat["sift_edge_threshold"]
        self.sift_sigma = feat["sift_sigma"]

        self.harris_block = feat["harris_block_size"]
        self.harris_ksize = feat["harris_ksize"]
        self.harris_k = feat["harris_k"]
        self.harris_thresh = feat["harris_threshold"]

        self.pyramid_levels = feat["pyramid_levels"]

    # ------------------------------------------------------------------
    # SIFT
    # ------------------------------------------------------------------
    def sift_detect(self, img: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect SIFT keypoints and compute descriptors."""
        gray = self._gray(img)
        sift = cv2.SIFT_create(
            nfeatures=self.sift_n_features,
            nOctaveLayers=self.sift_n_octave_layers,
            contrastThreshold=self.sift_contrast_threshold,
            edgeThreshold=self.sift_edge_threshold,
            sigma=self.sift_sigma,
        )
        kps, descs = sift.detectAndCompute(gray, None)
        logger.debug(f"SIFT: {len(kps)} keypoints detected")
        return kps, descs if descs is not None else np.array([])

    def sift_feature_vector(self, img: np.ndarray, vocab_size: int = 64) -> np.ndarray:
        """Compute a fixed-length BoVW histogram from SIFT descriptors.

        Uses k-means mini-batch on the descriptors themselves when called
        on a single image (for simplicity). For proper BoVW, train a
        vocabulary on the full dataset first.
        """
        _, descs = self.sift_detect(img)
        if len(descs) == 0:
            return np.zeros(vocab_size, dtype=np.float32)

        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(vocab_size, len(descs))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(descs)
        hist, _ = np.histogram(labels, bins=n_clusters, range=(0, n_clusters))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-8)
        if len(hist) < vocab_size:
            hist = np.pad(hist, (0, vocab_size - len(hist)))
        return hist

    # ------------------------------------------------------------------
    # ORB (open-source alternative to SURF)
    # ------------------------------------------------------------------
    def orb_detect(self, img: np.ndarray, n_features: int = 500) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect ORB keypoints and compute binary descriptors."""
        gray = self._gray(img)
        orb = cv2.ORB_create(nfeatures=n_features)
        kps, descs = orb.detectAndCompute(gray, None)
        logger.debug(f"ORB: {len(kps)} keypoints detected")
        return kps, descs if descs is not None else np.array([])

    # ------------------------------------------------------------------
    # Harris Corner Detection
    # ------------------------------------------------------------------
    def harris_corners(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Harris corner response and thresholded corner map."""
        gray = self._gray(img).astype(np.float32)
        harris_resp = cv2.cornerHarris(gray, self.harris_block, self.harris_ksize, self.harris_k)
        harris_resp = cv2.dilate(harris_resp, None)
        thresh = self.harris_thresh * harris_resp.max()
        corner_mask = (harris_resp > thresh).astype(np.uint8) * 255
        logger.debug(f"Harris corners: {np.sum(corner_mask > 0)} pixels above threshold")
        return harris_resp, corner_mask

    # ------------------------------------------------------------------
    # Hessian Blob Detection (via OpenCV SimpleBlobDetector)
    # ------------------------------------------------------------------
    def hessian_blobs(self, img: np.ndarray) -> List[cv2.KeyPoint]:
        """Detect blob/crack regions using Hessian-based approach."""
        gray = self._gray(img)
        # Compute Hessian determinant
        dx2 = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 2, 0, ksize=5)
        dy2 = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 2, ksize=5)
        dxy = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 1, ksize=5)
        hessian_det = dx2 * dy2 - dxy ** 2

        # Threshold on Hessian determinant
        det_norm = cv2.normalize(np.abs(hessian_det), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(det_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use blob detector on the thresholded map
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary)
        logger.debug(f"Hessian blobs: {len(keypoints)} detected")
        return keypoints

    # ------------------------------------------------------------------
    # Image Pyramid (Gaussian)
    # ------------------------------------------------------------------
    def gaussian_pyramid(self, img: np.ndarray, levels: Optional[int] = None) -> List[np.ndarray]:
        """Build a Gaussian pyramid."""
        n = levels or self.pyramid_levels
        pyramid = [img.copy()]
        for _ in range(n - 1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        logger.debug(f"Gaussian pyramid: {len(pyramid)} levels")
        return pyramid

    def laplacian_pyramid(self, img: np.ndarray, levels: Optional[int] = None) -> List[np.ndarray]:
        """Build a Laplacian pyramid from the Gaussian pyramid."""
        gp = self.gaussian_pyramid(img, levels)
        lap_pyr = []
        for i in range(len(gp) - 1):
            expanded = cv2.pyrUp(gp[i + 1],
                                 dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lap = cv2.subtract(gp[i], expanded)
            lap_pyr.append(lap)
        lap_pyr.append(gp[-1])  # smallest level
        return lap_pyr

    # ------------------------------------------------------------------
    # Feature matching
    # ------------------------------------------------------------------
    @staticmethod
    def match_features(desc1: np.ndarray, desc2: np.ndarray,
                       method: str = "bf", ratio: float = 0.75) -> List:
        """Match descriptors using BFMatcher with Lowe's ratio test."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        if method == "bf":
            norm = cv2.NORM_L2 if desc1.dtype == np.float32 else cv2.NORM_HAMMING
            bf = cv2.BFMatcher(norm, crossCheck=False)
            matches = bf.knnMatch(desc1, desc2, k=2)
        else:
            FLANN_INDEX_KDTREE = 1
            idx_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(idx_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)

        # Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good.append(m)
        logger.debug(f"Feature matching: {len(good)} good matches")
        return good

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
