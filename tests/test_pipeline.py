"""
Test Suite for CrackVision Pipeline
=====================================
Verifies correct functioning of each module.
"""

import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import ImageEnhancer, HistogramProcessor
from src.feature_extraction import EdgeDetector, TextureExtractor, KeypointExtractor, HOGExtractor
from src.segmentation import CrackSegmenter, MorphologicalProcessor
from src.analysis import SeverityAnalyzer, DepthAnalyzer
from src.shape_analysis import SurfaceNormalEstimator, ShapeFromTexture
from src.motion import VideoInspector


def _make_test_image(size=(256, 256)):
    """Create a simple test image with a synthetic crack."""
    img = np.ones(size, dtype=np.uint8) * 180
    # Draw a diagonal line as a "crack"
    cv2.line(img, (50, 50), (200, 200), 40, 3)
    cv2.line(img, (100, 30), (150, 220), 50, 2)
    noise = np.random.normal(0, 5, size).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.img = _make_test_image()

    def test_clahe(self):
        enh = ImageEnhancer()
        result = enh.apply_clahe(self.img)
        self.assertEqual(result.shape, self.img.shape)

    def test_fourier_highpass(self):
        enh = ImageEnhancer()
        result = enh.fourier_highpass(self.img)
        self.assertEqual(result.shape, self.img.shape)

    def test_histogram(self):
        hp = HistogramProcessor()
        hist = hp.compute_histogram(self.img)
        self.assertEqual(len(hist), 256)
        self.assertAlmostEqual(hist.sum(), 1.0, places=4)

    def test_histogram_stats(self):
        hp = HistogramProcessor()
        stats = hp.compute_stats(self.img)
        self.assertIn("mean", stats)
        self.assertIn("skewness", stats)


class TestEdgeDetection(unittest.TestCase):
    def setUp(self):
        self.img = _make_test_image()

    def test_canny(self):
        ed = EdgeDetector()
        edges = ed.canny(self.img)
        self.assertEqual(edges.shape, self.img.shape)
        self.assertTrue(np.sum(edges > 0) > 0)

    def test_log(self):
        ed = EdgeDetector()
        edges = ed.laplacian_of_gaussian(self.img)
        self.assertEqual(edges.shape, self.img.shape)

    def test_dog(self):
        ed = EdgeDetector()
        edges = ed.difference_of_gaussians(self.img)
        self.assertEqual(edges.shape, self.img.shape)

    def test_multi_scale(self):
        ed = EdgeDetector()
        combined = ed.multi_scale_edges(self.img)
        self.assertEqual(combined.shape, self.img.shape)

    def test_hough_lines(self):
        ed = EdgeDetector()
        edges = ed.canny(self.img)
        lines = ed.hough_lines(edges, threshold=30, min_length=20)
        self.assertIsInstance(lines, list)


class TestTextureFeatures(unittest.TestCase):
    def setUp(self):
        self.img = _make_test_image()

    def test_gabor(self):
        te = TextureExtractor()
        feat = te.gabor_features(self.img)
        self.assertGreater(len(feat), 0)

    def test_lbp(self):
        te = TextureExtractor()
        lbp = te.lbp(self.img)
        self.assertEqual(lbp.shape, self.img.shape)

    def test_dwt(self):
        te = TextureExtractor()
        feat = te.dwt_features(self.img)
        self.assertGreater(len(feat), 0)

    def test_extract_combined(self):
        te = TextureExtractor()
        feat = te.extract(self.img)
        self.assertGreater(len(feat), 100)


class TestKeypointFeatures(unittest.TestCase):
    def setUp(self):
        self.img = _make_test_image()

    def test_sift(self):
        kp = KeypointExtractor()
        kps, descs = kp.sift_detect(self.img)
        self.assertIsInstance(kps, (list, tuple))

    def test_orb(self):
        kp = KeypointExtractor()
        kps, descs = kp.orb_detect(self.img)
        self.assertIsInstance(kps, (list, tuple))

    def test_harris(self):
        kp = KeypointExtractor()
        resp, mask = kp.harris_corners(self.img)
        self.assertEqual(resp.shape, self.img.shape)

    def test_pyramid(self):
        kp = KeypointExtractor()
        pyr = kp.gaussian_pyramid(self.img)
        self.assertGreater(len(pyr), 1)


class TestHOG(unittest.TestCase):
    def test_hog_compute(self):
        img = _make_test_image()
        hog = HOGExtractor()
        desc = hog.compute(img)
        self.assertGreater(len(desc), 0)

    def test_hog_manual(self):
        img = _make_test_image()
        hog = HOGExtractor()
        desc, vis = hog.compute_manual(img)
        self.assertGreater(len(desc), 0)
        self.assertEqual(len(vis.shape), 2)


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.img = _make_test_image()

    def test_threshold_segment(self):
        seg = CrackSegmenter()
        mask = seg.threshold_segment(self.img)
        self.assertEqual(mask.shape, self.img.shape)

    def test_full_pipeline(self):
        seg = CrackSegmenter()
        mask, regions = seg.segment(self.img)
        self.assertEqual(mask.shape, self.img.shape)
        self.assertIsInstance(regions, list)

    def test_morphological(self):
        seg = CrackSegmenter()
        mask = seg.threshold_segment(self.img)
        morph = MorphologicalProcessor()
        cleaned = morph.clean_mask(mask)
        self.assertEqual(cleaned.shape, mask.shape)


class TestSeverity(unittest.TestCase):
    def test_severity_analysis(self):
        img = _make_test_image()
        seg = CrackSegmenter()
        mask, _ = seg.segment(img)
        sa = SeverityAnalyzer()
        result = sa.compute_severity(mask)
        self.assertIn("severity_name", result)
        self.assertIn("composite_score", result)


class TestDepth(unittest.TestCase):
    def test_stereo_simulation(self):
        img = _make_test_image()
        da = DepthAnalyzer()
        left, right = da.simulate_stereo(img)
        self.assertEqual(left.shape, right.shape)

    def test_disparity(self):
        img = _make_test_image()
        da = DepthAnalyzer()
        left, right = da.simulate_stereo(img)
        disp = da.compute_disparity(left, right)
        self.assertEqual(disp.shape[:2], img.shape[:2])


class TestShapeAnalysis(unittest.TestCase):
    def test_photometric_stereo(self):
        img = _make_test_image()
        sn = SurfaceNormalEstimator()
        imgs, dirs = sn.simulate_multi_light(img)
        normals, albedo, depth = sn.photometric_stereo(imgs, dirs)
        self.assertEqual(normals.shape[:2], img.shape)

    def test_shape_from_texture(self):
        img = _make_test_image()
        sft = ShapeFromTexture()
        seg = sft.texture_segmentation(img)
        self.assertEqual(seg.shape, img.shape)


class TestMotion(unittest.TestCase):
    def test_optical_flow(self):
        img1 = _make_test_image()
        img2 = _make_test_image()  # slightly different due to noise
        vi = VideoInspector()
        flow = vi.farneback_flow(img1, img2)
        self.assertEqual(flow.shape[:2], img1.shape[:2])

    def test_frame_difference(self):
        img1 = _make_test_image()
        img2 = _make_test_image()
        vi = VideoInspector()
        diff = vi.frame_difference(img1, img2)
        self.assertEqual(diff.shape, img1.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
