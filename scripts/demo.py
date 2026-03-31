"""
Demo Script
=============
End-to-end demonstration of every CrackVision module.
Generates synthetic data, runs all pipeline stages, and saves visualisations.

Usage:
    python scripts/demo.py
"""

import os
import sys
import numpy as np
import cv2
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, logger, project_root, ensure_dir
from src.preprocessing import ImageLoader, ImageEnhancer, HistogramProcessor
from src.feature_extraction import EdgeDetector, TextureExtractor, KeypointExtractor, HOGExtractor
from src.segmentation import CrackSegmenter, MorphologicalProcessor
from src.analysis import SeverityAnalyzer, DepthAnalyzer
from src.shape_analysis import SurfaceNormalEstimator, ShapeFromTexture
from src.motion import VideoInspector
from src.visualization import Visualizer
from scripts.generate_data import generate_crack_image


def main():
    cfg = load_config()
    results_dir = ensure_dir(str(project_root() / cfg["paths"]["results_dir"]))

    print("=" * 70)
    print("  🔍 CrackVision — Full Pipeline Demonstration")
    print("=" * 70)
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════
    # 1. Generate a synthetic crack image
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/10] Generating synthetic crack image...")
    image, gt_mask = generate_crack_image(severity="severe", n_cracks=4)
    cv2.imwrite(os.path.join(results_dir, "demo_original.png"), image)
    cv2.imwrite(os.path.join(results_dir, "demo_gt_mask.png"), gt_mask)
    print(f"       Image shape: {image.shape}, GT mask pixels: {np.sum(gt_mask > 0)}")

    # ══════════════════════════════════════════════════════════════════
    # 2. Preprocessing & Enhancement
    # ══════════════════════════════════════════════════════════════════
    print("[2/10] Preprocessing & Enhancement...")
    enhancer = ImageEnhancer(cfg)
    hist_proc = HistogramProcessor(cfg)

    enhanced = enhancer.enhance(image)
    fourier_hp = enhancer.fourier_highpass(image, cutoff=25)
    sharpened = enhancer.sharpen(enhanced)

    # Histogram analysis
    hist_proc.plot_histogram(image, "Original Histogram", "demo_hist_original")
    hist_proc.plot_histogram(enhanced, "Enhanced Histogram", "demo_hist_enhanced")
    stats = hist_proc.compute_stats(enhanced)
    print(f"       Stats: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

    # ══════════════════════════════════════════════════════════════════
    # 3. Edge Detection
    # ══════════════════════════════════════════════════════════════════
    print("[3/10] Edge Detection (Canny, LoG, DoG, Hough)...")
    edge_det = EdgeDetector(cfg)

    canny = edge_det.canny(enhanced)
    auto_canny = edge_det.auto_canny(enhanced)
    log_edges = edge_det.laplacian_of_gaussian(enhanced)
    dog_edges = edge_det.difference_of_gaussians(enhanced)
    multi_edges = edge_det.multi_scale_edges(enhanced)
    sobel_mag, _, sobel_dir = edge_det.sobel_gradient(enhanced)

    # Hough lines
    lines = edge_det.hough_lines(canny, threshold=50, min_length=30)
    print(f"       Canny pixels: {np.sum(canny > 0)}, Hough lines: {len(lines)}")

    # ══════════════════════════════════════════════════════════════════
    # 4. Texture Features (Gabor, LBP, GLCM, DWT)
    # ══════════════════════════════════════════════════════════════════
    print("[4/10] Texture Feature Extraction...")
    tex_ext = TextureExtractor(cfg)

    gabor_feat = tex_ext.gabor_features(enhanced)
    lbp_img = tex_ext.lbp(enhanced)
    glcm_feats = tex_ext.glcm_features(enhanced)
    dwt_feat = tex_ext.dwt_features(enhanced)
    full_tex = tex_ext.extract(enhanced)
    print(f"       Gabor: {len(gabor_feat)}, LBP hist: 256, DWT: {len(dwt_feat)}, Total: {len(full_tex)}")

    # ══════════════════════════════════════════════════════════════════
    # 5. Keypoint Features (SIFT, ORB, Harris, Hessian, Pyramids)
    # ══════════════════════════════════════════════════════════════════
    print("[5/10] Keypoint Feature Extraction...")
    kp_ext = KeypointExtractor(cfg)

    sift_kps, sift_descs = kp_ext.sift_detect(enhanced)
    orb_kps, orb_descs = kp_ext.orb_detect(enhanced)
    harris_resp, harris_mask = kp_ext.harris_corners(enhanced)
    hessian_blobs = kp_ext.hessian_blobs(enhanced)
    gp = kp_ext.gaussian_pyramid(enhanced)
    lp = kp_ext.laplacian_pyramid(enhanced)
    sift_bovw = kp_ext.sift_feature_vector(enhanced)
    print(f"       SIFT: {len(sift_kps)}, ORB: {len(orb_kps)}, Harris corners: {np.sum(harris_mask > 0)}, Pyramid levels: {len(gp)}")

    # ══════════════════════════════════════════════════════════════════
    # 6. HOG Features
    # ══════════════════════════════════════════════════════════════════
    print("[6/10] HOG Feature Extraction...")
    hog_ext = HOGExtractor(cfg)

    hog_desc = hog_ext.compute(enhanced)
    hog_manual, hog_vis = hog_ext.compute_manual(enhanced)
    print(f"       OpenCV HOG: {len(hog_desc)}, Manual HOG: {len(hog_manual)}")

    # ══════════════════════════════════════════════════════════════════
    # 7. Segmentation
    # ══════════════════════════════════════════════════════════════════
    print("[7/10] Crack Segmentation...")
    segmenter = CrackSegmenter(cfg)
    morph = MorphologicalProcessor(cfg)

    seg_mask, regions = segmenter.segment(enhanced)
    cleaned = morph.clean_mask(seg_mask)
    skeleton = morph.skeletonize(cleaned)
    topo = morph.analyse_skeleton(skeleton)

    # Mean-shift
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    ms_mask = segmenter.meanshift_segment(image_bgr)

    # Black-hat (for dark crack enhancement)
    bhat = morph.black_hat(enhanced)

    print(f"       Regions: {len(regions)}, Skeleton pixels: {topo['skeleton_pixels']}")

    # ══════════════════════════════════════════════════════════════════
    # 8. Severity Analysis
    # ══════════════════════════════════════════════════════════════════
    print("[8/10] Severity Analysis...")
    severity_analyzer = SeverityAnalyzer(cfg)
    overall = severity_analyzer.compute_severity(cleaned)
    region_sevs = severity_analyzer.analyse_regions(cleaned, regions)
    print(f"       Overall: {overall['severity_name']} (score={overall['composite_score']})")

    # ══════════════════════════════════════════════════════════════════
    # 9. Depth & Shape Analysis
    # ══════════════════════════════════════════════════════════════════
    print("[9/10] Depth Estimation & Shape Analysis...")
    depth_analyzer = DepthAnalyzer(cfg)
    sn_estimator = SurfaceNormalEstimator(cfg)
    sft = ShapeFromTexture(cfg)

    # Simulated stereo
    left, right = depth_analyzer.simulate_stereo(enhanced, shift_px=12)
    disparity = depth_analyzer.compute_disparity(left, right)
    depth_map = depth_analyzer.disparity_to_depth(disparity)
    crack_depth = depth_analyzer.crack_depth_profile(depth_map, cleaned)

    # Photometric stereo
    multi_light_imgs, light_dirs = sn_estimator.simulate_multi_light(enhanced)
    normals, albedo, ps_depth = sn_estimator.photometric_stereo(multi_light_imgs, light_dirs)
    normals_rgb = sn_estimator.normals_to_rgb(normals)
    irregularity = sn_estimator.detect_irregularities(normals)

    # Shape from texture
    tex_seg = sft.texture_segmentation(enhanced)
    tilt_map = sft.estimate_tilt(enhanced)

    print(f"       Depth range: {crack_depth.get('min_depth', 0):.3f} – {crack_depth.get('max_depth', 0):.3f}")
    print(f"       Mean tilt: {tilt_map.mean():.1f}°")

    # ══════════════════════════════════════════════════════════════════
    # 10. Motion Analysis (simulated frame sequence)
    # ══════════════════════════════════════════════════════════════════
    print("[10/10] Motion / Temporal Analysis...")
    vi = VideoInspector(cfg)

    # Generate a sequence simulating crack growth
    frames = []
    for i in range(5):
        img_i, _ = generate_crack_image(severity="moderate", n_cracks=2 + i)
        frames.append(img_i)

    if len(frames) >= 2:
        flow = vi.farneback_flow(frames[0], frames[1])
        flow_vis = vi.flow_to_hsv(flow)
        energy = vi.motion_energy(flow)
        prop = vi.detect_propagation(frames)
        bg_masks = vi.subtract_background(
            [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if len(f.shape) == 2 else f for f in frames]
        )
        print(f"       Optical flow energy: {energy:.1f}, Propagation: {prop['propagation_detected']}")

    # ══════════════════════════════════════════════════════════════════
    # Visualization
    # ══════════════════════════════════════════════════════════════════
    print("\n📊 Generating visualizations...")
    viz = Visualizer(cfg)

    # Pipeline stages
    stages = {
        "Original": image,
        "CLAHE Enhanced": enhanced,
        "Fourier HP": fourier_hp,
        "Canny Edges": canny,
        "LoG Edges": log_edges,
        "DoG Edges": dog_edges,
        "Sobel Gradient": sobel_mag,
        "Multi-Scale Edges": multi_edges,
        "Segmentation": cleaned,
        "Skeleton": skeleton,
        "Black-Hat": bhat,
        "HOG Visualization": hog_vis,
    }
    viz.plot_pipeline_results(stages, "CrackVision — Processing Stages", "demo_pipeline")

    # Additional visualizations
    stages2 = {
        "LBP": lbp_img,
        "Mean-Shift Seg": ms_mask,
        "Harris Corners": harris_mask,
        "Disparity Map": cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        "Normal Map": normals_rgb,
        "Albedo": albedo,
        "PS Depth": ps_depth,
        "Irregularities": irregularity,
        "Texture Seg": tex_seg,
        "Ground Truth": gt_mask,
    }
    viz.plot_pipeline_results(stages2, "CrackVision — Advanced Analysis", "demo_advanced")

    # Overlay result
    overlay = viz.overlay_mask(image, cleaned)
    if region_sevs:
        annotated = viz.draw_regions(overlay, region_sevs)
    else:
        annotated = overlay
    if lines:
        annotated = viz.draw_hough_lines(annotated, lines)
    viz.save_image(annotated, "demo_final_result")

    # Severity report
    if region_sevs:
        viz.plot_severity_report(region_sevs, "demo_severity")

    # Depth map
    viz.plot_depth_map(
        cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        "Stereo Disparity Map", "demo_depth"
    )

    # Normal map
    viz.plot_normal_map(normals_rgb, "demo_normals")

    elapsed = time.time() - t_start
    print(f"\n✅ Demo complete in {elapsed:.1f}s")
    print(f"📁 Results saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
