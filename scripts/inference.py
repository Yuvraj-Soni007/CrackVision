"""
Inference Script
=================
Load a trained model and run crack detection + severity analysis on new images.
"""

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, logger, project_root, ensure_dir
from src.preprocessing import ImageLoader, ImageEnhancer
from src.feature_extraction import EdgeDetector, TextureExtractor, KeypointExtractor, HOGExtractor
from src.segmentation import CrackSegmenter, MorphologicalProcessor
from src.analysis import CrackClassifier, SeverityAnalyzer
from src.visualization import Visualizer
from scripts.train import extract_features


def run_inference(image_path: str, model_name: str = "crack_classifier_svm"):
    """Full inference pipeline on a single image."""
    cfg = load_config()

    # Load modules
    loader = ImageLoader(cfg)
    enhancer = ImageEnhancer(cfg)
    edge_det = EdgeDetector(cfg)
    segmenter = CrackSegmenter(cfg)
    morph = MorphologicalProcessor(cfg)
    severity = SeverityAnalyzer(cfg)
    viz = Visualizer(cfg)
    hog_ext = HOGExtractor(cfg)
    tex_ext = TextureExtractor(cfg)
    kp_ext = KeypointExtractor(cfg)

    # Load and classify
    img = loader.load(image_path)
    enhanced = enhancer.enhance(img)

    # Feature extraction for classification
    feat = extract_features(img, enhancer, hog_ext, tex_ext, kp_ext)

    # Load trained classifier
    classifier = CrackClassifier(cfg)
    try:
        classifier.load(model_name)
        pred = classifier.predict(feat.reshape(1, -1))
        severity_names = cfg["classification"]["severity_levels"]
        predicted_class = severity_names[int(pred[0])] if int(pred[0]) < len(severity_names) else "unknown"
        logger.info(f"Classification result: {predicted_class}")
    except Exception as e:
        logger.warning(f"Could not load classifier: {e}. Skipping classification.")
        predicted_class = "unknown"

    # Edge detection
    edges = edge_det.multi_scale_edges(enhanced)

    # Segmentation
    seg_mask, regions = segmenter.segment(enhanced)

    # Morphological cleanup + skeleton
    cleaned = morph.clean_mask(seg_mask)
    skeleton = morph.skeletonize(cleaned)
    topo = morph.analyse_skeleton(skeleton)

    # Severity analysis
    sev_results = severity.analyse_regions(cleaned, regions)

    # Hough lines on edges
    lines = edge_det.hough_lines(edges)

    # ── Visualization ──
    overlay = viz.overlay_mask(img, cleaned)
    if sev_results:
        annotated = viz.draw_regions(overlay, sev_results)
    else:
        annotated = overlay

    if lines:
        annotated = viz.draw_hough_lines(annotated, lines)

    # Save all stages
    stages = {
        "Original": img,
        "Enhanced": enhanced,
        "Multi-Scale Edges": edges,
        "Segmentation Mask": cleaned,
        "Skeleton": skeleton,
        "Annotated Result": annotated,
    }
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    viz.plot_pipeline_results(stages, save_name=f"pipeline_{base_name}")
    if sev_results:
        viz.plot_severity_report(sev_results, save_name=f"severity_{base_name}")
    viz.save_image(annotated, f"result_{base_name}")

    # Print report
    print("\n" + "=" * 60)
    print(f"  CrackVision — Inference Report: {os.path.basename(image_path)}")
    print("=" * 60)
    print(f"  Classification: {predicted_class}")
    print(f"  Crack regions detected: {len(regions)}")
    print(f"  Skeleton: {topo}")
    for i, s in enumerate(sev_results):
        print(f"\n  Region {i + 1}:")
        print(f"    Severity: {s['severity_name']} (score={s['composite_score']:.4f})")
        print(f"    Width: mean={s['mean_width']:.1f}px, max={s['max_width']:.1f}px")
        print(f"    Length: {s['total_length_px']:.1f}px")
        print(f"    Area density: {s['area_density']:.6f}")
    print("=" * 60)

    return {
        "classification": predicted_class,
        "regions": regions,
        "severity": sev_results,
        "topology": topo,
    }


def main():
    parser = argparse.ArgumentParser(description="CrackVision — Inference")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="crack_classifier_svm",
                        help="Saved model name")
    args = parser.parse_args()

    run_inference(args.image, args.model)


if __name__ == "__main__":
    main()
