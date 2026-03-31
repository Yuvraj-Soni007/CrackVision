"""
Training Script
================
End-to-end training pipeline:
  1. Generate / load data
  2. Preprocessing & enhancement
  3. Feature extraction (HOG + Gabor + LBP + DWT + SIFT BoVW)
  4. Train classifier (SVM / KNN / GMM)
  5. Evaluate and save model
"""

import os
import sys
import argparse
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, logger, project_root, ensure_dir
from src.preprocessing import ImageLoader, ImageEnhancer, HistogramProcessor
from src.feature_extraction import EdgeDetector, TextureExtractor, KeypointExtractor, HOGExtractor
from src.segmentation import CrackSegmenter, MorphologicalProcessor
from src.analysis import CrackClassifier

from scripts.generate_data import generate_dataset


def extract_features(img: np.ndarray, enhancer: ImageEnhancer,
                     hog_ext: HOGExtractor, tex_ext: TextureExtractor,
                     kp_ext: KeypointExtractor) -> np.ndarray:
    """Extract a comprehensive feature vector from a single image."""
    # Enhance
    enhanced = enhancer.enhance(img)

    # HOG features
    hog_feat = hog_ext.compute(enhanced)
    # Truncate / pad to fixed size
    hog_size = 2048
    if len(hog_feat) > hog_size:
        hog_feat = hog_feat[:hog_size]
    else:
        hog_feat = np.pad(hog_feat, (0, max(0, hog_size - len(hog_feat))))

    # Texture features (Gabor + LBP + DWT)
    tex_feat = tex_ext.extract(enhanced)

    # SIFT BoVW
    sift_feat = kp_ext.sift_feature_vector(enhanced, vocab_size=64)

    # Concatenate
    feature = np.concatenate([hog_feat, tex_feat, sift_feat])
    return feature


def main():
    parser = argparse.ArgumentParser(description="CrackVision — Training Pipeline")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset (default: generates synthetic data)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Samples per severity class (synthetic)")
    parser.add_argument("--model-type", type=str, default="svm",
                        choices=["svm", "knn", "gmm"],
                        help="Classifier type")
    parser.add_argument("--use-pca", action="store_true", default=True,
                        help="Apply PCA dimensionality reduction")
    parser.add_argument("--use-lda", action="store_true", default=False,
                        help="Apply LDA dimensionality reduction (instead of PCA)")
    args = parser.parse_args()

    cfg = load_config()
    logger.info("=" * 60)
    logger.info("CrackVision — Training Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = str(project_root() / "data" / "synthetic")
        if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
            logger.info("Generating synthetic dataset...")
            generate_dataset(n_per_class=args.n_samples)

    # ------------------------------------------------------------------
    # 2. Load images and extract features
    # ------------------------------------------------------------------
    loader = ImageLoader(cfg)
    enhancer = ImageEnhancer(cfg)
    hog_ext = HOGExtractor(cfg)
    tex_ext = TextureExtractor(cfg)
    kp_ext = KeypointExtractor(cfg)

    severity_map = {"none": 0, "minor": 1, "moderate": 2, "severe": 3, "critical": 4}
    X_list = []
    y_list = []

    t0 = time.time()
    for severity_name in severity_map:
        img_dir = os.path.join(data_dir, severity_name, "images")
        if not os.path.exists(img_dir):
            logger.warning(f"Missing directory: {img_dir}")
            continue

        images = loader.load_directory(img_dir)
        label = severity_map[severity_name]

        for fname, img in images:
            try:
                feat = extract_features(img, enhancer, hog_ext, tex_ext, kp_ext)
                X_list.append(feat)
                y_list.append(label)
            except Exception as e:
                logger.warning(f"Skipping {fname}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    logger.info(f"Feature matrix: {X.shape}  ({time.time() - t0:.1f}s)")

    if len(X) < 10:
        logger.error("Not enough samples for training. Generate more data.")
        return

    # ------------------------------------------------------------------
    # 3. Train classifier
    # ------------------------------------------------------------------
    classifier = CrackClassifier(cfg)
    classifier.model_type = args.model_type
    metrics = classifier.train(X, y, use_pca=args.use_pca and not args.use_lda,
                               use_lda=args.use_lda)

    # Print results
    logger.info("=" * 60)
    logger.info("Training Results")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if k == "classification_report":
            print(f"\n{v}")
        elif k == "confusion_matrix":
            print(f"Confusion Matrix:\n{np.array(v)}")
        else:
            logger.info(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 4. Save model
    # ------------------------------------------------------------------
    classifier.save(f"crack_classifier_{args.model_type}")
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
