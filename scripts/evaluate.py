"""
Evaluation Script
==================
Comprehensive evaluation of the crack detection pipeline on a test dataset.
Generates metrics, confusion matrices, and comparison tables.
"""

import os
import sys
import argparse
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, logger, project_root, ensure_dir
from src.preprocessing import ImageLoader, ImageEnhancer
from src.feature_extraction import HOGExtractor, TextureExtractor, KeypointExtractor
from src.analysis import CrackClassifier
from scripts.train import extract_features

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)


def evaluate_model(data_dir: str, model_name: str = "crack_classifier_svm"):
    """Evaluate a trained model on the full dataset (or a hold-out set)."""
    cfg = load_config()
    results_dir = ensure_dir(str(project_root() / cfg["paths"]["results_dir"]))

    loader = ImageLoader(cfg)
    enhancer = ImageEnhancer(cfg)
    hog_ext = HOGExtractor(cfg)
    tex_ext = TextureExtractor(cfg)
    kp_ext = KeypointExtractor(cfg)

    severity_map = {"none": 0, "minor": 1, "moderate": 2, "severe": 3, "critical": 4}

    # Load all data and extract features
    X_list = []
    y_true = []
    filenames = []

    logger.info("Loading and extracting features for evaluation...")
    t0 = time.time()

    for severity_name, label in severity_map.items():
        img_dir = os.path.join(data_dir, severity_name, "images")
        if not os.path.exists(img_dir):
            continue

        images = loader.load_directory(img_dir)
        for fname, img in images:
            try:
                feat = extract_features(img, enhancer, hog_ext, tex_ext, kp_ext)
                X_list.append(feat)
                y_true.append(label)
                filenames.append(fname)
            except Exception as e:
                logger.warning(f"Skipping {fname}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_true, dtype=np.int32)
    logger.info(f"Evaluation set: {len(X)} samples  ({time.time() - t0:.1f}s)")

    # Load model and predict
    classifier = CrackClassifier(cfg)
    classifier.load(model_name)
    y_pred = classifier.predict(X)

    # Metrics
    class_names = list(severity_map.keys())
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names, zero_division=0)

    # Print results
    print("\n" + "=" * 60)
    print("  CrackVision — Evaluation Results")
    print("=" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    print("=" * 60)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix (Acc={acc:.3f}, F1={f1:.3f})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → confusion_matrix.png")

    # Save metrics JSON
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "n_samples": len(X),
        "model": model_name,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → evaluation_metrics.json")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CrackVision — Evaluation")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to evaluation dataset")
    parser.add_argument("--model", type=str, default="crack_classifier_svm",
                        help="Saved model name")
    args = parser.parse_args()

    data_dir = args.data_dir or str(project_root() / "data" / "synthetic")
    evaluate_model(data_dir, args.model)


if __name__ == "__main__":
    main()
