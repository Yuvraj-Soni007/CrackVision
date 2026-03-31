"""
Crack Classification Module
==============================
Classifies detected regions as crack / non-crack and categorises crack types
using classical ML classifiers:
  • SVM (Support Vector Machine)
  • K-Nearest Neighbours (KNN)
  • Gaussian Mixture Model (GMM)
  • PCA / LDA dimensionality reduction

CV Syllabus: Pattern Analysis —
  K-means, GMM, Bayes, KNN, ANN, PCA, LDA, ICA
"""

import os
import pickle
import numpy as np
from typing import Optional, Tuple, List, Dict

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from ..utils import load_config, logger, ensure_dir, project_root


class CrackClassifier:
    """Train and run crack classifiers with dimensionality reduction."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        cls = cfg["classification"]
        self.model_type = cls["model_type"]
        self.test_size = cls["test_size"]
        self.random_state = cls["random_state"]
        self.pca_n = cls["pca_n_components"]

        # SVM params
        self.svm_kernel = cls["svm_kernel"]
        self.svm_c = cls["svm_c"]
        self.svm_gamma = cls["svm_gamma"]

        # KNN params
        self.knn_k = cls["knn_neighbors"]
        self.knn_metric = cls["knn_metric"]

        # GMM params
        self.gmm_n = cls["gmm_n_components"]
        self.gmm_cov = cls["gmm_covariance_type"]

        # Cross-validation
        self.cv_folds = cfg["evaluation"]["cross_validation_folds"]

        self.model_dir = str(project_root() / cfg["paths"]["model_dir"])
        ensure_dir(self.model_dir)

        # Pipeline objects
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.lda: Optional[LDA] = None
        self.classifier = None

    # ------------------------------------------------------------------
    # Classifier factory
    # ------------------------------------------------------------------
    def _make_classifier(self, model_type: Optional[str] = None):
        mt = model_type or self.model_type
        if mt == "svm":
            return SVC(kernel=self.svm_kernel, C=self.svm_c,
                       gamma=self.svm_gamma, probability=True,
                       random_state=self.random_state)
        elif mt == "knn":
            return KNeighborsClassifier(n_neighbors=self.knn_k, metric=self.knn_metric)
        elif mt == "gmm":
            # GMM is generative; wrap it in a simple predict interface later
            return GaussianMixture(n_components=self.gmm_n,
                                   covariance_type=self.gmm_cov,
                                   random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier: {mt}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, y: np.ndarray,
              use_pca: bool = True, use_lda: bool = False,
              model_type: Optional[str] = None) -> Dict:
        """Train a classifier on feature matrix X and labels y.

        Returns a dict of evaluation metrics.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Standardise
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Dimensionality reduction
        if use_lda and len(np.unique(y)) > 1:
            n_comp = min(self.pca_n, len(np.unique(y)) - 1, X_train_s.shape[1])
            self.lda = LDA(n_components=n_comp)
            X_train_s = self.lda.fit_transform(X_train_s, y_train)
            X_test_s = self.lda.transform(X_test_s)
            logger.info(f"LDA reduction → {n_comp} components")
        elif use_pca:
            n_comp = min(self.pca_n, X_train_s.shape[0], X_train_s.shape[1])
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
            X_train_s = self.pca.fit_transform(X_train_s)
            X_test_s = self.pca.transform(X_test_s)
            ev = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA reduction → {n_comp} components, explained variance={ev:.3f}")

        # Train classifier
        mt = model_type or self.model_type
        self.classifier = self._make_classifier(mt)

        if mt == "gmm":
            self.classifier.fit(X_train_s)
            y_pred = self.classifier.predict(X_test_s)
        else:
            self.classifier.fit(X_train_s, y_train)
            y_pred = self.classifier.predict(X_test_s)

        # Cross-validation (for non-GMM)
        cv_scores = None
        if mt != "gmm":
            cv_scores = cross_val_score(
                self._make_classifier(mt), X_train_s, y_train, cv=self.cv_folds, scoring="f1_weighted"
            )

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
        }
        if cv_scores is not None:
            metrics["cv_f1_mean"] = float(cv_scores.mean())
            metrics["cv_f1_std"] = float(cv_scores.std())

        logger.info(f"Training complete ({mt}): accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        return metrics

    # ------------------------------------------------------------------
    # K-Means unsupervised clustering
    # ------------------------------------------------------------------
    def kmeans_cluster(self, X: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """K-means clustering on the feature matrix.

        Returns (labels, cluster_centers).
        """
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X_s)
        logger.info(f"K-means: {n_clusters} clusters, inertia={km.inertia_:.2f}")
        return labels, km.cluster_centers_

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new feature vectors."""
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        X_s = self.scaler.transform(X)
        if self.lda is not None:
            X_s = self.lda.transform(X_s)
        elif self.pca is not None:
            X_s = self.pca.transform(X_s)
        return self.classifier.predict(X_s)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (SVM/KNN with probability=True)."""
        if self.classifier is None:
            raise RuntimeError("Classifier not trained.")
        X_s = self.scaler.transform(X)
        if self.lda is not None:
            X_s = self.lda.transform(X_s)
        elif self.pca is not None:
            X_s = self.pca.transform(X_s)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_s)
        return np.array([])

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, name: str = "crack_classifier"):
        """Persist the trained pipeline to disk."""
        data = {
            "scaler": self.scaler,
            "pca": self.pca,
            "lda": self.lda,
            "classifier": self.classifier,
            "model_type": self.model_type,
        }
        path = os.path.join(self.model_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved → {path}")

    def load(self, name: str = "crack_classifier"):
        """Load a previously saved pipeline."""
        path = os.path.join(self.model_dir, f"{name}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.lda = data["lda"]
        self.classifier = data["classifier"]
        self.model_type = data["model_type"]
        logger.info(f"Model loaded ← {path}")
