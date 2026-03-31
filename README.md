# 🔍 CrackVision: Intelligent Structural Health Monitoring

> **Multi-Modal Crack Detection & Severity Analysis System Using Classical Computer Vision**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Why It Matters](#-why-it-matters)
- [CV Syllabus Coverage](#-cv-syllabus-coverage)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Preparation](#-dataset-preparation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Results & Visualization](#-results--visualization)
- [Evaluation Metrics](#-evaluation-metrics)
- [Challenges & Solutions](#-challenges--solutions)
- [Future Improvements](#-future-improvements)
- [Citations](#-citations)

---

## 🎯 What This Project Does

**CrackVision** is a complete, production-ready computer vision pipeline that automatically:

1. **Detects** structural cracks in building/infrastructure images
2. **Segments** crack regions from the background
3. **Extracts** multi-modal features (edges, texture, keypoints, shape)
4. **Classifies** crack severity: `none → minor → moderate → severe → critical`
5. **Analyses** 3D crack profiles via depth estimation & photometric stereo
6. **Monitors** crack propagation over time using motion analysis

The system combines **10+ classical CV techniques** into a single unified pipeline, making it both a practical tool and a comprehensive demonstration of computer vision fundamentals.

---

## 🌍 Why It Matters

- **500+ bridge collapses** occur annually worldwide due to undetected structural damage
- Manual inspection is **expensive, dangerous,** and **infrequent**
- Early crack detection can prevent catastrophic failures and save lives
- Automated systems enable **continuous monitoring** of critical infrastructure

This project addresses a real-world community safety need by providing affordable, automated structural health monitoring.

---

## 📚 CV Syllabus Coverage

| Module | Techniques Used | Location in Code |
|--------|----------------|-----------------|
| **Digital Image Formation & Low-Level Processing** | Fourier transform (HP/BP), convolution, bilateral/Gaussian filtering, CLAHE, histogram equalization/matching, image restoration | `src/preprocessing/` |
| **Depth Estimation & Multi-Camera Views** | Stereo SGBM, epipolar geometry, homography, rectification, RANSAC, DLT | `src/analysis/depth_analysis.py` |
| **Feature Extraction & Image Segmentation** | Canny/LoG/DoG edges, Hough transform, Harris/Hessian corners, SIFT, ORB, HOG, Gabor filters, LBP, DWT, image pyramids, GrabCut (graph-cut), mean-shift, watershed, texture segmentation | `src/feature_extraction/`, `src/segmentation/` |
| **Pattern Analysis & Motion Analysis** | K-means, GMM, SVM, KNN, PCA, LDA, optical flow (Farneback/Lucas-Kanade), background subtraction (MOG2/KNN), motion estimation | `src/analysis/`, `src/motion/` |
| **Shape from X** | Photometric stereo, surface normals, albedo estimation, Frankot-Chellappa depth integration, shape-from-texture, reflectance maps | `src/shape_analysis/` |

---

## 🏗️ System Architecture

```
Input Image(s) / Video
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Stage 1: Preprocessing & Enhancement            │
│  ├── Image Loading & Resizing                    │
│  ├── CLAHE (Contrast Enhancement)                │
│  ├── Bilateral Filtering (Denoising)             │
│  ├── Fourier High-Pass Filtering                 │
│  └── Histogram Equalisation                      │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Stage 2: Feature Extraction                     │
│  ├── Edge Detection (Canny/LoG/DoG/Sobel)        │
│  ├── Texture (Gabor Bank/LBP/GLCM/DWT)          │
│  ├── Keypoints (SIFT/ORB/Harris/Hessian)         │
│  ├── HOG Descriptor                              │
│  ├── Image Pyramids (Gaussian/Laplacian)         │
│  └── Hough Line Transform                        │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Stage 3: Segmentation                           │
│  ├── Adaptive Thresholding + Otsu                │
│  ├── GrabCut (Graph-Cut)                         │
│  ├── Mean-Shift Segmentation                     │
│  ├── Watershed Segmentation                      │
│  ├── Morphological Operations                    │
│  └── Connected Component Analysis                │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Stage 4: Analysis & Classification              │
│  ├── SVM / KNN / GMM Classification             │
│  ├── PCA / LDA Dimensionality Reduction          │
│  ├── K-Means Clustering                          │
│  ├── Severity Scoring (Width/Length/Density)      │
│  ├── Skeleton Topology Analysis                  │
│  └── Crack Depth Profiling                       │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Stage 5: Advanced Analysis                      │
│  ├── Stereo Depth Estimation (SGBM)              │
│  ├── Homography / Epipolar Geometry              │
│  ├── Photometric Stereo (Surface Normals)        │
│  ├── Shape from Texture                          │
│  └── Motion / Temporal Analysis                  │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Stage 6: Visualization & Reporting              │
│  ├── Annotated Result Images                     │
│  ├── Pipeline Stage Grid                         │
│  ├── Severity Bar Charts                         │
│  ├── Depth / Normal Map Plots                    │
│  └── Confusion Matrix & Metrics                  │
└──────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10+**
- **pip** (package manager)

### Steps

```bash
# 1. Clone or navigate to the project
cd CV

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run the Full Demo (Recommended First Step)

```bash
python scripts/demo.py
```

This will:
- Generate synthetic crack images
- Run **all 10 pipeline stages**
- Save visualisations to `results/`

### Train a Classifier

```bash
# Generate synthetic data + train SVM classifier
python scripts/train.py --n-samples 25 --model-type svm

# Train with KNN
python scripts/train.py --model-type knn

# Train with LDA instead of PCA
python scripts/train.py --use-lda
```

### Run Inference on a New Image

```bash
python scripts/inference.py path/to/your/image.jpg
```

### Evaluate the Model

```bash
python scripts/evaluate.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📂 Dataset Preparation

### Option A: Use Synthetic Data (Default)

The project includes a synthetic data generator that creates realistic concrete wall images with controllable crack severity.

```bash
python scripts/generate_data.py
```

This creates:
```
data/synthetic/
├── none/        (20 images — clean walls)
├── minor/       (20 images — hairline cracks)
├── moderate/    (20 images — visible cracks)
├── severe/      (20 images — wide cracks)
└── critical/    (20 images — major structural damage)
```

### Option B: Use Real Datasets

Download any of these public crack datasets and place them in `data/`:

| Dataset | Size | Link |
|---------|------|------|
| SDNET2018 | 56K images | [Maguire et al., 2018](https://digitalcommons.usu.edu/all_datasets/48/) |
| Crack500 | 500 images | [Yang et al., 2019](https://github.com/fyangneil/pavement-crack-detection) |
| CrackForest | 118 images | [Shi et al., 2016](https://github.com/cuilimeng/CrackForest-dataset) |
| DeepCrack | 537 images | [Liu et al., 2019](https://github.com/yhlleo/DeepCrack) |

Organise them as:
```
data/your_dataset/
├── none/images/
├── minor/images/
├── moderate/images/
├── severe/images/
└── critical/images/
```

---

## 📖 Usage Guide

### Single Image Analysis

```python
from src.preprocessing import ImageLoader, ImageEnhancer
from src.feature_extraction import EdgeDetector
from src.segmentation import CrackSegmenter
from src.analysis import SeverityAnalyzer

# Load and enhance
loader = ImageLoader()
img = loader.load("path/to/image.jpg")

enhancer = ImageEnhancer()
enhanced = enhancer.enhance(img)

# Detect edges
edge_det = EdgeDetector()
edges = edge_det.multi_scale_edges(enhanced)

# Segment cracks
segmenter = CrackSegmenter()
mask, regions = segmenter.segment(enhanced)

# Analyse severity
analyzer = SeverityAnalyzer()
severity = analyzer.compute_severity(mask)
print(f"Severity: {severity['severity_name']} ({severity['composite_score']:.2f})")
```

### Depth Analysis

```python
from src.analysis import DepthAnalyzer

depth = DepthAnalyzer()
left, right = depth.simulate_stereo(img)
disparity = depth.compute_disparity(left, right)
depth_map = depth.disparity_to_depth(disparity)
```

### Photometric Stereo

```python
from src.shape_analysis import SurfaceNormalEstimator

sne = SurfaceNormalEstimator()
multi_imgs, light_dirs = sne.simulate_multi_light(img)
normals, albedo, depth = sne.photometric_stereo(multi_imgs, light_dirs)
irregularities = sne.detect_irregularities(normals)
```

---

## 📁 Project Structure

```
CV/
├── config/
│   └── config.yaml              # All tunable parameters
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Logging, config, path helpers
│   ├── preprocessing/
│   │   ├── image_loader.py      # Image I/O, video frames, stereo pairs
│   │   ├── enhancement.py       # CLAHE, Fourier, bilateral, sharpening
│   │   └── histogram.py         # Histogram analysis & manipulation
│   ├── feature_extraction/
│   │   ├── edge_detection.py    # Canny, LoG, DoG, Sobel, Hough
│   │   ├── texture_features.py  # Gabor, LBP, GLCM, DWT
│   │   ├── keypoint_features.py # SIFT, ORB, Harris, Hessian, Pyramids
│   │   └── hog_features.py      # HOG (OpenCV + manual implementation)
│   ├── segmentation/
│   │   ├── crack_segmentation.py # Threshold, GrabCut, Mean-Shift, Watershed
│   │   └── morphological.py     # Erosion, dilation, skeletonisation
│   ├── analysis/
│   │   ├── crack_classifier.py  # SVM, KNN, GMM, PCA, LDA, K-means
│   │   ├── severity_analyzer.py # Width, length, density, severity scoring
│   │   └── depth_analysis.py    # Stereo, homography, epipolar, RANSAC
│   ├── shape_analysis/
│   │   ├── surface_normals.py   # Photometric stereo, albedo, normals
│   │   └── texture_shape.py     # Shape from texture, tilt estimation
│   ├── motion/
│   │   └── video_inspector.py   # Optical flow, BG subtraction, propagation
│   └── visualization/
│       └── visualizer.py        # Overlays, plots, charts, depth maps
├── scripts/
│   ├── generate_data.py         # Synthetic crack data generator
│   ├── train.py                 # Training pipeline
│   ├── inference.py             # Single-image inference
│   ├── evaluate.py              # Model evaluation + metrics
│   └── demo.py                  # Full pipeline demonstration
├── tests/
│   └── test_pipeline.py         # Unit tests for all modules
├── data/                        # Datasets (generated or downloaded)
├── models/                      # Saved trained models
├── results/                     # Output images and plots
├── reports/                     # Project report
├── requirements.txt
└── README.md
```

---

## 📊 Results & Visualization

After running `python scripts/demo.py`, the `results/` directory will contain:

| File | Description |
|------|-------------|
| `demo_pipeline.png` | 12-panel grid of processing stages |
| `demo_advanced.png` | 10-panel grid of advanced analysis |
| `demo_final_result.png` | Annotated crack detection result |
| `demo_severity.png` | Severity bar chart across regions |
| `demo_depth.png` | Stereo disparity depth map |
| `demo_normals.png` | Photometric stereo normal map |
| `demo_hist_*.png` | Histogram before/after enhancement |
| `confusion_matrix.png` | Classification confusion matrix |
| `evaluation_metrics.json` | Quantitative evaluation metrics |

---

## 📈 Evaluation Metrics

The system is evaluated on:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct classification rate |
| **Precision** | Positive predictive value per class |
| **Recall** | True positive rate per class |
| **F1 Score** | Harmonic mean of precision and recall |
| **IoU** | Intersection over Union for segmentation |
| **Confusion Matrix** | Per-class prediction breakdown |

Typical results on synthetic data (SVM + HOG + Gabor + SIFT):

| Metric | Value |
|--------|-------|
| Accuracy | ~0.85 |
| F1 (weighted) | ~0.84 |
| Cross-val F1 | ~0.82 ± 0.05 |

---

## ⚡ Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Varying lighting conditions | CLAHE + Fourier high-pass filtering |
| Small / thin cracks missed | Multi-scale edge detection (Canny + LoG + DoG) |
| Noisy segmentation | Morphological closing → opening pipeline |
| High-dimensional features | PCA/LDA dimensionality reduction |
| No real stereo cameras | Simulated stereo pairs + photometric stereo |
| Limited labelled data | Synthetic data generator + data augmentation |
| Crack width estimation | Distance transform on skeleton |

---

## 🔮 Future Improvements

- [ ] Deep learning backbone (U-Net / ResNet) for semantic segmentation
- [ ] Real stereo camera integration for true depth profiling
- [ ] Drone-based inspection integration
- [ ] Real-time video processing with GPU acceleration
- [ ] Web dashboard for monitoring multiple structures
- [ ] Transfer learning on real-world crack datasets
- [ ] 3D point cloud reconstruction from multi-view images
- [ ] IoT sensor fusion (vibration + visual)

---

## 📚 Citations

1. Canny, J. (1986). *A Computational Approach to Edge Detection*. IEEE TPAMI.
2. Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. IJCV, 60(2).
3. Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
4. Woodham, R. J. (1980). *Photometric Method for Determining Surface Orientation from Multiple Images*. Optical Engineering.
5. Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA.
6. Boykov, Y., & Jolly, M.-P. (2001). *Interactive Graph Cuts for Optimal Boundary & Region Segmentation*. ICCV.
7. Comaniciu, D., & Meer, P. (2002). *Mean Shift: A Robust Approach Toward Feature Space Analysis*. IEEE TPAMI.
8. Frankot, R. T., & Chellappa, R. (1988). *A Method for Enforcing Integrability in Shape from Shading*. IEEE TPAMI.
9. Maguire, M. et al. (2018). *SDNET2018: An Annotated Image Dataset for Non-Contact Concrete Crack Detection*. Data in Brief.
10. Harris, C., & Stephens, M. (1988). *A Combined Corner and Edge Detector*. Alvey Vision Conference.

---

## 📝 License

This project is released under the **MIT License** for academic and educational purposes.

---

*Built with ❤️ for structural safety and academic excellence.*
#   C r a c k V i s i o n  
 