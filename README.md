# CrackVision  
**Intelligent Structural Health Monitoring System using Classical Computer Vision**

CrackVision is a modular, production-ready computer vision framework designed for automated detection, segmentation, feature extraction, depth estimation, and severity analysis of structural cracks. It implements more than ten classical CV modules and supports academic research, university submissions, and real-world inspection workflows.

---

## Features
- Automated crack detection and segmentation  
- Multi-algorithm feature extraction (edges, texture, keypoints)  
- Severity classification using classical ML (SVM, KNN)  
- Stereo-based depth estimation (SGBM)  
- Photometric stereo for surface normals  
- Temporal crack progression tracking  
- Modular, extensible, research-friendly architecture  

---

## Table of Contents
- Overview  
- Motivation  
- Syllabus Coverage  
- System Architecture  
- Installation  
- Quick Start  
- Dataset Preparation  
- Usage Examples  
- Project Structure  
- Results  
- Evaluation Metrics  
- Challenges & Solutions  
- Future Work  
- License  

---

## Overview

CrackVision integrates multiple classical computer vision algorithms into an end-to-end crack analysis pipeline.  
It supports:

- Crack localisation  
- Segmentation and mask refinement  
- Feature extraction (SIFT, ORB, HOG, LBP)  
- Severity scoring using interpretable metrics  
- Depth estimation and surface normal recovery  
- Motion-based crack progression analysis  

The system is robust, modular, and easily extensible for research applications.

---

## Motivation

Structural cracks indicate early deterioration, safety hazards, corrosion, and material failure.  
Manual inspections are often subjective, slow, and challenging in hazardous or remote locations.

CrackVision aims to offer:

- A low-cost automated monitoring solution  
- A scalable and objective evaluation system  
- A practical tool for civil engineering and infrastructure maintenance  

---

## Syllabus Coverage

| Module | Techniques | Directory |
|--------|-----------|-----------|
| Image Processing | Convolution, CLAHE, bilateral, Fourier | `src/preprocessing/` |
| Stereo Vision | SGBM, homography, RANSAC | `src/analysis/depth_analysis.py` |
| Segmentation | Otsu, GrabCut, watershed, mean-shift, CCA | `src/segmentation/` |
| Feature Extraction | SIFT, ORB, Harris, HOG, LBP, Gabor, DWT | `src/feature_extraction/` |
| Motion Analysis | Optical flow, background subtraction | `src/motion/` |
| Pattern Recognition | SVM, KNN, PCA, LDA, GMM | `src/analysis/` |
| Shape-from-X | Photometric stereo | `src/shape_analysis/` |

---

## System Architecture
Input Image(s)
│
▼

Preprocessing (CLAHE, Bilateral, Fourier)
│
Feature Extraction (Edges, Texture, Keypoints)
│
Segmentation (GrabCut, Watershed, Thresholding)
│
Severity Classification (SVM/KNN, PCA/LDA)
│
Depth & Shape Analysis (Stereo SGBM, Photometric Stereo)
│
Visualization & Reporting

---

## Installation

### Prerequisites
- Python 3.10+  
- pip  

### Setup

```bash
git clone <your_repository_link>
cd CV

python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
Quick Start
Run Demo
python scripts/demo.py
Train Models
python scripts/train.py --model-type svm
python scripts/train.py --model-type knn
Run Inference
python scripts/inference.py path/to/image.jpg
Dataset Preparation
Option A: Generate Synthetic Data
python scripts/generate_data.py

Structure:

data/synthetic/
    none/
    minor/
    moderate/
    severe/
    critical/
Option B: Use Public Crack Datasets

Compatible with:

SDNET2018
Crack500
CrackForest
DeepCrack
Usage Examples
Basic Image Processing
from src.preprocessing import ImageLoader, ImageEnhancer
from src.feature_extraction import EdgeDetector
from src.segmentation import CrackSegmenter
from src.analysis import SeverityAnalyzer
Depth & Shape Analysis
from src.analysis.depth_analysis import StereoDepthEstimator
from src.shape_analysis.photometric import PhotometricStereo
Project Structure
CV/
 ├── src/
 │   ├── preprocessing/
 │   ├── feature_extraction/
 │   ├── segmentation/
 │   ├── analysis/
 │   ├── shape_analysis/
 │   ├── motion/
 │   └── visualization/
 ├── scripts/
 ├── data/
 ├── models/
 ├── results/
 ├── reports/
 └── README.md
Results

Output includes:

Crack segmentation masks
Depth maps and normal maps
Feature visualisations
Severity bar charts
Confusion matrices
Evaluation metrics (JSON)
Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Intersection-over-Union
Confusion Matrix

Synthetic dataset baseline:

Accuracy ≈ 0.85
Weighted F1 ≈ 0.84
Challenges & Solutions
Challenge	Solution
Lighting inconsistency	CLAHE + Fourier filtering
Thin crack detection	Multi-scale edge extraction
Noisy segmentation	Morphological refinement
Limited dataset	Synthetic generation + augmentation
Lack of stereo inputs	Simulated stereo + photometric stereo
Future Work
Deep learning segmentation (U-Net, ResNet variants)
Drone-based inspection integration
Real stereo camera interfacing
GPU-accelerated real-time pipeline
Cloud and web dashboard for monitoring
License

This project is released under the MIT License for academic and research use.


---

If you want, I can also generate:

- A shorter README version  
- A professional banner/header for the top  
- Badges (Python version, License, Build, etc.)  
- A CONTRIBUTING.md or project wiki  
