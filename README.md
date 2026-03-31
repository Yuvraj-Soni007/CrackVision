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
