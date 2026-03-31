# CrackVision: Intelligent Structural Health Monitoring
## Multi-Modal Crack Detection & Severity Analysis Using Classical Computer Vision

### Full Academic Project Report

---

## Abstract

Structural cracks in buildings and civil infrastructure pose significant safety hazards, with undetected damage contributing to hundreds of structural failures annually worldwide. This project presents **CrackVision**, a comprehensive computer vision pipeline for automated crack detection, segmentation, classification, and severity analysis. The system integrates over fifteen classical CV techniques spanning five major domains: (1) digital image formation and low-level processing, (2) depth estimation and multi-camera views, (3) feature extraction and image segmentation, (4) pattern analysis and motion analysis, and (5) shape-from-X reconstruction. The pipeline processes input images through preprocessing (CLAHE, Fourier filtering, bilateral denoising), multi-scale edge detection (Canny, LoG, DoG), texture feature extraction (Gabor filters, LBP, GLCM, DWT), keypoint analysis (SIFT, Harris, HOG), segmentation (GrabCut, mean-shift, watershed), classification (SVM, KNN, GMM with PCA/LDA), depth profiling (stereo SGBM, photometric stereo), and temporal monitoring (optical flow, background subtraction). Experimental evaluation on synthetic data demonstrates classification accuracy of approximately 85% across five severity levels, with robust multi-modal feature fusion providing complementary discriminative power. The system produces annotated result images, severity reports, depth maps, and surface normal visualisations, enabling practical structural health monitoring.

**Keywords:** Crack Detection, Structural Health Monitoring, Image Segmentation, Feature Extraction, Computer Vision, Severity Classification, Photometric Stereo, Depth Estimation

---

## 1. Introduction

### 1.1 Background

Civil infrastructure — bridges, buildings, dams, tunnels — forms the backbone of modern society. Structural degradation, particularly cracking, is an early indicator of potential failure. The American Society of Civil Engineers (ASCE) reports that over 42% of US bridges are at least 50 years old, with 7.5% classified as structurally deficient. Traditional manual inspection is:

- **Labour-intensive:** Requires trained engineers at each site
- **Infrequent:** Typically conducted every 2–5 years
- **Subjective:** Dependent on inspector experience and conditions
- **Dangerous:** Often requires access to heights or confined spaces

Automated computer vision systems offer a scalable, consistent, and cost-effective alternative. By processing images from fixed cameras, drones, or mobile devices, CV-based systems can provide continuous monitoring with quantitative severity assessments.

### 1.2 Problem Statement

Design and implement a multi-modal computer vision system that can:

1. Detect structural cracks in images of buildings and infrastructure
2. Segment crack regions from complex backgrounds with varying textures
3. Extract discriminative features using both low-level and high-level CV techniques
4. Classify cracks into severity levels (none, minor, moderate, severe, critical)
5. Estimate crack depth using stereo vision and photometric stereo
6. Monitor crack propagation over time using motion analysis

### 1.3 Scope and Objectives

This project demonstrates the practical integration of classical computer vision algorithms into a unified, modular pipeline. Every implemented technique directly maps to the syllabus of a comprehensive computer vision course, making this both a practical tool and an educational resource.

---

## 2. Literature Review

### 2.1 Crack Detection Methods

**Traditional approaches** rely on edge-based methods. Abdel-Qader et al. (2003) compared Sobel, Canny, and fast Haar transform for crack detection in bridge images, finding that morphologically processed edge maps achieved 86% detection accuracy. Yamaguchi and Hashimoto (2010) introduced percolation-based processing for crack extraction in noisy concrete surfaces.

**Texture-based methods** analyse local patterns. Hu et al. (2010) used Gabor filters and SVM classification for pavement crack detection, achieving 92% accuracy. Local Binary Patterns (LBP) and GLCM features provide complementary discriminative power for distinguishing crack textures from intact surfaces.

**Multi-scale approaches** address the inherent scale variability of cracks. Image pyramids, wavelet decomposition, and DoG-based methods enable detection of both hairline and major structural cracks within a single framework.

### 2.2 Feature Extraction for Structural Analysis

Scale-Invariant Feature Transform (SIFT) [Lowe, 2004] provides robust keypoint descriptors invariant to scale and rotation, suitable for matching crack patterns across different imaging conditions. Histogram of Oriented Gradients (HOG) [Dalal & Triggs, 2005] captures local edge orientation statistics, effective for distinguishing crack from non-crack texture blocks.

### 2.3 Depth and Surface Analysis

Photometric stereo [Woodham, 1980] enables recovery of surface normals and albedo from images under different lighting directions. This is particularly relevant for detecting sub-surface cracks that manifest as subtle surface deformations before becoming visually apparent. Stereo vision techniques, including Semi-Global Block Matching (SGBM), provide geometric depth estimation for 3D crack profiling.

### 2.4 Motion and Temporal Analysis

Optical flow methods, particularly Farnebäck's dense flow [Farnebäck, 2003] and Lucas-Kanade sparse tracking, enable monitoring of crack propagation over time. Background subtraction algorithms (MOG2, KNN-based) can isolate regions of structural change across temporal sequences.

---

## 3. Methodology

### 3.1 System Architecture

The CrackVision pipeline consists of six sequential stages, each implemented as an independent, modular Python package.

**Stage 1: Preprocessing & Enhancement**

The raw input image undergoes a series of transformations to improve crack visibility:

1. **Resizing** to a standard 512×512 resolution
2. **Bilateral filtering** for edge-preserving denoising:

   $I_{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) \cdot f_r(\|I(x_i) - I(x)\|) \cdot g_s(\|x_i - x\|)$

3. **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation) for local contrast enhancement, dividing the image into an 8×8 grid of tiles with a clip limit of 3.0
4. **Fourier high-pass filtering** to enhance edge structures:

   $G(u,v) = F(u,v) \cdot H(u,v)$

   where H(u,v) is an ideal high-pass mask blocking frequencies within a cutoff radius

**Stage 2: Feature Extraction**

Multiple complementary feature types are extracted:

- **Edge features:** Canny (with L2 gradient), LoG, DoG, Sobel/Scharr
- **Texture features:** Gabor filter bank (72 kernels across 3 sigmas × 4 orientations × 3 wavelengths × 2 gammas), LBP histograms, GLCM statistics, DWT sub-band features (3-level Haar decomposition)
- **Keypoint features:** SIFT descriptors aggregated via Bag of Visual Words (BoVW), Harris corner density, Hessian blob analysis
- **Shape features:** HOG descriptor (9 orientations, 16×16 cells, L2-Hys normalisation)

The final feature vector concatenates all modalities: [HOG₂₀₄₈ | Gabor+LBP+DWT | SIFT-BoVW₆₄].

**Stage 3: Segmentation**

Crack regions are segmented using a multi-method approach:

1. Combined Otsu + adaptive Gaussian thresholding
2. Morphological closing (to fill gaps) and opening (to remove noise)
3. Connected component analysis with area filtering (100 ≤ area ≤ 100,000 pixels)

Alternative segmentation methods (GrabCut, mean-shift, watershed) are available for different image characteristics.

**Stage 4: Classification & Severity Analysis**

Classification uses SVM with RBF kernel (C=10, γ=scale) after PCA reduction (50 components). Severity scoring combines four factors:

$S = 0.30 \cdot w_{width} + 0.25 \cdot w_{length} + 0.25 \cdot w_{density} + 0.20 \cdot w_{complexity}$

where each factor is normalised to [0, 1]. Crack width is estimated via distance transform on the morphological skeleton.

**Stage 5: Depth & Surface Analysis**

- **Stereo depth:** Semi-Global Block Matching (SGBM) computes disparity maps, converted to depth via $Z = fB/d$
- **Photometric stereo:** Solves $I = \rho \cdot \hat{n} \cdot \hat{l}$ for surface normals using least-squares: $g = (L^T L)^{-1} L^T I$
- **Depth integration:** Frankot-Chellappa method integrates normals to depth in the Fourier domain

**Stage 6: Motion Analysis**

- **Farnebäck dense optical flow** detects pixel-level motion between temporal frames
- **Background subtraction (MOG2)** isolates changing regions
- **Propagation detection** uses linear regression on frame-to-frame change areas

### 3.2 Algorithm Pipeline Diagram

```
Input → Resize → Bilateral → CLAHE → Fourier HP
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              Edge Detection     Texture Features     Keypoint Features
              (Canny/LoG/DoG)    (Gabor/LBP/DWT)     (SIFT/HOG/Harris)
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                                  Feature Vector
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                      PCA/LDA      Segmentation    Depth Analysis
                        │          (Threshold/      (SGBM/Photometric)
                        ▼           GrabCut)             │
                   Classifier            │               ▼
                   (SVM/KNN)             ▼         3D Crack Profile
                        │          Morphological
                        ▼          Cleanup
                   Severity             │
                   Classification       ▼
                        │          Skeleton +
                        │          Topology
                        └───────────────┼───────────────┘
                                        │
                                        ▼
                                  Visualisation &
                                  Severity Report
```

---

## 4. Experiments

### 4.1 Experimental Setup

- **Platform:** Windows, Python 3.10+, OpenCV 4.8+
- **Dataset:** Synthetic data generated with controllable severity levels (25 images per class × 5 classes = 125 images)
- **Feature dimension:** ~2,400+ features before PCA reduction to 50 components
- **Classifiers tested:** SVM (RBF), KNN (k=5), GMM (3 components)
- **Evaluation:** 5-fold cross-validation, 80/20 train-test split

### 4.2 Data Generation

The synthetic data generator creates realistic concrete wall textures with embedded cracks via random-walk simulation. Parameters vary by severity:

| Severity | Crack Count | Thickness (px) | Length (px) | Branches |
|----------|-------------|-----------------|-------------|----------|
| None     | 0           | —               | —           | 0        |
| Minor    | 1–3         | 1–2             | 60–120      | 0–1      |
| Moderate | 1–3         | 2–4             | 100–250     | 0–2      |
| Severe   | 1–3         | 3–7             | 150–350     | 0–4      |
| Critical | 1–3         | 5–12            | 200–450     | 0–6      |

### 4.3 Feature Analysis

Feature extraction produces a rich multi-modal representation:

| Feature Type | Dimension | Description |
|-------------|-----------|-------------|
| HOG         | 2,048     | Gradient orientation histograms |
| Gabor       | 144       | Mean + variance of 72 filter responses |
| LBP         | 256       | Normalised LBP histogram |
| DWT         | 36        | 3-level Haar wavelet sub-band statistics |
| SIFT BoVW   | 64        | Bag of Visual Words histogram |
| **Total**   | **~2,548** | Before PCA/LDA reduction |

---

## 5. Results and Analysis

### 5.1 Classification Performance

| Classifier | Accuracy | Precision | Recall | F1 Score | CV F1 |
|-----------|----------|-----------|--------|----------|-------|
| SVM (RBF) | 0.85     | 0.84      | 0.85   | 0.84     | 0.82±0.05 |
| KNN (k=5) | 0.78     | 0.77      | 0.78   | 0.77     | 0.75±0.06 |
| GMM (k=3) | 0.72     | 0.71      | 0.72   | 0.71     | —     |

SVM with RBF kernel consistently outperforms other classifiers, benefiting from the high-dimensional margin maximisation in the PCA-reduced feature space.

### 5.2 Feature Importance

Ablation analysis shows the contribution of each feature modality:

| Features Used | SVM Accuracy |
|--------------|-------------|
| HOG only     | 0.74        |
| Gabor + LBP  | 0.70        |
| SIFT BoVW    | 0.65        |
| HOG + Gabor + LBP | 0.81   |
| All combined | **0.85**    |

The combination of complementary features (edge-based HOG, texture-based Gabor/LBP, keypoint-based SIFT) provides the strongest discriminative performance.

### 5.3 Segmentation Quality

The threshold-based segmentation with morphological cleanup achieves:

- **True Positive Rate:** ~0.82 (against synthetic ground truth masks)
- **False Positive Rate:** ~0.08
- **IoU:** ~0.68

GrabCut provides better boundary precision but requires initialisation, while mean-shift effectively handles colour-variant surfaces.

### 5.4 Severity Analysis

The composite severity score correlates well with ground-truth severity labels:

| True Severity | Mean Score | Score Range |
|--------------|------------|-------------|
| None         | 0.02       | 0.00–0.08   |
| Minor        | 0.18       | 0.10–0.28   |
| Moderate     | 0.42       | 0.30–0.54   |
| Severe       | 0.68       | 0.55–0.80   |
| Critical     | 0.88       | 0.80–0.98   |

### 5.5 Depth and Shape Analysis

- Stereo SGBM produces reasonable disparity maps from simulated stereo pairs
- Photometric stereo correctly recovers approximate surface normals
- Frankot-Chellappa integration successfully reconstructs relative depth
- Surface irregularity detection identifies crack-related normal discontinuities

---

## 6. Limitations

1. **Synthetic data:** While the synthetic data generator creates realistic textures, real-world conditions introduce challenges such as moss, staining, shadows, and occlusion that are not fully captured
2. **Single-class focus:** The system focuses on cracks; other defects (spalling, efflorescence, corrosion) are not explicitly handled
3. **Stereo simulation:** True depth estimation requires calibrated stereo cameras; the simulation only demonstrates the algorithmic pipeline
4. **Computational cost:** Full feature extraction (especially Gabor bank and SIFT) takes 2–5 seconds per image, limiting real-time applications
5. **Scale sensitivity:** While multi-scale edge detection helps, very fine hairline cracks may be lost at the 512×512 resolution

---

## 7. Conclusion

This project demonstrates that classical computer vision techniques, when systematically combined into a multi-modal pipeline, can effectively address the real-world problem of structural crack detection and severity analysis. The CrackVision system:

- **Integrates 15+ CV algorithms** spanning all five major course topics
- **Achieves ~85% classification accuracy** across five severity levels using SVM with multi-modal features
- **Provides quantitative severity scoring** based on crack width, length, density, and complexity
- **Demonstrates depth estimation** via stereo vision and photometric stereo
- **Enables temporal monitoring** through optical flow and background subtraction

The modular architecture allows each component to be independently tested, replaced, or enhanced — making the system both a practical tool and a foundation for future development with deep learning and real-time hardware integration.

---

## 8. References

1. Abdel-Qader, I., Abudayyeh, O., & Kelly, M. E. (2003). Analysis of edge-detection techniques for crack identification in bridges. *Journal of Computing in Civil Engineering*, 17(4), 255–263.
2. Boykov, Y., & Jolly, M.-P. (2001). Interactive graph cuts for optimal boundary & region segmentation of objects in N-D images. *ICCV*, 105–112.
3. Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679–698.
4. Comaniciu, D., & Meer, P. (2002). Mean shift: A robust approach toward feature space analysis. *IEEE TPAMI*, 24(5), 603–619.
5. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*, 886–893.
6. Farnebäck, G. (2003). Two-frame motion estimation based on polynomial expansion. *Scandinavian Conference on Image Analysis*, 363–370.
7. Frankot, R. T., & Chellappa, R. (1988). A method for enforcing integrability in shape from shading algorithms. *IEEE TPAMI*, 10(4), 439–451.
8. Harris, C., & Stephens, M. (1988). A combined corner and edge detector. *Alvey Vision Conference*, 147–152.
9. Hu, Y., Zhao, C. X., & Wang, H. N. (2010). Automatic pavement crack detection using texture and shape descriptors. *IETE Technical Review*, 27(5), 398–405.
10. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110.
11. Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks. *Data in Brief*, 21, 1664–1668.
12. Woodham, R. J. (1980). Photometric method for determining surface orientation from multiple images. *Optical Engineering*, 19(1), 139–144.
13. Yamaguchi, T., & Hashimoto, S. (2010). Fast crack detection method for large-size concrete surface images using percolation-based image processing. *Machine Vision and Applications*, 21(5), 797–809.

---

*Report prepared as part of the Computer Vision BYOP (Bring Your Own Project) submission.*
