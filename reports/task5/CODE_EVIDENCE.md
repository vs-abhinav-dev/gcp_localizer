# Task 5 Report Code Evidence Guide (`CODE_EVIDENCE.md`)

This guide highlights the **5 most critical source code locations** in the repository suitable for inclusion as code snippets or evidence in the **CSE411 Computer Vision Project Task 5 Report**.

> [!IMPORTANT]
> Do NOT copy-paste whole source files into the main report text. Use the exact file references, line ranges, and recommended captions below to reference code cleanly.

---

## 1. GCP Localizer Model Architecture (Regression Head)

- **File Path**: `src/gcp_localizer/models/localizer.py`
- **Class Name**: `GCPLocalizer`
- **Line Range**: Lines 6–31
- **What It Demonstrates**: Defines the fine-tuned ResNet18 regression architecture. Replaces the standard classification head with a `Linear(512 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> 2)` sequential block to output relative `(x, y)` tile coordinates.
- **Recommended Report Caption**: *Listing 1: PyTorch model definition of GCPLocalizer with ResNet18 backbone and custom 2-coordinate regression output head.*

---

## 2. GCP Marker Presence Classifier Model

- **File Path**: `src/gcp_localizer/models/classifier.py`
- **Class Name**: `GCPClassifier`
- **Line Range**: Lines 6–26
- **What It Demonstrates**: ResNet18 transfer-learning architecture with a modified `Linear(in_features, num_classes)` head used for Stage 1 binary GCP presence filtering (GCP present vs background).
- **Recommended Report Caption**: *Listing 2: PyTorch model definition of GCPClassifier for binary marker presence detection.*

---

## 3. Fast Crop Dataset Loader & Normalization

- **File Path**: `src/gcp_localizer/data/datasets.py`
- **Class / Function**: `FastGCPDataset` & `preprocess_image_tensor`
- **Line Range**: Lines 11–36
- **What It Demonstrates**: High-performance dataset loader reading compressed `.npz` tiles, converting RGB numpy arrays to float tensors, and applying standard ImageNet mean/std normalization.
- **Recommended Report Caption**: *Listing 3: FastGCPDataset implementation for loading preprocessed aerial crop tiles and target GCP coordinates.*

---

## 4. Euclidean Localization Error Metric Computation

- **File Path**: `src/gcp_localizer/evaluation/metrics.py`
- **Functions**: `compute_localization_error` & `evaluate_shape_maes`
- **Line Range**: Lines 10–32, 43–77
- **What It Demonstrates**: Mathematical implementation of 2D Euclidean distance calculation $d = \sqrt{(x_{pred} - x_{gt})^2 + (y_{pred} - y_{gt})^2}$ in pixel space, and automatic aggregation of Mean Absolute Error (MAE) and median error broken down by marker shape type (L-Shape, Square, Cross).
- **Recommended Report Caption**: *Listing 4: Quantitative evaluation routines for calculating Euclidean pixel localization error and shape-wise MAE breakdown.*

---

## 5. Integrated Two-Stage Inference Pipeline Execution

- **File Path**: `src/gcp_localizer/evaluation/visualization.py`
- **Function**: `plot_end_to_end_pipeline`
- **Line Range**: Lines 244–299
- **What It Demonstrates**: Demonstrates the execution flow of the integrated system: Candidate crop tile $\rightarrow$ Stage 1 GCP Presence Classifier $\rightarrow$ Softmax Confidence Check $\rightarrow$ Stage 2 GCP Center Regression $\rightarrow$ Coordinate Error Overlay.
- **Recommended Report Caption**: *Listing 5: End-to-end integration logic executing candidate tile classification filter followed by localizer regression.*
