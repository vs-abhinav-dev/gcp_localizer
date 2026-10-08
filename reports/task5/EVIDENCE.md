# Task 5 Report Evidence Mapping (`EVIDENCE.md`)

This document maps all generated metric files, plots, diagrams, and figures in `reports/task5/` to the corresponding figures and tables required for the **CSE411 Computer Vision Project Task 5 – Implementation Part 2 Report**.

---

## Report Evidence & Figure Mapping Table

| Figure / Artifact ID | File Path | Type | Summary & Report Usage |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `reports/task5/diagrams/current_pipeline.mmd` | Mermaid Diagram | **Current Implemented Pipeline Architecture**: Illustrates the end-to-end flow from raw images to candidate crop generation (512x512 to 224x224), GCP presence classification filter, and localizer coordinate regression. |
| **Figure 2** | `reports/task5/diagrams/full_proposed_pipeline.mmd` | Mermaid Diagram | **Full Proposed Pipeline Architecture**: Clearly demarcation of completed modules (Presence Classifier + Localizer) versus future work (3-Class GCP Shape Classifier). |
| **Table 1** | `reports/task5/model_summary.txt` | Text Report | **Model & Pipeline Hyperparameters**: Detailed table of preprocessing dimensions (512x512 crop -> 224x224 input), model backbones (ResNet18), output heads, loss functions (MAE L1Loss & CrossEntropyLoss), learning rate (1e-4), batch size (32), and epochs. |
| **Table 2** | `reports/task5/dataset_statistics.txt` <br> `reports/task5/dataset_statistics.json` | Text / JSON | **Dataset & Data Leakage Statistics**: Summarizes original annotated image counts (1000 total: 850 train, 150 val), processed tile counts (4250 pos/neg train, 150 pos/neg val, 8800 total), and verifies zero leakage (`train source images ∩ val source images = ∅`). |
| **Table 3** | `reports/task5/presence_metrics.txt` <br> `reports/task5/presence_metrics.json` | Text / JSON | **GCP Presence Classification Performance**: Held-out validation evaluation metrics: Accuracy (100.00%), Precision (100.00%), Recall (100.00%), F1-Score (1.0000), and full confusion matrix (150 TN, 150 TP). |
| **Figure 3** | `reports/task5/figures/presence/presence_confusion_matrix.png` | PNG Chart | **Classifier Confusion Matrix Plot**: Visual confusion matrix displaying true vs predicted positive and negative GCP crop samples. |
| **Figure 4** | `reports/task5/figures/presence/presence_examples.png` | PNG Grid | **GCP Presence Classification Visual Examples**: Demonstrates correct positive detections (TP) with high confidence scores and correct background rejections (TN). |
| **Table 4** | `reports/task5/localization_metrics.txt` <br> `reports/task5/localization_metrics.json` | Text / JSON | **Localization Error Metrics**: Held-out validation metrics (150 samples): Overall MAE = 1.2674 px, Median Error = 0.9313 px, Max Error = 24.4557 px, Min Error = 0.0888 px, and per-shape MAE breakdown (L-Shape: 1.0602 px, Square: 1.1633 px, Cross: 1.9050 px). |
| **Figure 5** | `reports/task5/figures/localization/localization_grid.png` | PNG Grid | **Qualitative Localization Predictions Grid**: 3x3 grid showing crop images with ground-truth centers (lime green dots) and model predicted centers (red crosses) alongside pixel errors. |
| **Figure 6** | `reports/task5/figures/localization/localization_example_01.png` to `06.png` | PNG Visuals | **Individual Localization Examples**: High-resolution detailed figures demonstrating pin-point center predictions on aerial marker crops. |
| **Figure 7** | `reports/task5/figures/integrated/integrated_examples.png` | PNG Pipeline | **Integrated Two-Stage Inference Flow**: Demonstrates the active 2-stage inference pipeline on positive and negative candidate tiles, showcasing candidate tile -> classifier filtering -> localizer center prediction. |
| **Figure 8** | `reports/task5/figures/error_analysis/localization_failures.png` | PNG Analysis | **Localization Failure Mode Analysis**: Visualizes the top 4 maximum-error validation cases to analyze challenging illumination, partial marker occlusion, or blur conditions. |
| **Figure 9** | `reports/task5/figures/error_analysis/presence_failures.png` | PNG Analysis | **Presence Classifier Error Analysis**: Visualizes false positives / false negatives (or confirms 0 misclassifications on held-out split). |
| **Appendix A** | `reports/task5/figures/training/training_note.txt` | Text Document | **Training History Note**: Documents checkpoint status and notes training loss curve availability. |
| **Appendix B** | `reports/task5/integration_test.txt` | Text Log | **Automated Pipeline Integration Test Log**: Verifies 4/4 passing smoke tests (`dataset loading`, `presence classification`, `localization`, `integrated inference`). |

---

## Recommended Report Placement

1. **Section 1 — System Architecture**: Insert **Figure 1** (`current_pipeline.mmd`) and **Figure 2** (`full_proposed_pipeline.mmd`).
2. **Section 2 — Experimental Setup & Dataset**: Insert **Table 1** (`model_summary.txt`) and **Table 2** (`dataset_statistics.txt`).
3. **Section 3 — GCP Presence Classification Results**: Insert **Table 3** (`presence_metrics.txt`), **Figure 3** (`presence_confusion_matrix.png`), and **Figure 4** (`presence_examples.png`).
4. **Section 4 — GCP Localization Results**: Insert **Table 4** (`localization_metrics.txt`), **Figure 5** (`localization_grid.png`), and **Figure 6** (`localization_example_01.png`).
5. **Section 5 — Integrated Two-Stage Inference**: Insert **Figure 7** (`integrated_examples.png`).
6. **Section 6 — Error Analysis & Discussion**: Insert **Figure 8** (`localization_failures.png`) and **Figure 9** (`presence_failures.png`).
7. **Section 7 — Verification & Integration Tests**: Insert **Appendix B** (`integration_test.txt`).
