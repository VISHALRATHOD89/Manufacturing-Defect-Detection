# Steel Surface Defect Detection — Computer Vision

An end-to-end computer vision system that detects surface defects in steel sheets using transfer learning, with Grad-CAM explainability and an interactive Gradio demo.

**Dataset:** [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) — real production-line images from Severstal, a major operating steel manufacturer (Cherepovets, Russia). 12,568 images captured by high-frequency line-scan inspection cameras.

## The Problem

This is a **multi-label** classification problem — each image can have zero, one, or multiple defect types present simultaneously (4 defect classes total), unlike standard multi-class classification where categories are mutually exclusive.

## Approach

**Transfer learning:** Since labeled defects are inherently scarce (a well-run factory produces mostly good parts), training from scratch wasn't viable. A pretrained ResNet50 (ImageNet weights) was used as a frozen feature extractor, with only the final classification layer fine-tuned on the steel defect data.

## Key Results

Initial training with standard `BCEWithLogitsLoss` produced misleadingly high accuracy (94-98%) but very poor recall on rare defect classes:

| Defect | Recall (baseline) | Recall (after fix) |
|---|---|---|
| Defect 1 | 0.27 | **0.85** |
| Defect 2 | 0.09 | **0.93** |
| Defect 3 | 0.74 | 0.81 |
| Defect 4 | 0.22 | **0.91** |

**Root cause:** severe class imbalance (defect_2 had only ~2% positive rate) meant the loss function had little incentive to learn rare classes. Fixed by computing per-class `pos_weight` values and applying them to `BCEWithLogitsLoss`, directly penalizing missed rare-class positives during training — not just adjusting the decision threshold after the fact.

**Why recall was prioritized:** in manufacturing QC, a missed defect (false negative) can result in a defective part shipping to a customer — a far costlier outcome than an unnecessary manual re-inspection (false positive).

## Explainability — Grad-CAM

Every prediction is paired with a Grad-CAM heatmap showing which region of the image most influenced the model's decision — critical for building operator trust and enabling fast human verification, rather than a black-box "defective" label.

![Grad-CAM comparison](models/gradcam_comparison.png)

*Note: Grad-CAM heatmaps are normalized per-image, so visual intensity alone can be misleading — always cross-reference with the actual predicted probability (verified: 93.7% for the real defect vs. 4.6% for the clean image shown above).*

## Architecture


**Key components:**
- **`src/dataset.py` / `src/dataloaders.py`** — custom PyTorch `Dataset` and `DataLoader` for loading Severstal images and multi-label targets
- **`src/train.py`** — fine-tunes ResNet50 with class-weighted `BCEWithLogitsLoss`
- **`src/evaluate.py`** — computes per-class recall/precision to validate the imbalance fix
- **`src/gradcam.py`** — generates Grad-CAM heatmaps for model interpretability
- **`app/gradio_app.py`** — Gradio web interface for real-time image upload and inference
