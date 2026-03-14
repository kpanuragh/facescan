# Clinical-Grade rPPG Model with Multi-Metric Estimation

**Date:** 2026-03-14
**Project:** facescan_model
**Status:** Design Document
**Objective:** Build a clinically-relevant rPPG methodology for estimating multiple health biomarkers (heart rate, respiratory rate, SpO2) from facial video, publishable for medical research community adoption.

---

## Executive Summary

This project develops a **shared feature extractor with specialized task heads** architecture for extracting heart rate, respiratory rate, and blood oxygen saturation from facial video using remote photoplethysmography (rPPG). The model outputs not just biomarker values but also confidence scores, enabling clinical use with explicit reliability signals. Training leverages the MCD-rPPG dataset (3600 videos, 600 subjects, full ground truth). The methodology will be open-sourced and published in IEEE/ACM venues, enabling medical institutions with validation capabilities to adopt and clinically validate the approach.

---

## Problem Statement

**Current Landscape:**
- rPPG research has produced good results for single metrics (mainly heart rate)
- Existing public datasets and models predominantly feature light-skinned subjects (demographic bias)
- Clinical adoption of rPPG is limited by:
  - Lack of multi-parameter estimation from single video
  - Absence of confidence/reliability signals
  - Unclear methodology for researchers to validate and build upon
- No clinically-validated rPPG system currently available without proprietary hardware/software

**Gap:**
Building an *open, reproducible, multi-metric rPPG methodology* that the medical community can validate, publish about, and build clinical products upon.

---

## Architecture Design

### Core Concept

**Three-component system:**

```
Video Input (30fps facial video)
    ↓
[Face ROI Extraction] → Preprocessed frames
    ↓
[Shared Feature Extractor] → Rich physiological features (128-256 dims)
    ├→ [HR Head] → Heart Rate (bpm) + Confidence
    ├→ [RR Head] → Respiratory Rate (breaths/min) + Confidence
    └→ [SpO2 Head] → Oxygen Saturation (%) + Confidence
    ↓
Output: 3 biomarkers with confidence scores
```

### Component Details

#### 1. Shared Feature Extractor (Base Network)

**Purpose:** Learn generalizable facial features capturing physiological signals relevant to all three metrics.

**Architecture:**
- Input: 30 frames of 128×128 RGB face ROI
- Stack: Spatial-temporal CNN (inspired by rPPG conventions)
  - 3D convolutions to capture motion + color changes
  - Batch normalization + ReLU activations
  - Max pooling for feature compression
- Output: Feature vector (256 dimensions)

**Why Shared:**
- HR, RR, and SpO2 all originate from blood flow dynamics in facial skin
- Shared feature extraction forces the network to learn unified physiological representations
- Multi-task learning acts as regularization (prevents overfitting)

#### 2. Task-Specific Prediction Heads

Each head is lightweight (2-3 fully connected layers):

**HR Head (Heart Rate):**
- Input: 256-dim feature vector
- Layers: FC(256→128) → ReLU → FC(128→2)
- Output: [hr_value, hr_confidence]
- Unit: bpm (beats per minute)

**RR Head (Respiratory Rate):**
- Input: 256-dim feature vector
- Layers: FC(256→128) → ReLU → FC(128→2)
- Output: [rr_value, rr_confidence]
- Unit: breaths per minute

**SpO2 Head (Oxygen Saturation):**
- Input: 256-dim feature vector
- Layers: FC(256→128) → ReLU → FC(128→2)
- Output: [spo2_value, spo2_confidence]
- Unit: percentage (90-100%)

**Confidence Score:**
- Sigmoid activation constrains to [0, 1]
- Trained using uncertainty estimation (calibration loss)
- Higher score = network is more certain of prediction

#### 3. Multi-Task Loss Function

```
Total Loss = α·L_HR + β·L_RR + γ·L_SpO2 + λ·L_confidence

where:
  L_HR = MSE(predicted_hr, ground_truth_hr)
  L_RR = MSE(predicted_rr, ground_truth_rr)
  L_SpO2 = MSE(predicted_spo2, ground_truth_spo2)
  L_confidence = calibration loss (to align confidence with accuracy)
  α, β, γ = task weights (initially equal, tunable)
  λ = confidence calibration weight
```

**Why Multi-Task:**
- Shared gradients help base network learn better features
- Tasks regularize each other (prevent overfitting)
- More efficient than 3 separate models

---

## Data Pipeline

### Dataset: MCD-rPPG

**Source:** [Gaze into the Heart](https://huggingface.co/datasets/kyegorov/mcd_rppg)

**Specifications:**
- 3600 synchronized video recordings
- 600 unique subjects
- Multiple camera angles
- Conditions: resting + post-exercise
- Ground truth: ECG (for HR), respiratory signal, SpO2, blood pressure, stress level

**Video Specs:**
- Resolution: Consumer-grade (varies, typically 640×480 to 1920×1080)
- Frame rate: 30 fps
- Duration: 2-5 minutes per video
- Format: RGB color

### Preprocessing Pipeline

1. **Face Detection & ROI Extraction**
   - Use OpenCV/MediaPipe to detect facial landmarks
   - Extract face bounding box with padding
   - Resize to 128×128 for consistency
   - Store ROI coordinates for reproducibility

2. **Normalization**
   - Subtract per-frame channel mean (per subject, to remove illumination bias)
   - Divide by per-frame channel std dev
   - Clamp to [-5, 5] to handle outliers

3. **Temporal Windowing**
   - Segment videos into 30-frame windows (1 second @ 30fps)
   - Overlap: 50% (for data augmentation and continuous monitoring)
   - Result: ~2-3 windows per second of video

4. **Ground Truth Extraction**
   - **HR:** Extract from ECG signal in MCD-rPPG
     - Compute instantaneous HR every 1 second via peak detection
     - Average over the 30-frame window
   - **RR:** Extract from respiratory signal
     - Detect breathing cycles from synchronized respiratory trace
     - Count cycles per minute
   - **SpO2:** Extract from synchronized PPG/oximetry
     - Use ground truth directly from MCD-rPPG metadata

### Train/Validation/Test Split

```
Total: 3600 videos from 600 subjects

Training:   70% = 2520 videos (~420 subjects)
Validation: 15% = 540 videos (~90 subjects)
Test:       15% = 540 videos (~90 subjects)

Strategy: Subject-level split (not frame-level)
Reason: Prevents overfitting to individual subject characteristics
```

### Data Augmentation

**During training:**
- Random horizontal flip (simulate camera angle variation)
- Random brightness/contrast adjustment (simulate lighting variation)
- Random frame skipping (simulate variable frame rate)
- Temporal jittering (±1 frame shift)

**Not applied to test set** (to ensure clean evaluation)

---

## Training Procedure

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 32 | Balance GPU memory vs. convergence |
| **Learning Rate** | 1e-4 | Standard for medical signal tasks; use Adam decay |
| **Optimizer** | Adam | Works well for multi-task learning |
| **Epochs** | Up to 150 | Train until validation plateau |
| **Early Stopping** | 15 epochs patience | Prevent overfitting |
| **Loss Weights** | α=β=γ=1.0 initially | Equal importance; tune if needed |

### Training Loop

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        frames, hr_true, rr_true, spo2_true = batch

        # Forward pass
        features = feature_extractor(frames)
        hr_pred, hr_conf = hr_head(features)
        rr_pred, rr_conf = rr_head(features)
        spo2_pred, spo2_conf = spo2_head(features)

        # Compute loss
        loss = (
            mse(hr_pred, hr_true) + kl_div(hr_conf, mae(hr_pred, hr_true)) +
            mse(rr_pred, rr_true) + kl_div(rr_conf, mae(rr_pred, rr_true)) +
            mse(spo2_pred, spo2_true) + kl_div(spo2_conf, mae(spo2_pred, spo2_true))
        )

        # Backprop
        loss.backward()
        optimizer.step()

    # Validation
    val_loss, val_metrics = validate(model, val_loader)
    if val_loss improved:
        save_checkpoint()
    if patience_counter > 15:
        break
```

### Checkpointing & Reproducibility

- Save best model by validation loss
- Save training logs (loss curves, metrics over epochs)
- Save hyperparameter config (JSON)
- Use fixed random seeds for reproducibility
- Log code version (git commit hash)

---

## Evaluation & Validation Strategy

### Test Set Evaluation

**For each metric (HR, RR, SpO2), compute:**

1. **Mean Absolute Error (MAE)** — primary clinical metric
   - HR: target < 2 bpm
   - RR: target < 2 breaths/min
   - SpO2: target < 1%

2. **Root Mean Squared Error (RMSE)**
   - Penalizes large errors (outliers)
   - Compare against MAE to detect outliers

3. **Pearson Correlation Coefficient**
   - Measures signal tracking quality
   - Target: > 0.95 for clinical-grade

4. **Bland-Altman Agreement**
   - Clinical standard for measurement comparison
   - Plot: difference vs. average
   - Compute: mean difference ± 1.96 std dev

### Confidence Calibration

**Validate that confidence scores are meaningful:**

1. Partition test set by predicted confidence:
   - High confidence: predictions where confidence > 0.9
   - Medium confidence: 0.5 < confidence < 0.9
   - Low confidence: confidence < 0.5

2. Measure actual MAE in each partition:
   - High confidence should have lowest MAE
   - Low confidence should have highest MAE
   - Compute Expected Calibration Error (ECE)

3. Visualize: Confidence vs. Actual Error (scatter plot)
   - Should show positive correlation

### Demographic Bias Analysis

**Address clinical inclusivity:**

1. Annotate test set with Fitzpatrick skin tone
2. Compute metrics separately for each skin tone group:
   - Compare MAE across groups
   - Flag if any group has >2× worse error
3. Report: table with per-group metrics

**Example output:**
```
Fitzpatrick I-II:   HR MAE = 1.2 bpm,  RR MAE = 0.8,  SpO2 MAE = 0.5%
Fitzpatrick III-IV: HR MAE = 1.5 bpm,  RR MAE = 1.1,  SpO2 MAE = 0.7%
Fitzpatrick V-VI:   HR MAE = 1.8 bpm,  RR MAE = 1.4,  SpO2 MAE = 0.9%
```

### Physiological Consistency Check

**Validate multi-metric plausibility:**

1. Extract post-exercise videos from test set
2. During exercise, expect:
   - HR increases
   - RR increases
   - SpO2 may decrease slightly
3. Compute correlation between predicted metrics:
   - Expected: positive correlation (HR ↑ → RR ↑)
   - Flag if predictions violate physiology

### Ablation Study

**Demonstrate scientific novelty:**

1. Train three single-task models:
   - Model_HR: only predicts heart rate
   - Model_RR: only predicts respiratory rate
   - Model_SpO2: only predicts oxygen saturation

2. Compare against multi-task model:
   - Table: single-task MAE vs. multi-task MAE
   - Expected: multi-task performs ≥ single-task (benefit of shared features)
   - Metric: % improvement

3. Publish results:
   - Demonstrates that multi-task learning + confidence scoring is novel contribution

### Final Output Report

Structure:
```
1. Summary Table
   - MAE, RMSE, Pearson r for each metric

2. Bland-Altman Plots (3 plots, one per metric)

3. Per-Subject Performance
   - Mean ± std error across 90 test subjects

4. Confidence Calibration Plot
   - Confidence vs. actual error

5. Demographic Bias Analysis
   - Table of metrics per Fitzpatrick group

6. Ablation Results
   - Single-task vs. multi-task comparison

7. Failure Case Analysis
   - Videos where prediction failed (lighting, motion, etc.)
```

---

## Web Deployment

### Application Architecture

```
┌─────────────────────────────────────────┐
│         Browser (Frontend)              │
│  React/Vue + WebRTC + HTML5 Canvas      │
│  • Webcam capture                       │
│  • Real-time display                    │
│  • Results visualization                │
└────────────────┬────────────────────────┘
                 │ HTTP/WebSocket
┌────────────────▼────────────────────────┐
│      Python Backend (FastAPI)           │
│  • Model inference (GPU/CPU)            │
│  • Frame processing                     │
│  • Result formatting                    │
└────────────────┬────────────────────────┘
                 │
         ┌───────▼───────┐
         │  Trained Model│
         │  (PyTorch)    │
         └───────────────┘
```

### Processing Pipeline

1. **Capture:** Get 30-frame buffer from webcam (1 second)
2. **Preprocess:** Extract face ROI, normalize
3. **Inference:** Pass through network (< 100ms on CPU)
4. **Output:** HR, RR, SpO2 + confidence scores
5. **Display:** Update UI with results + confidence bars
6. **History:** Keep 60-second rolling history, plot trends

### Frontend Display

- **Large readouts:** Heart Rate, Respiratory Rate, SpO2
- **Confidence indicators:** Color-coded (green = high, yellow = medium, red = low)
- **Trend graphs:** 60-second history for each metric
- **Status messages:** "Initializing...", "Ready", "Processing", etc.
- **Disclaimers:** "For research/wellness use only. Not for medical diagnosis."

### Model Export & Compatibility

- **Primary format:** PyTorch (.pt)
- **Secondary format:** ONNX (cross-platform inference)
- **Size:** ~4-10 MB (lightweight for deployment)
- **Inference time:** ~50-100ms per 30-frame window on CPU
- **Requirements:** Python 3.8+, PyTorch 1.9+, OpenCV

---

## Publication & Open-Source Strategy

### Target Venues

**Primary:**
- IEEE Journal of Biomedical and Health Informatics (IEEE JBHI)
- ACM Transactions on Computing for Healthcare (TOCH)
- Conference: IEEE EMBS or ACM International Conference on Multimedia

**Parallel:**
- arXiv preprint (for early visibility and community feedback)
- GitHub release with full code + trained models

### Paper Outline

1. **Introduction**
   - rPPG potential for non-contact vital monitoring
   - Gap: lack of multi-metric, open methodology
   - Contribution: shared-feature multi-task rPPG model with confidence

2. **Related Work**
   - rPPG methods (chrominance, motion, hybrid)
   - Existing datasets and their limitations
   - Multi-task learning in medical applications

3. **Methodology**
   - Architecture (shared extractor + 3 heads)
   - Multi-task learning approach
   - Confidence calibration for clinical use

4. **Experiments**
   - Dataset description (MCD-rPPG)
   - Training details
   - Ablation study (single-task vs. multi-task)
   - Demographic bias analysis

5. **Results**
   - Test set performance (MAE, RMSE, Pearson r)
   - Confidence calibration results
   - Bland-Altman plots
   - Comparison with prior rPPG methods (if applicable)

6. **Discussion**
   - What worked, what didn't
   - Limitations (e.g., demographic bias still exists)
   - Path to clinical validation (for others)
   - Future work

7. **Open-Source Release**
   - GitHub link
   - HuggingFace model link
   - Reproducibility instructions

### Public Deliverables

1. **GitHub Repository**
   - Training code + notebooks
   - Preprocessing utilities
   - Model architecture definitions
   - Web app (frontend + backend)
   - README with setup + usage
   - License: MIT or Apache 2.0

2. **HuggingFace Hub**
   - Trained model weights (PyTorch)
   - ONNX export
   - Model card with limitations/use cases
   - Inference code

3. **Documentation**
   - Blog post: "Why Open rPPG Matters for Healthcare"
   - Technical report: methodology details + validation results
   - Video demo: 2-3 min showing live usage

---

## Timeline & Milestones

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data prep & exploration | 2 weeks | Preprocessed dataset, data statistics |
| Model development | 3 weeks | Trained architecture, checkpoints |
| Evaluation & analysis | 2 weeks | Test results, demographic bias analysis |
| Ablation studies | 1 week | Single-task comparison |
| Web app development | 2 weeks | Working web interface |
| Documentation & publication | 2 weeks | Paper draft, code comments, README |
| **Total** | **~12 weeks** | Publishable model + working web app |

---

## Success Criteria

✅ **Model Performance:**
- HR: MAE < 3 bpm, Pearson r > 0.93 on test set
- RR: MAE < 2 breaths/min, Pearson r > 0.90
- SpO2: MAE < 1%, Pearson r > 0.88

✅ **Confidence Calibration:**
- High-confidence predictions have 2× lower error than low-confidence
- Expected Calibration Error (ECE) < 0.05

✅ **Demographic Inclusivity:**
- No group has >2× worse performance than best-performing group
- Results reported per Fitzpatrick skin tone

✅ **Reproducibility:**
- Code is fully open-source with setup instructions
- Ablation study demonstrates novelty
- All hyperparameters documented

✅ **Deployment:**
- Web app runs in browser
- Inference < 100ms per frame
- User gets real-time readings + confidence scores

✅ **Publication:**
- Paper submitted to tier-1 venue (IEEE JBHI, TOCH, or conference)
- Preprint on arXiv
- Code on GitHub with documentation

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| SpO2 estimation poor | Medium | High | Focus on color-based signal; may need synthetic augmentation |
| Demographic bias persists | High | Medium | Report honestly; publish as limitation; propose future work |
| Model overfits to MCD-rPPG | Medium | High | Use ablation, cross-dataset validation, strong regularization |
| Web app latency issues | Low | Medium | Optimize inference, cache frames, consider GPU inference |
| Confidence scores poorly calibrated | Medium | Medium | Use explicit calibration loss; validate on separate hold-out set |

---

## Files & Directory Structure (Post-Implementation)

```
facescan_model/
├── docs/
│   └── superpowers/specs/
│       └── 2026-03-14-rppg-clinical-methodology-design.md (this file)
├── data/
│   ├── raw/                 (MCD-rPPG downloaded files)
│   ├── processed/           (preprocessed ROI frames + ground truth)
│   └── splits/              (train/val/test split metadata)
├── models/
│   ├── architecture.py      (network definition)
│   ├── checkpoints/         (saved weights)
│   └── trained_model.pt     (final model)
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── inference.py
├── web/
│   ├── frontend/            (React app)
│   └── backend/             (FastAPI server)
├── results/
│   ├── metrics.json         (test set results)
│   ├── plots/               (Bland-Altman, calibration plots)
│   └── demographic_analysis.csv
└── README.md
```

---

## Next Steps

1. **User Approval:** Review and approve this design
2. **Implementation Plan:** Once approved, transition to writing-plans skill to create detailed implementation roadmap
3. **Execution:** Begin with data preparation and exploration
4. **Iteration:** Regular evaluation checkpoints to catch issues early

