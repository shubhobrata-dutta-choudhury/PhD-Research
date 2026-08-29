# Restoring Statistical Coverage in Quantized Medical Segmentation

**An Adaptive Conformal Prediction Framework for Trustworthy Edge AI**

---

## 1. Problem Statement & Motivation

### The Edge AI Challenge in Point-of-Care Medical Imaging
* **The Challenge:** Portable medical diagnostic systems (e.g., handheld point-of-care ultrasound, mobile dermatoscopy) require low-latency, on-device inference without cloud connectivity.
* **The Compression Bottleneck:** Standard deep segmentation architectures (e.g., U-Net, Attention U-Net) exceed the memory and power budgets of embedded Neural Processing Units (NPUs).
* **Hardware Truncation (INT8 PTQ):** Post-Training Quantization (PTQ) reduces weights and activations from 32-bit floating-point ($\text{FP32}$) to 8-bit integers ($\text{INT8}$), reducing the memory footprint by $\approx 75\%$ and slashing Multiply-Accumulate (MAC) energy consumption.

### The Safety Hazard: Configuration Shift & Calibration Collapse
* **Loss of Predictive Distribution Fidelity:** While top-line segmentation accuracy (Dice / mIoU) degrades only marginally under INT8 PTQ ($\approx 1\text{--}3\%$), the continuous output logit distributions undergo severe non-linear distortion.
* **Overconfidence in Clinical Failures:** The quantization noise acts as an internal **configuration shift**, driving the model to produce overconfident predictions on ambiguous lesion boundaries and false negatives.
* **Breakdown of Conformal Guarantees:** Conformal Prediction (CP) guarantees marginal coverage $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha$ under the assumption of exchangeability and stable calibration. Under quantization shift, standard static non-conformity scoring fails, causing empirical coverage to drop significantly below safety margins (e.g., failing to meet a $90\%$ target).

---

## 2. Work Already Completed: Methodology & Profiling Pipeline

### Experimental Testbed & Dataset Setup
* **Dataset:** ISIC 2018 Skin Lesion Boundary Segmentation Task ($\approx 2,594$ dermatoscopic images) cross-validated with the Breast Ultrasound Images (BUSI) dataset.
* **Architecture:** MobileNetV3-UNet ($\approx 3.5\text{M}$ parameters) selected for mobile and edge NPU compatibility.
* **Software-Simulated Quantization Engine:** Implemented via PyTorch's native quantization engine (`torch.ao.quantization`) using symmetric per-channel weight quantization and asymmetric per-tensor activation quantization:
  $$X_{\text{quant}} = \text{clamp}\left(\left\lfloor \frac{X}{\text{Scale}} \right\rceil + \text{ZeroPoint}, -128, 127\right)$$

### Evaluation Protocol
1. **Model Optimization:** Trained the baseline architecture in full precision ($\text{FP32}$) using a hybrid Binary Cross-Entropy (BCE) and Dice loss until convergence.
2. **Quantization Profiling:** Applied PTQ calibration over a held-out calibration set ($N = 500$) to calculate activation scaling factors without retraining weights.
3. **Statistical Uncertainty Calibration:** Evaluated pixel-level Conformal Prediction targeting a nominal error rate $\alpha = 0.10$ ($90\%$ statistical coverage) using the standard non-conformity score:
   $$s(x_i, y_i) = 1 - \hat{P}(y_i \mid x_i)$$
   where the non-conformity quantile $\hat{q}$ is computed as:
   $$\hat{q} = \text{Quantile}\left(\frac{\lceil (n+1)(1-\alpha) \rceil}{n}; \{s_i\}_{i=1}^n\right)$$
4. **Baseline Comparison Pipeline:** Benchmarked against uncalibrated Maximum Softmax Probability (MSP) and post-hoc Temperature Scaling (TS):
   $$\hat{P}_{\text{cal}}(y \mid x) = \sigma\left(\frac{z(x)}{T}\right)$$

---

## 3. Results of Current Work

### Preliminary Empirical Benchmarks

The preliminary profiling shows a severe degradation in calibration and statistical coverage when transitioning from $\text{FP32}$ to simulated $\text{INT8}$ execution, despite marginal changes in segmentation Dice score.

| Precision & Setup | Calibration Method | Dice Score ($\uparrow$) | Expected Calibration Error ($\text{ECE} \downarrow$) | Empirical Conformal Coverage ($1-\alpha = 0.90$) | Mean Set Size (Relative to FP32) | Footprint / Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 (Baseline)** | Uncalibrated (MSP) | 0.891 | 0.084 | 88.7% | 1.00× | 14.2 MB / 42 ms |
| **FP32 (Baseline)** | Temperature Scaling | 0.891 | **0.031** | **90.2%** *(Nominal)* | 0.94× | 14.2 MB / 42 ms |
| **INT8 (Simulated PTQ)** | Uncalibrated (MSP) | 0.864 | 0.238 *(+183% error)* | 71.4% *(Safety Failure)* | 0.78× *(Under-covering)*| **3.6 MB** / **11 ms** |
| **INT8 (Simulated PTQ)** | Static Temp Scaling | 0.864 | 0.142 | 81.6% *(Under-covering)* | 1.15× *(Over-inflated)* | **3.6 MB** / **11 ms** |

### Key Findings from Profiling
1. **Quantization Induces Calibration Failure:** Converting weights to $\text{INT8}$ causes the Expected Calibration Error to jump from $0.084$ to $0.238$, indicating that the model becomes pathologically overconfident.
2. **Static Calibration Is Insufficient:** Standard post-hoc Temperature Scaling optimized on global validation statistics fails on quantized outputs because quantization distortion is spatially non-uniform across feature maps, missing the target $90\%$ coverage by $8.4\%$.
3. **Safety Violation:** Under $\text{INT8}$, the uncalibrated prediction sets contract artificially ($0.78\times$), omitting true lesion boundaries in $28.6\%$ of clinical test pixels.

---

## 4. Proposed Future Work: Reinforcement Learning-Adaptive Calibration

### Core Proposed Novelty
Rather than using static global scalar scaling, we propose a **Dynamic Contextual Calibration Layer** parameterized by a lightweight Reinforcement Learning (RL) agent. The RL agent observes local quantization entropy and dynamically shifts the non-conformity threshold per sample to maintain exact $1-\alpha$ coverage.