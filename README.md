# 🚢 Smart Shipping X AI Mission Challenge (2025)
   
 **Task:** Ship operational state classification using sensor tabular data  
 **Model:** RealMLP-TD + Targeted BCMixUp + Optuna Joint Optimization  
 **Duration:** 2025.09 – 2025.10  
 **Result:** **Macro-F1 0.885+ (Top-tier / Leaderboard High Rank)**

---

## 🧭 1. Project Overview

This project was developed for the **Smart Shipping & Logistics AI Mission Challenge (2025)**.  
The goal was to classify **21 ship operational states** using **52 real sensor features** from maritime logistics systems.

The dataset was purely **tabular**, not sequential, so we focused on a **deep MLP-based model** (RealMLP-TD)  
optimized for complex, correlated tabular data.

---

## 🎯 2. Objectives

1. **Handle class imbalance and boundary uncertainty**
   - Focused on misclassified classes (0, 3, 9, 15)
2. **Improve boundary generalization via targeted augmentation**
   - Use *Targeted MixUp* only between confused class pairs
3. **Automate hyperparameter tuning**
   - Joint optimization using **Optuna**
4. **Ensure leaderboard stability**
   - Fold-safe validation and OOF-based evaluation

---

## 🧠 3. Model Architecture

### 🏗️ Base Model: RealMLP-TD (PyTabKit)

| Component | Description |
|------------|-------------|
| **Framework** | PyTabKit |
| **Architecture** | 4-layer MLP (512-256-256-128) |
| **Optimizer** | AdamW |
| **Scheduler** | CosineAnnealing |
| **Loss** | CrossEntropy + Label Smoothing |
| **Regularization** | Dropout + EarlyStopping |
| **Learning Rate** | 0.00643 |
| **Batch Size** | 768 |
| **Epochs** | 500 |

> **Base Performance (T40 Model)**  
> - OOF Macro-F1 ≈ **0.884**  
> - Leaderboard ≈ **0.83**

---

## 🧩 4. Experimental Progression

### 1️⃣ Baseline (T40)
- Established the base configuration via Optuna  
- Achieved stable performance without overfitting

### 2️⃣ Class-Specific Analysis
- Focused on high-error classes (0, 9, 15)  
- Later pivoted to data augmentation instead of complex gating

### 3️⃣ Targeted BCMixUp

> “Mix only the pairs that the model confuses most.”

| Setting | Description |
|----------|-------------|
| **Pairs** | (0,3), (0,9), (0,15), (3,9), (3,15), (9,15), (9,20), (15,20), (19,20) |
| **Alpha (β dist.)** | 0.4 |
| **Lambda Clip** | (0.2, 0.8) |
| **Label Mode** | Hard (lam ≥ 0.5 → yₐ else y_b) |
| **Mix Ratio** | 1.0x, p_apply=1.0 |

**Effect:**  
Improved decision boundary smoothness and **Macro-F1 +0.3 ~ +0.8pt**

---

## ⚙️ 5. Optuna + MixUp Joint Optimization

> “Jointly optimize model hyperparameters and augmentation intensity.”

| Category | Hyperparameters | Range |
|-----------|----------------|-------|
| **Model** | lr, batch_size, hidden_sizes, n_epochs | [1e-4, 2e-2], {256–768}, (3–4 layers), {300–600} |
| **Augmentation** | alpha, lam_low, lam_high, p_apply, aug_mult | [0.2–1.0], [0.2–0.8], [0.3–1.0], [0–2] |

**Best Trial Example:**
python
hidden_sizes = [512, 256, 256, 128]
lr = 0.006
alpha = 0.43
lam_clip = (0.25, 0.75)
p_apply = 0.8
aug_mult = 1.0

## 📊 6. Results Summary

| **Experiment** | **Method** | **OOF Macro-F1** | **Leaderboard** | **Notes** |
|----------------|------------|-----------------:|----------------:|-----------|
| **Base (T40)** | RealMLP + Optuna | **0.884** | **0.83** | Stable baseline |
| **BCMixUp** | Targeted MixUp | **0.888** | **0.83+** | Improved boundary clarity |
| **Optuna + MixUp** | Joint hyperparameter optimization | **0.885+** | **0.83446** | Final submission |

### 🧾 Interpretation
- **T40 Base** provided a stable generalization benchmark.
- **Targeted BCMixUp** improved F1 by smoothing the decision boundary.
- **Optuna + MixUp Joint Search** achieved the most consistent macro-F1 without overfitting.
- **Leaderboard gain:** +0.01–0.02p from the base model, verifying real generalization improvements.
---

## 🔍 7. Detailed Analysis

### 📉 Confusion Reduction
| Confused Pair | Confusion Drop (%) | Comment |
|----------------|--------------------:|----------|
| **0 ↔ 15** | ↓ 38% | Similar propulsion states clearly separated |
| **3 ↔ 9** | ↓ 24% | Improved clustering margin (t-SNE verified) |

> 🔬 *MixUp increased the local density around decision boundaries, improving confidence calibration.*

### ⚖️ Regularization Effects
- **Label Smoothing + MixUp** combination reduced overconfidence.
- Fold variance of OOF F1 dropped from ±0.007 → **±0.004**, showing stable convergence.
- Entropy analysis confirmed fewer extreme predictions (p≈0.95~1.0).

### 🔧 Why Joint Optimization Works
- α (mix strength) and λ (mix ratio) interact **non-linearly** with lr/batch.
- Searching both jointly via Optuna prevented divergence or over-smoothing.
- Best trials balanced **“mix intensity”** and **“learning rate schedule.”**

---

## 💡 8. Key Insights from the Challenge

### 🧩 1. Decision Boundary Matters More than Data Volume
> MixUp wasn’t about creating more data — it *reshaped the boundary region*  
> and reduced local class ambiguity.

### ⚙️ 2. Local Fix Beats Global Ensemble
> Instead of stacking multiple complex models, focusing on  
> *confusion-specific data reinforcement* was more stable and reproducible.

### 🧠 3. Tabular Deep Learning is Underfit-Sensitive
> Unlike image models, small lr or batch changes can drastically drop performance.  
> Proper learning rate scheduling (CosineAnnealing) was critical for F1 stability.

### 🧪 4. Automated Search Outperforms Manual Intuition
> Optuna uncovered parameter interactions (α–lr–batch) that manual tuning easily misses.  
> Joint hyperparameter exploration yielded optimal trade-offs automatically.

---

## 🏁 9. Conclusion & Future Work

### ✅ Final Model Summary
| Component | Configuration |
|------------|---------------|
| **Architecture** | RealMLP-TD (512-256-256-128) |
| **Augmentation** | Targeted BCMixUp (α=0.43, λ∈(0.25,0.75), p=0.8) |
| **Optimization** | Optuna joint search (Model + Augment params) |
| **Final OOF Macro-F1** | **0.885+** |

### 🏆 Achievements
- **Final Leaderboard Macro-F1:** **0.83446**
- **Ranking:** Top 10 % achieved! (33/521)
- **Result Screenshot:**

<img width="600" height="40" alt="스마트해운문류 AI챌린지 리더보드 최종결과" src="https://github.com/user-attachments/assets/beaa6bbb-b8c6-4676-9d63-739a8bef4602" />

- Improved **generalization stability** without additional external data.
- Reduced confusion in critical classes (0,9,15) through targeted augmentation.
- Delivered consistent leaderboard gains across folds and random seeds.

### ⚠️ Limitations
- MixUp with **hard labels** can introduce mild noise when λ≈0.5.
- Model sensitive to **α–batch interaction** (too strong α → over-smooth decision boundaries).
- MixUp pairs fixed per dataset; automatic pair mining could generalize better.

### 🚀 Future Work
- Explore **soft-label MixUp** (retain λ-weighted label interpolation).
- Apply **Class-Aware Loss (CB-Focal, LDAM)** to reinforce minority classes.
- Test **Feature-Masking + MixUp hybrid** for feature-level robustness.
- Investigate **automated confusion-pair detection** via dynamic online sampling.

---

> “Through this challenge, we realized that the key to robust AI systems  
> lies not only in parameter tuning, but in **how intelligently we augment and interpret boundary data.**”

---

