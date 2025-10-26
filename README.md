# 🚢 Smart Shipping & Logistics X AI Mission Challenge (2025)

> **Team:** Smart Maritime AI Lab  
> **Task:** Ship operational state classification using sensor tabular data  
> **Model:** RealMLP-TD + Targeted BCMixUp + Optuna Joint Optimization  
> **Duration:** 2025.06 – 2025.10  
> **Result:** **Macro-F1 0.885+ (Top-tier / Leaderboard High Rank)**

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

✅ **Effect:**  
Improved decision boundary smoothness and **Macro-F1 +0.3 ~ +0.8pt**

---

## ⚙️ 5. Optuna + MixUp Joint Optimization

> “Jointly optimize model hyperparameters and augmentation intensity.”

| Category | Hyperparameters | Range |
|-----------|----------------|-------|
| **Model** | lr, batch_size, hidden_sizes, n_epochs | [1e-4, 2e-2], {256–768}, (3–4 layers), {300–600} |
| **Augmentation** | alpha, lam_low, lam_high, p_apply, aug_mult | [0.2–1.0], [0.2–0.8], [0.3–1.0], [0–2] |

**Best Trial Example:**
```python
hidden_sizes = [512, 256, 256, 128]
lr = 0.006
alpha = 0.43
lam_clip = (0.25, 0.75)
p_apply = 0.8
aug_mult = 1.0
