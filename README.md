# 🚢 Smart Shipping & Logistics (AI Mission Challenge)

## 📌 Project Overview
This repository presents our solution for the **Smart Shipping & Logistics X AI Mission Challenge (2025)**.  
The competition involved classifying the **operational state of vessels** using complex **sensor tabular data (52 features)** collected from real maritime logistics systems.  
We developed a **RealMLP-TD (PyTabKit)**–based deep learning model optimized with **Optuna** and enhanced through **Targeted BCMixUp augmentation** to improve macro-F1 and generalization performance.

---

## 🎯 Objectives
- Build a robust tabular deep learning classifier for 21 ship operational states  
- Handle **class imbalance** and **feature correlation** inherent in sensor data  
- Integrate **Targeted MixUp augmentation** to focus on confusing class pairs  
- Optimize both model & augmentation hyperparameters via **Optuna**  
- Implement **meta acceptors (gating classifiers)** for ambiguous classes (0, 9, 15)  
- Deliver reproducible training, validation, and submission pipelines

---

## 🧩 Dataset Description
| Type | Description |
|------|--------------|
| **Input Features (f1–f52)** | 52 continuous sensor variables (pressure, vibration, temperature, current, etc.) |
| **Target** | 21 discrete classes representing ship operational states |
| **Train Samples** | ~21,000 |
| **Test Samples** | ~5,000 |
| **Evaluation Metric** | Macro-F1 Score |
| **Split Strategy** | 10-Fold Stratified Cross Validation (OOF aggregation) |

**Goal:** Accurately classify ship operational modes to enhance real-world logistics monitoring and predictive maintenance.

---

## ⚙️ Data Preprocessing
We applied the following data preparation steps before model training:

1. **Feature Normalization:**  
   - StandardScaler (fold-safe within CV loop)
   - Prevented data leakage by fitting only on train folds

2. **Feature Selection:**  
   - Removed constant/low-variance columns (if any)
   - Analyzed feature correlation matrix to ensure balance between vibration & temperature sensors

3. **Outlier Handling:**  
   - IQR-based clipping within sensor ranges  
   - Replaced extreme values (>99th percentile) with capped limits

4. **Target Balancing (Optional):**  
   - Considered random oversampling for rare states (<1% frequency)  
   - Final model relied on loss-weighting inside RealMLP-TD

---

## 🧠 Model Architecture: RealMLP-TD
The **RealMLP-TD (Tabular Deep)** model was implemented via **PyTabKit** library.  
It uses sequential dense layers with label smoothing, dropout, and adaptive learning rates.

```python
model = RealMLP_TD_Classifier(
    device='cuda',
    random_state=42,
    n_cv=1,
    n_refit=1,
    n_epochs=500,
    batch_size=768,
    hidden_sizes=[512, 256, 256, 128],
    val_metric_name='cross_entropy',
    lr=0.00643,
    use_ls=True,
    verbosity=1,
    use_early_stopping=True
)
