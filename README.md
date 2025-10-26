# 🚢 Smart Shipping & Logistics (AI Mission Challenge)

## 📌 Project Overview
This project was part of the **Smart Shipping & Logistics X AI Mission Challenge (2025)**.  
The goal was to develop an **AI model that classifies ship operation states** using real-world sensor data collected from maritime logistics systems.  
We focused on **multi-class classification (21 classes)** using **RealMLP-TD (PyTabKit)**,  
and applied **Optuna-based hyperparameter optimization** and **Targeted MixUp augmentation** to achieve stable performance improvements.

---

## 🎯 Objectives
- Analyze and preprocess 52-dimensional sensor tabular data  
- Develop a deep MLP-based model (RealMLP-TD) for 21-class classification  
- Optimize model hyperparameters using **Optuna (60 trials)**  
- Enhance performance via **Targeted BCMixUp augmentation**  
- Design class-specific experts (acceptors) for classes **0, 9, 15**  
- Achieve the highest **Macro-F1** on the leaderboard

---

## 🛠️ Techniques Used
- **Model**: RealMLP-TD (PyTabKit implementation)
- **Optimization**: Optuna search across lr, batch size, depth, epoch  
- **Augmentation**: Targeted BCMixUp (Beta(α, α), lam_clip(0.2~0.8))  
- **Meta-Classifiers**: Logistic regression–based gating for confusing classes (0, 9, 15)  
- **Loss Function**: Cross-Entropy with Label Smoothing  
- **Validation**: 10-Fold Stratified Cross Validation (OOF aggregation)  

---

## ⚙️ Model Configuration
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
