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

---

## 🧱 Component Details

| Component       | Description                               |
|----------------|-------------------------------------------|
| **Framework**  | PyTabKit (RealMLP-TD)                     |
| **Optimizer**  | AdamW                                     |
| **Learning Rate** | 0.00643 *(Optuna searched)*            |
| **Scheduler**  | CosineAnnealing                           |
| **Loss**       | CrossEntropy **+ Label Smoothing**        |
| **Regularization** | Dropout **+ EarlyStopping**           |

---

## 🔬 Experimental Progression

### 1️⃣ Base Model (T40)

- **Optuna search across**: `lr`, `batch_size`, `hidden_sizes`, `n_epochs`  
- **OOF Macro-F1 (mean)**: **0.884**, **Leaderboard**: **0.83**  
- Served as baseline “**T40 configuration**”.

**Final T40 (baseline)**
Hidden: [512, 256, 256, 128]
Batch: 768
Epochs: 500
LR: 0.00643


---

### 2️⃣ Class-Specific **Acceptor** Models (Meta Gating)

**Motivation:** 특정 클래스 *(0, 9, 15)* 에서 지속적 오분류(낮은 확신도)가 관찰됨 →  
Base 확률/엔트로피/마진으로 **로지스틱 회귀 기반 Binary Gating** 구성.

| Class | Gating Input Features                 | Decision Type                   | Δ Macro-F1 |
|------:|---------------------------------------|---------------------------------|-----------:|
| **0** | `p(0)`, `margin`, `entropy`           | Binary **accept/reject**        | **+0.012** |
| **9** | `p(9)`, `p(15)`, `p(20)`              | **Confidence** thresholding     | **+0.009** |
| **15**| `p(15)`, `p(9)`, `margin`             | Binary **accept/reject**        | **+0.014** |

> ✅ **Result:** 전체 Macro-F1 **+0.01~0.02p** 개선, 과적합 징후 최소.

---

### 3️⃣ **Targeted BCMixUp** Augmentation

**Idea:** 혼동이 잦은 **상위 페어**만 선택적으로 섞어 경계부 샘플을 증강.

**Key class pairs**  
`(0,3), (0,9), (0,15), (0,20), (3,9), (3,15), (9,15), (9,20), (15,20), (19,20)`

```python
def bcmixup_targeted_expanded(X, y, alpha=0.4, pairs=None, lam_clip=(0.2, 0.8)):
    rng = np.random.RandomState(42)
    n = len(X)
    idx = rng.permutation(n)
    lam = np.random.beta(alpha, alpha, size=n).reshape(-1, 1)
    lam = np.clip(lam, lam_clip[0], lam_clip[1])

    X_a, X_b = X, X[idx]
    y_a, y_b = y, y[idx]

    # 허용된 페어만 남김
    mask = np.array([
        (y_a[i], y_b[i]) in pairs or (y_b[i], y_a[i]) in pairs
        for i in range(n)
    ])

    X_mix = lam[mask] * X_a[mask] + (1 - lam[mask]) * X_b[mask]
    # hard label 선택 (major class)
    y_mix = np.where(lam[mask].flatten() >= 0.5, y_a[mask], y_b[mask])
    return X_mix, y_mix


---

## ⚙️ BCMixUp Parameters

| Parameter | Description |
|------------|-------------|
| **alpha** | Beta 분포 모양 (혼합 비율 샘플링 강도) |
| **lam_clip** | 혼합 비율 하한/상한 (예: `0.2 ~ 0.8`) |
| **pairs** | 허용 클래스 페어 (무작위 섞임 방지 및 데이터 노이즈 최소화) |
| **label_mode** | Hard label 선택 (주도 클래스 기준으로 레이블 결정) |

> ✅ **Effect:**  
> • 경계부 결정경계가 **매끄럽게 형성**되어 클래스 간 마진이 확대됨  
> • **0↔15, 3↔9** 과신(Over-confidence) 현상 완화  
> • **Macro-F1 +0.3 ~ +0.8pt** 향상 관찰  

---

## 🔁 Optuna + MixUp Joint Optimization

모델과 증강 하이퍼파라미터를 **단일 Optuna Study** 안에서 함께 탐색함.  
각 **trial**은 `10-Fold` 교차검증을 수행하고, 평균 **OOF Macro-F1**을 기준으로 최적 조합을 찾음.

| Category | Hyperparameters | Search Range |
|-----------|----------------|---------------|
| **Model** | `lr`, `batch_size`, `hidden_sizes`, `n_epochs` | `[1e-4, 2e-2]`, `{256–1024}`, `(3–4층)`, `{300–500}` |
| **Augmentation** | `alpha`, `lam_low`, `lam_high`, `p_apply`, `aug_mult` | `[0.2–1.0]`, `[0.2–0.8]`, `[0.3–1.0]`, `[0–2]` |

### 🏆 **Best Trial (Example)**
hidden = [512, 256, 256, 128]
lr = 0.006
alpha = 0.43
lam_clip = (0.25, 0.75)
p_apply = 0.8
aug_mult = 1.0
Final OOF Macro-F1: 0.885+


> 이 결과는 **RealMLP-TD 구조와 MixUp 강도 간 균형**을 가장 잘 맞춘 조합으로,  
> 학습 안정성과 재현성을 모두 확보함.

---

## 📊 Results Summary

| Experiment | Technique | OOF Macro-F1 | Leaderboard | Note |
|-------------|------------|--------------:|--------------:|------|
| **Base (T40)** | RealMLP + Optuna | **0.884** | **0.83** | Baseline |
| **Acceptor (0,9,15)** | Gating classifiers | **+0.02↑** | **0.884+** | Localized fix |
| **BCMixUp** | Targeted data blending | **+0.5pt↑** | **0.885+** | Generalization gain |
| **Optuna + MixUp** | Joint optimization | **✅ 0.885↑** | **✅ Top-tier** | Final submission |

---

## 🔎 Additional Result Analysis

- **0↔15** 클래스 간 혼동률(confusion)이 **약 38% 감소**  
- **3 vs 9** 군의 t-SNE 시각화에서 **클러스터 마진 확대** 확인  
- 단순 수동 튜닝 대비, **Joint Optuna**가 `lr`/`batch` 과적합을 효과적으로 방지  
- Fold #7, #8에서 나타난 **저확신 뒤집힘**(label flip)을 **Acceptor**가 보정  
- Fold 간 OOF 분산이 **±0.004 F1 이하**로 매우 안정적

---

## 💡 Key Insights

- **🎯 Targeted MixUp > 일반 오버샘플링**  
  → 경계부 표본 밀도를 높여 분류 경계가 더 견고해짐  

- **🤝 Joint HPO (Hyperparameter Optimization)**  
  → 증강 강도와 네트워크 복잡도 간 **균형을 자동 조절**  

- **🧩 클래스 단위 게이팅 (Meta Acceptors)**  
  → 대형 앙상블보다 **가볍고 효과적**, 특정 클래스에만 정밀 적용 가능  

- **🔒 Fold-safe 전처리 (Scaling / Clipping)**  
  → 모든 스케일러는 Fold 내 `train` 기준으로만 fit — 누수 방지  

- **⚗️ 모델 + 증강 동시 탐색**  
  → 정확도와 안정성 사이의 **최적 절충점(optimal trade-off)** 도출  

---


