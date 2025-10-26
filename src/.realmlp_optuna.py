# ============================================================
#  9개 후보 모델 → 10-Fold Full Training & Ensemble/Stacking
#  - aug_mult / aug_mult_v2: NaN/0 → 증강 OFF (안전 처리)
#  - CSV 컬럼: params_* / 무접두 모두 지원
#  - stacking: LogisticRegression C 그리드서치
# ============================================================

!pip -q install pytabkit scikit-learn

import os, itertools, json, math, ast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from pytabkit import RealMLP_TD_Classifier

# ---------------------
# 경로 / 데이터 로드
# ---------------------
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

DATA_DIR = "/content"
SAVE_DIR = "/content/drive/MyDrive/t40_final_ensemble"
os.makedirs(SAVE_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_t = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

X = train.drop(columns=["ID", "target"]).values
y = train["target"].values
X_test = test.drop(columns=["ID"]).values
n_classes = len(np.unique(y))

# 후보 모델 하이퍼파라미터 CSV (number 40 포함)
csv_path = "/content/drive/MyDrive/t40_mixup_optuna_v2/trials_selected_dedup.csv"
models_df = pd.read_csv(csv_path)

# ---------------------
# 유틸: 안전 파라미터 읽기
# ---------------------
def _to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in ("", "none", "nan", "null"):
                return np.nan
        return float(x)
    except Exception:
        return np.nan

def getp(params: dict, key: str, default=None):
    """params[key] 또는 params['params_'+key] 우선순위대로 반환"""
    if key in params:
        return params[key]
    pkey = f"params_{key}"
    if pkey in params:
        return params[pkey]
    return default

def get_hidden_sizes(params: dict):
    hs = getp(params, "hidden_sizes")
    if isinstance(hs, str):
        try:
            return list(ast.literal_eval(hs))
        except Exception:
            # 혹시 문자열이지만 파싱 안 되면 쉼표 분리 시도
            try:
                return [int(s) for s in hs.strip("[]()").split(",")]
            except Exception:
                raise ValueError(f"hidden_sizes 파싱 실패: {hs}")
    return list(hs)

def get_aug_mult_from_params(params: dict) -> float:
    """aug_mult_v2 → aug_mult → default 0.0, NaN이면 0.0"""
    x = _to_float_or_nan(getp(params, "aug_mult_v2", np.nan))
    if pd.isna(x):
        x = _to_float_or_nan(getp(params, "aug_mult", np.nan))
    if pd.isna(x):
        return 0.0
    return float(x)

# ---------------------
# Targeted MixUp (Optuna 로직 동일, soft→argmax fallback)
# ---------------------
ALLOWED_PAIRS = {(0,3), (0,9), (0,15), (0,20),
                 (3,9), (3,15), (9,15), (9,20),
                 (15,20), (19,20)}

def bcmixup_targeted(X, y,
                     alpha, lam_low, lam_high,
                     pairs, p_apply, aug_mult,
                     label_mode, n_classes, rng):
    if aug_mult <= 0:
        return X, y

    n = len(X)
    m = int(n * aug_mult)
    idx_a = rng.randint(0, n, size=m)
    idx_b = rng.randint(0, n, size=m)
    y_a, y_b = y[idx_a], y[idx_b]

    mask = np.array([(ya, yb) in pairs or (yb, ya) in pairs for ya, yb in zip(y_a, y_b)])
    if mask.sum() == 0:
        return X, y
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    y_a, y_b = y[idx_a], y[idx_b]

    lam = rng.beta(alpha, alpha, size=len(idx_a))
    lam = np.clip(lam, lam_low, lam_high)

    Xm = lam[:,None]*X[idx_a] + (1.0 - lam)[:,None]*X[idx_b]

    if label_mode == "soft":
        ym = (lam[:,None]*np.eye(n_classes)[y_a] +
              (1.0-lam)[:,None]*np.eye(n_classes)[y_b])
        y_mix = ym.argmax(1).astype(int)  # fallback (Optuna와 동일)
    else:
        y_mix = np.where(lam >= 0.5, y_a, y_b).astype(int)

    if p_apply < 1.0:
        k = max(1, int(len(Xm) * p_apply))
        Xm, y_mix = Xm[:k], y_mix[:k]

    X_aug = np.vstack([X, Xm])
    y_aug = np.concatenate([y, y_mix])
    return X_aug, y_aug

# ---------------------
# 모델 학습 (10-Fold Full)
# ---------------------
def train_model_from_params(params, save_name):
    """주어진 파라미터 dict로 10-Fold full training (증강 자동 판별)"""
    hidden  = get_hidden_sizes(params)
    lr      = float(getp(params, "lr"))
    batch   = int(getp(params, "batch_size"))
    epochs  = int(getp(params, "n_epochs"))
    label_md= str(getp(params, "label_mode", "hard"))

    # 증강 여부/하이퍼 파싱 (NaN 안전 처리)
    aug_mult = get_aug_mult_from_params(params)
    use_aug  = (aug_mult > 0.0)

    if use_aug:
        alpha   = _to_float_or_nan(getp(params, "alpha"))
        lam_low = _to_float_or_nan(getp(params, "lam_low"))
        lam_high= _to_float_or_nan(getp(params, "lam_high"))
        p_apply = _to_float_or_nan(getp(params, "p_apply"))

        # NaN이면 기본값으로 소폭 보정 (Optuna 탐색 범위의 합리적 중앙값)
        if pd.isna(alpha):   alpha = 0.4
        if pd.isna(lam_low): lam_low = 0.2
        if pd.isna(lam_high):lam_high = 0.8
        if lam_high <= lam_low: lam_high = lam_low + 1e-3
        if pd.isna(p_apply): p_apply = 1.0
    else:
        # 증강 OFF 시 쓰이진 않지만, 인터페이스 통일
        alpha, lam_low, lam_high, p_apply = 0.4, 0.2, 0.8, 1.0

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof = np.zeros((len(X), n_classes), dtype=np.float32)
    test_pred = np.zeros((len(X_test), n_classes), dtype=np.float32)
    accs, f1s = [], []
    rng = np.random.RandomState(42)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xva = sc.transform(Xva)
        Xte = sc.transform(X_test)

        Xtr_aug, ytr_aug = bcmixup_targeted(
            Xtr, ytr,
            alpha=alpha, lam_low=lam_low, lam_high=lam_high,
            pairs=ALLOWED_PAIRS, p_apply=p_apply, aug_mult=aug_mult,
            label_mode=label_md, n_classes=n_classes, rng=rng
        )

        model = RealMLP_TD_Classifier(
            device="cuda", random_state=42,
            n_cv=1, n_refit=1,
            n_epochs=epochs, batch_size=batch,
            hidden_sizes=list(hidden),
            val_metric_name="cross_entropy",
            lr=lr, use_ls=True, verbosity=0, use_early_stopping=True
        )
        model.fit(Xtr_aug, ytr_aug)

        pv = model.predict_proba(Xva)
        pt = model.predict_proba(Xte)
        oof[va] = pv
        test_pred += pt / skf.n_splits

        accs.append(accuracy_score(yva, pv.argmax(1)))
        f1s.append(f1_score(yva, pv.argmax(1), average="macro"))

    print(f"[{save_name}] Mean ACC={np.mean(accs):.4f}, F1={np.mean(f1s):.4f}, aug_mult={aug_mult}")

    # 저장
    np.save(os.path.join(SAVE_DIR, f"{save_name}_oof.npy"), oof)
    np.save(os.path.join(SAVE_DIR, f"{save_name}_test.npy"), test_pred)
    return oof, test_pred

# ---------------------
# 1) 후보 모델 9개 풀학습
# ---------------------
oof_dict, test_dict = {}, {}
for _, row in models_df.iterrows():
    num = int(row["number"])
    save_name = f"model_{num}"
    params = row.to_dict()
    oof, test_pred = train_model_from_params(params, save_name)
    oof_dict[num] = oof
    test_dict[num] = test_pred

# ---------------------
# 2) 앙상블 / 스태킹 탐색
# ---------------------
def hard_vote(pred_mats):
    # pred_mats: [n_models x n_samples x n_classes]
    # argmax 후 다수결
    preds = [pm.argmax(1) for pm in pred_mats]
    preds = np.stack(preds, axis=0)  # [n_models, n_samples]
    maj = []
    for i in range(preds.shape[1]):
        maj.append(np.bincount(preds[:, i], minlength=n_classes).argmax())
    return np.array(maj)

def evaluate_combo(combo, method="hard"):
    """combo: model number list"""
    oofs = [oof_dict[c] for c in combo]
    tests = [test_dict[c] for c in combo]

    if method == "hard":
        oof_preds  = hard_vote(oofs)
        test_preds = hard_vote(tests)
    elif method == "stacking":
        # Logistic Regression (확률 스태킹)
        X_stack = np.hstack(oofs)         # [n_train, n_classes*|combo|]
        X_test_stack = np.hstack(tests)   # [n_test,  n_classes*|combo|]
        best_f1, best_preds = -1, None
        for C in [0.01, 0.1, 1, 10]:
            meta = LogisticRegression(C=C, max_iter=2000, multi_class="multinomial")
            meta.fit(X_stack, y)
            oof_preds_tmp = meta.predict(X_stack)
            f1_tmp = f1_score(y, oof_preds_tmp, average="macro")
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_preds = meta.predict(X_test_stack)
        oof_preds, test_preds = meta.predict(X_stack), best_preds
    else:
        raise ValueError("Unknown method")

    f1 = f1_score(y, oof_preds, average="macro")
    return f1, test_preds

results = []
model_ids = sorted(oof_dict.keys())
for r in range(2, len(model_ids)+1):
    for combo in itertools.combinations(model_ids, r):
        for method in ["hard", "stacking"]:
            f1, test_preds = evaluate_combo(combo, method)
            results.append({
                "combo": combo, "method": method, "macro_f1": f1, "test_preds": test_preds
            })

# ---------------------
# 3) 상위 10개 저장
# ---------------------
results.sort(key=lambda x: x["macro_f1"], reverse=True)
top10 = results[:10]

rank_rows = []
for i, res in enumerate(top10, 1):
    sub = sub_t.copy()
    sub["target"] = res["test_preds"]
    out_path = os.path.join(SAVE_DIR, f"submission_top{i}.csv")
    sub.to_csv(out_path, index=False)
    rank_rows.append({"rank": i, "combo": res["combo"], "method": res["method"], "macro_f1": res["macro_f1"], "path": out_path})
    print(f"[SAVE] Top{i}: combo={res['combo']} method={res['method']} F1={res['macro_f1']:.4f}")

pd.DataFrame(rank_rows).to_csv(os.path.join(SAVE_DIR, "ensemble_top10_summary.csv"), index=False)
print(f"\n[Done] All artifacts saved -> {SAVE_DIR}")

