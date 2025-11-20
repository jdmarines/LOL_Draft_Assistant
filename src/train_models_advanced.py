"""
Versión robusta del entrenamiento:
- Si XGBoost soporta early_stopping_rounds → lo usa.
- Si NO lo soporta → usa fit() normal sin early stopping.
- El resto del pipeline no cambia.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    f1_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import joblib

from xgboost import XGBClassifier
import xgboost
from packaging import version

# -------------------------
# Config
# -------------------------
PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "match_features_15_20_1.csv"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_JSON = MODELS_DIR / "features_list_advanced.json"
METRICS_JSON  = MODELS_DIR / "metrics_advanced.json"
REPORT_TXT    = MODELS_DIR / "report_advanced.txt"

RANDOM_STATE = 42
N_SPLITS = 5

# FLAGS
USE_DELTAS_ONLY = True
DROP_COUNTER_FEATURES = True

# -------------------------
# Utils
# -------------------------

def summarize_metrics(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1-1e-7))),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }

def save_report(path: Path, header: str, metrics: dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header.strip() + "\n\n")
        for model_name, m in metrics.items():
            f.write(f"[{model_name}]\n")
            for k, v in m.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

def select_features(df):
    cols = df.columns.tolist()
    cols = [c for c in cols if c != "Winner"]

    if USE_DELTAS_ONLY:
        cols = [c for c in cols if c.startswith("Delta_")]
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c in num_cols]

    if DROP_COUNTER_FEATURES:
        cols = [c for c in cols if "counter" not in c.lower()]

    return cols

# -------------------------
# Chequeo automático de XGBoost
# -------------------------

def xgb_supports_early_stopping():
    """
    Detecta si la versión instalada soporta early_stopping_rounds en .fit().
    """
    ver = xgboost.__version__
    print(f"🔍 XGBoost version detectada: {ver}")

    # Desde XGBoost 1.6 en adelante soporta early_stopping_rounds sin problemas.
    return version.parse(ver) >= version.parse("1.6.0")

SUPPORTS_ESTOP = xgb_supports_early_stopping()

# -------------------------
# Entrenamiento K-Fold
# -------------------------

def train_kfold_models(X, y, feature_names):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {
        "logreg": [],
        "rf": [],
        "xgb": [],
    }

    best_logreg = None
    best_rf = None
    best_xgb = None

    fold_idx = 0

    for tr, te in skf.split(X, y):
        fold_idx += 1
        print(f"\n===== Fold {fold_idx}/{N_SPLITS} =====")

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        classes = np.unique(y_tr)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

        # --------------------- Logistic Regression ---------------------
        logreg = LogisticRegression(
            solver="liblinear", class_weight=cw_dict,
            max_iter=1000, random_state=RANDOM_STATE
        )
        logreg.fit(X_tr, y_tr)
        p_lr = logreg.predict_proba(X_te)[:, 1]
        metrics_lr = summarize_metrics(y_te, p_lr)
        results["logreg"].append(metrics_lr)
        best_logreg = logreg
        print("LOGREG:", metrics_lr)

        # --------------------- Random Forest ---------------------
        rf = RandomForestClassifier(
            n_estimators=800,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=cw_dict,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_te)[:, 1]
        metrics_rf = summarize_metrics(y_te, p_rf)
        results["rf"].append(metrics_rf)
        best_rf = rf
        print("RF:", metrics_rf)

        # --------------------- XGBoost ---------------------
        xgb = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        if SUPPORTS_ESTOP:
            # ✅ usar early stopping si está soportado
            xgb.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False,
                early_stopping_rounds=200
            )
        else:
            # ⚠️ fallback sin early stopping
            print("⚠️ XGBoost sin early_stopping_rounds (versión antigua).")
            xgb.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                verbose=False
            )

        p_xgb = xgb.predict_proba(X_te)[:, 1]
        metrics_xgb = summarize_metrics(y_te, p_xgb)
        results["xgb"].append(metrics_xgb)
        best_xgb = xgb

        print("XGB:", metrics_xgb)

    # ======================
    # Métricas agregadas
    # ======================
    metrics_summary = {}
    for model_name, folds in results.items():
        agg = {}
        for k in folds[0].keys():
            vals = [m[k] for m in folds]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        metrics_summary[model_name] = agg

    # -------------------------
    # Selección del mejor modelo
    # -------------------------
    best_model_name = max(
        metrics_summary.items(),
        key=lambda kv: kv[1]["roc_auc_mean"]
    )[0]

    print(f"\n⭐ Mejor modelo: {best_model_name}")

    base_model = (
        best_rf if best_model_name == "rf" else
        best_xgb if best_model_name == "xgb" else
        best_logreg
    )

    # -------------------------
    # Calibración isotónica
    # -------------------------
    cal_model = CalibratedClassifierCV(
        base_model, cv=3, method="isotonic"
    )
    cal_model.fit(X, y)

    p_cal = cal_model.predict_proba(X)[:, 1]
    metrics_summary[f"{best_model_name}_calibrated_global"] = summarize_metrics(y, p_cal)

    # -------------------------
    # Guardar artefactos
    # -------------------------
    joblib.dump(cal_model, MODELS_DIR / f"{best_model_name}_calibrated_cv.joblib")
    joblib.dump(base_model, MODELS_DIR / f"{best_model_name}_base_cv.joblib")

    with open(FEATURES_JSON, "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    save_report(
        REPORT_TXT,
        header=f"Entrenamiento avanzado con fallback XGBoost\nUSE_DELTAS_ONLY={USE_DELTAS_ONLY}",
        metrics=metrics_summary
    )

    print("✅ Entrenamiento avanzado completado.")
    print(metrics_summary)


def main():
    df = pd.read_csv(IN_PATH)
    feature_cols = select_features(df)

    print("🔧 Features usadas:", feature_cols)

    X = df[feature_cols].values
    y = df["Winner"].astype(int).values

    train_kfold_models(X, y, feature_cols)


if __name__ == "__main__":
    main()
