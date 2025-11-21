# src/train_models_advanced.py

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

# -------------------------
# Config
# -------------------------
PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "match_features_synergy_loo_15_20_1.csv"


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_JSON = MODELS_DIR / "features_list_advanced.json"
METRICS_JSON  = MODELS_DIR / "metrics_advanced.json"
REPORT_TXT    = MODELS_DIR / "report_advanced.txt"

RANDOM_STATE = 42
N_SPLITS = 5

# FLAGS
USE_DELTAS_ONLY = True          # usar solo columnas Delta_*
DROP_COUNTER_FEATURES = False    # quitar columnas con "counter" en el nombre

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
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

def select_features(df: pd.DataFrame):
    cols = df.columns.tolist()
    cols = [c for c in cols if c != "Winner"]

    if USE_DELTAS_ONLY:
        cols = [c for c in cols if c.startswith("Delta_")]
    else:
        # incluir numéricas
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in cols if c in num_cols]

    if DROP_COUNTER_FEATURES:
        cols = [c for c in cols if "counter" not in c.lower()]

    return cols

# -------------------------
# Entrenamiento con K-Fold
# -------------------------

def train_kfold_models(X, y, feature_names):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {
        "logreg": [],
        "rf": [],
        "xgb": [],
    }

    # Guardar modelos entrenados en el último fold (como referencia)
    best_logreg = None
    best_rf = None
    best_xgb = None

    fold_idx = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1
        print(f"\n===== Fold {fold_idx}/{N_SPLITS} =====")

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Pesos de clase (por si acaso)
        classes = np.unique(y_tr)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

        # ----------------- Logistic Regression (baseline) -----------------
        logreg = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            class_weight=cw_dict,
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
        logreg.fit(X_tr, y_tr)
        proba_lr = logreg.predict_proba(X_te)[:, 1]
        metrics_lr = summarize_metrics(y_te, proba_lr)
        results["logreg"].append(metrics_lr)
        best_logreg = logreg

        print("LOGREG fold metrics:", metrics_lr)

        # ----------------- Random Forest -----------------
        rf = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight=cw_dict,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(X_tr, y_tr)
        proba_rf = rf.predict_proba(X_te)[:, 1]
        metrics_rf = summarize_metrics(y_te, proba_rf)
        results["rf"].append(metrics_rf)
        best_rf = rf

        print("RF fold metrics:", metrics_rf)

        # ----------------- XGBoost -----------------
        # sin scale_pos_weight si clases ~balanceadas
        xgb = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.0,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False,
            #early_stopping_rounds=200
        )

        proba_xgb = xgb.predict_proba(X_te)[:, 1]
        metrics_xgb = summarize_metrics(y_te, proba_xgb)
        results["xgb"].append(metrics_xgb)
        best_xgb = xgb

        print("XGB fold metrics:", metrics_xgb)

    # Agregar medias de cada modelo
    metrics_summary = {}
    for model_name, folds in results.items():
        if len(folds) == 0:
            continue
        agg = {}
        for k in folds[0].keys():
            vals = [fold[k] for fold in folds]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        metrics_summary[model_name] = agg

    # Calibración isotónica del mejor modelo según ROC AUC mean
    best_model_name = max(
        metrics_summary.items(),
        key=lambda kv: kv[1]["roc_auc_mean"]
    )[0]
    print(f"\n⭐ Mejor modelo por ROC AUC (mean): {best_model_name}")

    # Elige el objeto a calibrar
    base_model = best_rf if best_model_name == "rf" else (
        best_xgb if best_model_name == "xgb" else best_logreg
    )

    # Calibramos con todo el dataset usando CV interno
    cal_model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    )
    cal_model.fit(X, y)

    # Métrica global post-calibración (out-of-sample aprox usando CV interno)
    # Aquí, para referencia, evaluamos sobre el mismo X,y (ojo: optimista)
    proba_cal = cal_model.predict_proba(X)[:, 1]
    metrics_cal = summarize_metrics(y, proba_cal)
    metrics_summary[f"{best_model_name}_calibrated_global"] = metrics_cal

    # Guardar modelos
    joblib.dump(base_model, MODELS_DIR / f"{best_model_name}_base_cv.joblib")
    joblib.dump(cal_model, MODELS_DIR / f"{best_model_name}_calibrated_cv.joblib")

    # Guardar features
    with open(FEATURES_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # Guardar métricas
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    header = f"""Entrenamiento avanzado K-Fold
- Archivo: {IN_PATH}
- Splits: {N_SPLITS}
- USE_DELTAS_ONLY={USE_DELTAS_ONLY}
- DROP_COUNTER_FEATURES={DROP_COUNTER_FEATURES}
"""
    save_report(REPORT_TXT, header, metrics_summary)

    print("\n✅ Entrenamiento K-Fold completado.")
    print("Resumen métricas (mean/std):")
    for name, m in metrics_summary.items():
        print(name, "->", m)

def main():
    df = pd.read_csv(IN_PATH)
    feature_cols = select_features(df)

    print("🔧 Features usadas:", feature_cols)

    X = df[feature_cols].values
    y = df["Winner"].astype(int).values

    # 1) Entrenamiento con K-Fold (lo que ya tenías)
    train_kfold_models(X, y, feature_cols)

    # 2) ENTRENAMIENTO FINAL DE LOGISTIC REGRESSION CON TODO EL DATASET
    print("\n🧩 Entrenando modelo final (Logistic Regression) con TODO el dataset...")

    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    logreg_final = LogisticRegression(
        solver="liblinear",
        class_weight=cw_dict,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    logreg_final.fit(X, y)

    # Guardamos el modelo y la lista de features
    joblib.dump(
        {
            "model": logreg_final,
            "features": feature_cols,
        },
        MODELS_DIR / "logreg_model.pkl",
    )

    print(f"✅ Modelo final (LogReg) guardado en: {MODELS_DIR / 'logreg_model.pkl'}")


if __name__ == "__main__":
    main()
