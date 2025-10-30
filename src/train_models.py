"""
Entrena modelos clásicos (RandomForest, XGBoost) sobre match_features_15_20_1.csv.
- Split estratificado Train/Valid/Test
- Métricas: Accuracy, ROC AUC, LogLoss, Brier, F1
- Calibración de probabilidades (Platt / isotonic opcional) para RF
- Early stopping para XGB
- Guardado de artefactos: modelo, columnas de features, métricas y report.txt

Requiere:
  pip install xgboost scikit-learn joblib shap (shap es opcional)

Entrada:
  data/processed/match_features_15_20_1.csv  (producido por build_match_features.py)
Salida:
  models/
    rf_model.joblib
    rf_calibrated.joblib
    xgb_model.json
    features_list.json
    metrics.json
    report.txt
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, f1_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

import joblib

# XGBoost
from xgboost import XGBClassifier

# Opcional: SHAP (si está instalado)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -------------------------
# Paths
# -------------------------
PROC_DIR = Path("data/processed")
IN_PATH  = PROC_DIR / "match_features_15_20_1.csv"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_JSON = MODELS_DIR / "features_list.json"
METRICS_JSON  = MODELS_DIR / "metrics.json"
REPORT_TXT    = MODELS_DIR / "report.txt"

# -------------------------
# Utilidades
# -------------------------
RANDOM_STATE = 42

def select_feature_columns(df: pd.DataFrame):
    """Selecciona columnas de entrada (X). Excluye target, ids y columnas auxiliares."""
    drop_exact = {
        "Winner",
        # Identidad (si existen):
        "Blue_1","Blue_2","Blue_3","Blue_4","Blue_5",
        "Red_1","Red_2","Red_3","Red_4","Red_5",
    }
    cols = []
    for c in df.columns:
        if c in drop_exact:
            continue
        # Excluir columnas obviamente textuales
        if df[c].dtype.kind in "biufc":  # numéricas
            cols.append(c)
    return cols

def summarize_metrics(y_true, proba, threshold=0.5):
    """Devuelve dict con métricas claves."""
    y_pred = (proba >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1-1e-7))),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    return out

def save_report(path: Path, header: str, metrics: dict):
    def write_kv(f, k, v, indent=0):
        pad = "  " * indent
        # Si es dict, escribir sección y recorrer recursivamente
        if isinstance(v, dict):
            f.write(f"{pad}[{k}]\n")
            for kk, vv in v.items():
                write_kv(f, kk, vv, indent + 1)
            return
        # Formatear números con 4 decimales
        if isinstance(v, (int, float, np.integer, np.floating)):
            f.write(f"{pad}{k}: {float(v):.4f}\n")
        else:
            # Otros tipos: listas, strings, None, etc.
            f.write(f"{pad}{k}: {v}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(header.strip() + "\n\n")
        for k, v in metrics.items():
            write_kv(f, k, v)

# -------------------------
# Main
# -------------------------
def main():
    # 1) Cargar dataset
    df = pd.read_csv(IN_PATH)
    assert "Winner" in df.columns, "Falta columna Winner en match_features"

    # 2) Selección de X e y
    feature_cols = select_feature_columns(df)
    X = df[feature_cols].values
    y = df["Winner"].astype(int).values

    # 3) Split estratificado (60/20/20)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RANDOM_STATE
    )
    # -> train 60%, val 20%, test 20%

    # 4) Pesos de clase (por si está desbalanceado)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    # ==========================
    # Modelo 1: RandomForest
    # ==========================
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=cw_dict,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_val_proba = rf.predict_proba(X_val)[:,1]
    rf_test_proba = rf.predict_proba(X_test)[:,1]

    rf_val_metrics  = summarize_metrics(y_val,  rf_val_proba)
    rf_test_metrics = summarize_metrics(y_test, rf_test_proba)

    # Calibración (Platt por defecto)
    rf_cal = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
    rf_cal.fit(X_val, y_val)  # calibramos con el set de validación

    rf_cal_test_proba = rf_cal.predict_proba(X_test)[:,1]
    rf_cal_test_metrics = summarize_metrics(y_test, rf_cal_test_proba)

    # Guardar RF
    joblib.dump(rf, MODELS_DIR / "rf_model.joblib")
    joblib.dump(rf_cal, MODELS_DIR / "rf_calibrated.joblib")

    # ==========================
    # Modelo 2: XGBoost
    # ==========================
    # Estimar scale_pos_weight (neg/pos en train)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = max(1.0, neg / max(1, pos))

    xgb = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        scale_pos_weight=spw,
        n_jobs=-1,
    )

# Ajuste de entrenamiento: algunas versiones de xgboost no aceptan early_stopping_rounds en .fit()
    try:
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=200
        )
    except TypeError:
        # Fallback a la API de callbacks (nueva en algunas versiones)
        try:
            from xgboost import callback
            xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=[callback.EarlyStopping(rounds=200)]
            )
        except Exception:
            # Último recurso: entrenar sin early stopping
            xgb.fit(X_train, y_train)

    xgb_test_proba = xgb.predict_proba(X_test)[:,1]
    xgb_test_metrics = summarize_metrics(y_test, xgb_test_proba)

    # Guardar XGB
    xgb.save_model(str(MODELS_DIR / "xgb_model.json"))

    # ==========================
    # (Opcional) SHAP quick-look
    # ==========================
    shap_summary = {}
    if HAS_SHAP:
        try:
            # En arboles, usar TreeExplainer
            explainer = shap.TreeExplainer(xgb)
            # muestreo para velocidad
            samp = min(2048, X_train.shape[0])
            Xs = X_train[:samp]
            shap_values = explainer.shap_values(Xs)
            # top mean |SHAP| por feature
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(-mean_abs)[:20]
            shap_summary = { feature_cols[i]: float(mean_abs[i]) for i in top_idx }
        except Exception:
            shap_summary = {"info": "SHAP failed or not available."}

    # ==========================
    # Guardar columnas y métricas
    # ==========================
    with open(FEATURES_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    metrics = {
        "rf_val": rf_val_metrics,
        "rf_test": rf_test_metrics,
        "rf_calibrated_test": rf_cal_test_metrics,
        "xgb_test": xgb_test_metrics,
        "class_weights": cw_dict,
        "shap_top20" : shap_summary
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Reporte legible
    text_header = f"""Modelos entrenados (seed={RANDOM_STATE})
- Datos: {IN_PATH}
- Features: {len(feature_cols)}
- Train/Val/Test = 60/20/20
- RF: n_estimators=600 (calibrado con Platt)
- XGB: early_stopping, hist, lr=0.02
"""
    save_report(REPORT_TXT, text_header, metrics)

    print("✅ Entrenamiento finalizado.")
    print("  RF test:", rf_test_metrics)
    print("  RF calibrated test:", rf_cal_test_metrics)
    print("  XGB test:", xgb_test_metrics)
    if HAS_SHAP:
        print("  SHAP top20:", list(shap_summary.items())[:5])

if __name__ == "__main__":
    main()
