import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "logreg_model.pkl"

def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    coef = model.coef_[0]
    df_coef = pd.DataFrame({
        "feature": features,
        "coef": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)

    print("📊 Coeficientes de Logistic Regression (ordenados por importancia absoluta):\n")
    print(df_coef.to_string(index=False))

    print("\n🔎 Interpretación rápida:")
    print("- coef > 0  → favorece al equipo AZUL cuando esa feature aumenta.")
    print("- coef < 0  → favorece al equipo ROJO cuando esa feature aumenta.")
    print("- |coef| grande → feature más influyente en la decisión del modelo.")

if __name__ == "__main__":
    main()
