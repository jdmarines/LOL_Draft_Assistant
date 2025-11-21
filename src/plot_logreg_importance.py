import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Rutas
ROOT = Path(__file__).resolve().parents[1]  # /workspaces/LOL_Draft_Assistant
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "reports" / "figures"

MODEL_PATH = MODELS_DIR / "logreg_model.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) Cargar modelo
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    coef = model.coef_[0]
    df_coef = pd.DataFrame({
        "feature": features,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False)

    # 2) Seleccionar top N features más importantes
    TOP_N = 15
    df_top = df_coef.head(TOP_N).copy()

    # 3) Gráfico de importancia absoluta
    plt.figure(figsize=(8, 6))
    plt.barh(df_top["feature"], df_top["abs_coef"])
    plt.title("Importancia absoluta de las features (Logistic Regression)")
    plt.xlabel("Valor absoluto del coeficiente")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_abs_path = OUT_DIR / "logreg_importancia_absoluta.png"
    plt.savefig(fig_abs_path, dpi=300)
    plt.close()

    # 4) Gráfico de coeficientes con signo
    plt.figure(figsize=(8, 6))
    plt.barh(df_top["feature"], df_top["coef"])
    plt.title("Coeficientes (con signo) - Logistic Regression")
    plt.xlabel("Coeficiente")
    plt.axvline(0, linewidth=1)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_signed_path = OUT_DIR / "logreg_coeficientes_signo.png"
    plt.savefig(fig_signed_path, dpi=300)
    plt.close()

    print("✅ Figuras generadas:")
    print(f" - {fig_abs_path}")
    print(f" - {fig_signed_path}")
    print("\nTop coeficientes:")
    print(df_top.to_string(index=False))


if __name__ == "__main__":
    main()
