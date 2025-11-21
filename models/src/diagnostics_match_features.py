# src/diagnostics_match_features.py

import pandas as pd
import numpy as np
from pathlib import Path

PROC_DIR = Path("data/processed")
IN_PATH = PROC_DIR / "match_features_15_20_1.csv"

def main():
    df = pd.read_csv(IN_PATH)
    print(f"📦 Dataset: {IN_PATH}")
    print(f"- Filas: {len(df)}, Columnas: {len(df.columns)}\n")

    # 1) Balance de clases
    assert "Winner" in df.columns, "Falta columna Winner"
    vc = df["Winner"].value_counts(normalize=True)
    print("🎯 Balance de clases (Winner):")
    for k, v in vc.items():
        print(f"  {k}: {v:.3f}")
    print()

    # 2) NaNs por columna
    na_counts = df.isna().sum()
    na_pct = (na_counts / len(df)) * 100
    na_df = pd.DataFrame({
        "na_count": na_counts,
        "na_pct": na_pct
    }).sort_values("na_pct", ascending=False)

    print("🧪 Top 20 columnas con más NaN:")
    print(na_df.head(20))
    print()

    # 3) Columnas Delta_*
    delta_cols = [c for c in df.columns if c.startswith("Delta_")]
    print(f"🔁 Columnas Delta_*: {len(delta_cols)}")
    print(delta_cols[:20], "..." if len(delta_cols) > 20 else "")
    print()

    # 4) Columnas de counters
    counter_cols = [c for c in df.columns if "counter" in c.lower()]
    print(f"⚔️ Columnas relacionadas con counters: {len(counter_cols)}")
    print(counter_cols)
    print()

    # 5) Columnas numéricas potencialmente útiles (core)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    core_candidates = [c for c in num_cols if c.startswith("Delta_")]
    core_candidates += [c for c in num_cols if c.startswith("Blue_") or c.startswith("Red_")]
    core_candidates = sorted(set(core_candidates) - {"Winner"})
    print(f"🧩 Columnas numéricas (core candidates): {len(core_candidates)}")
    print(core_candidates[:30], "..." if len(core_candidates) > 30 else "")

    # 6) Sugerir columnas con NA>20% como candidatas a eliminar
    high_na = na_df[na_df["na_pct"] > 20].index.tolist()
    print("\n⚠️ Columnas con más de 20% NaN (candidatas a eliminar):")
    print(high_na)

if __name__ == "__main__":
    main()
