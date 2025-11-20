import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

PROC_DIR = Path("data/processed")
FEATS_PATH = PROC_DIR / "match_features_15_20_1.csv"
OUT_PATH = PROC_DIR / "pair_stats.npz"

TEAM_BLUE = ["Blue_1", "Blue_2", "Blue_3", "Blue_4", "Blue_5"]
TEAM_RED  = ["Red_1", "Red_2", "Red_3", "Red_4", "Red_5"]

def main():
    df = pd.read_csv(FEATS_PATH)

    # Aseguramos tipos int
    for c in TEAM_BLUE + TEAM_RED:
        df[c] = df[c].astype(int)

    max_id = int(df[TEAM_BLUE + TEAM_RED].max().max())
    n_champs = max_id + 1
    print(f"n_champs detectados: {n_champs}")

    synergy_games = np.zeros((n_champs, n_champs), dtype=np.int32)
    synergy_wins  = np.zeros((n_champs, n_champs), dtype=np.int32)

    vs_games = np.zeros((n_champs, n_champs), dtype=np.int32)
    vs_wins  = np.zeros((n_champs, n_champs), dtype=np.int32)

    for _, row in df.iterrows():
        blue = [int(row[c]) for c in TEAM_BLUE]
        red  = [int(row[c]) for c in TEAM_RED]
        win_blue = int(row["Winner"])  # 1 = gana azul, 0 = gana rojo

        # Synergy interna: pares dentro de cada equipo
        for i, j in combinations(blue, 2):
            synergy_games[i, j] += 1
            synergy_games[j, i] += 1
            if win_blue == 1:
                synergy_wins[i, j] += 1
                synergy_wins[j, i] += 1

        for i, j in combinations(red, 2):
            synergy_games[i, j] += 1
            synergy_games[j, i] += 1
            if win_blue == 0:
                synergy_wins[i, j] += 1
                synergy_wins[j, i] += 1

        # VS: cada champ de azul contra cada champ de rojo
        for i in blue:
            for j in red:
                vs_games[i, j] += 1
                vs_games[j, i] += 1  # simetría
                if win_blue == 1:
                    vs_wins[i, j] += 1
                else:
                    vs_wins[j, i] += 1

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        synergy_games=synergy_games,
        synergy_wins=synergy_wins,
        vs_games=vs_games,
        vs_wins=vs_wins,
    )
    print(f"✅ Guardado {OUT_PATH}")

if __name__ == "__main__":
    main()
