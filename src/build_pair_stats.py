# src/build_pair_stats.py

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

DATA_DIR = Path("data/matches")
PROC_DIR = Path("data/processed")
MATCHES_PATH = DATA_DIR / "matches_15_20_1.csv"
OUT_PATH = PROC_DIR / "pair_stats.npz"

TEAM_BLUE = ["BB1","BB2","BB3","BB4","BB5"]
TEAM_RED  = ["RB1","RB2","RB3","RB4","RB5"]

def main():
    df = pd.read_csv(MATCHES_PATH)
    # asumimos champion_id máximo razonable (puedes ajustarlo)
    max_id = int(max(df[TEAM_BLUE + TEAM_RED].max()))
    n_champs = max_id + 1

    synergy_games = np.zeros((n_champs, n_champs), dtype=np.int32)
    synergy_wins  = np.zeros((n_champs, n_champs), dtype=np.int32)

    vs_games = np.zeros((n_champs, n_champs), dtype=np.int32)
    vs_wins  = np.zeros((n_champs, n_champs), dtype=np.int32)

    for _, row in df.iterrows():
        blue = [int(row[c]) for c in TEAM_BLUE]
        red  = [int(row[c]) for c in TEAM_RED]
        win_blue = int(row["Winner"])  # 1 = gana azul, 0 = gana rojo

        # Synergy: pares dentro del mismo equipo (simétricos)
        for i, j in combinations(blue, 2):
            synergy_games[i, j] += 1
            synergy_games[j, i] += 1
            if win_blue == 1:
                synergy_wins[i, j] += 1
                synergy_wins[j, i] += 1

        for i, j in combinations(red, 2):
            synergy_games[i, j] += 1
            synergy_games[j, i] += 1
            if win_blue == 0:  # gana rojo
                synergy_wins[i, j] += 1
                synergy_wins[j, i] += 1

        # VS: i de azul contra j de rojo
        for i in blue:
            for j in red:
                vs_games[i, j] += 1
                vs_games[j, i] += 1  # también cuenta la inversa si quieres simetría
                if win_blue == 1:
                    vs_wins[i, j] += 1    # victoria de i sobre j
                else:
                    vs_wins[j, i] += 1    # victoria de j sobre i

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        synergy_games=synergy_games,
        synergy_wins=synergy_wins,
        vs_games=vs_games,
        vs_wins=vs_wins,
    )
    print(f"✅ Guardado {OUT_PATH} para n_champs={n_champs}")

if __name__ == "__main__":
    main()
