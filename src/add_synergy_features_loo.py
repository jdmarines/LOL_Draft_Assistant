# src/add_synergy_features_loo.py

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

PROC_DIR = Path("data/processed")
MATCH_FEATS_IN  = PROC_DIR / "match_features_15_20_1.csv"
PAIR_STATS_PATH = PROC_DIR / "pair_stats.npz"
MATCH_FEATS_OUT = PROC_DIR / "match_features_synergy_loo_15_20_1.csv"

TEAM_BLUE = ["Blue_1", "Blue_2", "Blue_3", "Blue_4", "Blue_5"]
TEAM_RED  = ["Red_1", "Red_2", "Red_3", "Red_4", "Red_5"]

ALPHA_SYNERGY = 10.0
ALPHA_VS      = 10.0

def smoothed_rate(wins, games, alpha):
    """Laplace smoothing: (wins + alpha) / (games + 2*alpha)."""
    return (wins + alpha) / (games + 2.0 * alpha)

def main():
    # 1) Cargar datos
    df = pd.read_csv(MATCH_FEATS_IN)
    data = np.load(PAIR_STATS_PATH)

    synergy_games_global = data["synergy_games"]
    synergy_wins_global  = data["synergy_wins"]
    vs_games_global      = data["vs_games"]
    vs_wins_global       = data["vs_wins"]

    # Aseguramos tipos int en los picks
    for c in TEAM_BLUE + TEAM_RED:
        df[c] = df[c].astype(int)

    blue_syn_loo = []
    red_syn_loo  = []
    blue_vs_loo  = []
    red_vs_loo   = []

    for _, row in df.iterrows():
        blue = [int(row[c]) for c in TEAM_BLUE]
        red  = [int(row[c]) for c in TEAM_RED]
        win_blue = int(row["Winner"])  # 1 gana Azul, 0 gana Rojo

        # -------------------------
        # 1) Synergy LOO (equipo)
        # -------------------------
        # Blue
        b_syn_vals = []
        for i, j in combinations(blue, 2):
            g_global = synergy_games_global[i, j]
            w_global = synergy_wins_global[i, j]

            # Esta partida siempre aporta 1 juego a ese par
            contrib_games = 1
            contrib_wins  = 1 if win_blue == 1 else 0

            g_loo = max(0, g_global - contrib_games)
            w_loo = max(0, w_global - contrib_wins)

            p_loo = smoothed_rate(w_loo, g_loo, ALPHA_SYNERGY)
            b_syn_vals.append(p_loo)
        b_syn = float(np.sum(b_syn_vals)) if b_syn_vals else 0.0

        # Red
        r_syn_vals = []
        for i, j in combinations(red, 2):
            g_global = synergy_games_global[i, j]
            w_global = synergy_wins_global[i, j]

            contrib_games = 1
            contrib_wins  = 1 if win_blue == 0 else 0  # gana rojo

            g_loo = max(0, g_global - contrib_games)
            w_loo = max(0, w_global - contrib_wins)

            p_loo = smoothed_rate(w_loo, g_loo, ALPHA_SYNERGY)
            r_syn_vals.append(p_loo)
        r_syn = float(np.sum(r_syn_vals)) if r_syn_vals else 0.0

        # -------------------------
        # 2) VS LOO (entre equipos)
        # -------------------------
        b_vs_vals = []
        r_vs_vals = []

        for i in blue:
            for j in red:
                # Blue_vs_Red: ventaja de i (blue) sobre j (red)
                g_ij_global = vs_games_global[i, j]
                w_ij_global = vs_wins_global[i, j]

                # esta partida siempre aporta 1 juego a (i,j)
                contrib_g_ij = 1
                contrib_w_ij = 1 if win_blue == 1 else 0

                g_ij_loo = max(0, g_ij_global - contrib_g_ij)
                w_ij_loo = max(0, w_ij_global - contrib_w_ij)

                p_ij_loo = smoothed_rate(w_ij_loo, g_ij_loo, ALPHA_VS)
                b_vs_vals.append(p_ij_loo)

                # Red_vs_Blue: ventaja de j (red) sobre i (blue)
                g_ji_global = vs_games_global[j, i]
                w_ji_global = vs_wins_global[j, i]

                contrib_g_ji = 1
                contrib_w_ji = 1 if win_blue == 0 else 0

                g_ji_loo = max(0, g_ji_global - contrib_g_ji)
                w_ji_loo = max(0, w_ji_global - contrib_w_ji)

                p_ji_loo = smoothed_rate(w_ji_loo, g_ji_loo, ALPHA_VS)
                r_vs_vals.append(p_ji_loo)

        b_vs = float(np.sum(b_vs_vals)) if b_vs_vals else 0.0
        r_vs = float(np.sum(r_vs_vals)) if r_vs_vals else 0.0

        blue_syn_loo.append(b_syn)
        red_syn_loo.append(r_syn)
        blue_vs_loo.append(b_vs)
        red_vs_loo.append(r_vs)

    # Añadir columnas al DataFrame
    df["Blue_Synergy_loo"] = blue_syn_loo
    df["Red_Synergy_loo"]  = red_syn_loo
    df["Delta_Synergy_loo"] = df["Blue_Synergy_loo"] - df["Red_Synergy_loo"]

    df["Blue_vs_Red_loo"] = blue_vs_loo
    df["Red_vs_Blue_loo"] = red_vs_loo
    df["Delta_vs_loo"]    = df["Blue_vs_Red_loo"] - df["Red_vs_Blue_loo"]

    df["Delta_Total_SynergyVS_loo"] = df["Delta_Synergy_loo"] + df["Delta_vs_loo"]

    df.to_csv(MATCH_FEATS_OUT, index=False)
    print(f"✅ Guardado {MATCH_FEATS_OUT} con sinergia/VS LOO (sin leakage).")

if __name__ == "__main__":
    main()
