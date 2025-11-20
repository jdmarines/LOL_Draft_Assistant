# src/add_synergy_features.py

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

PROC_DIR = Path("data/processed")
MATCH_FEATS_IN  = PROC_DIR / "match_features_15_20_1.csv"
PAIR_STATS_PATH = PROC_DIR / "pair_stats.npz"
MATCH_FEATS_OUT = PROC_DIR / "match_features_synergy_15_20_1.csv"

TEAM_BLUE = ["Blue_1", "Blue_2", "Blue_3", "Blue_4", "Blue_5"]
TEAM_RED  = ["Red_1", "Red_2", "Red_3", "Red_4", "Red_5"]

ALPHA_SYNERGY = 10.0
ALPHA_VS      = 10.0

def smoothed_rate(wins, games, alpha):
    return (wins + alpha) / (games + 2.0 * alpha)

def compute_team_synergy(team_ids, synergy_games, synergy_wins, alpha):
    vals = []
    for i, j in combinations(team_ids, 2):
        g = synergy_games[i, j]
        w = synergy_wins[i, j]
        p_hat = smoothed_rate(w, g, alpha)
        vals.append(p_hat)
    return float(np.sum(vals)) if vals else 0.0

def compute_vs_advantage(blue_ids, red_ids, vs_games, vs_wins, alpha):
    vals_blue = []
    vals_red = []
    for i in blue_ids:
        for j in red_ids:
            g_ij = vs_games[i, j]
            w_ij = vs_wins[i, j]
            p_ij = smoothed_rate(w_ij, g_ij, alpha)
            vals_blue.append(p_ij)

            g_ji = vs_games[j, i]
            w_ji = vs_wins[j, i]
            p_ji = smoothed_rate(w_ji, g_ji, alpha)
            vals_red.append(p_ji)
    return float(np.sum(vals_blue)), float(np.sum(vals_red))

def main():
    df = pd.read_csv(MATCH_FEATS_IN)
    data = np.load(PAIR_STATS_PATH)

    synergy_games = data["synergy_games"]
    synergy_wins  = data["synergy_wins"]
    vs_games      = data["vs_games"]
    vs_wins       = data["vs_wins"]

    # aseguramos que las columnas de picks sean int
    for c in TEAM_BLUE + TEAM_RED:
        df[c] = df[c].astype(int)

    blue_syn_list = []
    red_syn_list  = []
    blue_vs_list  = []
    red_vs_list   = []

    for _, row in df.iterrows():
        blue = [int(row[c]) for c in TEAM_BLUE]
        red  = [int(row[c]) for c in TEAM_RED]

        b_syn = compute_team_synergy(blue, synergy_games, synergy_wins, ALPHA_SYNERGY)
        r_syn = compute_team_synergy(red,  synergy_games, synergy_wins, ALPHA_SYNERGY)

        b_vs, r_vs = compute_vs_advantage(blue, red, vs_games, vs_wins, ALPHA_VS)

        blue_syn_list.append(b_syn)
        red_syn_list.append(r_syn)
        blue_vs_list.append(b_vs)
        red_vs_list.append(r_vs)

    df["Blue_Synergy"] = blue_syn_list
    df["Red_Synergy"]  = red_syn_list
    df["Delta_Synergy"] = df["Blue_Synergy"] - df["Red_Synergy"]

    df["Blue_vs_Red"] = blue_vs_list
    df["Red_vs_Blue"] = red_vs_list
    df["Delta_vs"]    = df["Blue_vs_Red"] - df["Red_vs_Blue"]

    df["Delta_Total_SynergyVS"] = df["Delta_Synergy"] + df["Delta_vs"]

    df.to_csv(MATCH_FEATS_OUT, index=False)
    print(f"✅ Guardado {MATCH_FEATS_OUT} con columnas de sinergia y VS añadidas.")

if __name__ == "__main__":
    main()
