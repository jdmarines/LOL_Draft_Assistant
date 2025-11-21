"""
Motor de recomendación de picks para el draft de LoL.

Dado un estado parcial de composición (campeones azul/rojo),
simula añadir cada campeón posible y usa el modelo de regresión
logística entrenado para estimar la probabilidad de victoria.

Salidas:
- Top K picks recomendados.
- Probabilidad de victoria esperada.
- Mini explicación basada en las features más importantes.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import joblib


# =========================
# Configuración de rutas
# =========================

ROOT = Path(__file__).resolve().parents[1]  # /workspaces/LOL_Draft_Assistant
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "logreg_model.pkl"
CHAMPS_PATH = DATA_PROC / "champs_extended_15_20_1.csv"
PAIR_STATS_PATH = DATA_PROC / "pair_stats.npz"  # synergy / vs (global)


# =========================
# Data classes
# =========================

@dataclass
class Recommendation:
    champ_id: int
    champ_name: str
    prob_blue_win: float
    prob_red_win: float
    score: float
    explanation: str


# =========================
# Carga de recursos
# =========================

class Resources:
    def __init__(self):
        # Modelo
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.model_features: List[str] = bundle["features"]

        # Campeones extendidos (stats + meta + etiquetas)
        # Se asume al menos:
        #  - champ_id (int)
        #  - name (nombre "bonito")
        #  - apiname (nombre API / key)
        #  - attackrange, cc_score, engage_score, peel_score, zone_control_score,
        #    magic_ratio, phys_ratio, tankiness, win_rate_role, pick_rate_role,
        #    ban_rate_role, meta_strength, etc.
        self.champs_df = pd.read_csv(CHAMPS_PATH)

        # Normalizamos nombres clave
        # Ajusta aquí los nombres de columnas si en tu CSV son distintos
        expected_cols = [
            "champ_id", "name", "apiname",
            "attackrange",
            "cc_score", "engage_score", "peel_score", "zone_control_score",
            "magic_ratio", "phys_ratio", "tankiness",
            "win_rate_role", "pick_rate_role", "ban_rate_role",
            "meta_strength",
        ]
        missing = [c for c in expected_cols if c not in self.champs_df.columns]
        if missing:
            print(f"⚠️ Aviso: faltan columnas en champs_extended: {missing}")
            print("   Ajusta los nombres de columna en CHAMPS_PATH o en este script.")

        # Mapeos id ↔ nombre
        self.id2name: Dict[int, str] = (
            self.champs_df.set_index("champ_id")["name"].to_dict()
        )
        # name y apiname en minúsculas → champ_id
        name_map = {}
        for _, row in self.champs_df[["champ_id", "name", "apiname"]].iterrows():
            cid = int(row["champ_id"])
            for k in [str(row["name"]), str(row["apiname"])]:
                name_map[k.strip().lower()] = cid
        self.name2id: Dict[str, int] = name_map

        # Champion features para agregación por equipo
        self.champ_features = self.champs_df.set_index("champ_id")

        # Perfil promedio (para placeholders)
        self.champ_mean = self.champ_features.mean(numeric_only=True)

        # Synergy / VS matrices (si existen)
        self.synergy_available = False
        self.synergy_games = None
        self.synergy_wins = None
        self.vs_games = None
        self.vs_wins = None
        try:
            pair_data = np.load(PAIR_STATS_PATH)
            self.synergy_games = pair_data["synergy_games"]
            self.synergy_wins = pair_data["synergy_wins"]
            self.vs_games = pair_data["vs_games"]
            self.vs_wins = pair_data["vs_wins"]
            self.synergy_available = True
        except FileNotFoundError:
            print("⚠️ pair_stats.npz no encontrado. Sinergias y VS estarán a 0.")


RES = Resources()


# =========================
# Utilidades
# =========================

def normalize_champion(spec: Union[int, str]) -> int:
    """
    Convierte un campeón a champ_id interno.
    Acepta:
      - int: se devuelve tal cual
      - str: se busca en name2id usando name/apiname lowercase
    """
    if isinstance(spec, int):
        return spec
    if isinstance(spec, float) and spec.is_integer():
        return int(spec)

    key = str(spec).strip().lower()
    if key not in RES.name2id:
        raise ValueError(f"No se encontró campeón '{spec}' en el mapping name2id.")
    return RES.name2id[key]


def smoothed_rate(wins: int, games: int, alpha: float = 10.0) -> float:
    return (wins + alpha) / (games + 2.0 * alpha)


def team_stat_agg(team_ids: List[int], col: str, agg: str = "mean") -> float:
    """
    Agrega una columna de stats por equipo (mean/sum/min/max).

    - Si la columna no existe, retornamos 0.0.
    - Si el champ_id es -1 (placeholder), usamos un perfil NEUTRO
      que no aporte CC, engage, peel ni tankiness exagerado.
    """
    if col not in RES.champ_features.columns:
        return 0.0

    vals = []
    for cid in team_ids:
        if cid == -1:
            # Placeholder neutro (hueco en la comp)
            if col in ["cc_score", "engage_score", "peel_score", "zone_control_score"]:
                vals.append(0.0)
            elif col in ["tankiness"]:
                vals.append(0.0)
            elif "magic_ratio" in col:
                vals.append(0.5)  # daño mixto neutro
            elif "phys_ratio" in col:
                vals.append(0.5)
            elif "attackrange" in col:
                vals.append(450.0)  # rango medio estándar
            else:
                vals.append(0.0)
        else:
            vals.append(RES.champ_features.at[cid, col])

    if not vals:
        return 0.0

    vals = np.array(vals, dtype=float)
    if agg == "mean":
        return float(vals.mean())
    if agg == "sum":
        return float(vals.sum())
    if agg == "min":
        return float(vals.min())
    if agg == "max":
        return float(vals.max())
    raise ValueError(f"Agregación desconocida: {agg}")


def get_primary_role(cid: int) -> str:
    if "primary_role" in RES.champ_features.columns:
        return str(RES.champ_features.at[cid, "primary_role"])
    # si todavía no tenemos primary_role, tratamos todo como FLEX
    return "FLEX"


def role_penalty(cand_id: int, current_ids: List[int]) -> float:
    """
    Penaliza picks que repiten demasiado un rol (ej. segundo ADC).
    Devuelve un valor en [0, 1], donde 1 = sin penalización.
    """
    cand_role = get_primary_role(cand_id)
    if cand_role == "FLEX":
        return 1.0

    roles = [get_primary_role(cid) for cid in current_ids if cid != -1]
    count_same = sum(r == cand_role for r in roles)

    if count_same == 0:
        return 1.0      # primera vez ese rol -> ok
    if count_same == 1:
        return 0.8      # segundo campeón del mismo rol -> penalización ligera
    if count_same == 2:
        return 0.5      # ya exagerado
    return 0.3          # muy forzado (triple ADC, etc.)


def team_attackrange_top2_mean(team_ids: List[int]) -> float:
    vals = []
    for cid in team_ids:
        if cid == -1:
            vals.append(RES.champ_mean["attackrange"])
        else:
            vals.append(RES.champ_features.at[cid, "attackrange"])
    if not vals:
        return 0.0
    vals = np.sort(np.array(vals, dtype=float))
    top2 = vals[-2:] if len(vals) >= 2 else vals
    return float(top2.mean())


def compute_synergy_and_vs(
    blue_ids: List[int],
    red_ids: List[int],
    alpha_synergy: float = 10.0,
    alpha_vs: float = 10.0,
) -> Dict[str, float]:
    """
    Calcula Blue_Synergy, Red_Synergy, Blue_vs_Red, Red_vs_Blue
    usando las matrices globales (NO LOO, pero sirve para inferencia).
    """
    if not RES.synergy_available:
        return {
            "Blue_Synergy_loo": 0.0,
            "Red_Synergy_loo": 0.0,
            "Delta_Synergy_loo": 0.0,
            "Blue_vs_Red_loo": 0.0,
            "Red_vs_Blue_loo": 0.0,
            "Delta_vs_loo": 0.0,
            "Delta_Total_SynergyVS_loo": 0.0,
        }

    from itertools import combinations

    # Synergy interna
    blue_syn_vals = []
    for i, j in combinations(blue_ids, 2):
        if i < 0 or j < 0:
            continue
        g = RES.synergy_games[i, j]
        w = RES.synergy_wins[i, j]
        blue_syn_vals.append(smoothed_rate(w, g, alpha_synergy))

    red_syn_vals = []
    for i, j in combinations(red_ids, 2):
        if i < 0 or j < 0:
            continue
        g = RES.synergy_games[i, j]
        w = RES.synergy_wins[i, j]
        red_syn_vals.append(smoothed_rate(w, g, alpha_synergy))

    blue_syn = float(np.sum(blue_syn_vals)) if blue_syn_vals else 0.0
    red_syn = float(np.sum(red_syn_vals)) if red_syn_vals else 0.0

    # VS entre equipos
    blue_vs_vals = []
    red_vs_vals = []
    for i in blue_ids:
        for j in red_ids:
            if i < 0 or j < 0:
                continue
            # ventaja de i (blue) sobre j (red)
            g_ij = RES.vs_games[i, j]
            w_ij = RES.vs_wins[i, j]
            blue_vs_vals.append(smoothed_rate(w_ij, g_ij, alpha_vs))

            # ventaja de j (red) sobre i (blue)
            g_ji = RES.vs_games[j, i]
            w_ji = RES.vs_wins[j, i]
            red_vs_vals.append(smoothed_rate(w_ji, g_ji, alpha_vs))

    blue_vs = float(np.sum(blue_vs_vals)) if blue_vs_vals else 0.0
    red_vs = float(np.sum(red_vs_vals)) if red_vs_vals else 0.0

    return {
        "Blue_Synergy_loo": blue_syn,
        "Red_Synergy_loo": red_syn,
        "Delta_Synergy_loo": blue_syn - red_syn,
        "Blue_vs_Red_loo": blue_vs,
        "Red_vs_Blue_loo": red_vs,
        "Delta_vs_loo": blue_vs - red_vs,
        "Delta_Total_SynergyVS_loo": (blue_syn - red_syn) + (blue_vs - red_vs),
    }


# =========================
# Construcción de features
# =========================

def build_features_for_draft(
    blue_ids_raw: List[Union[int, str]],
    red_ids_raw: List[Union[int, str]],
) -> Dict[str, float]:
    """
    Construye el diccionario de features (igual estructura que entrenamiento)
    a partir de una composición de 0-5 campeones azul y 0-5 rojos.

    blue_ids_raw / red_ids_raw pueden ser ints (champ_id) o strings (nombre).
    """
    # Normalizamos a champ_id
    blue_ids = [normalize_champion(c) for c in blue_ids_raw]
    red_ids = [normalize_champion(c) for c in red_ids_raw]

    # Rellenamos a tamaño 5 con placeholders (-1)
    while len(blue_ids) < 5:
        blue_ids.append(-1)
    while len(red_ids) < 5:
        red_ids.append(-1)

    # =============== Stats base y meta ===============
    feats: Dict[str, float] = {}

    # Ban / pick / win rate role (media)
    feats["Blue_ban_rate_role_mean"] = team_stat_agg(blue_ids, "ban_rate_role", "mean")
    feats["Red_ban_rate_role_mean"] = team_stat_agg(red_ids, "ban_rate_role", "mean")
    feats["Delta_ban_rate_role_mean"] = (
        feats["Blue_ban_rate_role_mean"] - feats["Red_ban_rate_role_mean"]
    )

    feats["Blue_win_rate_role_mean"] = team_stat_agg(blue_ids, "win_rate_role", "mean")
    feats["Red_win_rate_role_mean"] = team_stat_agg(red_ids, "win_rate_role", "mean")
    feats["Delta_win_rate_role_mean"] = (
        feats["Blue_win_rate_role_mean"] - feats["Red_win_rate_role_mean"]
    )

    feats["Blue_pick_rate_role_mean"] = team_stat_agg(blue_ids, "pick_rate_role", "mean")
    feats["Red_pick_rate_role_mean"] = team_stat_agg(red_ids, "pick_rate_role", "mean")
    feats["Delta_pick_rate_role_mean"] = (
        feats["Blue_pick_rate_role_mean"] - feats["Red_pick_rate_role_mean"]
    )

    feats["Blue_meta_strength"] = team_stat_agg(blue_ids, "meta_strength", "mean")
    feats["Red_meta_strength"] = team_stat_agg(red_ids, "meta_strength", "mean")
    feats["Delta_meta_strength"] = (
        feats["Blue_meta_strength"] - feats["Red_meta_strength"]
    )

    # CC, engage, peel, zoning
    for col in ["cc_score", "engage_score", "peel_score", "zone_control_score"]:
        b_key = f"Blue_{col}_sum"
        r_key = f"Red_{col}_sum"
        d_key = f"Delta_{col}_sum"

        feats[b_key] = team_stat_agg(blue_ids, col, "sum")
        feats[r_key] = team_stat_agg(red_ids, col, "sum")
        feats[d_key] = feats[b_key] - feats[r_key]

    # Re-etiquetamos para que coincida EXACTAMENTE con tus nombres:
    feats["Delta_cc_score_sum"] = feats["Delta_cc_score_sum"]
    feats["Delta_engage_score_sum"] = feats["Delta_engage_score_sum"]
    feats["Delta_peel_score_sum"] = feats["Delta_peel_score_sum"]
    feats["Delta_zone_control_score_sum"] = feats["Delta_zone_control_score_sum"]

    # Tankiness (suma)
    feats["Blue_tankiness_sum"] = team_stat_agg(blue_ids, "tankiness", "sum")
    feats["Red_tankiness_sum"] = team_stat_agg(red_ids, "tankiness", "sum")
    feats["Delta_tankiness_sum"] = (
        feats["Blue_tankiness_sum"] - feats["Red_tankiness_sum"]
    )

    # Ratios de daño (sum)
    feats["Blue_phys_ratio_sum"] = team_stat_agg(blue_ids, "phys_ratio", "sum")
    feats["Red_phys_ratio_sum"] = team_stat_agg(red_ids, "phys_ratio", "sum")
    feats["Delta_phys_ratio_sum"] = (
        feats["Blue_phys_ratio_sum"] - feats["Red_phys_ratio_sum"]
    )

    feats["Blue_magic_ratio_sum"] = team_stat_agg(blue_ids, "magic_ratio", "sum")
    feats["Red_magic_ratio_sum"] = team_stat_agg(red_ids, "magic_ratio", "sum")
    feats["Delta_magic_ratio_sum"] = (
        feats["Blue_magic_ratio_sum"] - feats["Red_magic_ratio_sum"]
    )

    # Rango (mean / min / max / top2_mean)
    feats["Blue_attackrange_mean"] = team_stat_agg(blue_ids, "attackrange", "mean")
    feats["Red_attackrange_mean"] = team_stat_agg(red_ids, "attackrange", "mean")
    feats["Delta_attackrange_mean"] = (
        feats["Blue_attackrange_mean"] - feats["Red_attackrange_mean"]
    )

    feats["Blue_attackrange_min"] = team_stat_agg(blue_ids, "attackrange", "min")
    feats["Red_attackrange_min"] = team_stat_agg(red_ids, "attackrange", "min")
    feats["Delta_attackrange_min"] = (
        feats["Blue_attackrange_min"] - feats["Red_attackrange_min"]
    )

    feats["Blue_attackrange_max"] = team_stat_agg(blue_ids, "attackrange", "max")
    feats["Red_attackrange_max"] = team_stat_agg(red_ids, "attackrange", "max")
    feats["Delta_attackrange_max"] = (
        feats["Blue_attackrange_max"] - feats["Red_attackrange_max"]
    )

    feats["Blue_range_top2_mean"] = team_attackrange_top2_mean(blue_ids)
    feats["Red_range_top2_mean"] = team_attackrange_top2_mean(red_ids)
    feats["Delta_range_top2_mean"] = (
        feats["Blue_range_top2_mean"] - feats["Red_range_top2_mean"]
    )

    # Synergy y VS (usando matrices globales, aproximación LOO-compatible)
    syn_vs = compute_synergy_and_vs(blue_ids, red_ids)
    feats.update(syn_vs)

    # Counters: de momento los ponemos en 0 (si quieres podemos conectar op.gg aquí)
    feats["Delta_counters_count"] = 0.0
    feats["Delta_counters_adv"] = 0.0

    # Ahora filtramos SOLO las features que el modelo espera
    x_feats = {}
    for f in RES.model_features:
        if f in feats:
            x_feats[f] = float(feats[f])
        else:
            # Si falta alguna, la ponemos a 0.0 (mejor que lanzar error).
            x_feats[f] = 0.0

    return x_feats


# =========================
# Predicción y recomendación
# =========================

def predict_blue_win_prob(features: Dict[str, float]) -> float:
    """
    Devuelve P(ganar Azul) usando el modelo logístico.
    """
    x = np.array([[features[f] for f in RES.model_features]], dtype=float)
    proba = RES.model.predict_proba(x)[0, 1]  # clase 1 = azul gana
    return float(proba)


def explain_candidate(champ_id, new_feats, base_feats):
    # deltas entre draft base y draft con el pick hipotético
    deltas = {k: new_feats[k] - base_feats.get(k, 0) for k in new_feats}
    factors = []

    def add_if(key, label, threshold=0.1):
        if key in deltas and abs(deltas[key]) > threshold:
            val = deltas[key]
            sign = "+" if val > 0 else "-"
            factors.append((abs(val), f"{sign} {label} (Δ {val:.2f})"))

    # ---- CC / engage / peel / zona ----
    add_if("Delta_cc_score_sum", "más CC total", 0.3)
    add_if("Delta_engage_score_sum", "más herramientas de engage", 0.3)
    add_if("Delta_peel_score_sum", "mejor peel / protección", 0.3)
    add_if("Delta_zone_control_score_sum", "mejor control de zona", 0.3)

    # ---- Rango: tratamos distinto, sin mostrar el número bruto ----
    def add_range(key, label_base):
        if key not in deltas:
            return
        val = deltas[key]
        if val <= 0:
            return
        # Clasificamos el impacto en leve / medio / fuerte
        if val > 150:
            etiqueta = f"{label_base} (aumento MUY fuerte)"
        elif val > 75:
            etiqueta = f"{label_base} (aumento moderado)"
        else:
            etiqueta = f"{label_base} (aumento leve)"
        factors.append((abs(val), f"+ {etiqueta}"))

    add_range("Delta_range_top2_mean", "mejor rango en tus carries")
    add_range("Delta_attackrange_mean", "mayor rango promedio")

    # ---- Daño físico / mágico ----
    add_if("Delta_magic_ratio_sum", "mejor balance de daño mágico", 0.2)
    add_if("Delta_phys_ratio_sum", "mejor balance de daño físico", 0.2)

    # ordenar por importancia y quedarnos con top 4
    factors.sort(key=lambda x: x[0], reverse=True)
    textos = [txt for _, txt in factors[:4]]

    if not textos:
        return "(pick neutro, sin cambios tácticos fuertes)"
    return " | ".join(textos)



def recommend_for(
    blue_champs: List[Union[int, str]],
    red_champs: List[Union[int, str]],
    side: str = "blue",
    top_k: int = 5,
) -> List[Recommendation]:
    """
    Recomienda top_k picks para el lado indicado ("blue" o "red"),
    dado el estado actual del draft.

    blue_champs / red_champs pueden ser:
      - champ_id (int)
      - nombre "Ahri", "Miss Fortune", "Aurelion Sol", etc.
    """
    # Normalizamos listas actuales
    blue_ids = [normalize_champion(c) for c in blue_champs]
    red_ids = [normalize_champion(c) for c in red_champs]
    used_ids = set(blue_ids + red_ids)

    # Lista de todos los campeones posibles
    all_ids = list(RES.champ_features.index.astype(int))
    candidate_ids = [cid for cid in all_ids if cid not in used_ids]

    # Baseline: composición con placeholders donde falten picks
    base_blue = blue_ids.copy()
    base_red = red_ids.copy()
    while len(base_blue) < 5:
        base_blue.append(-1)
    while len(base_red) < 5:
        base_red.append(-1)
    base_feats = build_features_for_draft(base_blue, base_red)
    base_p = predict_blue_win_prob(base_feats)

    recs: List[Recommendation] = []

    for cid in candidate_ids:
        if side == "blue":
            new_blue = blue_ids + [cid]
            new_red = red_ids
        else:
            new_blue = blue_ids
            new_red = red_ids + [cid]

        feats = build_features_for_draft(new_blue, new_red)
        p_blue = predict_blue_win_prob(feats)
        p_red = 1.0 - p_blue

        # Score: si es azul, queremos maximizar p_blue.
        # Si es rojo, queremos maximizar p_red.
        base_score = p_blue if side == "blue" else p_red    
        if side == "blue":
            current_ids = blue_ids
        else:
            current_ids = red_ids

        pen = role_penalty(cid, current_ids)
        score = base_score * pen


        expl = explain_candidate(cid, feats, base_feats)
        recs.append(
            Recommendation(
                champ_id=cid,
                champ_name=RES.id2name.get(cid, f"champ_{cid}"),
                prob_blue_win=p_blue,
                prob_red_win=p_red,
                score=score,
                explanation=expl,
            )
        )

    recs_sorted = sorted(recs, key=lambda r: r.score, reverse=True)
    return recs_sorted[:top_k]


# =========================
# Ejemplo de uso CLI
# =========================

if __name__ == "__main__":
    # Ejemplo: estado parcial de draft
    # Azul: Orianna, Jinx
    # Rojo: Malphite
    blue = ["Orianna", "Jinx"]
    red = ["Malphite"]

    print("🔵 Recomendaciones para BLUE (siguiente pick):")
    recs_blue = recommend_for(blue, red, side="blue", top_k=5)
    for r in recs_blue:
        print(
            f"- {r.champ_name:15s} | P(Blue win)={r.prob_blue_win:.3f} | "
            f"P(Red win)={r.prob_red_win:.3f}\n  {r.explanation}"
        )

    print("\n🔴 Recomendaciones para RED (siguiente pick):")
    recs_red = recommend_for(blue, red, side="red", top_k=5)
    for r in recs_red:
        print(
            f"- {r.champ_name:15s} | P(Blue win)={r.prob_blue_win:.3f} | "
            f"P(Red win)={r.prob_red_win:.3f}\n  {r.explanation}"
        )

