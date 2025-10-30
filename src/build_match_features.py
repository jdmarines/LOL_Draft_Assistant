import pandas as pd
import numpy as np
from pathlib import Path

# Rutas
PROC_DIR = Path("data/processed")
RAW_DIR = Path("data")
MATCHES_PATH = RAW_DIR / "matches" / "matches_15_20_1.csv"
CHAMPS_ENRICHED = PROC_DIR / "champs_base_enriched.csv"
OUT_PATH = PROC_DIR / "match_features_15_20_1.csv"

# -------------------------------
# Utilidades
# -------------------------------

TEAM_BLUE = ["BB1","BB2","BB3","BB4","BB5"]
TEAM_RED  = ["RB1","RB2","RB3","RB4","RB5"]

# columnas de champion-level que agregaremos por equipo
AGG_MEAN_COLS = [
    "attackrange",
    "cc_score","engage_score","peel_score","zone_control_score","mobility_score",
    "magic_ratio","physical_ratio","damage_mix",
    "hp","armor","spellblock",
    "hpperlevel","attackdamageperlevel","attackspeedperlevel"
]

AGG_SUM_COLS = [
    "cc_score","engage_score","peel_score","zone_control_score"
]

# meta opcional (puede no existir, entonces las ignoramos si faltan)
META_COLS = ["win_rate_role","pick_rate_role","ban_rate_role","tier_role"]

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def compute_tankiness(df_row):
    # Heurística simple y estable (si no tienes tankiness explícito):
    # escala lineal de hp + armor + mr (spellblock). Ajusta si quieres pesos.
    return df_row["hp"] + 2.0*df_row["armor"] + 2.0*df_row["spellblock"]

def reduce_counters_series(counters_ids, counters_wr):
    """
    counters_ids: "950;133;10"
    counters_wr:  "0.4425;0.4810;0.4863"
    Devuelve lista de (id, wr) como tuplas.
    """
    if pd.isna(counters_ids) or len(str(counters_ids)) == 0:
        return []
    ids = str(counters_ids).split(";")
    wrs = str(counters_wr).split(";") if (pd.notna(counters_wr) and len(str(counters_wr))>0) else []
    wrs = wrs + [""]*(len(ids)-len(wrs))
    out = []
    for i, w in zip(ids, wrs):
        try:
            cid = int(i)
        except:
            continue
        try:
            wr = float(w) if w not in ("", "NA") else np.nan
        except:
            wr = np.nan
        out.append((cid, wr))
    return out

def build_counter_lookup(df_ch):
    """
    Construye un diccionario: champion_id -> {counter_id: counter_wr_vs_me, ...}
    donde 'counter_wr_vs_me' es el winrate del counter ENFRENTANDO a este campeón (op.gg).
    """
    lookup = {}
    for _, r in df_ch.iterrows():
        me = int(r["champion_id"])
        pairs = reduce_counters_series(r.get("counters_ids", np.nan), r.get("counters_wr", np.nan))
        if pairs:
            lookup[me] = {cid: wr for cid, wr in pairs if not np.isnan(wr)}
        else:
            lookup[me] = {}
    return lookup

def count_enemy_counters(team_ids, enemy_ids, counter_lookup):
    """
    Cuenta cuántos campeones del enemigo son counters directos de mis picks, y su 'fuerza' media.
    Retorna:
      - count_counters (int)
      - mean_counter_wr (float)  -> promedio de winrate del counter vs mí (más alto = peor para mí)
    """
    wrs = []
    cnt = 0
    for my_id in team_ids:
        lookup = counter_lookup.get(int(my_id), {})
        for e in enemy_ids:
            if int(e) in lookup:
                cnt += 1
                wrs.append(lookup[int(e)])
    mean_wr = float(np.mean(wrs)) if len(wrs) > 0 else np.nan
    return cnt, mean_wr

def team_aggregate(df_champs, ids, prefix):
    """
    ids: iterable de champion_id (5 picks)
    Devuelve serie con agregados prefijados (mean/sum/max/min/var en algunas columnas).
    """
    sub = df_champs.loc[df_champs["champion_id"].isin([int(x) for x in ids])].copy()
    # fallback si algo falta
    if len(sub) == 0:
        return pd.Series(dtype="float64")

    # Tankiness (si no la tienes ya)
    if "tankiness" not in sub.columns:
        sub["tankiness"] = sub.apply(compute_tankiness, axis=1)

    # Agregados
    feats = {}
    for c in safe_cols(sub, AGG_MEAN_COLS + META_COLS + ["tankiness"]):
        feats[f"{prefix}_{c}_mean"] = sub[c].mean()
        feats[f"{prefix}_{c}_max"]  = sub[c].max()
        feats[f"{prefix}_{c}_min"]  = sub[c].min()
        feats[f"{prefix}_{c}_std"]  = sub[c].std(ddof=0)

    for c in safe_cols(sub, AGG_SUM_COLS + ["tankiness"]):
        feats[f"{prefix}_{c}_sum"] = sub[c].sum()

    # Rango específico útil
    if "attackrange" in sub.columns:
        feats[f"{prefix}_range_top2_mean"] = sub["attackrange"].nlargest(2).mean()
        feats[f"{prefix}_range_min"] = sub["attackrange"].min()

    # Mezcla de daño resumida
    if set(["magic_ratio","physical_ratio"]).issubset(sub.columns):
        feats[f"{prefix}_magic_ratio_sum"] = sub["magic_ratio"].sum()
        feats[f"{prefix}_phys_ratio_sum"]  = sub["physical_ratio"].sum()

    # Meta composites (si existen)
    if set(["win_rate_role","pick_rate_role"]).issubset(sub.columns):
        feats[f"{prefix}_meta_strength"] = (sub["win_rate_role"] * (sub["pick_rate_role"]+1e-6)).mean()

    return pd.Series(feats)

def build_features(df_matches, df_champs):
    # Preconstruimos lookup de counters si hay columnas
    has_counters = set(["counters_ids","counters_wr"]).issubset(df_champs.columns)
    counter_lookup = build_counter_lookup(df_champs) if has_counters else {}

    rows = []
    for idx, row in df_matches.iterrows():
        blue_ids = [row[c] for c in TEAM_BLUE]
        red_ids  = [row[c] for c in TEAM_RED ]

        # Agregados por equipo
        blue = team_aggregate(df_champs, blue_ids, prefix="Blue")
        red  = team_aggregate(df_champs, red_ids,  prefix="Red")

        feat = pd.concat([blue, red])

        # Deltas (Blue - Red) para algunas métricas clave
        delta_keys = [
            "cc_score_sum","engage_score_sum","peel_score_sum","zone_control_score_sum","mobility_score_sum",
            "tankiness_sum",
            "attackrange_mean","attackrange_max","attackrange_min","range_top2_mean",
            "magic_ratio_sum","phys_ratio_sum",
            "win_rate_role_mean","pick_rate_role_mean","ban_rate_role_mean","meta_strength"
        ]

        for k in delta_keys:
            bk = f"Blue_{k}"
            rk = f"Red_{k}"
            if bk in feat.index and rk in feat.index:
                feat[f"Delta_{k}"] = feat[bk] - feat[rk]

        # Features de counters (si existen)
        if has_counters:
            b_cnt, b_wr = count_enemy_counters(blue_ids, red_ids, counter_lookup)
            r_cnt, r_wr = count_enemy_counters(red_ids, blue_ids, counter_lookup)
            feat["Blue_counters_faced_count"] = b_cnt
            feat["Blue_counters_faced_wr_mean"] = b_wr
            feat["Red_counters_faced_count"] = r_cnt
            feat["Red_counters_faced_wr_mean"] = r_wr
            feat["Delta_counters_count"] = b_cnt - r_cnt
            # Nota: menor mean_wr es mejor para el equipo (el counter tiene menor winrate vs mí)
            # Convertimos a "ventaja" aproximada:
            if not np.isnan(b_wr) and not np.isnan(r_wr):
                feat["Delta_counters_adv"] = (1.0 - b_wr) - (1.0 - r_wr)  # positivo = ventaja Azul

        # Target
        feat["Winner"] = row["Winner"]

        # Identidad (opcional para debug)
        for i, cid in enumerate(blue_ids, 1):
            feat[f"Blue_{i}"] = cid
        for i, cid in enumerate(red_ids, 1):
            feat[f"Red_{i}"] = cid

        rows.append(feat)

    df = pd.DataFrame(rows).reset_index(drop=True)

    # Limpieza de inf/NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    # imputación mínima (promedio de columna)
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    return df

def main():
    # Cargar campeones enriquecidos y partidas
    df_ch = pd.read_csv(CHAMPS_ENRICHED)
    df_mt = pd.read_csv(MATCHES_PATH)

    # Asegurar tipos
    for c in TEAM_BLUE + TEAM_RED:
        df_mt[c] = df_mt[c].astype(int)

    # Construir features
    out = build_features(df_mt, df_ch)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Guardado {OUT_PATH} | filas={len(out)} cols={len(out.columns)}")

if __name__ == "__main__":
    main()
