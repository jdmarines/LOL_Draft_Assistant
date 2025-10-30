import re
import json
import yaml
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
EXT_DIR = Path("data/external")
OUT_DIR = Path("data/processed")

CONF_PATH = Path("conf/spell_keywords.yaml")

DD_FULL = RAW_DIR / "championFull.json"
BASE_RAW = RAW_DIR / "champs_base_raw.csv"
OPGG_META = EXT_DIR / "champs_meta_15_20_1.csv"  # opcional, si ya lo generaste

LOWER = lambda s: s.lower() if isinstance(s, str) else ""

def load_conf():
    with open(CONF_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dd():
    with open(DD_FULL, "r", encoding="utf-8") as f:
        return json.load(f)["data"]  # dict api_name -> champ_obj

def text_from_spell(sp):
    # Concatenamos campos relevantes por spell
    parts = []
    for k in ("name", "description", "tooltip"):
        if k in sp and sp[k]:
            parts.append(sp[k])
    return LOWER(" ".join(parts))

def count_hits(text, keywords):
    hits = 0
    for kw in keywords:
        if kw in text:
            hits += 1
    return hits

def flag_any(text, keywords):
    return any(kw in text for kw in keywords)

def extract_damage_mix_from_text(text, dmg_conf):
    t = text
    m_hits = sum(1 for kw in dmg_conf["magic"] if kw in t)
    p_hits = sum(1 for kw in dmg_conf["physical"] if kw in t)
    tr_hits = sum(1 for kw in dmg_conf["true"] if kw in t)
    total = m_hits + p_hits + tr_hits
    if total == 0:
        return 0.0, 0.0, 0.0
    return m_hits/total, p_hits/total, tr_hits/total

def normalize_score(value, max_value):
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0, value / max_value))

def main():
    conf = load_conf()
    dd = load_dd()
    df_raw = pd.read_csv(BASE_RAW)

    # Opcional: meta op.gg por rol (si existe)
    df_meta = None
    if OPGG_META.exists():
        df_meta = pd.read_csv(OPGG_META)

    # Preparamos salida
    rows = []

    for api_name, champ in dd.items():
        champ_id = int(champ["key"])
        spells = champ.get("spells", [])
        passive = champ.get("passive", {})
        passive_text = LOWER(passive.get("description", ""))

        # Parseamos spell por spell
        cc_hard_contrib = 0.0
        cc_soft_contrib = 0.0
        displacement_contrib = 0.0
        engage_contrib = 0.0
        mobility_contrib = 0.0
        peel_contrib = 0.0
        heal_contrib = 0.0
        zone_contrib = 0.0
        execute_contrib = 0.0

        aoe_any = False
        point_click_any = False
        projectile_any = False

        # Para damage mix, contamos hits sobre todo el kit
        magic_hits = physical_hits = true_hits = 0

        for sp in spells:
            st = text_from_spell(sp)

            # Señales principales
            cc_hard_contrib       += count_hits(st, conf["signals"]["cc_hard"]["keywords"])      * conf["signals"]["cc_hard"]["score"]
            cc_soft_contrib       += count_hits(st, conf["signals"]["cc_soft"]["keywords"])      * conf["signals"]["cc_soft"]["score"]
            displacement_contrib  += count_hits(st, conf["signals"]["displacement"]["keywords"]) * conf["signals"]["displacement"]["score"]
            engage_contrib        += count_hits(st, conf["signals"]["engage"]["keywords"])       * conf["signals"]["engage"]["score"]
            mobility_contrib      += count_hits(st, conf["signals"]["mobility"]["keywords"])     * conf["signals"]["mobility"]["score"]
            peel_contrib          += count_hits(st, conf["signals"]["peel_defense"]["keywords"]) * conf["signals"]["peel_defense"]["score"]
            heal_contrib          += count_hits(st, conf["signals"]["heal_sustain"]["keywords"]) * conf["signals"]["heal_sustain"]["score"]
            zone_contrib          += count_hits(st, conf["signals"]["zone_control"]["keywords"]) * conf["signals"]["zone_control"]["score"]
            execute_contrib       += count_hits(st, conf["signals"]["execute"]["keywords"])      * conf["signals"]["execute"]["score"]

            # Flags auxiliares
            if flag_any(st, conf["rules"]["aoe_keywords"]): aoe_any = True
            if flag_any(st, conf["rules"]["point_and_click_keywords"]): point_click_any = True
            if flag_any(st, conf["rules"]["projectile_keywords"]): projectile_any = True

            # Damage mix hits
            m, p, tr = extract_damage_mix_from_text(st, conf["signals"]["damage_types"])
            # re-convert from fractional-per-spell to "hits" by counting presence (0/1) if any of the keywords matched
            if m > 0:  magic_hits += 1
            if p > 0:  physical_hits += 1
            if tr > 0: true_hits += 1

        # Incluir pasiva en conteos suaves (peel/heal/zone, etc.)
        if passive_text:
            peel_contrib     += count_hits(passive_text, conf["signals"]["peel_defense"]["keywords"]) * conf["signals"]["peel_defense"]["score"]
            heal_contrib     += count_hits(passive_text, conf["signals"]["heal_sustain"]["keywords"]) * conf["signals"]["heal_sustain"]["score"]
            zone_contrib     += count_hits(passive_text, conf["signals"]["zone_control"]["keywords"]) * conf["signals"]["zone_control"]["score"]
            if flag_any(passive_text, conf["rules"]["aoe_keywords"]): aoe_any = True

            m, p, tr = extract_damage_mix_from_text(passive_text, conf["signals"]["damage_types"])
            if m > 0:  magic_hits += 1
            if p > 0:  physical_hits += 1
            if tr > 0: true_hits += 1

        # Scores agregados (antes de normalizar)
        cc_total = cc_hard_contrib + cc_soft_contrib + displacement_contrib

        # Normalización 0..1 usando “máximos razonables”
        norm = conf["normalization"]
        cc_score      = normalize_score(cc_total,            norm["max_cc_contrib"])
        engage_score  = normalize_score(engage_contrib,      norm["max_engage_contrib"])
        peel_score    = normalize_score(peel_contrib,        norm["max_peel_contrib"])
        zone_score    = normalize_score(zone_contrib,        norm["max_zone_contrib"])
        mobility_score= normalize_score(mobility_contrib,    norm["max_mobility_contrib"])

        # Damage mix ratios a partir de hits por spell (simple pero útil)
        dmg_total_hits = magic_hits + physical_hits + true_hits
        if dmg_total_hits == 0:
            magic_ratio = 0.0
            physical_ratio = 0.0
            true_ratio = 0.0
        else:
            magic_ratio = magic_hits / dmg_total_hits
            physical_ratio = physical_hits / dmg_total_hits
            true_ratio = true_hits / dmg_total_hits

        damage_mix = 1.0 - abs(magic_ratio - physical_ratio)  # 1=mixto, 0=polarizado

        rows.append({
            "champion_id": champ_id,
            "apiname": api_name,
            "cc_score": cc_score,
            "engage_score": engage_score,
            "peel_score": peel_score,
            "zone_control_score": zone_score,
            "mobility_score": mobility_score,
            "magic_ratio": magic_ratio,
            "physical_ratio": physical_ratio,
            "true_ratio": true_ratio,
            "damage_mix": damage_mix,
            "has_aoe_flag": int(aoe_any),
            "has_pointclick_flag": int(point_click_any),
            "has_projectile_flag": int(projectile_any),
        })

    df_feat = pd.DataFrame(rows)

    # Merge con base raw (stats numéricas y tags) por apiname o id
    df_raw_clean = df_raw.rename(columns={"id": "champion_id"})
    out = df_raw_clean.merge(df_feat, on=["champion_id", "apiname"], how="left")

    # Opcional: merge con meta op.gg (por nombre display o apiname_guess previamente normalizado)
    if df_meta is not None:
        # Intento de merge por display_name (df_raw) vs apiname_guess (meta)
        # Primero normalizamos para campeones con apóstrofes (Kai'Sa, K'Sante)
        # Simple heuristic: quita apóstrofes del display para join auxiliar
        def norm_name(s):
            return str(s).replace("'", "").lower()
        out["display_norm"] = out["display_name"].apply(norm_name)
        df_meta["display_norm"] = df_meta["apiname_guess"].apply(lambda x: norm_name(x))
        out = out.merge(df_meta, on="display_norm", how="left", suffixes=("", "_meta"))
        out.drop(columns=["display_norm"], inplace=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "champs_base_enriched.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Guardado {out_path} con {len(out)} filas y {len(out.columns)} columnas.")

if __name__ == "__main__":
    main()
