import json
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/championFull.json")
OUT_PATH = Path("data/processed/champs_extended_15_20_1.csv")

def compute_magic_ratio(stats):
    ap_sources = ["magic", "magicDamage", "spellDamage"]
    # naive heuristic
    return 1.0 if stats["spellblock"] > stats["armor"] else 0.5

def compute_phys_ratio(stats):
    return 1.0 - compute_magic_ratio(stats)

def compute_tankiness(stats):
    # escala balanceada para ML: ~1–3
    raw = stats["hp"] + stats["armor"] * 20 + stats["spellblock"] * 20
    return raw / 1000.0


def main():
    print("Leyendo championFull.json...")
    data = json.load(open(RAW_PATH, "r", encoding="utf-8"))

    rows = []
    for cname, cdata in data["data"].items():
        info = {}
        info["champ_id"] = int(cdata["key"])          # ej 25
        info["apiname"] = cdata["id"]                 # Morgana
        info["name"] = cdata["name"]                  # "Morgana"

        stats = cdata["stats"]
        for k, v in stats.items():
            info[k] = v

        # === Features derivadas ===
        info["attackrange"] = stats["attackrange"]

        # semánticas tipo CC/engage
        info["cc_score"] = 0.0
        info["engage_score"] = 0.0
        info["peel_score"] = 0.0
        info["zone_control_score"] = 0.0

        # heurísticas basadas en sus spells:
        for sp in cdata["spells"]:
            desc = sp["description"].lower()
            if "stun" in desc or "root" in desc or "snare" in desc:
                info["cc_score"] += 1
            if "dash" in desc or "leap" in desc or "gap" in desc:
                info["engage_score"] += 1
            if "shield" in desc or "heal" in desc:
                info["peel_score"] += 1
            if "zone" in desc or "area" in desc or "field" in desc:
                info["zone_control_score"] += 1

        # ratios
        info["magic_ratio"] = compute_magic_ratio(stats)
        info["phys_ratio"] = compute_phys_ratio(stats)

        # tankiness
        info["tankiness"] = compute_tankiness(stats)

        # meta strength (luego lo llenaremos con OP.GG)
        info["meta_strength"] = 0.0

        rows.append(info)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Archivo generado: {OUT_PATH}")
    print(f"Columnas: {list(df.columns)}")

if __name__ == "__main__":
    main()
