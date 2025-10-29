import json
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/external/opgg_meta_raw_15_20_1.json")
OUT_PATH = Path("data/external/champs_meta_15_20_1.csv")

def load_opgg_raw():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data debería ser una lista de campeones+rol tal como me pegaste.
    # Si en tu dump el JSON está envuelto tipo {"data":[...]} ajusta aquí.
    return data

def extract_counter_info(counter_list):
    """
    counter_list es algo como:
    [
      {"name": "Naafiri", "champion_id": 950, "play": 4072, "win": 1802},
      {"name": "Quinn", "champion_id": 133, ...},
      ...
    ]
    Puede contener "$undefined". Filtramos eso.
    Devolvemos:
      ids_str -> "950;133;10"
      wr_str  -> "0.4425;0.481;0.486"
    """
    clean = [c for c in counter_list if isinstance(c, dict)]
    ids = []
    wrs = []
    for c in clean:
        champ_id = c.get("champion_id")
        play = c.get("play", 0)
        win = c.get("win", 0)
        wr = (win / play) if play else None
        ids.append(str(champ_id))
        wrs.append(f"{wr:.4f}" if wr is not None else "NA")

    ids_str = ";".join(ids) if ids else ""
    wr_str = ";".join(wrs) if wrs else ""
    return ids_str, wr_str


def normalize(data):
    rows = []
    for entry in data:
        # entry es el bloque tipo Morgana/Briar/Milio que pegaste
        # primero necesitamos un champion_id numérico consistente
        # lo podemos tomar del primer counter si no está en la raíz.
        # pero op.gg a veces no lo da en la raíz...
        # solución: intentemos inferirlo de los counters o lo dejamos NA si no existe.
        # Más adelante lo uniremos por "name" -> apiname -> nuestro id Riot.

        # counters
        counters = entry.get("positionCounters", [])
        counters_ids, counters_wr = extract_counter_info(counters)

        row = {
            "apiname_guess": entry.get("name"),  # "Morgana", "Briar", etc.
            "key_lower": entry.get("key"),       # "morgana", "briar"
            "role": entry.get("positionName"),   # "MID", "JUNGLE", ...
            "win_rate_role": entry.get("positionWinRate"),     # %
            "pick_rate_role": entry.get("positionPickRate"),   # %
            "ban_rate_role": entry.get("positionBanRate"),     # %
            "role_rate": entry.get("positionRoleRate"),        # % de ese champ que va a ese rol
            "tier_role": entry.get("positionTier"),            # 1,2,3...
            "rank_role": entry.get("positionRank"),            # ranking dentro del rol
            "counters_ids": counters_ids,                      # "950;133;10"
            "counters_wr": counters_wr                         # "0.4425;0.4810;0.4863"
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Limpieza / normalización:
    # - pasamos role a mayúsculas consistentes (ya viene así tipo MID/JUNGLE/SUPPORT)
    # - normalizamos tasas como proporciones (0-1) en lugar de porcentajes enteros
    df["win_rate_role"] = df["win_rate_role"] / 100.0
    df["pick_rate_role"] = df["pick_rate_role"] / 100.0
    df["ban_rate_role"] = df["ban_rate_role"] / 100.0
    df["role_rate"] = df["role_rate"]  # ya parece proporción (0-1)

    # Nota: "apiname_guess" = display_name tipo "Morgana".
    # En nuestro mundo Data Dragon tiene:
    #   apiname  = "Morgana"
    #   display  = "Morgana"
    # Para campeones con apóstrofes Kai'Sa vs Kaisa
    # haremos un pequeño fix luego al merge.

    return df

def main():
    raw = load_opgg_raw()
    df = normalize(raw)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Guardado {OUT_PATH} con {len(df)} filas.")

if __name__ == "__main__":
    main()
