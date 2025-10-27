import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/raw")  # por ahora lo dejamos en raw; luego podemos mover a processed

def load_champion_full() -> dict:
    """
    Lee el JSON activo championFull.json (ya descargado para el parche elegido)
    y retorna el diccionario de campeones.
    """
    with open(RAW_DIR / "championFull.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]  # dict { "Morgana": { ... }, "Malphite": { ... }, ... }

def flatten_champ_data(champ_api: str, champ_obj: dict) -> dict:
    """
    Toma la entrada de un campeón dentro de championFull.json
    y devuelve un dict plano con las columnas que nos interesan.
    """
    stats = champ_obj["stats"]
    info = champ_obj["info"]
    tags = champ_obj.get("tags", [])
    spells = champ_obj.get("spells", [])
    passive = champ_obj.get("passive", {})

    # concatenamos texto de spells y passive para features de NLP/regex después
    spell_texts = []
    for s in spells:
        # Podemos usar description (larga, con color), tooltip (más técnica) o ambas
        spell_texts.append(s.get("name", ""))
        spell_texts.append(s.get("description", ""))
        spell_texts.append(s.get("tooltip", ""))

    passive_text = passive.get("description", "")

    return {
        # Identidad
        "id": int(champ_obj["key"]),               # id numérico tipo 25
        "apiname": champ_obj["id"],                # "Morgana"
        "display_name": champ_obj["name"],         # "Morgana"
        "title": champ_obj.get("title", ""),       # "the Fallen"
        "partype": champ_obj.get("partype", ""),   # "Mana", "Energy", "None"

        # Roles básicos reportados por Riot
        "tag_primary": tags[0] if len(tags) > 0 else None,
        "tag_secondary": tags[1] if len(tags) > 1 else None,

        # Info block (Riot "info" = perfil muy resumido: ataque/defensa/magia/dificultad)
        "info_attack": info.get("attack", None),
        "info_defense": info.get("defense", None),
        "info_magic": info.get("magic", None),
        "info_difficulty": info.get("difficulty", None),

        # Stats base
        "hp": stats.get("hp", None),
        "hpperlevel": stats.get("hpperlevel", None),
        "armor": stats.get("armor", None),
        "armorperlevel": stats.get("armorperlevel", None),
        "spellblock": stats.get("spellblock", None),
        "spellblockperlevel": stats.get("spellblockperlevel", None),
        "movespeed": stats.get("movespeed", None),
        "attackrange": stats.get("attackrange", None),
        "attackdamage": stats.get("attackdamage", None),
        "attackdamageperlevel": stats.get("attackdamageperlevel", None),
        "attackspeed": stats.get("attackspeed", None),
        "attackspeedperlevel": stats.get("attackspeedperlevel", None),

        # Texto crudo para posterior análisis táctico (engage, peel, cc, zone control...)
        "spells_text": " ".join(spell_texts),
        "passive_text": passive_text,
    }

def main():
    champs = load_champion_full()

    rows = []
    for api_name, champ_obj in champs.items():
        row = flatten_champ_data(api_name, champ_obj)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ordenar por id numérica para que sea estable y bonito
    df = df.sort_values("id").reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "champs_base_raw.csv"
    df.to_csv(out_csv, index=False)

    print(f"✅ Exportado {out_csv} con {len(df)} campeones")
    print("Columnas:", list(df.columns))

if __name__ == "__main__":
    main()
