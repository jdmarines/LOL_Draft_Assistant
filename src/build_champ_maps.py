import json
from pathlib import Path

DD_CHAMPS_PATH = Path("data/raw/champion.json")

OUT_SIMPLE = Path("champ_id_map.json")
OUT_EXTENDED = Path("champ_id_map_extended.json")


def load_ddragon(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # ddragon tiene estructura {"type": "...", "format": "...", "version": "...", "data": {<apiName>: {...}}}
    return raw["data"]


def build_maps(dd_data: dict):
    """
    dd_data: dict con llaves tipo "MissFortune", "Ahri", etc.
    Cada valor tiene campos como:
      - id (apiname, ej "MissFortune")
      - key (string numérica ej "21")
      - name (display ej "Miss Fortune")
    """
    simple_map = {}
    extended_map = {}

    for api_name, info in dd_data.items():
        champ_id_str = info["key"]        # "21"
        api = info["id"]                  # "MissFortune"
        display = info["name"]            # "Miss Fortune"

        simple_map[champ_id_str] = api
        extended_map[champ_id_str] = {
            "api": api,
            "display": display
        }

    return simple_map, extended_map


def write_json(obj: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    dd = load_ddragon(DD_CHAMPS_PATH)
    simple_map, extended_map = build_maps(dd)

    write_json(simple_map, OUT_SIMPLE)
    write_json(extended_map, OUT_EXTENDED)

    print(f"✅ Generado {OUT_SIMPLE} con {len(simple_map)} campeones")
    print(f"✅ Generado {OUT_EXTENDED} con {len(extended_map)} campeones")


if __name__ == "__main__":
    main()

