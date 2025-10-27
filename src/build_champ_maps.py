import pandas as pd
import json
from pathlib import Path

# Ruta al csv maestro con columnas: id, apiname, champion
SOURCE_CSV = Path("data/raw/champ_list.csv")

# Rutas de salida
OUT_SIMPLE = Path("champ_id_map.json")
OUT_EXTENDED = Path("champ_id_map_extended.json")


def load_champ_table(csv_path: Path) -> pd.DataFrame:
    """
    Lee la tabla de campeones y asegura tipos consistentes.
    Espera columnas:
      - id (int)
      - apiname (string estilo Riot API, sin espacios)
      - champion (string legible para humanos, con espacios)
    """
    df = pd.read_csv(csv_path, sep=';', dtype={"id": int, "apiname": str, "champion": str})

    # sanity check básico
    required_cols = {"id", "apiname", "champion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {missing}")

    # quitar espacios accidentales en apiname / champion
    df["apiname"] = df["apiname"].str.strip()
    df["champion"] = df["champion"].str.strip()

    return df


def build_simple_map(df: pd.DataFrame) -> dict:
    """
    Construye el mapeo simple:
        { "266": "Aatrox", "103": "Ahri", ... }
    Claves como string para evitar problemas de serialización JSON.
    """
    return {str(row.id): row.apiname for _, row in df.iterrows()}


def build_extended_map(df: pd.DataFrame) -> dict:
    """
    Construye el mapeo extendido:
        {
          "266": {"api": "Aatrox", "display": "Aatrox"},
          "136": {"api": "AurelionSol", "display": "Aurelion Sol"},
          ...
        }
    """
    extended = {}
    for _, row in df.iterrows():
        extended[str(row.id)] = {
            "api": row.apiname,
            "display": row.champion
        }
    return extended


def save_json(obj: dict, path: Path):
    """
    Guarda el diccionario como JSON con indentación linda.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    df = load_champ_table(SOURCE_CSV)

    simple_map = build_simple_map(df)
    extended_map = build_extended_map(df)

    save_json(simple_map, OUT_SIMPLE)
    save_json(extended_map, OUT_EXTENDED)

    print(f"✅ Generado {OUT_SIMPLE} ({len(simple_map)} campeones)")
    print(f"✅ Generado {OUT_EXTENDED} ({len(extended_map)} campeones)")


if __name__ == "__main__":
    main()
