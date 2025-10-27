import json
import os
from pathlib import Path
import urllib.request

RAW_DIR = Path("data/raw")

def build_ddragon_url(patch: str) -> str:
    """
    Devuelve la URL a championFull.json para un parche dado.
    Ej: patch="15.20.1"
    """
    return f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/championFull.json"

def download_champion_full(patch: str) -> Path:
    """
    Descarga championFull.json para el parche dado y lo guarda en:
    data/raw/championFull_<patch>.json

    También guarda/reescribe:
    data/raw/championFull.json
    como copia "activa" del parche elegido.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = build_ddragon_url(patch)
    out_versioned = RAW_DIR / f"championFull_{patch}.json"
    out_active = RAW_DIR / "championFull.json"

    print(f"Descargando {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    # Guardar versión específica del parche
    with open(out_versioned, "wb") as f:
        f.write(data)

    # Guardar/actualizar versión activa
    with open(out_active, "wb") as f:
        f.write(data)

    print(f"✅ Guardado {out_versioned}")
    print(f"✅ Actualizado {out_active} (parche activo)")

    return out_active

def main():
    # Parche objetivo del proyecto:
    # Worlds 2025 -> usamos 15.20.1 según tu definición
    patch = "15.20.1"

    out_path = download_champion_full(patch)

    # sanity check mínimo
    with open(out_path, "r", encoding="utf-8") as f:
        dd = json.load(f)
    champs = dd["data"]
    print(f"✅ Campeones cargados: {len(champs)}")

if __name__ == "__main__":
    main()
