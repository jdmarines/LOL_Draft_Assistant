# src/preprocess.py

import pandas as pd
import json
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

def load_champ_data():
    champs = pd.read_csv(DATA_RAW / "champs_base.csv")
    with open("champ_id_map.json") as f:
        id_map = json.load(f)
    return champs, id_map

def load_matches():
    matches = pd.read_csv(DATA_RAW / "matches.csv")
    return matches

if __name__ == "__main__":
    champs, id_map = load_champ_data()
    matches = load_matches()
    print("Champs loaded:", champs.shape)
    print("Matches loaded:", matches.shape)

