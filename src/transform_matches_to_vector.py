import ast
import pandas as pd
from pathlib import Path

IN_PATH  = Path("data/matches/drafts.csv")           # <-- tu archivo de entrada
OUT_PATH = Path("data/matches/matches_15_20_1.csv")      # <-- salida estándar para el pipeline
OUT_AUG  = Path("data/matches/matches_15_20_1_aug.csv")  # <-- (opcional) con augment

REQUIRE_5 = True       # fuerza exactamente 5 picks por lado
DO_AUGMENT = True      # duplica dataset con Azul↔Rojo y Winner invertido

def parse_list(x):
    """Convierte strings tipo '[1, 2, 3]' → lista; deja listas tal cual; maneja NaN."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def to_row(bb, rr, win_flag):
    """Devuelve dict con columnas BB1..BB5, RB1..RB5, Winner."""
    row = {f"BB{i+1}": int(bb[i]) for i in range(5)}
    row.update({f"RB{i+1}": int(rr[i]) for i in range(5)})
    row["Winner"] = int(win_flag)
    return row

def main():
    df = pd.read_csv(IN_PATH)

    # Parsear arrays desde columnas de texto
    for col in ["picks_blue","picks_red","bans_blue","bans_red","vector"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)

    out_rows = []
    bad_rows = 0

    for _, r in df.iterrows():
        # 1) Prioriza 'picks_blue'/'picks_red'; si no existen, intenta con 'vector'
        bb = r.get("picks_blue", None)
        rr = r.get("picks_red", None)

        if bb is None or rr is None:
            vec = r.get("vector", None)
            if isinstance(vec, list) and len(vec) >= 11:
                bb = vec[0:5]
                rr = vec[5:10]
                # Winner de vector (último elemento) si no tenemos 'winner'
                label_from_vector = vec[10]
            else:
                bb, rr, label_from_vector = None, None, None
        else:
            label_from_vector = None

        # 2) Validaciones básicas
        if bb is None or rr is None:
            bad_rows += 1
            continue
        if REQUIRE_5 and (len(bb) != 5 or len(rr) != 5):
            bad_rows += 1
            continue
        try:
            bb = [int(x) for x in bb]
            rr = [int(x) for x in rr]
        except Exception:
            bad_rows += 1
            continue

        # 3) Winner
        winner = r.get("winner", None)
        label  = r.get("label", None)

        if isinstance(winner, str):
            win_flag = 1 if winner.upper() == "BLUE" else 0
        elif pd.notna(label):
            win_flag = int(label)
        elif label_from_vector is not None:
            win_flag = int(label_from_vector)
        else:
            bad_rows += 1
            continue

        out_rows.append(to_row(bb, rr, win_flag))

    out = pd.DataFrame(out_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"✅ Guardado {OUT_PATH} | filas={len(out)} | descartadas={bad_rows}")

    # (Opcional) Augment por permutación Azul↔Rojo
    if DO_AUGMENT:
        aug = out.copy()
        blue_cols = [f"BB{i}" for i in range(1,6)]
        red_cols  = [f"RB{i}" for i in range(1,6)]

        # swap equipos y voltea Winner
        aug[blue_cols], aug[red_cols] = out[red_cols].values, out[blue_cols].values
        aug["Winner"] = 1 - out["Winner"]

        full = pd.concat([out, aug], ignore_index=True)
        full.to_csv(OUT_AUG, index=False)
        print(f"✅ Guardado {OUT_AUG} | filas={len(full)} (augment on)")

if __name__ == "__main__":
    main()
