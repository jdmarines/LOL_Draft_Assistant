import streamlit as st
from pathlib import Path
import sys

# =====================================
# CONFIG Y IMPORTS
# =====================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

# Importamos el motor actual
from recommender import (
    RES,                      # recursos: champs_df, modelo, etc.
    build_features_for_draft,
    predict_blue_win_prob,
    recommend_for,
)

st.set_page_config(
    page_title="LoL Draft Recommender",
    page_icon="🎮",
    layout="wide",
)

# =====================================
# UTILIDADES
# =====================================

def get_champion_list():
    """Lista ordenada de nombres de campeón."""
    return sorted(RES.champs_df["name"].tolist())

def normalize_selection(selection_list):
    """Quita '(vacío)' y devuelve solo los nombres de campeones."""
    return [c for c in selection_list if c != "(vacío)"]

# =====================================
# UI
# =====================================

st.title("🎮 LoL Draft Recommender — MVP")
st.markdown(
    """
    Versión actual:
    - **Sin counters de OP.GG** (todas las features de counters fijas en 0).
    - **Placeholders neutros internos** solo para completar los equipos hasta 5 campeones.
    """
)

st.divider()

champ_list = get_champion_list()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Equipo BLUE")
    b1 = st.selectbox("Blue 1", ["(vacío)"] + champ_list, key="b1")
    b2 = st.selectbox("Blue 2", ["(vacío)"] + champ_list, key="b2")
    b3 = st.selectbox("Blue 3", ["(vacío)"] + champ_list, key="b3")
    b4 = st.selectbox("Blue 4", ["(vacío)"] + champ_list, key="b4")
    b5 = st.selectbox("Blue 5", ["(vacío)"] + champ_list, key="b5")

with col2:
    st.subheader("🔴 Equipo RED")
    r1 = st.selectbox("Red 1", ["(vacío)"] + champ_list, key="r1")
    r2 = st.selectbox("Red 2", ["(vacío)"] + champ_list, key="r2")
    r3 = st.selectbox("Red 3", ["(vacío)"] + champ_list, key="r3")
    r4 = st.selectbox("Red 4", ["(vacío)"] + champ_list, key="r4")
    r5 = st.selectbox("Red 5", ["(vacío)"] + champ_list, key="r5")

blue_sel = normalize_selection([b1, b2, b3, b4, b5])
red_sel  = normalize_selection([r1, r2, r3, r4, r5])

st.divider()

st.markdown(
    """
    - Para **evaluar un draft completo**, selecciona hasta 5 campeones por lado y pulsa **Calcular probabilidad**.
    - Para **ver recomendaciones top-5**, usa **máximo 4 campeones por lado** (como si el draft estuviera en progreso).
    """
)

if st.button("🔍 Calcular probabilidad y recomendaciones"):
    if len(blue_sel) == 0 or len(red_sel) == 0:
        st.error("Debes seleccionar al menos un campeón en cada equipo.")
    else:
        # ==========================
        # 1) PROBABILIDAD DE VICTORIA
        # ==========================
        feats = build_features_for_draft(blue_sel, red_sel)
        p_blue = predict_blue_win_prob(feats)
        p_red = 1.0 - p_blue

        st.markdown("## 📊 Resultado global del draft")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilidad de victoria BLUE", f"{p_blue*100:.1f}%")
        with c2:
            st.metric("Probabilidad de victoria RED", f"{p_red*100:.1f}%")

        # ==========================
        # 2) RECOMENDACIONES TOP-5
        # ==========================
        st.markdown("## 🧠 Recomendaciones de siguiente pick (Top-5)")

        # Restricción para evitar drafts con 6 campeones por lado
        if len(blue_sel) > 4 or len(red_sel) > 4:
            st.warning(
                "Para ver recomendaciones top-5, usa máximo **4 campeones por lado**.\n"
                "Con 5 picks ya consideramos el draft completo."
            )
        else:
            col_blue, col_red = st.columns(2)

            # 🔵 Recomendaciones para BLUE
            with col_blue:
                st.subheader("🔵 Sugerencias para BLUE")
                try:
                    recs_blue = recommend_for(blue_sel, red_sel, side="blue", top_k=5)
                    if not recs_blue:
                        st.info("No hay candidatos disponibles (quedan muy pocos campeones libres).")
                    else:
                        for r in recs_blue:
                            st.markdown(
                                f"""
                                **{r.champ_name}**  
                                P(Blue win): **{r.prob_blue_win*100:.1f}%**  
                                P(Red win): **{r.prob_red_win*100:.1f}%**  
                                _{r.explanation}_
                                """
                            )
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error calculando recomendaciones para BLUE: {e}")

            # 🔴 Recomendaciones para RED
            with col_red:
                st.subheader("🔴 Sugerencias para RED")
                try:
                    recs_red = recommend_for(blue_sel, red_sel, side="red", top_k=5)
                    if not recs_red:
                        st.info("No hay candidatos disponibles (quedan muy pocos campeones libres).")
                    else:
                        for r in recs_red:
                            st.markdown(
                                f"""
                                **{r.champ_name}**  
                                P(Blue win): **{r.prob_blue_win*100:.1f}%**  
                                P(Red win): **{r.prob_red_win*100:.1f}%**  
                                _{r.explanation}_
                                """
                            )
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Error calculando recomendaciones para RED: {e}")
