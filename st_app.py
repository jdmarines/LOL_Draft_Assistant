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

def coach_summary(feats: dict) -> str:
    """
    Devuelve un texto tipo coach en función de los deltas de la compo
    (siempre interpretando desde el lado BLUE).
    """
    lines = []

    def val(k, default=0.0):
        return float(feats.get(k, default))

    d_cc    = val("Delta_cc_score_sum")
    d_eng   = val("Delta_engage_score_sum")
    d_peel  = val("Delta_peel_score_sum")
    d_zone  = val("Delta_zone_control_score_sum")
    d_tank  = val("Delta_tankiness_sum")
    d_mag   = val("Delta_magic_ratio_sum")
    d_phys  = val("Delta_phys_ratio_sum")
    d_range = val("Delta_attackrange_mean")
    d_meta  = val("Delta_meta_strength")

    # CC
    if d_cc > 0.5:
        lines.append("BLUE tiene **más CC total** que RED, lo que favorece peleas largas y picks aislados.")
    elif d_cc < -0.5:
        lines.append("RED tiene **más CC total**, por lo que BLUE debe cuidar los ángulos de entrada y visión.")

    # Engage
    if d_eng > 0.5:
        lines.append("La compo BLUE cuenta con **mejores herramientas de engage**, puede proponer las peleas.")
    elif d_eng < -0.5:
        lines.append("BLUE tiene **menos engage directo**; conviene jugar a counter-engage o front-to-back.")

    # Peel
    if d_peel > 0.5:
        lines.append("BLUE posee **mejor peel y herramientas defensivas**, sus carries deberían estar más protegidos.")
    elif d_peel < -0.5:
        lines.append("RED tiene **mejor peel**, lo que dificulta ejecutar dives profundos contra sus carries.")

    # Zona / control de oleadas
    if d_zone > 0.5:
        lines.append("BLUE destaca en **control de zona** (trampas, zonas, poke), ideal para objetivos neutrales.")
    elif d_zone < -0.5:
        lines.append("RED controla mejor el espacio, BLUE debería evitar pelear en zonas estrechas u objetivos forzados.")

    # Tankiness
    if d_tank > 200:
        lines.append("BLUE cuenta con **más frontline/tankiness**, puede aguantar peleas extendidas.")
    elif d_tank < -200:
        lines.append("RED tiene **frontline más sólida**, BLUE debería apoyarse en rango, poke o kiteo.")

    # Daño mágico / físico
    if d_mag > 0.5:
        lines.append("BLUE está más cargado hacia **daño mágico**, es clave que RED arme resistencia mágica.")
    elif d_phys > 0.5:
        lines.append("BLUE está más cargado hacia **daño físico**, RED debe priorizar armadura.")

    # Rango
    if d_range > 75:
        lines.append("La compo BLUE tiene **mayor rango promedio**, puede desgastar antes de entrar en melee.")
    elif d_range < -75:
        lines.append("RED tiene **mejor rango**, BLUE necesita flancos, TP o engages explosivos.")

    # Meta strength
    if d_meta > 0.05:
        lines.append("En términos de **meta**, BLUE está ligeramente favorecido según estadísticas globales.")
    elif d_meta < -0.05:
        lines.append("La compo de RED está más alineada con el **meta actual** en win/pick/ban rates.")

    if not lines:
        return "Las composiciones están bastante equilibradas; ningún eje táctico destaca de forma extrema."

    return " ".join(lines)


# =====================================
# UI
# =====================================

st.title("🎮 LoL Draft Recommender — MVP V2")
st.markdown(
    """
    Versión actual:
    - **Sin counters de OP.GG** (todas las features de counters fijas en 0).
    - **Placeholders neutros internos** solo para completar los equipos hasta 5 campeones.
    - Motor entrenado con features de composición (CC, engage, rango, tankiness, daño físico/mágico, meta).
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

        # Barra visual sencilla
        st.progress(p_blue)  # interpreta p_blue en [0,1]

        # ==========================
        # 1.1 PANEL TÁCTICO DE FEATURES
        # ==========================
        st.markdown("### ⚙️ Resumen táctico de la composición (BLUE vs RED)")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Δ CC total (BLUE-RED)", f"{feats.get('Delta_cc_score_sum', 0.0):.2f}")
            st.metric("Δ Engage (BLUE-RED)", f"{feats.get('Delta_engage_score_sum', 0.0):.2f}")
            st.metric("Δ Peel/Protección", f"{feats.get('Delta_peel_score_sum', 0.0):.2f}")
            st.metric("Δ Control de zona", f"{feats.get('Delta_zone_control_score_sum', 0.0):.2f}")

        with colB:
            st.metric("Δ Tankiness", f"{feats.get('Delta_tankiness_sum', 0.0):.1f}")
            st.metric("Δ Rango medio", f"{feats.get('Delta_attackrange_mean', 0.0):.1f}")
            st.metric("Δ Daño mágico", f"{feats.get('Delta_magic_ratio_sum', 0.0):.2f}")
            st.metric("Δ Daño físico", f"{feats.get('Delta_phys_ratio_sum', 0.0):.2f}")

        # ==========================
        # 1.2 TEXTO ESTILO COACH
        # ==========================
        resumen = coach_summary(feats)
        st.markdown(f"**Comentario tipo coach:** {resumen}")

        # ==========================
        # 2) RECOMENDACIONES TOP-5
        # ==========================

st.markdown("## 🧠 Recomendaciones de siguiente pick (Top-5)")

blue_needs_pick = len(blue_sel) < 5
red_needs_pick = len(red_sel) < 5

if not blue_needs_pick and not red_needs_pick:
    # Ambos lados ya tienen 5 → no hay más draft
    st.info("Ambos equipos ya tienen 5 campeones. El draft está completo, no hay picks por recomendar.")
else:
    col_blue, col_red = st.columns(2)

    # 🔵 Recomendaciones para BLUE (solo si BLUE aún tiene huecos)
    with col_blue:
        st.subheader("🔵 Sugerencias para BLUE")
        if blue_needs_pick:
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
        else:
            st.info("BLUE ya tiene 5 campeones. No hay más picks para este lado.")

    # 🔴 Recomendaciones para RED (solo si RED aún tiene huecos)
    with col_red:
        st.subheader("🔴 Sugerencias para RED")
        if red_needs_pick:
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
        else:
            st.info("RED ya tiene 5 campeones. No hay más picks para este lado.")
