# LOL_Draft_Assistant

# 🧭 Backlog — Sistema de Predicción y Recomendación de Drafts (League of Legends)

| Fase | Objetivo | Tareas principales | Entregable | Duración estimada | Estado |
|------|-----------|-------------------|-------------|-------------------|---------|
| **0. Setup y datos base** | Preparar entorno reproducible y datos del parche objetivo | • Definir parche y estructura del repo (`data/`, `src/`, `notebooks/`) <br> • Cargar `champs_base.csv` y `matches.csv` <br> • Normalizar IDs y nombres de campeones <br> • Crear `champ_id_map.json` | Datos base limpios y entorno funcional | 1 semana (≈10 h) | 🔲 |
| **1. Feature engineering táctico** | Generar variables agregadas por equipo y deltas estratégicos | • Agregar features: `tankiness`, `engage_score`, `peel_score`, `cc_score`, `range_norm`, `phys_ratio`, `magic_ratio`, `scaling_score` <br> • Calcular promedios por equipo y banderas (`has_frontline`, `full_ad`, `has_waveclear`) <br> • Crear deltas Azul–Rojo (`delta_engage`, `delta_tankiness`, etc.) <br> • Exportar `draft_features.csv` | Dataset tabular con features tácticas y deltas | 2 semanas (≈20 h) | 🔲 |
| **2. Matriz de winrate y sinergia** | Capturar interacciones entre campeones y sinergias de equipo | • Generar matriz `W[i,j]` con suavizado bayesiano (α≈20) <br> • Establecer umbral mínimo de muestras (T=30) y fallback por clase (Tank vs Marksman, etc.) <br> • Calcular sinergia interna `S[i,j]` (pares aliados) <br> • Derivar features: `Blue_vs_Red`, `Red_vs_Blue`, `Delta_vs`, `Blue_Synergy`, `Red_Synergy`, `Delta_Synergy`, `Delta_Total` <br> • Integrar al dataset final | Dataset enriquecido con sinergia interna y ventaja cruzada | 2 semanas (≈20 h) | 🔲 |
| **3. Entrenamiento del modelo ML** | Entrenar y comparar modelos de predicción de victoria | • Split 80/20 estratificado <br> • Entrenar y comparar: Logistic Regression, RF, XGBoost, LightGBM, CatBoost, SVM/MLP opcional <br> • Métricas: Accuracy, AUC, Brier, F1, LogLoss <br> • Guardar `model_winprob.pkl` y reporte de comparación <br> • Analizar importancia de variables (SHAP) | Modelo final y reporte comparativo de desempeño | 1.5 semanas (≈15 h) | 🔲 |
| **4. Recomendador de pick** | Construir el motor que sugiere picks óptimos durante el draft | • Implementar `recommender.py` <br> • Simular drafts parciales con modelo `P(win|C)` <br> • Calcular `synergy_if_C` y `counter_value_if_C` <br> • Combinar scores: `score_total = α·P(win)+β·synergy+γ·counter` <br> • Integrar uso dual de matriz (Ruta A estable, Ruta B situacional) <br> • Retornar top 3 picks con explicación | Recomendador funcional con análisis cuantitativo y situacional | 2 semanas (≈20 h) | 🔲 |
| **5. Reporte tipo coach** | Generar análisis táctico en lenguaje natural | • Implementar `reporter.py` <br> • Mostrar composición, winrate estimado y fortalezas/debilidades (full AD, falta de frontline, etc.) <br> • Identificar win condition enemiga <br> • Integrar recomendador para picks sugeridos <br> • (Opcional) Interfaz Streamlit para prueba en vivo | Reporte narrativo y/o dashboard tipo staff técnico | 1 semana (≈10 h) | 🔲 |
| **6. Evaluación y documentación** | Validar, documentar y preparar la entrega final | • Evaluar con validación cruzada <br> • Documentar supuestos, alcance y limitaciones <br> • Incluir comparativa de modelos y gráficos SHAP <br> • Redactar anexo técnico o notebook resumen | Informe final + notebooks reproducibles | 1 semana (≈10 h) | 🔲 |

---

## ⏱️ **Resumen de planificación**

- **Total estimado:** 10.5 semanas (≈105 horas con dedicación de 10 h/semana)
- **Hitos clave:**
  - Semana 1 → Setup completo  
  - Semana 3 → Dataset con features tácticas  
  - Semana 5 → Dataset con sinergia y winrate  
  - Semana 6.5 → Modelos comparativos entrenados  
  - Semana 8.5 → Recomendador funcional  
  - Semana 9.5 → Reporte tipo coach  
  - Semana 10.5 → Evaluación y documentación final  

---

## 🚀 Entregable final esperado

> **Asistente de Draft Estratégico para League of Legends**
>
> - **Entrada:** picks del equipo azul, picks del equipo rojo (completo o parcial).  
> - **Salida:**  
>   - Probabilidad de victoria predicha.  
>   - Identidad táctica de composición (engage, poke, scaling…).  
>   - Win condition detectada y riesgos (ej. “falta frontline”).  
>   - Recomendación de picks óptimos con justificación cuantitativa y situacional.  
>   - Análisis narrativo tipo staff profesional (GenG/T1-style).

---
