# -*- coding: utf-8 -*-
"""
Interface Streamlit — Analyseur de tendances sur commentaires (lefaso.net).
Lancement : streamlit run app.py

ÉTAPE 6 (Alertes automatiques) est intégrée ici : voir onglet "Alertes".
"""

import os
import sys

# Permet à Python de trouver les scripts locaux propres à ce projet
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
    
import json

import streamlit as st

from comments_data import COMMENTS
from sentiment import full_analysis
from trends import top_terms_overall, daily_term_frequency, detect_spikes
from visualization import plot_daily_activity, plot_top_terms_bar

st.set_page_config(page_title="Analyseur de tendances", layout="wide")
st.title("📈 Analyseur de tendances — commentaires lefaso.net")
st.caption("Nettoyage spécialisé · Détection de tendances · Sentiment · Alertes automatiques")

tab_overview, tab_trends, tab_alerts, tab_export = st.tabs(
    ["📊 Vue d'ensemble", "🔥 Tendances", "🚨 Alertes", "📤 Export"]
)

with tab_overview:
    st.subheader("Activité quotidienne par sentiment")
    path = plot_daily_activity(COMMENTS, "/tmp/daily_activity.png")
    st.image(path, use_container_width=True)

    st.subheader("Détail des commentaires analysés")
    rows = []
    for c in COMMENTS:
        analysis = full_analysis(c["text"])
        rows.append({
            "Date": c["date"], "Auteur": c["author"], "Texte": c["text"][:60] + "...",
            "Sentiment": analysis["sentiment"], "Type": analysis["type_expression"],
        })
    st.dataframe(rows, use_container_width=True)

with tab_trends:
    term_type = st.radio("Type de terme", ["hashtag", "mot"], horizontal=True)
    top_n = st.slider("Nombre de termes à afficher", 3, 20, 10)

    top_terms = top_terms_overall(COMMENTS, term_type, top_n)
    path = plot_top_terms_bar(COMMENTS, "/tmp/top_terms.png", term_type)
    if path:
        st.image(path, use_container_width=True)
    else:
        st.info("Aucun terme trouvé pour ce type.")

    if top_terms:
        selected_term = st.selectbox("Voir l'évolution d'un terme dans le temps", [t for t, _ in top_terms])
        series = daily_term_frequency(COMMENTS, selected_term)
        if not series.empty:
            st.line_chart(series)

with tab_alerts:
    st.write(
        "Détecte automatiquement les jours où un terme dépasse un pic "
        "de mentions par rapport à sa moyenne habituelle."
    )
    multiplier = st.slider("Sensibilité de l'alerte (x fois la moyenne)", 1.5, 5.0, 2.0, step=0.5)

    all_hashtags = [t for t, _ in top_terms_overall(COMMENTS, "hashtag", top_n=20)]
    alerts_found = []
    for term in all_hashtags:
        alerts_found.extend(detect_spikes(COMMENTS, term, spike_multiplier=multiplier))

    if alerts_found:
        for alert in alerts_found:
            st.warning(
                f"🚨 Pic détecté le {alert['date']} pour **#{alert['term']}** : "
                f"{alert['count']} mentions (moyenne habituelle : {alert['baseline']})"
            )
    else:
        st.success("Aucun pic anormal détecté avec ce seuil de sensibilité.")

with tab_export:
    st.write("Export du rapport d'analyse complet au format JSON.")
    report = {
        "top_hashtags": top_terms_overall(COMMENTS, "hashtag", 10),
        "commentaires_analyses": [
            {**c, "date": str(c["date"]), **full_analysis(c["text"])} for c in COMMENTS
        ],
    }
    st.download_button(
        "📥 Télécharger le rapport (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
        file_name="rapport_tendances.json",
        mime="application/json",
    )
