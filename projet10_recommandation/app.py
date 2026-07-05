# -*- coding: utf-8 -*-
"""
Interface Streamlit — Système de recommandation de contenu textuel.
Lancement : streamlit run app.py
"""

import streamlit as st

from data import ARTICLES, USER_HISTORY
from recommend import content_based_recommend, collaborative_recommend, hybrid_recommend
from evaluation import leave_one_out_evaluation, satisfaction_proxy

st.set_page_config(page_title="Recommandation de contenu", layout="wide")
st.title("📚 Système de recommandation de contenu textuel")
st.caption("Content-based · Filtrage collaboratif · Hybride")

with st.sidebar:
    st.header("⚙️ Paramètres")
    user_id = st.selectbox("Utilisateur", list(USER_HISTORY.keys()))
    algorithm = st.selectbox("Algorithme", ["Content-based", "Filtrage collaboratif", "Hybride"])
    top_n = st.slider("Nombre de recommandations", 1, 8, 3)
    content_weight = 0.5
    if algorithm == "Hybride":
        content_weight = st.slider("Poids du content-based (vs collaboratif)", 0.0, 1.0, 0.5)

st.subheader(f"📖 Historique de lecture de {user_id}")
history_rows = [
    {"Article": ARTICLES[aid]["title"], "Note d'engagement": rating}
    for aid, rating in USER_HISTORY[user_id]
]
st.table(history_rows)

if st.button("Générer les recommandations", type="primary"):
    if algorithm == "Content-based":
        results = content_based_recommend(user_id, top_n=top_n)
    elif algorithm == "Filtrage collaboratif":
        results = collaborative_recommend(user_id, top_n=top_n)
    else:
        results = hybrid_recommend(user_id, top_n=top_n, content_weight=content_weight)

    st.subheader(f"✨ Recommandations ({algorithm})")
    for rank, r in enumerate(results, start=1):
        st.markdown(f"**{rank}. {r['title']}** — score : `{r['score']:.3f}`")
        st.caption(ARTICLES[r["article_id"]]["text"][:150] + "...")
        st.divider()

with st.expander("📊 Évaluation du système (leave-one-out)"):
    if st.button("Lancer l'évaluation"):
        evaluation = leave_one_out_evaluation(top_n=top_n)
        col1, col2 = st.columns(2)
        col1.metric(f"Précision@{top_n} moyenne", f"{evaluation['avg_precision_at_n']:.2f}")
        col2.metric(f"Rappel@{top_n} moyenne", f"{evaluation['avg_recall_at_n']:.2f}")

        st.write("Détail par utilisateur :")
        st.table([
            {
                "Utilisateur": uid,
                "Article retenu (held-out)": r["held_out_title"],
                "Retrouvé dans le Top-N ?": "✅" if r["hit"] else "❌",
            }
            for uid, r in evaluation["per_user"].items()
        ])

        st.write("Satisfaction (note moyenne d'engagement par utilisateur) :")
        st.bar_chart(satisfaction_proxy())
