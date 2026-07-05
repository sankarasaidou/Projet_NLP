# -*- coding: utf-8 -*-
"""
Interface Streamlit — Modélisation de sujets (LDA / NMF / BERTopic).
Lancement : streamlit run app.py
"""
import os
import sys

# Permet à Python de trouver les scripts locaux propres à ce projet
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
    
import pandas as pd
import streamlit as st

from corpus import CORPUS
from topic_models import run_lda, run_nmf, run_bertopic
from interpretation import interpret_topics, compare_methods_coherence

st.set_page_config(page_title="Modélisation de sujets", layout="wide")
st.title("🧩 Modélisation de sujets — LDA / NMF / BERTopic")
st.caption(f"Corpus de démo : {len(CORPUS)} articles couvrant 5 thématiques.")

with st.sidebar:
    st.header("⚙️ Paramètres")
    method = st.selectbox("Technique de modélisation", ["LDA", "NMF", "BERTopic"])
    n_topics = None
    if method in ("LDA", "NMF"):
        n_topics = st.slider("Nombre de topics", min_value=2, max_value=8, value=5)
    n_words = st.slider("Mots-clés par topic", min_value=3, max_value=12, value=8)


@st.cache_data(show_spinner="Entraînement du modèle de topics...")
def compute_topics(method_name, n_topics_val, n_words_val):
    if method_name == "LDA":
        return run_lda(CORPUS, n_topics=n_topics_val, n_words=n_words_val)
    if method_name == "NMF":
        return run_nmf(CORPUS, n_topics=n_topics_val, n_words=n_words_val)
    return run_bertopic(CORPUS, n_words=n_words_val)


try:
    result = compute_topics(method, n_topics, n_words)
except ImportError as e:
    st.error(str(e))
    st.stop()

topics = interpret_topics(result["topics"])

st.subheader(f"Topics extraits — {method}")
for topic in topics:
    with st.container(border=True):
        st.markdown(f"**Topic {topic['id']}** — thème probable : `{topic['theme_probable']}`")
        st.write(", ".join(topic["top_words"]))

st.subheader("Répartition des documents par topic")
assignment_df = pd.DataFrame({
    "Document": [doc[:60] + "..." for doc in CORPUS],
    "Topic assigné": result["doc_topic_assignment"],
})
st.dataframe(assignment_df, use_container_width=True)

st.bar_chart(assignment_df["Topic assigné"].value_counts().sort_index())

with st.expander("📊 Comparer la cohérence des méthodes (LDA vs NMF)"):
    st.write(
        "La cohérence mesure si les mots-clés d'un même topic apparaissent "
        "souvent ensemble dans le corpus (plus la valeur est haute, mieux "
        "c'est). BERTopic n'est pas inclus ici par défaut (paquet optionnel)."
    )
    if st.button("Calculer la cohérence LDA vs NMF"):
        results = {
            "LDA": run_lda(CORPUS, n_topics=n_topics or 5, n_words=n_words),
            "NMF": run_nmf(CORPUS, n_topics=n_topics or 5, n_words=n_words),
        }
        coherence = compare_methods_coherence(results, CORPUS)
        st.table({
            "Méthode": list(coherence.keys()),
            "Cohérence moyenne": [round(c["avg_coherence"], 3) for c in coherence.values()],
        })
