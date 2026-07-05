# -*- coding: utf-8 -*-
"""
Interface Streamlit — Analyse de sentiment comparative.
Lancement : streamlit run app.py
"""
import os
import sys

# Permet à Python de trouver les scripts locaux propres à ce projet
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

import streamlit as st

from corpus import TRAIN_DATA, TEST_DATA
from lexical_approach import analyze_lexical
from statistical_approach import StatisticalSentimentClassifier, MODELS

st.set_page_config(page_title="Analyse de sentiment", layout="wide")
st.title("💬 Analyse de sentiment — approches comparatives")
st.caption("Lexicale · Statistique (Naive Bayes / SVM / Régression logistique) · Neuronale (CamemBERT / FlauBERT)")


@st.cache_resource(show_spinner="Entraînement des modèles statistiques...")
def load_statistical_models():
    train_texts = [t for t, _ in TRAIN_DATA]
    train_labels = [l for _, l in TRAIN_DATA]
    return {
        name: StatisticalSentimentClassifier(name).fit(train_texts, train_labels)
        for name in MODELS
    }


statistical_models = load_statistical_models()

text = st.text_area(
    "Texte à analyser :",
    value="Je suis très satisfait, le service était excellent et rapide.",
    height=100,
)
statistical_choice = st.selectbox("Modèle statistique à utiliser", list(MODELS.keys()))

if st.button("Analyser", type="primary") and text.strip():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📖 Lexicale")
        result = analyze_lexical(text)
        st.metric("Sentiment", result["label"], f"score = {result['score']:.2f}")
        if result["details"]:
            st.write("Mots contribuant au score :")
            for d in result["details"]:
                sign = "🔴" if d["negated"] else ("🟢" if d["polarity"] > 0 else "🔵")
                st.write(f"{sign} `{d['word']}` : {d['polarity']:+.2f}")

    with col2:
        st.subheader("📊 Statistique")
        pred = statistical_models[statistical_choice].predict([text])[0]
        st.metric(f"Sentiment ({statistical_choice})", pred)

    with col3:
        st.subheader("🧠 Neuronale (CamemBERT/FlauBERT)")
        try:
            from neural_approach import predict_neural
            pred = predict_neural([text])[0]
            st.metric("Sentiment", pred)
        except ImportError as e:
            st.info(f"Non disponible dans cet environnement : {e}")

with st.expander("📈 Évaluation comparative sur le jeu de test"):
    if st.button("Lancer l'évaluation comparative"):
        from evaluation import evaluate_all
        results = evaluate_all(TRAIN_DATA, TEST_DATA, statistical_choice)
        for approach, r in results.items():
            st.markdown(f"**{approach}**")
            if "error" in r:
                st.info(r["error"])
                continue
            st.write(f"Accuracy globale : `{r['accuracy']:.2f}`")
            st.table(r["report"])
            if r["errors"]:
                st.write("Erreurs de classification :")
                for e in r["errors"]:
                    st.write(f"- « {e['text']} » → prédit **{e['predicted']}**, attendu **{e['gold']}**")
