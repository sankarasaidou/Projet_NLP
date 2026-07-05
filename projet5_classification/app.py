# -*- coding: utf-8 -*-
"""
ÉTAPE 6 : Déploiement — Interface de classification en temps réel.
Lancement : streamlit run app.py
"""
import os
import sys

# Permet à Python de trouver les scripts locaux propres à ce projet
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

import streamlit as st

from data import LABELED_CORPUS, TEST_CORPUS, CATEGORIES
from classification import train_and_predict, VECTORIZERS, CLASSIFIERS, _INCOMPATIBLE
from optimization import optimize_and_select_best
from preprocessing import preprocess

st.set_page_config(page_title="Classification d'articles", layout="wide")
st.title("📰 Classification automatique d'articles de presse")
st.caption(f"Catégories : {', '.join(CATEGORIES)}")

with st.sidebar:
    st.header("⚙️ Paramètres")
    vectorizer_name = st.selectbox("Technique de vectorisation", list(VECTORIZERS.keys()))
    compatible_classifiers = [
        c for c in CLASSIFIERS if (c, vectorizer_name) not in _INCOMPATIBLE
    ]
    classifier_name = st.selectbox("Algorithme de classification", compatible_classifiers)


tab_classify, tab_optimize = st.tabs(["📤 Classifier un article", "🔧 Optimisation & meilleur modèle"])

with tab_classify:
    uploaded_file = st.file_uploader("Uploader un article (.txt)", type=["txt"])
    default_text = (
        "Le gouvernement a annoncé une réforme fiscale destinée à soutenir "
        "les petites entreprises face à la hausse des prix."
    )
    if uploaded_file is not None:
        article_text = uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        article_text = st.text_area("Ou collez le texte de l'article :", value=default_text, height=150)

    if st.button("Classifier", type="primary") and article_text.strip():
        try:
            result = train_and_predict(
                LABELED_CORPUS, [(article_text, "?")],  # "?" = label inconnu, à prédire
                vectorizer_name, classifier_name,
            )
            predicted_label = result["predictions"][0]
            confidence = result["confidences"][0]

            st.success(f"Catégorie prédite : **{predicted_label}**")
            if confidence is not None:
                st.metric("Score de confiance", f"{confidence * 100:.1f}%")
                st.progress(min(confidence, 1.0))
            else:
                st.info("Ce classifieur ne fournit pas de score de confiance calibré.")
        except ValueError as e:
            st.error(str(e))

with tab_optimize:
    st.write(
        "Recherche automatique des meilleurs hyperparamètres (GridSearchCV) "
        "pour chaque algorithme, puis sélection du meilleur modèle global."
    )
    if st.button("Lancer l'optimisation"):
        output = optimize_and_select_best(LABELED_CORPUS, TEST_CORPUS)
        rows = [
            {
                "Modèle": name,
                "Meilleurs hyperparamètres": str(r["best_params"]),
                "Accuracy CV": round(r["cv_accuracy"], 2),
                "Accuracy test": round(r["test_accuracy"], 2),
            }
            for name, r in output["results_by_model"].items()
        ]
        st.table(rows)
        st.success(
            f"Meilleur modèle : **{output['best_model_name']}** "
            f"(accuracy test = {output['best_test_accuracy']:.2f})"
        )
