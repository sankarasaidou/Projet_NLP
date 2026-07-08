# -*- coding: utf-8 -*-
"""
Interface Streamlit pour tester le modèle NER de domaine.

Lancement : streamlit run app.py
"""

import streamlit as st
import spacy
from spacy import displacy

from train_ner import MODEL_DIR, regex_entity_component, build_pipeline, train  # noqa: F401
from evaluate import evaluate

st.set_page_config(page_title="NER Domaine IA", layout="wide")
st.title("🔎 Reconnaissance d'entités nommées — Domaine IA")
st.caption(
    "Pipeline hybride : EntityRuler (termes connus) + NER entraîné "
    "(généralisation) + couche Regex (dates, versions, codes projet)."
)

LABEL_COLORS = {
    "TECHNIQUE": "#8ef",
    "FRAMEWORK": "#fea",
    "DATASET": "#faa",
    "METRIC": "#afa",
    "DATE": "#dda0dd",
    "VERSION": "#ffd1a9",
    "CODE_PROJET": "#c9c9ff",
}


@st.cache_resource(show_spinner="Chargement / entraînement du modèle...")
def load_model():
    if not MODEL_DIR.exists():
        nlp = train(n_iter=30)
    else:
        nlp = spacy.load(MODEL_DIR)
    return nlp


nlp = load_model()

tab_test, tab_eval = st.tabs(["🧪 Tester le modèle", "📊 Évaluation"])

with tab_test:
    default_text = (
        "Le projet PROJ-2025-042 utilise TensorFlow 2.15 pour entraîner un "
        "modèle Transformer depuis le 12 janvier 2024, avec une précision "
        "de 0.91 sur le dataset ImageNet."
    )
    text = st.text_area("Texte à analyser :", value=default_text, height=120)

    if st.button("Analyser", type="primary"):
        doc = nlp(text)

        if doc.ents:
            html = displacy.render(
                doc, style="ent", options={"colors": LABEL_COLORS}, jupyter=False
            )
            st.markdown(html, unsafe_allow_html=True)

            st.subheader("Entités détectées")
            st.table(
                [
                    {
                        "Texte": ent.text,
                        "Label": ent.label_,
                        "Début": ent.start_char,
                        "Fin": ent.end_char,
                    }
                    for ent in doc.ents
                ]
            )
        else:
            st.info("Aucune entité détectée dans ce texte.")

with tab_eval:
    st.write(
        "Évaluation du modèle sur le jeu de test (textes jamais vus à "
        "l'entraînement, annotés manuellement dans `data/corpus.py`)."
    )
    if st.button("Lancer l'évaluation"):
        results = evaluate(nlp)
        col1, col2, col3 = st.columns(3)
        col1.metric("Précision globale", f"{results['precision']:.2f}")
        col2.metric("Rappel global", f"{results['recall']:.2f}")
        col3.metric("F1-score global", f"{results['f1']:.2f}")

        st.subheader("Détail par label")
        rows = []
        for label, s in results["by_label"].items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            rows.append({"Label": label, "Précision": round(p, 2), "Rappel": round(r, 2), "F1": round(f1, 2)})
        st.table(rows)
