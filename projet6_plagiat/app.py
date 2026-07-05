# -*- coding: utf-8 -*-
"""
Interface Streamlit — Détecteur de plagiat.
Lancement : streamlit run app.py
"""

import streamlit as st

from data import DOCUMENTS, KNOWN_CASES
from similarity_metrics import METRICS
from threshold_optimization import find_optimal_thresholds
from validation import validate
from visualization import similarity_matrix, plot_similarity_heatmap, plot_dendrogram

st.set_page_config(page_title="Détecteur de plagiat", layout="wide")
st.title("🕵️ Détecteur de plagiat — mesures de similarité")
st.caption("Cosinus (n-grammes) · Jaccard (n-grammes) · Distance de Levenshtein")

tab_compare, tab_matrix, tab_validation = st.tabs(
    ["📝 Comparer 2 documents", "🔥 Matrice de similarité", "✅ Validation"]
)

with tab_compare:
    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_area("Document A", value=DOCUMENTS["doc_original_1"], height=150)
    with col2:
        text_b = st.text_area("Document B", value=DOCUMENTS["doc_plagiat_reformule_1"], height=150)

    if st.button("Comparer", type="primary") and text_a.strip() and text_b.strip():
        thresholds = find_optimal_thresholds()
        for metric_name, metric_fn in METRICS.items():
            score = metric_fn(text_a, text_b)
            threshold = thresholds[metric_name]["threshold"]
            is_plagiarism = score >= threshold
            st.metric(
                metric_name,
                f"{score:.3f}",
                f"{'⚠️ Plagiat probable' if is_plagiarism else '✅ Pas de plagiat détecté'} (seuil={threshold:.2f})",
            )

with tab_matrix:
    metric_choice = st.selectbox("Métrique pour la matrice", list(METRICS.keys()))
    matrix, names = similarity_matrix(DOCUMENTS, metric_choice)

    heatmap_path = plot_similarity_heatmap(matrix, names, f"Similarité — {metric_choice}", "/tmp/heatmap.png")
    st.image(heatmap_path, use_container_width=True)

    if st.checkbox("Afficher le clustering hiérarchique (dendrogramme)"):
        dendro_path = plot_dendrogram(matrix, names, "/tmp/dendrogram.png")
        st.image(dendro_path, use_container_width=True)

with tab_validation:
    st.write(
        "Validation des seuils optimisés sur des cas connus (plagiat "
        "total, partiel, reformulé, et documents indépendants)."
    )
    if st.button("Lancer la validation"):
        report = validate()
        for metric_name, info in report.items():
            st.subheader(f"{metric_name} (seuil = {info['threshold']:.2f}, accuracy = {info['accuracy']:.2f})")
            st.table([
                {
                    "Doc A": c["doc_a"], "Doc B": c["doc_b"],
                    "Score": c["score"], "Attendu": c["expected"],
                    "Prédit": c["predicted"], "Correct": "✅" if c["correct"] else "❌",
                }
                for c in info["cases"]
            ])
