# -*- coding: utf-8 -*-
"""
Interface Streamlit — Générateur de résumés multi-techniques.
Lancement : streamlit run app.py
"""
import os
import sys

# Permet à Python de trouver les scripts locaux propres à ce projet
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)
    
import streamlit as st

from corpus import DOCUMENTS, REFERENCE_SUMMARIES
from extractive import EXTRACTIVE_METHODS
from optimization import summarize_adaptive, summarize_hybrid, compare_all_methods, STYLE_TO_N_SENTENCES
from evaluation import rouge_scores, coherence_score

st.set_page_config(page_title="Générateur de résumés", layout="wide")
st.title("📝 Générateur de résumés automatiques multi-techniques")
st.caption("Extractif (TextRank, TF-IDF) · Abstractif (T5, mBART) · Hybride")

with st.sidebar:
    st.header("⚙️ Paramètres")
    doc_choice = st.selectbox("Document à résumer", list(DOCUMENTS.keys()))
    style = st.select_slider("Style / longueur du résumé", options=["Court", "Moyen", "Long"], value="Moyen")

    all_methods = list(EXTRACTIVE_METHODS.keys()) + ["T5 (français)", "mBART (multilingue)", "Hybride (extractif + abstractif)"]
    method = st.selectbox("Méthode de résumé", all_methods)

text = DOCUMENTS[doc_choice]
reference = REFERENCE_SUMMARIES[doc_choice]

st.subheader("📄 Texte original")
st.write(text)

if st.button("Générer le résumé", type="primary"):
    try:
        if method == "Hybride (extractif + abstractif)":
            summary = summarize_hybrid(text, style)
        else:
            summary = summarize_adaptive(text, method, style)

        st.subheader(f"✨ Résumé ({method}, style {style})")
        st.success(summary)

        st.subheader("📊 Évaluation (ROUGE vs référence humaine)")
        scores = rouge_scores(summary, reference)
        cols = st.columns(3)
        for col, (name, s) in zip(cols, scores.items()):
            col.metric(name, f"F1={s['f1']:.2f}", f"P={s['precision']:.2f} R={s['recall']:.2f}")
        st.metric("Cohérence intra-résumé", f"{coherence_score(summary):.2f}")

    except ImportError as e:
        st.error(str(e))

with st.expander("🔬 Comparer toutes les méthodes disponibles"):
    if st.button("Comparer"):
        comparison = compare_all_methods(text, reference, style)
        for name, result in comparison.items():
            if "error" in result:
                st.info(f"{name} : {result['error']}")
                continue
            st.markdown(f"**{name}** — ROUGE-1 F1 = `{result['rouge']['ROUGE-1']['f1']:.2f}`, "
                        f"cohérence = `{result['coherence']:.2f}`")
            st.write(result["summary"])
            st.divider()
