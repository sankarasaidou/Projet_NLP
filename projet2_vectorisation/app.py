# -*- coding: utf-8 -*-
"""
Interface Streamlit — Comparaison des techniques de vectorisation.
"""
import os
import sys

# ==============================================================================
# 1. CORRECTIF DE CHEMINS LOCAL (Placé STRICTEMENT au début)
# ==============================================================================
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

import streamlit as st

# Imports locaux sécurisés grâce au sys.path ci-dessus
from corpus_sample import SAMPLE_CORPUS
from scraper import scrape_lefaso, SEED_KEYWORDS
from similarity import top_x_documents

# ==============================================================================
# 2. FONCTION PRINCIPALE APPELÉE PAR L'APP GLOBAL
# ==============================================================================
def main():
    # Suppression de st.set_page_config (géré par l'app.py racine)
    
    st.title("📚 Comparaison des techniques de vectorisation")
    st.caption(
        "BoW, TF-IDF, Word2Vec, BERT — comparez leur pertinence pour retrouver "
        "les documents les plus proches d'une liste de mots-clés."
    )

    with st.sidebar:
        st.header("⚙️ Paramètres")
        source = st.radio(
            "Source du corpus",
            ["Corpus de démo (hors-ligne)", "Scraper lefaso.net (réseau requis)"],
            key="p2_source_radio"
        )
        method = st.selectbox("Technique de vectorisation", ["BoW", "TF-IDF", "Word2Vec", "BERT"], key="p2_method_select")
        top_x = st.slider("Nombre de documents à afficher (TopX)", min_value=1, max_value=20, value=5, key="p2_topx_slider")
        st.caption(f"Mots-clés seed utilisés pour orienter la collecte : {', '.join(SEED_KEYWORDS)}")

    @st.cache_data(show_spinner="Chargement du corpus...")
    def load_corpus(source_choice: str):
        if source_choice.startswith("Scraper"):
            try:
                articles = scrape_lefaso(max_articles=20)
                if articles:
                    return articles
                st.warning("Aucun article récupéré, utilisation du corpus de démo.")
            except Exception as e:
                st.warning(f"Scraping indisponible ({e}), utilisation du corpus de démo.")
        return SAMPLE_CORPUS

    documents = load_corpus(source)
    st.write(f"Corpus chargé : **{len(documents)} documents**.")

    keywords_input = st.text_input(
        "Mots-clés à rechercher (séparés par des virgules)",
        value="agricole, production, rendement",
        key="p2_keywords_input"
    )
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    if st.button("Rechercher", type="primary", key="p2_search_btn") and keywords:
        try:
            results = top_x_documents(documents, keywords, method, x=top_x)
        except ImportError as e:
            st.error(str(e))
            results = []

        if results:
            st.subheader(f"Top {top_x} documents — méthode {method}")
            for rank, doc in enumerate(results, start=1):
                st.markdown(f"**{rank}. {doc['title']}** \n"
                            f"Score de similarité : `{doc['score']:.3f}`  \n"
                            f"[{doc['url']}]({doc['url']})")
                st.divider()
        else:
            st.info("Aucun résultat à afficher.")

    with st.expander("🔬 Comparer plusieurs techniques en une fois"):
        methods_to_compare = st.multiselect(
            "Techniques à comparer", ["BoW", "TF-IDF", "Word2Vec", "BERT"], default=["BoW", "TF-IDF"], key="p2_compare_multiselect"
        )
        if st.button("Comparer", key="p2_compare_btn") and keywords:
            from similarity import compare_techniques
            comparison = compare_techniques(documents, keywords, x=top_x, methods=methods_to_compare)
            cols = st.columns(len(comparison))
            for col, (method_name, docs) in zip(cols, comparison.items()):
                with col:
                    st.markdown(f"**{method_name}**")
                    if isinstance(docs, dict) and "error" in docs:
                        st.error(docs["error"])
                    else:
                        for doc in docs:
                            st.write(f"`{doc['score']:.3f}` — {doc['title']}")

# Permet de tester le projet de manière autonome si exécuté directement
if __name__ == "__main__":
    st.set_page_config(page_title="Comparaison de vectorisation", layout="wide")
    main()