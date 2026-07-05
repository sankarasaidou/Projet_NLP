# -*- coding: utf-8 -*-
"""
Interface Streamlit — Comparaison des techniques de vectorisation.
Lancement : streamlit run app.py
"""

import streamlit as st

from corpus_sample import SAMPLE_CORPUS
from scraper import scrape_lefaso, SEED_KEYWORDS
from similarity import top_x_documents

st.set_page_config(page_title="Comparaison de vectorisation", layout="wide")
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
    )
    method = st.selectbox("Technique de vectorisation", ["BoW", "TF-IDF", "Word2Vec", "BERT"])
    top_x = st.slider("Nombre de documents à afficher (TopX)", min_value=1, max_value=20, value=5)
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
)
keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

if st.button("Rechercher", type="primary") and keywords:
    try:
        results = top_x_documents(documents, keywords, method, x=top_x)
    except ImportError as e:
        st.error(str(e))
        results = []

    if results:
        st.subheader(f"Top {top_x} documents — méthode {method}")
        for rank, doc in enumerate(results, start=1):
            st.markdown(f"**{rank}. {doc['title']}**  \n"
                        f"Score de similarité : `{doc['score']:.3f}`  \n"
                        f"[{doc['url']}]({doc['url']})")
            st.divider()
    else:
        st.info("Aucun résultat à afficher.")

with st.expander("🔬 Comparer plusieurs techniques en une fois"):
    methods_to_compare = st.multiselect(
        "Techniques à comparer", ["BoW", "TF-IDF", "Word2Vec", "BERT"], default=["BoW", "TF-IDF"]
    )
    if st.button("Comparer") and keywords:
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
