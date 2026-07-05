# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Application de différentes techniques de modélisation de sujets
------------------------------------------------------------------------
Chaque fonction renvoie une structure commune :

    {
        "topics": [
            {"id": 0, "top_words": ["mot1", "mot2", ...], "weight_per_doc": [...]},
            ...
        ],
        "doc_topic_assignment": [topic_id_du_doc_0, topic_id_du_doc_1, ...],
        "method": "LDA" | "NMF" | "BERTopic",
    }

afin de pouvoir comparer facilement les 3 méthodes (étape 5) et les
afficher de façon uniforme dans Streamlit.

- LDA (Latent Dirichlet Allocation) et NMF (Non-negative Matrix
  Factorization) : implémentées avec scikit-learn, sur une matrice
  BoW/TF-IDF classique.
- BERTopic : bibliothèque dédiée (embeddings + clustering + c-TF-IDF),
  optionnelle (paquet plus lourd, nécessite `pip install bertopic`).
"""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from preprocessing import preprocess_corpus


def _top_words_per_topic(model, feature_names, n_words=8):
    topics = []
    for topic_id, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:n_words]
        top_words = [feature_names[i] for i in top_indices]
        topics.append({"id": topic_id, "top_words": top_words})
    return topics


def run_lda(documents: list[str], n_topics: int = 5, n_words: int = 8, seed: int = 42):
    clean_docs = preprocess_corpus(documents)
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(clean_docs)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=seed, learning_method="batch", max_iter=50
    )
    doc_topic_matrix = lda.fit_transform(doc_term_matrix)

    topics = _top_words_per_topic(lda, feature_names, n_words)
    assignment = doc_topic_matrix.argmax(axis=1).tolist()
    return {"topics": topics, "doc_topic_assignment": assignment, "method": "LDA"}


def run_nmf(documents: list[str], n_topics: int = 5, n_words: int = 8, seed: int = 42):
    clean_docs = preprocess_corpus(documents)
    vectorizer = TfidfVectorizer()
    doc_term_matrix = vectorizer.fit_transform(clean_docs)
    feature_names = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=n_topics, random_state=seed, init="nndsvda", max_iter=500)
    doc_topic_matrix = nmf.fit_transform(doc_term_matrix)

    topics = _top_words_per_topic(nmf, feature_names, n_words)
    assignment = doc_topic_matrix.argmax(axis=1).tolist()
    return {"topics": topics, "doc_topic_assignment": assignment, "method": "NMF"}


def run_bertopic(documents: list[str], n_words: int = 8):
    """BERTopic gère lui-même l'embedding + le clustering + le choix du
    nombre de topics (contrairement à LDA/NMF où n_topics est fixé a
    priori). Nécessite `pip install bertopic`."""
    try:
        from bertopic import BERTopic
    except ImportError as e:
        raise ImportError("bertopic n'est pas installé : `pip install bertopic`") from e

    clean_docs = preprocess_corpus(documents)
    # min_topic_size bas car notre corpus de démo est petit ; à ajuster
    # (valeur par défaut 10) sur un corpus réel de centaines de documents.
    topic_model = BERTopic(language="french", min_topic_size=2, verbose=False)
    assignment, _ = topic_model.fit_transform(clean_docs)

    topics = []
    for topic_id in set(assignment):
        if topic_id == -1:
            continue  # -1 = documents considérés comme "bruit" par BERTopic
        words_scores = topic_model.get_topic(topic_id)
        top_words = [w for w, _ in words_scores[:n_words]]
        topics.append({"id": topic_id, "top_words": top_words})

    return {"topics": topics, "doc_topic_assignment": list(assignment), "method": "BERTopic"}


TOPIC_METHODS = {"LDA": run_lda, "NMF": run_nmf, "BERTopic": run_bertopic}


if __name__ == "__main__":
    from corpus import CORPUS

    for name in ["LDA", "NMF"]:
        print(f"\n=== {name} ===")
        result = TOPIC_METHODS[name](CORPUS, n_topics=5) if name != "BERTopic" else TOPIC_METHODS[name](CORPUS)
        for topic in result["topics"]:
            print(f"Topic {topic['id']}: {', '.join(topic['top_words'])}")
