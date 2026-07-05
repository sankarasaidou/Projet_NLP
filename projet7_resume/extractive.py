# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Résumé extractif
--------------------------------
Deux approches de résumé extractif (on sélectionne les phrases les plus
importantes du texte original, sans en générer de nouvelles) :

1. TextRank : adaptation de PageRank aux phrases. On construit un
   graphe où chaque phrase est un nœud, et le poids des arêtes est la
   similarité entre phrases (cosinus sur TF-IDF). Les phrases les plus
   "centrales" (proches de beaucoup d'autres) sont considérées comme
   les plus importantes.

2. Scoring TF-IDF : on score chaque phrase par la somme des scores
   TF-IDF de ses mots -> approche plus simple, favorise les phrases
   contenant des mots rares mais importants dans le document.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text: str) -> list[str]:
    """Découpage simple en phrases (sur ponctuation forte)."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def summarize_textrank(text: str, n_sentences: int = 3, damping: float = 0.85, iterations: int = 30) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= n_sentences:
        return text

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)

    n = len(sentences)
    # Normalisation des lignes pour obtenir une matrice de transition
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = similarity_matrix / row_sums

    scores = np.ones(n) / n
    for _ in range(iterations):
        scores = (1 - damping) / n + damping * transition_matrix.T @ scores

    top_indices = sorted(np.argsort(scores)[::-1][:n_sentences])  # ordre chronologique
    return " ".join(sentences[i] for i in top_indices)


def summarize_tfidf_scoring(text: str, n_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= n_sentences:
        return text

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    top_indices = sorted(np.argsort(sentence_scores)[::-1][:n_sentences])
    return " ".join(sentences[i] for i in top_indices)


EXTRACTIVE_METHODS = {
    "TextRank": summarize_textrank,
    "Scoring TF-IDF": summarize_tfidf_scoring,
}


if __name__ == "__main__":
    from corpus import DOCUMENTS

    text = DOCUMENTS["Article de presse"]
    print("=== TextRank ===")
    print(summarize_textrank(text, n_sentences=2))
    print("\n=== Scoring TF-IDF ===")
    print(summarize_tfidf_scoring(text, n_sentences=2))
