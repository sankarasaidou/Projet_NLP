# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Calcul de la similarité
ÉTAPE 5 : Évaluation et comparaison
--------------------------------------
Pour une technique de vectorisation donnée et une liste de mots-clés,
`top_x_documents` renvoie les X documents les plus proches (similarité
cosinus). `compare_techniques` fait tourner les 4 techniques sur les 20
premiers documents et permet de comparer les classements obtenus.
"""

import numpy as np

from preprocessing import preprocess, preprocess_corpus
from vectorizers import VECTORIZERS, similarity_ranking


def top_x_documents(documents: list[dict], keywords: list[str], method: str, x: int = 5):
    """Retourne les x documents les plus similaires aux mots-clés donnés,
    selon la technique de vectorisation `method` ("BoW", "TF-IDF",
    "Word2Vec" ou "BERT")."""
    if method not in VECTORIZERS:
        raise ValueError(f"Méthode inconnue : {method}. Choix possibles : {list(VECTORIZERS)}")

    corpus_texts = preprocess_corpus(documents)
    query_terms = preprocess(" ".join(keywords)) if method != "BERT" else keywords

    vectorize_fn = VECTORIZERS[method]
    result = vectorize_fn(corpus_texts, query_terms)
    scores = similarity_ranking(result)

    ranking = np.argsort(scores)[::-1][:x]
    return [
        {
            "title": documents[i]["title"],
            "url": documents[i]["url"],
            "score": float(scores[i]),
        }
        for i in ranking
    ]


def compare_techniques(documents: list[dict], keywords: list[str], x: int = 5,
                        methods: list[str] | None = None) -> dict:
    """Compare les classements TopX obtenus par chaque technique demandée
    (par défaut : celles qui ne nécessitent pas de paquet supplémentaire
    non garanti, i.e. BoW et TF-IDF ; ajouter Word2Vec/BERT si gensim /
    sentence-transformers sont installés)."""
    methods = methods or ["BoW", "TF-IDF"]
    comparison = {}
    for method in methods:
        try:
            comparison[method] = top_x_documents(documents[:20], keywords, method, x)
        except ImportError as e:
            comparison[method] = {"error": str(e)}
    return comparison


if __name__ == "__main__":
    from corpus_sample import SAMPLE_CORPUS

    keywords = ["agriculture", "production", "rendement"]
    for method in ["BoW", "TF-IDF"]:
        print(f"\n=== {method} — mots-clés : {keywords} ===")
        for doc in top_x_documents(SAMPLE_CORPUS, keywords, method, x=3):
            print(f"  {doc['score']:.3f}  {doc['title']}")
