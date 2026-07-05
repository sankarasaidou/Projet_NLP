# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Vectorisation des données
---------------------------------------
On implémente les 4 techniques demandées, chacune renvoyant :
    - une matrice de vecteurs documents (n_docs x dim)
    - un vecteur pour la requête (liste de mots-clés)
afin qu'on puisse ensuite calculer une similarité cosinus (étape 4)
de façon identique quelle que soit la technique choisie.

    1. Bag of Words (BoW)      -> sklearn CountVectorizer
    2. TF-IDF                  -> sklearn TfidfVectorizer
    3. Word2Vec                -> gensim, moyenne des vecteurs de mots
    4. BERT embeddings         -> sentence-transformers (embedding de phrase)

BoW et TF-IDF sont testables avec seulement scikit-learn. Word2Vec et
BERT nécessitent des paquets plus lourds (gensim, sentence-transformers)
qui peuvent ne pas être installés partout : on lève un ImportError clair
si c'est le cas, plutôt que de planter silencieusement.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class VectorizationResult:
    doc_vectors: np.ndarray          # (n_docs, dim)
    query_vector: np.ndarray         # (1, dim)
    method: str


def vectorize_bow(corpus_texts: list[str], query_terms: list[str]) -> VectorizationResult:
    vectorizer = CountVectorizer()
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform([" ".join(query_terms)])
    return VectorizationResult(doc_matrix.toarray(), query_matrix.toarray(), "BoW")


def vectorize_tfidf(corpus_texts: list[str], query_terms: list[str]) -> VectorizationResult:
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform([" ".join(query_terms)])
    return VectorizationResult(doc_matrix.toarray(), query_matrix.toarray(), "TF-IDF")


def vectorize_word2vec(corpus_texts: list[str], query_terms: list[str],
                        vector_size: int = 100, window: int = 5, min_count: int = 1) -> VectorizationResult:
    """Entraîne un Word2Vec sur le corpus (petit corpus -> modèle simple,
    dans un vrai projet on utiliserait un modèle pré-entraîné plus large),
    puis représente chaque document par la MOYENNE des vecteurs de ses mots."""
    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise ImportError(
            "gensim n'est pas installé : `pip install gensim`"
        ) from e

    tokenized_docs = [text.split() for text in corpus_texts]
    model = Word2Vec(
        sentences=tokenized_docs, vector_size=vector_size,
        window=window, min_count=min_count, workers=1, seed=42,
    )

    def _average_vector(tokens):
        vectors = [model.wv[tok] for tok in tokens if tok in model.wv]
        if not vectors:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)

    doc_vectors = np.array([_average_vector(doc) for doc in tokenized_docs])
    query_vector = _average_vector(query_terms).reshape(1, -1)
    return VectorizationResult(doc_vectors, query_vector, "Word2Vec")


def vectorize_bert(corpus_texts: list[str], query_terms: list[str],
                    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> VectorizationResult:
    """Utilise sentence-transformers pour obtenir des embeddings de phrase
    contextuels (BERT). Modèle multilingue pour bien gérer le français."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers n'est pas installé : "
            "`pip install sentence-transformers`"
        ) from e

    model = SentenceTransformer(model_name)
    doc_vectors = model.encode(corpus_texts)
    query_vector = model.encode([" ".join(query_terms)])
    return VectorizationResult(np.array(doc_vectors), np.array(query_vector), "BERT")


VECTORIZERS = {
    "BoW": vectorize_bow,
    "TF-IDF": vectorize_tfidf,
    "Word2Vec": vectorize_word2vec,
    "BERT": vectorize_bert,
}


def similarity_ranking(result: VectorizationResult) -> np.ndarray:
    """Retourne le tableau des similarités cosinus (un score par document)
    entre chaque document et le vecteur requête."""
    return cosine_similarity(result.doc_vectors, result.query_vector).flatten()
