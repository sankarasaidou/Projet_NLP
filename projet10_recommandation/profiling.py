# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Profiling utilisateur
ÉTAPE 3 : Analyse de contenu
------------------------------------
- Analyse de contenu : chaque article est représenté par un vecteur
  TF-IDF (features textuelles). C'est un choix volontairement simple et
  robuste ; on pourrait substituer des embeddings BERT sans changer le
  reste du pipeline (même interface vectorielle).

- Profiling utilisateur : le profil d'un utilisateur est la MOYENNE
  PONDÉRÉE des vecteurs des articles qu'il a lus, pondérée par sa note
  d'engagement (1 à 5) -> les articles qui l'ont le plus engagé pèsent
  davantage dans son profil que ceux lus rapidement.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data import ARTICLES, USER_HISTORY


def build_content_vectors():
    """Vectorise tous les articles avec TF-IDF -> (matrice, vectorizer, liste d'ids)."""
    article_ids = list(ARTICLES.keys())
    texts = [ARTICLES[aid]["text"] for aid in article_ids]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer, article_ids


def build_user_profile(user_id: str, content_matrix, article_ids: list[str]) -> np.ndarray:
    """Profil utilisateur = moyenne pondérée des vecteurs des articles lus."""
    history = USER_HISTORY.get(user_id, [])
    if not history:
        return np.zeros(content_matrix.shape[1])

    vectors, weights = [], []
    for article_id, rating in history:
        if article_id in article_ids:
            idx = article_ids.index(article_id)
            vectors.append(content_matrix[idx].toarray().flatten())
            weights.append(rating)

    if not vectors:
        return np.zeros(content_matrix.shape[1])

    vectors = np.array(vectors)
    weights = np.array(weights) / sum(weights)  # normalisation
    return np.average(vectors, axis=0, weights=weights)


if __name__ == "__main__":
    matrix, vectorizer, article_ids = build_content_vectors()
    profile = build_user_profile("user_A", matrix, article_ids)
    top_terms_idx = profile.argsort()[::-1][:5]
    feature_names = vectorizer.get_feature_names_out()
    print("Termes les plus caractéristiques du profil de user_A :")
    print([feature_names[i] for i in top_terms_idx])
