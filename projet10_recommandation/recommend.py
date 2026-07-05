# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Similarité de contenu
ÉTAPE 5 : Algorithmes de recommandation
------------------------------------------------
Deux approches, comme demandé ("filtrage collaboratif et basé sur le
contenu") :

1. Content-based : similarité cosinus entre le PROFIL utilisateur
   (étape 2) et chaque article -> recommande les articles les plus
   proches du profil, en excluant ceux déjà lus.

2. Filtrage collaboratif (item-based) : construit une matrice
   utilisateur x article à partir des notes d'engagement, calcule la
   similarité entre ARTICLES en fonction des utilisateurs qui les ont
   appréciés en commun ("les gens qui ont aimé X ont aussi aimé Y"),
   puis recommande les articles proches de ceux déjà appréciés par
   l'utilisateur -- indépendamment du contenu textuel.

Un vrai système de production combine souvent les deux (approche
hybride) : voir `hybrid_recommend()`.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data import ARTICLES, USER_HISTORY
from profiling import build_content_vectors, build_user_profile


def content_based_recommend(user_id: str, top_n: int = 3) -> list[dict]:
    matrix, vectorizer, article_ids = build_content_vectors()
    profile = build_user_profile(user_id, matrix, article_ids)
    already_read = {aid for aid, _ in USER_HISTORY.get(user_id, [])}

    similarities = cosine_similarity(profile.reshape(1, -1), matrix.toarray()).flatten()

    ranking = []
    for idx, score in sorted(enumerate(similarities), key=lambda x: -x[1]):
        article_id = article_ids[idx]
        if article_id in already_read:
            continue
        ranking.append({"article_id": article_id, "title": ARTICLES[article_id]["title"], "score": float(score)})
        if len(ranking) >= top_n:
            break
    return ranking


def _build_rating_matrix():
    """Matrice utilisateur x article (0 si non noté)."""
    users = list(USER_HISTORY.keys())
    articles = list(ARTICLES.keys())
    matrix = np.zeros((len(users), len(articles)))
    for i, user in enumerate(users):
        for article_id, rating in USER_HISTORY[user]:
            j = articles.index(article_id)
            matrix[i, j] = rating
    return matrix, users, articles


def collaborative_recommend(user_id: str, top_n: int = 3) -> list[dict]:
    """Filtrage collaboratif item-based : similarité entre articles
    calculée sur les colonnes de la matrice de notes."""
    rating_matrix, users, articles = _build_rating_matrix()
    if user_id not in users:
        return []

    # Similarité article-article (colonnes de la matrice)
    item_similarity = cosine_similarity(rating_matrix.T)

    user_idx = users.index(user_id)
    user_ratings = rating_matrix[user_idx]
    already_read = {aid for aid, _ in USER_HISTORY.get(user_id, [])}

    # Score prédit pour chaque article non lu = moyenne pondérée des
    # similarités avec les articles déjà notés par l'utilisateur.
    scores = {}
    for j, article_id in enumerate(articles):
        if article_id in already_read:
            continue
        sims = item_similarity[j]
        weighted_sum = np.dot(sims, user_ratings)
        sim_sum = np.sum(np.abs(sims[user_ratings > 0])) if np.any(user_ratings > 0) else 0
        scores[article_id] = weighted_sum / sim_sum if sim_sum > 0 else 0.0

    ranking = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [{"article_id": aid, "title": ARTICLES[aid]["title"], "score": float(s)} for aid, s in ranking]


def hybrid_recommend(user_id: str, top_n: int = 3, content_weight: float = 0.5) -> list[dict]:
    """Combine content-based et filtrage collaboratif par moyenne pondérée
    des scores (les deux sont normalisés entre 0 et 1 au préalable)."""
    content_results = {r["article_id"]: r["score"] for r in content_based_recommend(user_id, top_n=len(ARTICLES))}
    collab_results = {r["article_id"]: r["score"] for r in collaborative_recommend(user_id, top_n=len(ARTICLES))}

    all_ids = set(content_results) | set(collab_results)
    combined = {}
    for aid in all_ids:
        c_score = content_results.get(aid, 0.0)
        cf_score = collab_results.get(aid, 0.0)
        combined[aid] = content_weight * c_score + (1 - content_weight) * cf_score

    ranking = sorted(combined.items(), key=lambda x: -x[1])[:top_n]
    return [{"article_id": aid, "title": ARTICLES[aid]["title"], "score": float(s)} for aid, s in ranking]


if __name__ == "__main__":
    for user_id in USER_HISTORY:
        print(f"\n=== Recommandations pour {user_id} ===")
        print("Content-based :", [r["title"] for r in content_based_recommend(user_id)])
        print("Collaboratif  :", [r["title"] for r in collaborative_recommend(user_id)])
        print("Hybride       :", [r["title"] for r in hybrid_recommend(user_id)])
