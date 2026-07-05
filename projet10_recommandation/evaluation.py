# -*- coding: utf-8 -*-
"""
ÉTAPE 6 : Évaluation
-------------------------
Validation LEAVE-ONE-OUT : pour chaque utilisateur ayant au moins 2
articles dans son historique, on retire TEMPORAIREMENT un article de
son profil (le "held-out"), on reconstruit son profil sans lui, puis on
vérifie si le système le recommande malgré tout dans son Top-N. Si oui,
c'est un signe que le système capture bien les préférences de
l'utilisateur (on a réussi à "deviner" un article qu'il avait
réellement apprécié, à partir du reste de son historique).

Métriques :
    - Précision@N : proportion des N recommandations qui sont
      pertinentes (ici, l'article held-out qu'on cherche à retrouver)
    - Rappel@N : proportion des articles pertinents qui sont dans le
      Top-N (avec un seul article held-out par utilisateur, rappel = 1
      si trouvé, 0 sinon)
    - Satisfaction utilisateur : proxy simple = note moyenne (1-5)
      donnée aux articles de l'historique complet -> indicateur du
      niveau d'engagement général de l'utilisateur avec les contenus
      recommandés par le passé.
"""

from data import ARTICLES, USER_HISTORY
from profiling import build_content_vectors, build_user_profile
from recommend import content_based_recommend


def leave_one_out_evaluation(top_n: int = 3) -> dict:
    results = {}

    for user_id, history in USER_HISTORY.items():
        if len(history) < 2:
            continue  # pas assez d'articles pour faire un split train/held-out

        held_out_article, held_out_rating = history[-1]
        reduced_history = history[:-1]

        # On simule temporairement un historique réduit (sans le held-out)
        original_history = USER_HISTORY[user_id]
        USER_HISTORY[user_id] = reduced_history
        try:
            recommendations = content_based_recommend(user_id, top_n=top_n)
        finally:
            USER_HISTORY[user_id] = original_history  # on restaure l'état original

        recommended_ids = [r["article_id"] for r in recommendations]
        hit = held_out_article in recommended_ids

        results[user_id] = {
            "held_out_article": held_out_article,
            "held_out_title": ARTICLES[held_out_article]["title"],
            "recommended": recommended_ids,
            "hit": hit,
            "precision_at_n": (1 / top_n) if hit else 0.0,
            "recall_at_n": 1.0 if hit else 0.0,
        }

    n_users = len(results)
    avg_precision = sum(r["precision_at_n"] for r in results.values()) / n_users if n_users else 0.0
    avg_recall = sum(r["recall_at_n"] for r in results.values()) / n_users if n_users else 0.0

    return {"per_user": results, "avg_precision_at_n": avg_precision, "avg_recall_at_n": avg_recall}


def satisfaction_proxy() -> dict:
    """Note moyenne d'engagement par utilisateur (proxy de satisfaction)."""
    return {
        user_id: sum(rating for _, rating in history) / len(history)
        for user_id, history in USER_HISTORY.items()
    }


if __name__ == "__main__":
    evaluation = leave_one_out_evaluation(top_n=3)
    for user_id, r in evaluation["per_user"].items():
        status = "✅ trouvé" if r["hit"] else "❌ manqué"
        print(f"{user_id}: held-out = {r['held_out_title']!r} -> {status} (recommandés: {r['recommended']})")

    print(f"\nPrécision@3 moyenne : {evaluation['avg_precision_at_n']:.2f}")
    print(f"Rappel@3 moyenne    : {evaluation['avg_recall_at_n']:.2f}")
    print(f"\nSatisfaction (note moyenne) par utilisateur : {satisfaction_proxy()}")
