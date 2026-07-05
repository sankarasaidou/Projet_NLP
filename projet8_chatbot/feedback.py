# -*- coding: utf-8 -*-
"""
ÉTAPE 6 : Apprentissage continu
--------------------------------------
Mécanisme de feedback simple mais réaliste :
    - chaque interaction est journalisée (question, intention prédite,
      réponse donnée, feedback utilisateur 👍/👎)
    - les feedbacks négatifs sont accumulés dans une liste de "cas à
      revoir" pour un humain (ex. l'équipe pédagogique de l'UV-BF)
    - si l'utilisateur indique la BONNE intention lors d'un feedback
      négatif, on peut l'ajouter directement à la base d'entraînement
      (data augmentation supervisée) et ré-entraîner le classifieur

En production, ce journal serait persisté en base de données ; ici on
utilise une liste en mémoire (voir aussi app.py pour la version
Streamlit avec state persistant le temps de la session).
"""

import json
from datetime import datetime, timezone

from faq_data import FAQ_DATABASE
from intent_classifier import IntentClassifier

FEEDBACK_LOG = []


def log_interaction(question: str, predicted_intent: str, answer: str, feedback: str | None = None):
    """feedback : "positif", "négatif", ou None si pas encore donné."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "predicted_intent": predicted_intent,
        "answer": answer,
        "feedback": feedback,
    }
    FEEDBACK_LOG.append(entry)
    return entry


def get_negative_feedback_cases() -> list[dict]:
    """Retourne les cas où le chatbot a été jugé incorrect -> à examiner
    par un humain (équipe pédagogique) ou à ré-injecter en entraînement."""
    return [e for e in FEEDBACK_LOG if e["feedback"] == "négatif"]


def retrain_with_correction(question: str, correct_intent: str) -> IntentClassifier:
    """Ajoute une correction humaine à la base d'entraînement et
    ré-entraîne le classifieur d'intentions -> amélioration continue."""
    FAQ_DATABASE.append({
        "intent": correct_intent,
        "question": question,
        "answer": "(réponse à compléter par l'équipe pédagogique)",
    })
    return IntentClassifier().fit(FAQ_DATABASE)


def analytics_summary() -> dict:
    """Statistiques simples pour un tableau de bord."""
    total = len(FEEDBACK_LOG)
    positive = sum(1 for e in FEEDBACK_LOG if e["feedback"] == "positif")
    negative = sum(1 for e in FEEDBACK_LOG if e["feedback"] == "négatif")
    pending = total - positive - negative

    intent_counts = {}
    for e in FEEDBACK_LOG:
        intent_counts[e["predicted_intent"]] = intent_counts.get(e["predicted_intent"], 0) + 1

    return {
        "total_interactions": total,
        "positive": positive,
        "negative": negative,
        "pending": pending,
        "satisfaction_rate": positive / (positive + negative) if (positive + negative) else None,
        "intent_distribution": intent_counts,
    }


if __name__ == "__main__":
    log_interaction("Comment m'inscrire ?", "inscription", "Réponse sur les inscriptions...", "positif")
    log_interaction("Le cours de stats bug", "administratif", "Réponse administrative...", "négatif")
    print(json.dumps(analytics_summary(), indent=2, ensure_ascii=False))
