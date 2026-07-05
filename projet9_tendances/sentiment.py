# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Analyse de sentiment
--------------------------------------
Deux classifications complémentaires, comme demandé dans le sujet
("voir si les commentaires sont neutres et/ou libres") :

1. Sentiment classique : positif / négatif / neutre, à partir d'un
   petit dictionnaire de polarité (même principe que le projet 4),
   enrichi ici pour prendre en compte les EMOJIS comme signal fort de
   sentiment (souvent plus fiable que le texte seul sur des commentaires
   courts).

2. "Neutre" vs "libre" (expression d'opinion) : un commentaire est classé
   "libre" (expression d'opinion / point de vue personnel) s'il contient
   des marqueurs subjectifs (pronoms "je"/"on", ponctuation expressive,
   exclamation, emoji, hashtag d'opinion) ; sinon il est classé "neutre"
   (énoncé factuel, ex. "Le budget est mentionné dans le communiqué.").
   C'est un indicateur utile pour la modération éditoriale : distinguer
   un commentaire factuel d'un commentaire qui exprime librement une
   opinion (qui peut nécessiter une modération différente).
"""

import re

from preprocessing import clean_comment, extract_emojis

POSITIVE_EMOJIS = {"👏", "🙏", "❤️", "🌾", "🙌", "😊", "👍"}
NEGATIVE_EMOJIS = {"😢", "😡", "😒", "🙄"}

POLARITY_LEXICON = {
    "excellente": 2.0, "bravo": 2.0, "merci": 1.0, "content": 1.5, "super": 1.5,
    "espoir": 1.0, "plaisir": 1.5, "aider": 1.0, "bonne": 1.0,
    "déception": -2.0, "promesses": -0.5, "colère": -2.0, "ras": -1.0,
    "trop": 0.0,  # neutre seul, dépend du mot suivant (intensificateur)
    "élevé": -1.0, "encore": -0.3,
}

SUBJECTIVITY_MARKERS = re.compile(
    r"\b(je|j'|on|nous|mon|ma|mes|franchement|vraiment)\b|!{1,}|\?{2,}", re.IGNORECASE
)


def analyze_sentiment(text: str) -> dict:
    cleaned = clean_comment(text)
    words = cleaned["clean_text"].split()

    score = sum(POLARITY_LEXICON.get(w, 0.0) for w in words)
    for emoji in cleaned["emojis"]:
        if emoji in POSITIVE_EMOJIS:
            score += 1.5
        elif emoji in NEGATIVE_EMOJIS:
            score -= 1.5

    if score > 0.5:
        sentiment = "positif"
    elif score < -0.5:
        sentiment = "négatif"
    else:
        sentiment = "neutre"

    return {"sentiment": sentiment, "score": score}


def classify_neutral_or_free(text: str) -> str:
    """Retourne "libre" (expression d'opinion) ou "neutre" (énoncé factuel)."""
    cleaned = clean_comment(text)
    has_subjectivity = bool(SUBJECTIVITY_MARKERS.search(text))
    has_emoji = len(cleaned["emojis"]) > 0
    has_hashtag_opinion = len(cleaned["hashtags"]) > 0

    return "libre" if (has_subjectivity or has_emoji or has_hashtag_opinion) else "neutre"


def full_analysis(text: str) -> dict:
    sentiment_result = analyze_sentiment(text)
    return {
        **sentiment_result,
        "type_expression": classify_neutral_or_free(text),
    }


if __name__ == "__main__":
    from comments_data import COMMENTS

    for c in COMMENTS[:5]:
        result = full_analysis(c["text"])
        print(f"{c['text'][:50]!r:<55} -> {result['sentiment']:<8} "
              f"(score={result['score']:.1f}) | {result['type_expression']}")
