# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Matching intelligent
ÉTAPE 5 : Génération de réponses
--------------------------------------
1. On classe d'abord l'intention de la question (étape 3).
2. On restreint la recherche de réponse aux entrées de la FAQ ayant
   la MÊME intention (matching intelligent = pas de recherche globale
   inefficace, on exploite la classification pour filtrer).
3. Parmi ces entrées, on calcule la similarité cosinus (TF-IDF) entre
   la question posée et chaque question-type de la FAQ, on prend la
   plus proche.
4. On insère l'entité détectée (ex. MATIERE="NLP") dans le template de
   réponse ({MATIERE} -> "NLP") avant de renvoyer la réponse finale.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import FAQ_DATABASE
from ner import extract_entities
from intent_classifier import IntentClassifier

_intent_classifier = None


def _get_classifier():
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier().fit()
    return _intent_classifier


def _fill_template(template: str, entities: list[dict]) -> str:
    filled = template
    for ent in entities:
        placeholder = "{" + ent["label"] + "}"
        if placeholder in filled:
            filled = filled.replace(placeholder, ent["text"])
    # S'il reste des placeholders non résolus (aucune entité détectée),
    # on les remplace par une formule générique plutôt que de les laisser
    # visibles tels quels dans la réponse.
    filled = filled.replace("{MATIERE}", "la matière concernée")
    return filled


def answer_question(user_question: str, faq_database=None) -> dict:
    faq_database = faq_database or FAQ_DATABASE
    classifier = _get_classifier()

    intent_result = classifier.predict(user_question)
    predicted_intent = intent_result["intent"]

    candidates = [item for item in faq_database if item["intent"] == predicted_intent]
    if not candidates:
        candidates = faq_database  # fallback si aucune entrée pour cette intention

    vectorizer = TfidfVectorizer()
    candidate_questions = [item["question"] for item in candidates]
    tfidf_matrix = vectorizer.fit_transform(candidate_questions + [user_question])

    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    best_index = similarities.argmax()
    best_match = candidates[best_index]

    entities = extract_entities(user_question)
    final_answer = _fill_template(best_match["answer"], entities)

    return {
        "intent": predicted_intent,
        "intent_confidence": intent_result["confidence"],
        "matched_question": best_match["question"],
        "match_similarity": float(similarities[best_index]),
        "entities": entities,
        "answer": final_answer,
    }


if __name__ == "__main__":
    tests = [
        "Comment accéder au cours de NLP ?",
        "Je veux m'inscrire, quels documents faut-il ?",
        "Ma vidéo de cours d'informatique ne se charge pas.",
    ]
    for t in tests:
        result = answer_question(t)
        print(f"\nQ: {t}")
        print(f"  Intention détectée : {result['intent']} (confiance={result['intent_confidence']:.2f})")
        print(f"  Entités : {result['entities']}")
        print(f"  Réponse : {result['answer']}")
