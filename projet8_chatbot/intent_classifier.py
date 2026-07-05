# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Classification d'intentions
--------------------------------------------
Entraîne un classifieur (TF-IDF + SVM, choix robuste sur peu de données)
à reconnaître l'INTENTION d'une question (inscription, cours, examen,
technique, administratif), à partir des questions-types de la FAQ.

Point important : avant classification, on remplace les entités
détectées (ex. "NLP", "certificat de scolarité") par un placeholder
générique ("{ENTITE}") -> le classifieur apprend l'INTENTION indépendamment
de la matière ou du document précis mentionné, ce qui généralise
beaucoup mieux avec peu d'exemples (sinon il faudrait un exemple par
matière x intention, ce qui n'est pas réaliste).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from faq_data import FAQ_DATABASE
from ner import extract_entities


def _mask_entities(text: str) -> str:
    """Remplace chaque entité détectée par un placeholder générique."""
    for ent in extract_entities(text):
        text = text.replace(ent["text"], "ENTITE")
    return text


class IntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel="linear", probability=True, random_state=42)
        self._is_fitted = False

    def fit(self, faq_database=None):
        faq_database = faq_database or FAQ_DATABASE
        questions = [_mask_entities(item["question"]) for item in faq_database]
        intents = [item["intent"] for item in faq_database]

        X = self.vectorizer.fit_transform(questions)
        self.model.fit(X, intents)
        self._is_fitted = True
        return self

    def predict(self, text: str) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Le classifieur doit être entraîné (fit()) avant de prédire.")
        masked = _mask_entities(text)
        X = self.vectorizer.transform([masked])
        intent = self.model.predict(X)[0]
        confidence = self.model.predict_proba(X).max()
        return {"intent": intent, "confidence": float(confidence)}


if __name__ == "__main__":
    clf = IntentClassifier().fit()
    tests = [
        "Comment je fais pour m'inscrire cette année ?",
        "Le cours de statistiques ne s'affiche pas.",
        "Je n'arrive pas à me connecter, ça bug.",
        "Quand est l'examen d'anglais ?",
    ]
    for t in tests:
        result = clf.predict(t)
        print(f"{t!r} -> {result['intent']} (confiance={result['confidence']:.2f})")
