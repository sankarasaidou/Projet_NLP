# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Approche statistique
-----------------------------------
Classification supervisée classique : on vectorise le texte prétraité
(TF-IDF) puis on entraîne 3 classifieurs à comparer entre eux :
    - Naive Bayes (MultinomialNB)     -> rapide, bonne baseline sur du texte
    - SVM (LinearSVC)                 -> souvent très performant en texte
    - Régression logistique           -> interprétable, probabilités calibrées
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from preprocessing import tokenize_lexical

MODELS = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(random_state=42),
    "Régression logistique": LogisticRegression(max_iter=1000, random_state=42),
}


class StatisticalSentimentClassifier:
    def __init__(self, model_name: str = "SVM"):
        if model_name not in MODELS:
            raise ValueError(f"Modèle inconnu : {model_name}. Choix : {list(MODELS)}")
        self.model_name = model_name
        self.vectorizer = TfidfVectorizer()
        self.model = MODELS[model_name]
        self._is_fitted = False

    def _prepare(self, texts):
        return [" ".join(tokenize_lexical(t)) for t in texts]

    def fit(self, texts: list[str], labels: list[str]):
        clean_texts = self._prepare(texts)
        X = self.vectorizer.fit_transform(clean_texts)
        self.model.fit(X, labels)
        self._is_fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if not self._is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant de prédire (fit()).")
        clean_texts = self._prepare(texts)
        X = self.vectorizer.transform(clean_texts)
        return list(self.model.predict(X))


if __name__ == "__main__":
    from corpus import TRAIN_DATA, TEST_DATA

    train_texts, train_labels = zip(*TRAIN_DATA)
    test_texts, test_labels = zip(*TEST_DATA)

    for name in MODELS:
        clf = StatisticalSentimentClassifier(name).fit(list(train_texts), list(train_labels))
        preds = clf.predict(list(test_texts))
        accuracy = sum(p == g for p, g in zip(preds, test_labels)) / len(test_labels)
        print(f"{name:<24} accuracy sur le test = {accuracy:.2f}  | préds : {preds}")
