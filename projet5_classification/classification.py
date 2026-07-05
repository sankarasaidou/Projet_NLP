# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Vectorisation multiple
ÉTAPE 4 : Classification comparative
------------------------------------------
On combine chaque technique de vectorisation avec chaque algorithme de
classification, et on garde le meilleur couple (voir aussi étape 5,
optimisation, dans optimization.py).

Vectorisations : BoW, TF-IDF, Word2Vec (moyenne), BERT (sentence-transformers)
Classifieurs   : Naive Bayes, SVM, MLP (réseau de neurones sklearn)

Naive Bayes suppose des features positives (comptes ou fréquences) : il
n'est donc appliqué qu'aux vectorisations BoW/TF-IDF, pas à
Word2Vec/BERT dont les vecteurs peuvent être négatifs.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from preprocessing import preprocess


def _vectorize_bow(train_texts, test_texts):
    vec = CountVectorizer()
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec


def _vectorize_tfidf(train_texts, test_texts):
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec


def _vectorize_word2vec(train_texts, test_texts, vector_size=100):
    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise ImportError("gensim n'est pas installé : `pip install gensim`") from e

    tokenized_train = [t.split() for t in train_texts]
    model = Word2Vec(sentences=tokenized_train, vector_size=vector_size, min_count=1, seed=42)

    def avg_vector(tokens):
        vecs = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)

    X_train = np.array([avg_vector(t) for t in tokenized_train])
    X_test = np.array([avg_vector(t.split()) for t in test_texts])
    return X_train, X_test, model


def _vectorize_bert(train_texts, test_texts, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("sentence-transformers n'est pas installé.") from e

    model = SentenceTransformer(model_name)
    X_train = np.array(model.encode(train_texts))
    X_test = np.array(model.encode(test_texts))
    return X_train, X_test, model


VECTORIZERS = {
    "BoW": _vectorize_bow,
    "TF-IDF": _vectorize_tfidf,
    "Word2Vec": _vectorize_word2vec,
    "BERT": _vectorize_bert,
}

CLASSIFIERS = {
    "Naive Bayes": lambda: MultinomialNB(),
    "SVM": lambda: SVC(kernel="linear", probability=True, random_state=42),
    "Réseau de neurones (MLP)": lambda: MLPClassifier(
        hidden_layer_sizes=(32,), max_iter=2000, random_state=42
    ),
}

# Naive Bayes exige des valeurs non négatives : incompatible avec
# Word2Vec/BERT (vecteurs pouvant être négatifs).
_INCOMPATIBLE = {("Naive Bayes", "Word2Vec"), ("Naive Bayes", "BERT")}


def train_and_predict(train_data, test_data, vectorizer_name, classifier_name):
    if (classifier_name, vectorizer_name) in _INCOMPATIBLE:
        raise ValueError(
            f"{classifier_name} n'est pas compatible avec des vecteurs {vectorizer_name} "
            "(valeurs potentiellement négatives). Choisir BoW ou TF-IDF avec Naive Bayes."
        )

    train_texts_raw = [t for t, _ in train_data]
    train_labels = [l for _, l in train_data]
    test_texts_raw = [t for t, _ in test_data]
    test_labels = [l for _, l in test_data]

    clean_train = [preprocess(t) for t in train_texts_raw]
    clean_test = [preprocess(t) for t in test_texts_raw]

    X_train, X_test, _ = VECTORIZERS[vectorizer_name](clean_train, clean_test)
    clf = CLASSIFIERS[classifier_name]()
    clf.fit(X_train, train_labels)

    predictions = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        confidences = proba.max(axis=1).tolist()
    else:
        confidences = [None] * len(predictions)

    accuracy = sum(p == g for p, g in zip(predictions, test_labels)) / len(test_labels)
    return {
        "predictions": list(predictions),
        "confidences": confidences,
        "gold": test_labels,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    from data import LABELED_CORPUS, TEST_CORPUS

    for vec_name in ["BoW", "TF-IDF"]:
        for clf_name in ["Naive Bayes", "SVM", "Réseau de neurones (MLP)"]:
            if (clf_name, vec_name) in _INCOMPATIBLE:
                continue
            result = train_and_predict(LABELED_CORPUS, TEST_CORPUS, vec_name, clf_name)
            print(f"{vec_name:<8} + {clf_name:<28} accuracy = {result['accuracy']:.2f}")
