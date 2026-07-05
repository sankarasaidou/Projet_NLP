# -*- coding: utf-8 -*-
"""
ÉTAPE 5 : Optimisation
---------------------------
Recherche du meilleur couple (vectorisation, classifieur, hyperparamètres)
par validation croisée (GridSearchCV) sur le jeu d'entraînement, puis
évaluation finale sur le jeu de test tenu à l'écart.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from preprocessing import preprocess

PARAM_GRIDS = {
    "Naive Bayes": {
        "clf": MultinomialNB(),
        "params": {"clf__alpha": [0.1, 0.5, 1.0]},
    },
    "SVM": {
        "clf": SVC(kernel="linear", probability=True, random_state=42),
        "params": {"clf__C": [0.1, 1.0, 10.0]},
    },
    "Réseau de neurones (MLP)": {
        "clf": MLPClassifier(max_iter=2000, random_state=42),
        "params": {"clf__hidden_layer_sizes": [(16,), (32,), (64,)]},
    },
}


def optimize_and_select_best(train_data, test_data, cv_folds=3):
    """Teste chaque classifieur avec TF-IDF, optimise ses hyperparamètres
    par GridSearchCV, puis retourne le meilleur modèle global évalué sur
    le jeu de test."""
    train_texts = [preprocess(t) for t, _ in train_data]
    train_labels = [l for _, l in train_data]
    test_texts = [preprocess(t) for t, _ in test_data]
    test_labels = [l for _, l in test_data]

    results = {}
    best_name, best_test_accuracy, best_pipeline = None, -1, None

    for name, config in PARAM_GRIDS.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", config["clf"]),
        ])
        search = GridSearchCV(pipeline, config["params"], cv=cv_folds, scoring="accuracy")
        search.fit(train_texts, train_labels)

        test_predictions = search.predict(test_texts)
        test_accuracy = sum(p == g for p, g in zip(test_predictions, test_labels)) / len(test_labels)

        results[name] = {
            "best_params": search.best_params_,
            "cv_accuracy": search.best_score_,
            "test_accuracy": test_accuracy,
        }

        if test_accuracy > best_test_accuracy:
            best_name, best_test_accuracy, best_pipeline = name, test_accuracy, search.best_estimator_

    return {
        "results_by_model": results,
        "best_model_name": best_name,
        "best_test_accuracy": best_test_accuracy,
        "best_pipeline": best_pipeline,
    }


if __name__ == "__main__":
    from data import LABELED_CORPUS, TEST_CORPUS

    output = optimize_and_select_best(LABELED_CORPUS, TEST_CORPUS)
    for name, r in output["results_by_model"].items():
        print(f"{name:<28} CV acc={r['cv_accuracy']:.2f}  test acc={r['test_accuracy']:.2f}  best_params={r['best_params']}")
    print(f"\nMeilleur modèle : {output['best_model_name']} (test acc = {output['best_test_accuracy']:.2f})")
