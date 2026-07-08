#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script d'entraînement — à exécuter UNE FOIS (ou à chaque mise à jour
du dataset) pour entraîner et sauvegarder les modèles statistiques.

Séparer entraînement (`train.py`, exécuté ponctuellement, potentiellement
sur une machine différente) et serving (`app.py`, exécuté en continu)
est une pratique standard de production : l'application ne réentraîne
jamais un modèle à chaque démarrage, elle charge un artefact déjà validé.

Usage :
    python train.py                       # entraîne les 3 modèles statistiques
    python train.py --model SVM           # entraîne seulement SVM
    python train.py --report              # affiche aussi le rapport d'évaluation
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sentiment_analysis.config import METADATA_PATH
from sentiment_analysis.data_loader import load_dataset
from sentiment_analysis.statistical import StatisticalSentimentClassifier, MODEL_FACTORIES, _tokenized_text
from sentiment_analysis.evaluation import classification_report, error_analysis
from sentiment_analysis.logging_config import get_logger

logger = get_logger("train")


def cross_validate_model(model_name: str, dataset, n_folds: int = 5) -> float:
    """Validation croisée stratifiée sur TRAIN+TEST réunis : donne une
    estimation bien plus fiable de la performance réelle qu'un simple
    split unique, surtout sur un petit dataset (120 exemples) où un seul
    découpage train/test peut être optimiste ou pessimiste par hasard."""
    all_texts = dataset.train_texts + dataset.test_texts
    all_labels = dataset.train_labels + dataset.test_labels
    clean_texts = [_tokenized_text(t) for t in all_texts]

    pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MODEL_FACTORIES[model_name]())])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, clean_texts, all_labels, cv=cv, scoring="accuracy")
    return float(scores.mean())


def train_model(model_name: str, dataset, show_report: bool) -> None:
    classifier = StatisticalSentimentClassifier(model_name)
    classifier.fit(dataset.train_texts, dataset.train_labels)
    classifier.save()

    predictions = classifier.predict(dataset.test_texts)
    report = classification_report(predictions, dataset.test_labels)

    logger.info("=== %s : accuracy test = %.2f ===", model_name, report["accuracy"])
    if show_report:
        for label, metrics in report["per_label"].items():
            logger.info(
                "  %-10s précision=%.2f rappel=%.2f f1=%.2f (support=%d)",
                label, metrics["precision"], metrics["recall"], metrics["f1"], metrics["support"],
            )
        errors = error_analysis(dataset.test_texts, predictions, dataset.test_labels)
        if errors:
            logger.info("  Erreurs de classification (%d) :", len(errors))
            for e in errors:
                logger.info("    %r -> prédit=%s attendu=%s", e["text"][:60], e["predicted"], e["expected"])


def main():
    parser = argparse.ArgumentParser(description="Entraîne et sauvegarde les modèles statistiques.")
    parser.add_argument(
        "--model", choices=list(MODEL_FACTORIES) + ["all"], default="all",
        help="Modèle à entraîner (par défaut : all = les 3 modèles).",
    )
    parser.add_argument("--report", action="store_true", help="Afficher le rapport détaillé par classe.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Nombre de folds pour la validation croisée.")
    args = parser.parse_args()

    dataset = load_dataset()

    models_to_train = list(MODEL_FACTORIES) if args.model == "all" else [args.model]
    cv_scores = {}
    for model_name in models_to_train:
        train_model(model_name, dataset, args.report)
        cv_scores[model_name] = cross_validate_model(model_name, dataset, args.cv_folds)
        logger.info("%-20s validation croisée (%d-fold) : accuracy = %.2f", model_name, args.cv_folds, cv_scores[model_name])

    if len(models_to_train) > 1:
        best_model = max(cv_scores, key=cv_scores.get)
        logger.info("=== Meilleur modèle (validation croisée) : %s (%.2f) ===", best_model, cv_scores[best_model])

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
        metadata["_best_model"] = best_model
        metadata["_cv_scores"] = cv_scores
        METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Entraînement terminé. Modèles sauvegardés dans models/.")


if __name__ == "__main__":
    main()
