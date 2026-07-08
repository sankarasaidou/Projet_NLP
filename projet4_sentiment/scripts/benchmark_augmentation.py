# -*- coding: utf-8 -*-
"""
Benchmark de l'augmentation de données — SANS FUITE train/test.
---------------------------------------------------------------------
Ce script répond à une question simple et essentielle : le dataset
synthétique généré (scripts/generate_synthetic_dataset.py) aide-t-il
VRAIMENT le modèle à généraliser sur du texte réaliste, ou le modèle
apprend-il simplement à reconnaître les formules fixes du générateur
("Je recommande vivement.", "À éviter absolument."...) ?

Méthodologie rigoureuse :
    1. On entraîne UNIQUEMENT sur les données synthétiques.
    2. On évalue UNIQUEMENT sur les 120 exemples originaux (écrits à la
       main, AUCUN gabarit) : un jeu de test totalement indépendant,
       jamais vu ni en train ni en validation croisée.

Comparé à la simple validation croisée sur le dataset fusionné (qui
donne ~0.99 -- un chiffre TROMPEUR, gonflé par la régularité des
gabarits), ce benchmark donne une mesure honnête de la généralisation
réelle.

Usage : python scripts/benchmark_augmentation.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analysis.statistical import StatisticalSentimentClassifier
from sentiment_analysis.evaluation import classification_report
from sentiment_analysis.logging_config import get_logger

logger = get_logger("benchmark")

ORIGINAL_120_PATH = Path(__file__).parent.parent / "data" / "dataset_original_120.csv"
SYNTHETIC_PATH = Path(__file__).parent.parent / "data" / "dataset_synthetique.csv"


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def main():
    if not ORIGINAL_120_PATH.exists() or not SYNTHETIC_PATH.exists():
        logger.error(
            "Fichiers requis manquants (%s, %s). Ce benchmark a besoin des "
            "deux fichiers séparés (avant fusion) pour tester sans fuite.",
            ORIGINAL_120_PATH, SYNTHETIC_PATH,
        )
        return

    train_texts, train_labels = load_csv(SYNTHETIC_PATH)
    test_texts, test_labels = load_csv(ORIGINAL_120_PATH)

    logger.info("Entraînement sur %d exemples SYNTHÉTIQUES uniquement...", len(train_texts))
    classifier = StatisticalSentimentClassifier("NaiveBayes").fit(train_texts, train_labels)

    predictions = classifier.predict(test_texts)
    report = classification_report(predictions, test_labels)

    logger.info("=" * 70)
    logger.info("RÉSULTAT (aucune fuite train/test) : accuracy = %.2f", report["accuracy"])
    logger.info("=" * 70)
    for label, m in report["per_label"].items():
        logger.info(
            "  %-10s précision=%.2f rappel=%.2f f1=%.2f (support=%d)",
            label, m["precision"], m["recall"], m["f1"], m["support"],
        )

    logger.info(
        "\nPour comparaison : le dataset original seul (120 exemples, "
        "split interne) obtenait ~0.42-0.46 d'accuracy en validation "
        "croisée avant augmentation."
    )


if __name__ == "__main__":
    main()
