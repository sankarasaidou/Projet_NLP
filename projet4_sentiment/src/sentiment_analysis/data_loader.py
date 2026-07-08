# -*- coding: utf-8 -*-
"""Chargement des données depuis /data (dataset + lexique), au lieu de
les coder en dur dans le code source -> plus facile à faire grandir,
versionner et remplacer sans toucher au code (ex. brancher un vrai
lexique publié comme FEEL ou Blogoscopie)."""

import csv
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from sentiment_analysis.config import DATASET_PATH, LEXICON_PATH, TEST_SIZE, RANDOM_STATE
from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Dataset:
    train_texts: list[str]
    train_labels: list[str]
    test_texts: list[str]
    test_labels: list[str]


def load_dataset() -> Dataset:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}. "
            "Vérifie que data/dataset.csv existe bien à la racine du projet."
        )

    texts, labels = [], []
    with open(DATASET_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])

    if not texts:
        raise ValueError(f"Le dataset {DATASET_PATH} est vide.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )
    logger.info(
        "Dataset chargé : %d exemples (train=%d, test=%d).",
        len(texts), len(train_texts), len(test_texts),
    )
    return Dataset(train_texts, train_labels, test_texts, test_labels)


def load_lexicon() -> dict[str, float]:
    if not LEXICON_PATH.exists():
        raise FileNotFoundError(
            f"Lexique introuvable : {LEXICON_PATH}. "
            "Vérifie que data/lexicon_fr.csv existe bien à la racine du projet."
        )

    lexicon = {}
    with open(LEXICON_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lexicon[row["word"]] = float(row["polarity"])

    logger.info("Lexique chargé : %d mots.", len(lexicon))
    return lexicon
