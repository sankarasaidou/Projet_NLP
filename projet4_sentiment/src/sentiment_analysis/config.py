# -*- coding: utf-8 -*-
"""Configuration centralisée du projet.

Bonne pratique de production : tous les chemins et paramètres
ajustables sont ici, dans UN SEUL endroit, plutôt que dispersés (codés
en dur) dans chaque module. Peut être surchargée par variables
d'environnement pour un déploiement Docker/Cloud sans modifier le code.
"""

import os
from pathlib import Path

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("SENTIMENT_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("SENTIMENT_MODELS_DIR", PROJECT_ROOT / "models"))

DATASET_PATH = DATA_DIR / "dataset.csv"
LEXICON_PATH = DATA_DIR / "lexicon_fr.csv"

STATISTICAL_MODEL_PATH = MODELS_DIR / "statistical_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# --- Paramètres du split train/test ---
TEST_SIZE = float(os.environ.get("SENTIMENT_TEST_SIZE", 0.2))
RANDOM_STATE = int(os.environ.get("SENTIMENT_RANDOM_STATE", 42))

# --- Paramètres de l'approche lexicale ---
NEUTRAL_THRESHOLD = float(os.environ.get("SENTIMENT_NEUTRAL_THRESHOLD", 0.5))
NEGATION_WINDOW = 3       # nb de mots précédents examinés pour détecter une négation
POSITIVE_EMOJI_SCORE = 1.5
NEGATIVE_EMOJI_SCORE = -1.5

# --- Modèle statistique par défaut ---
DEFAULT_STATISTICAL_MODEL = os.environ.get("SENTIMENT_DEFAULT_MODEL", "SVM")  # SVM | NaiveBayes | LogisticRegression

# --- Approche neuronale (optionnelle) ---
NEURAL_MODEL_CHECKPOINT = os.environ.get("SENTIMENT_NEURAL_CHECKPOINT", "camembert-base")
ENABLE_NEURAL = os.environ.get("SENTIMENT_ENABLE_NEURAL", "false").lower() == "true"

LABELS = ["négatif", "neutre", "positif"]

# --- Logging ---
LOG_LEVEL = os.environ.get("SENTIMENT_LOG_LEVEL", "INFO")
