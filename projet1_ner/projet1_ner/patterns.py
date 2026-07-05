# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Développement de motifs Regex
------------------------------------------
Deux familles de motifs sont construites ici :

1. PHRASE_PATTERNS : des motifs "à base de tokens" pour l'EntityRuler de
   spaCy. Ils couvrent les entités lexicales (TECHNIQUE, FRAMEWORK,
   DATASET, METRIC) : listes de termes connus, insensibles à la casse.
   -> Rapide, précis, mais ne généralise pas à des termes jamais listés
      (c'est pour ça qu'on ajoute AUSSI un NER entraîné, étape 5).

2. REGEX_PATTERNS : des expressions régulières "brutes" appliquées
   directement sur le texte (caractère par caractère), pour les entités
   structurées où un pattern est bien plus fiable qu'un modèle statistique :
      - DATE         : "12 janvier 2024", "03/06/2025"
      - VERSION      : "TensorFlow 2.15", "Python 3.11", "v2.1"
      - CODE_PROJET  : "MOD-2024-001", "PROJ-2025-042", "CODE-778"

Ces regex sont ensuite injectées comme entités via un composant spaCy
personnalisé (voir train_ner.py -> régle "regex_entity_component").
"""

import re

# --- 1. Motifs lexicaux pour l'EntityRuler --------------------------------

_TECHNIQUES = [
    "Machine Learning", "Deep Learning", "Unsupervised Learning",
    "apprentissage supervisé", "apprentissage non supervisé",
    "réseaux de neurones", "réseau de neurones",
    "réseaux de neurones convolutifs", "réseau de neurones récurrent",
    "NN", "RNN", "CNN", "Transformer",
]
_FRAMEWORKS = [
    "TensorFlow", "PyTorch", "spaCy", "Keras", "Scikit-learn",
    "Hugging Face",
]
_DATASETS = [
    "ImageNet", "CoNLL-2003", "MNIST", "IMDB",
]
_METRICS = [
    "précision", "rappel", "F1-score", "accuracy",
]


def _make_phrase_patterns(terms, label):
    """Construit des patterns EntityRuler multi-tokens à partir d'une
    liste de termes (gère les termes composés de plusieurs mots)."""
    patterns = []
    for term in terms:
        tokens = term.split(" ")
        pattern = [{"LOWER": tok.lower()} for tok in tokens]
        patterns.append({"label": label, "pattern": pattern})
    return patterns


PHRASE_PATTERNS = (
    _make_phrase_patterns(_TECHNIQUES, "TECHNIQUE")
    + _make_phrase_patterns(_FRAMEWORKS, "FRAMEWORK")
    + _make_phrase_patterns(_DATASETS, "DATASET")
    + _make_phrase_patterns(_METRICS, "METRIC")
)

# --- 2. Motifs regex "brut caractère" -------------------------------------

REGEX_PATTERNS = {
    "DATE": re.compile(
        r"\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|"
        r"août|septembre|octobre|novembre|décembre)\s+\d{4}\b"
        r"|\b\d{2}/\d{2}/\d{4}\b",
        re.IGNORECASE,
    ),
    "VERSION": re.compile(
        r"\b(?:TensorFlow|PyTorch|Python|spaCy|Keras)\s+\d+(?:\.\d+)+\b"
        r"|\bv\d+(?:\.\d+)+\b",
        re.IGNORECASE,
    ),
    "CODE_PROJET": re.compile(
        r"\b(?:MOD|PROJ|CODE)-\d{2,4}(?:-\d{2,4})?\b"
    ),
}
