# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Annotation manuelle des données
-------------------------------------------
On définit ici les entités que le modèle doit apprendre à reconnaître.
Elles sont volontairement EN DEHORS des entités classiques (PER, ORG, LOC...) :

    - TECHNIQUE     : une technique / méthode d'IA (Machine Learning, RNN, CNN...)
    - FRAMEWORK     : une bibliothèque / outil logiciel (TensorFlow, spaCy...)
    - DATASET       : un jeu de données de référence (ImageNet, CoNLL-2003...)
    - METRIC        : une métrique d'évaluation (précision, F1-score...)

Les entités structurées et régulières (DATE, VERSION, CODE_PROJET) sont
volontairement laissées à la couche Regex (étape 3 / patterns.py) : un
modèle statistique n'apporte rien pour reconnaître "12 janvier 2024" ou
"MOD-2024-001", une expression régulière est plus fiable et plus rapide.
C'est un choix de conception à assumer et expliquer dans un rapport de projet.

Plutôt que de compter les caractères à la main (source d'erreurs), on
localise chaque entité avec str.find() : plus robuste et plus lisible.
"""

from .corpus import CORPUS_TRAIN


def _find_all(text, entities_as_terms):
    """Convertit une liste de (terme_exact, label) en triplets
    (start, end, label) en cherchant le terme dans le texte."""
    spans = []
    for term, label in entities_as_terms:
        start = text.find(term)
        if start == -1:
            raise ValueError(f"Terme introuvable : {term!r} dans {text!r}")
        end = start + len(term)
        spans.append((start, end, label))
    # spaCy exige que les spans ne se chevauchent pas et soient triés
    spans.sort(key=lambda s: s[0])
    return spans


# Association explicite texte -> (terme exact, label) à annoter.
_RAW_ANNOTATIONS = [
    (0, [("Machine Learning", "TECHNIQUE")]),
    (1, [("réseaux de neurones", "TECHNIQUE"), ("Deep Learning", "TECHNIQUE")]),
    (2, [("RNN", "TECHNIQUE")]),
    (4, [
        ("apprentissage non supervisé", "TECHNIQUE"),
        ("Unsupervised Learning", "TECHNIQUE"),
        ("apprentissage supervisé", "TECHNIQUE"),
    ]),
    (5, [("TensorFlow", "FRAMEWORK"), ("PyTorch", "FRAMEWORK"), ("Deep Learning", "TECHNIQUE")]),
    (6, [("F1-score", "METRIC"), ("précision", "METRIC")]),
    (7, [("spaCy", "FRAMEWORK")]),
    (8, [("ImageNet", "DATASET")]),
    (11, [("CNN", "TECHNIQUE"), ("réseaux de neurones convolutifs", "TECHNIQUE")]),
    (12, [("Hugging Face", "FRAMEWORK"), ("Transformer", "TECHNIQUE")]),
    (13, [("rappel", "METRIC"), ("précision", "METRIC")]),
    (14, [("Scikit-learn", "FRAMEWORK"), ("Machine Learning", "TECHNIQUE")]),
]

TRAIN_DATA = []
for idx, terms in _RAW_ANNOTATIONS:
    text = CORPUS_TRAIN[idx]
    entities = _find_all(text, terms)
    TRAIN_DATA.append((text, {"entities": entities}))


ENTITY_LABELS = sorted({label for _, ann in TRAIN_DATA for _, _, label in ann["entities"]})
