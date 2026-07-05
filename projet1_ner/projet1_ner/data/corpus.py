# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Collecte des données
-------------------------------
On constitue un petit corpus francophone sur le domaine de l'Intelligence
Artificielle. Dans un vrai projet, ce corpus serait bien plus grand (des
centaines de documents scrapés ou issus de datasets publics) ; ici on en
garde un échantillon volontairement lisible pour pouvoir vérifier chaque
annotation à la main.

On sépare :
- CORPUS_TRAIN : textes qui serviront à entraîner le composant NER personnalisé.
- CORPUS_TEST  : textes JAMAIS vus à l'entraînement, utilisés à l'étape 6
                 (évaluation) pour mesurer la vraie performance du modèle.
"""

CORPUS_TRAIN = [
    "Le Machine Learning permet aux ordinateurs d'apprendre à partir de données "
    "sans être explicitement programmés.",

    "Les réseaux de neurones (NN) sont à la base du Deep Learning moderne.",

    "Un RNN (réseau de neurones récurrent) est particulièrement adapté au "
    "traitement de séquences comme le texte ou la parole.",

    "Le projet MOD-2024-001 a été lancé le 12 janvier 2024 pour tester un "
    "modèle de classification de texte.",

    "L'apprentissage non supervisé (Unsupervised Learning) ne nécessite pas de "
    "données étiquetées, contrairement à l'apprentissage supervisé.",

    "TensorFlow et PyTorch sont les deux frameworks les plus utilisés pour "
    "entraîner des modèles de Deep Learning.",

    "Le modèle a obtenu un score F1-score de 0.89 et une précision de 0.91 "
    "sur le jeu de test.",

    "spaCy propose un pipeline complet pour la reconnaissance d'entités "
    "nommées (NER) en plusieurs langues.",

    "Le dataset ImageNet a longtemps servi de référence pour évaluer les "
    "modèles de vision par ordinateur.",

    "La version 2.15 de TensorFlow apporte des optimisations pour "
    "l'entraînement distribué.",

    "Le code du projet CODE-778 a été mis à jour le 03/06/2025 après "
    "l'intégration d'un nouveau module d'attention.",

    "Les CNN (réseaux de neurones convolutifs) sont très efficaces pour "
    "l'analyse d'images.",

    "Hugging Face propose une bibliothèque très populaire pour utiliser des "
    "modèles Transformer pré-entraînés.",

    "Le rappel obtenu était de 0.85, ce qui reste inférieur à la précision "
    "mesurée sur ce même modèle.",

    "Scikit-learn reste une référence pour les algorithmes classiques de "
    "Machine Learning comme la régression logistique ou le SVM.",
]

# Chaque exemple de test est accompagné de ses entités gold (vérité terrain),
# annotées manuellement, pour pouvoir calculer précision/rappel/F1 à l'étape 6.
# Format : (texte, [(start_char, end_char, label), ...])
CORPUS_TEST = [
    (
        "Le Deep Learning et les réseaux de neurones ont transformé le "
        "traitement automatique du langage.",
        [(3, 16, "TECHNIQUE"), (24, 44, "TECHNIQUE")],
    ),
    (
        "Le projet PROJ-2025-042 utilise PyTorch pour entraîner un modèle "
        "de type Transformer depuis le 15 mars 2025.",
        [
            (10, 23, "CODE_PROJET"),
            (33, 40, "FRAMEWORK"),
            (75, 86, "TECHNIQUE"),
            (98, 111, "DATE"),
        ],
    ),
    (
        "Le modèle atteint une précision de 0.93 et un rappel de 0.88 sur "
        "le dataset CoNLL-2003.",
        [
            (23, 32, "METRIC"),
            (47, 53, "METRIC"),
            (78, 89, "DATASET"),
        ],
    ),
    (
        "TensorFlow 2.16 et Keras facilitent la construction de réseaux de "
        "neurones profonds.",
        [(0, 15, "VERSION"), (19, 24, "FRAMEWORK"), (52, 72, "TECHNIQUE")],
    ),
]
