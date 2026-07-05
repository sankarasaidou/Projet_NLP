# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Métriques de similarité
--------------------------------------
Trois métriques complémentaires, calculées sur des n-grammes de mots
(plus robuste qu'au mot isolé pour détecter du texte copié/reformulé) :

    - Similarité cosinus : angle entre les vecteurs de fréquence de
      n-grammes -> bonne pour détecter des réutilisations importantes de
      vocabulaire, insensible à la longueur du document.
    - Indice de Jaccard : |intersection| / |union| des ensembles de
      n-grammes -> simple et strict, pénalise fortement les
      reformulations qui changent beaucoup de mots.
    - Distance de Levenshtein (normalisée en similarité) : nombre
      minimal d'opérations (insertion/suppression/substitution de
      caractères) pour passer d'un texte à l'autre -> détecte bien les
      quasi-copies avec de petites modifications, mais coûteux en calcul
      et peu adapté aux reformulations profondes (changement de mots).
"""

from collections import Counter

import numpy as np

from preprocessing import tokenize, ngrams


def cosine_similarity_ngrams(text_a: str, text_b: str, n: int = 3) -> float:
    grams_a = Counter(ngrams(tokenize(text_a), n))
    grams_b = Counter(ngrams(tokenize(text_b), n))

    vocabulary = set(grams_a) | set(grams_b)
    if not vocabulary:
        return 0.0

    vec_a = np.array([grams_a.get(g, 0) for g in vocabulary])
    vec_b = np.array([grams_b.get(g, 0) for g in vocabulary])

    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def jaccard_similarity_ngrams(text_a: str, text_b: str, n: int = 3) -> float:
    set_a = set(ngrams(tokenize(text_a), n))
    set_b = set(ngrams(tokenize(text_b), n))

    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Distance d'édition classique (programmation dynamique)."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_similarity(text_a: str, text_b: str) -> float:
    """Normalise la distance de Levenshtein en similarité entre 0 et 1."""
    a, b = tokenize(text_a), tokenize(text_b)
    a_str, b_str = " ".join(a), " ".join(b)
    if not a_str and not b_str:
        return 1.0
    distance = levenshtein_distance(a_str, b_str)
    max_len = max(len(a_str), len(b_str))
    return 1 - distance / max_len if max_len else 1.0


METRICS = {
    "Cosinus (n-grammes)": cosine_similarity_ngrams,
    "Jaccard (n-grammes)": jaccard_similarity_ngrams,
    "Levenshtein": lambda a, b: levenshtein_similarity(a, b),
}


if __name__ == "__main__":
    from data import DOCUMENTS

    a = DOCUMENTS["doc_original_1"]
    b = DOCUMENTS["doc_plagiat_reformule_1"]
    for name, fn in METRICS.items():
        print(f"{name:<22}: {fn(a, b):.3f}")
