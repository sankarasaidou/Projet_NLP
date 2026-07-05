# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Constitution du corpus
-------------------------------------
Pour valider un détecteur de plagiat, il faut des CAS CONNUS :
    - des paires de documents identiques (plagiat total)
    - des paires reformulées (plagiat partiel / paraphrase)
    - des paires totalement différentes (pas de plagiat, "négatifs")

C'est indispensable pour l'étape 6 (validation) : sans cas connus, on ne
peut pas savoir si les seuils choisis à l'étape 4 sont pertinents.
"""

DOCUMENTS = {
    "doc_original_1": (
        "L'intelligence artificielle transforme profondément la société "
        "moderne, notamment à travers l'automatisation des tâches "
        "répétitives et l'analyse de grandes quantités de données."
    ),
    "doc_plagiat_exact_1": (
        "L'intelligence artificielle transforme profondément la société "
        "moderne, notamment à travers l'automatisation des tâches "
        "répétitives et l'analyse de grandes quantités de données."
    ),
    "doc_plagiat_reformule_1": (
        "L'IA change en profondeur nos sociétés contemporaines, en "
        "particulier grâce à l'automatisation de tâches répétitives et au "
        "traitement de vastes volumes de données."
    ),
    "doc_original_2": (
        "La gestion durable des ressources en eau constitue un enjeu "
        "majeur pour l'agriculture dans les régions arides, en particulier "
        "face aux effets du changement climatique."
    ),
    "doc_plagiat_partiel_2": (
        "Le changement climatique impacte fortement l'agriculture des "
        "zones arides. La gestion durable des ressources en eau constitue "
        "un enjeu majeur pour l'agriculture dans les régions arides."
    ),
    "doc_independant_1": (
        "Le championnat national de football a repris ce week-end avec "
        "plusieurs matchs disputés dans des stades combles à travers le "
        "pays."
    ),
    "doc_independant_2": (
        "Le nouveau musée d'art contemporain propose une exposition "
        "consacrée aux artistes émergents de la sous-région, avec des "
        "œuvres originales et engagées."
    ),
}

# Vérité terrain pour l'évaluation (étape 6) : (doc_a, doc_b, plagiat_attendu)
KNOWN_CASES = [
    ("doc_original_1", "doc_plagiat_exact_1", True),      # plagiat total
    ("doc_original_1", "doc_plagiat_reformule_1", True),  # paraphrase -> plagiat
    ("doc_original_2", "doc_plagiat_partiel_2", True),     # plagiat partiel
    ("doc_original_1", "doc_independant_1", False),        # aucun rapport
    ("doc_original_2", "doc_independant_2", False),        # aucun rapport
    ("doc_independant_1", "doc_independant_2", False),     # aucun rapport
]
