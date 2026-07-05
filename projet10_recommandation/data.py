# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Corpus de contenu
--------------------------------
On réutilise un corpus d'articles multi-thématiques (mêmes thèmes que le
projet 3 : agriculture, santé, économie, éducation, sécurité), avec un
identifiant par article. On simule aussi des utilisateurs avec un
HISTORIQUE DE LECTURE (liste d'articles déjà lus), pour pouvoir
construire des profils utilisateurs (étape 2) et évaluer les
recommandations (étape 6).
"""

ARTICLES = {
    "art_01": {"title": "Hausse de la production céréalière",
               "text": "La campagne agricole s'annonce prometteuse grâce à une "
                       "pluviométrie favorable. Rendements en hausse pour le maïs et le sorgho."},
    "art_02": {"title": "Foire agricole régionale",
               "text": "La foire agricole régionale a réuni des agriculteurs venus "
                       "présenter leurs techniques de culture et d'irrigation."},
    "art_03": {"title": "Le prix des engrais impacte les agriculteurs",
               "text": "La hausse du prix des engrais pèse sur les petits agriculteurs "
                       "pour financer leurs intrants."},
    "art_04": {"title": "Campagne de vaccination",
               "text": "Le ministère de la santé lance une campagne de vaccination "
                       "infantile contre la rougeole dans plusieurs régions."},
    "art_05": {"title": "Nouveau centre de santé communautaire",
               "text": "Un centre de santé communautaire a ouvert pour les femmes "
                       "enceintes et les enfants en bas âge."},
    "art_06": {"title": "Plan de relance économique",
               "text": "Un plan de relance économique soutient les petites entreprises "
                       "et stimule l'investissement national."},
    "art_07": {"title": "Croissance économique révisée à la hausse",
               "text": "Les prévisions de croissance ont été révisées à la hausse, "
                       "portées par les exportations agricoles."},
    "art_08": {"title": "Rentrée scolaire, défis de l'éducation",
               "text": "La rentrée scolaire approche avec son lot de défis : manque "
                       "d'enseignants et classes surchargées."},
    "art_09": {"title": "Programme de bourses d'études",
               "text": "Un programme de bourses encourage les filières scientifiques "
                       "et techniques dans l'enseignement supérieur."},
    "art_10": {"title": "Renforcement des mesures de sécurité",
               "text": "Les autorités renforcent les dispositifs de sécurité face aux "
                       "menaces persistantes dans plusieurs localités."},
    "art_11": {"title": "Patrouilles conjointes à la frontière",
               "text": "Des patrouilles conjointes luttent contre les trafics "
                       "transfrontaliers le long de la frontière."},
    "art_12": {"title": "Inflation et pouvoir d'achat",
               "text": "L'inflation érode le pouvoir d'achat des ménages, poussant à "
                       "de nouvelles mesures économiques."},
}

# Historique de lecture : liste d'articles lus par utilisateur, avec une
# note implicite (1 = lu rapidement / peu engageant, 5 = lu en entier /
# très engageant) simulant un signal d'intérêt.
USER_HISTORY = {
    "user_A": [("art_01", 5), ("art_02", 4), ("art_03", 5)],           # profil "agriculture"
    "user_B": [("art_04", 5), ("art_05", 4)],                          # profil "santé"
    "user_C": [("art_06", 5), ("art_07", 4), ("art_12", 3)],           # profil "économie"
    "user_D": [("art_01", 3), ("art_08", 5), ("art_09", 4)],           # profil mixte agriculture/éducation
}

# NB : la séparation "held-out" pour l'évaluation (étape 6) se fait de
# façon dynamique par leave-one-out directement dans evaluation.py,
# plutôt que d'être dupliquée ici (pour éviter toute incohérence entre
# USER_HISTORY et les articles "retirés pour test").
