# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Constitution du corpus
-------------------------------------
Le sujet demande d'utiliser des ensembles de données publics / articles
de presse en ligne (ex. l'AIB) pour créer un corpus thématique.

Comme pour le projet 2, on fournit un corpus de démo hors-ligne assez
diversifié (plusieurs thématiques mélangées) pour que la modélisation de
sujets ait quelque chose d'intéressant à découvrir : si tous les
documents parlent du même sujet, LDA/NMF/BERTopic n'ont rien à séparer.
En conditions réelles, on remplacerait cette liste par les résultats
du scraper (voir projet 2 : scraper.py, réutilisable tel quel sur l'AIB
ou lefaso.net).
"""

CORPUS = [
    # --- Thème : agriculture ---
    "La campagne agricole s'annonce prometteuse grâce à une pluviométrie "
    "favorable. Les producteurs de maïs et de sorgho anticipent une hausse "
    "des rendements cette saison.",
    "Le prix des engrais pèse sur les petits agriculteurs qui peinent à "
    "financer leurs intrants pour la prochaine saison de culture.",
    "La foire agricole régionale a réuni des centaines d'agriculteurs venus "
    "présenter leurs techniques de culture et d'irrigation.",
    "Les producteurs de coton demandent un meilleur accompagnement pour "
    "améliorer la productivité de leurs exploitations agricoles.",

    # --- Thème : santé ---
    "Le ministère de la santé lance une campagne de vaccination infantile "
    "contre la rougeole dans plusieurs régions du pays.",
    "Un nouveau centre de santé communautaire a ouvert ses portes pour les "
    "femmes enceintes et les enfants en bas âge.",
    "Les autorités sanitaires appellent à la vigilance face à la "
    "recrudescence du paludisme pendant la saison des pluies.",
    "Une pénurie de médicaments essentiels est signalée dans plusieurs "
    "centres de santé du pays, inquiétant le personnel médical.",

    # --- Thème : économie ---
    "Un nouveau plan de relance économique a été présenté pour soutenir "
    "les petites entreprises et stimuler l'investissement.",
    "Les prévisions de croissance économique ont été révisées à la hausse, "
    "portées par le secteur des services et les exportations.",
    "L'inflation continue d'éroder le pouvoir d'achat des ménages, "
    "poussant le gouvernement à envisager de nouvelles mesures.",
    "La Banque centrale a annoncé une révision de ses taux directeurs pour "
    "soutenir l'activité économique nationale.",

    # --- Thème : éducation ---
    "La rentrée scolaire approche avec son lot de défis : manque "
    "d'enseignants, classes surchargées et infrastructures vétustes.",
    "Un programme de bourses d'études a été lancé pour encourager les "
    "filières scientifiques et techniques dans l'enseignement supérieur.",
    "Les syndicats d'enseignants réclament de meilleures conditions de "
    "travail et une revalorisation salariale.",
    "Une nouvelle réforme du système éducatif prévoit l'introduction du "
    "numérique dans les salles de classe dès le primaire.",

    # --- Thème : sécurité ---
    "Les autorités ont annoncé un renforcement des dispositifs de sécurité "
    "dans plusieurs localités face aux menaces persistantes.",
    "Des patrouilles conjointes ont été mises en place le long de la "
    "frontière pour lutter contre les trafics transfrontaliers.",
    "Un dispositif de sécurité renforcé a été déployé à l'approche des "
    "festivités de fin d'année dans la capitale.",
    "Les forces de sécurité ont mené une opération de ratissage suite à "
    "plusieurs incidents signalés dans la zone.",
]
