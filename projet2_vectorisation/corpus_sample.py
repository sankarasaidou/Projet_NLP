# -*- coding: utf-8 -*-
"""
Corpus de secours (fallback hors-ligne)
-------------------------------------------
Utilisé quand scraper.py ne peut pas accéder au réseau (démo, tests,
sandbox sans accès internet). Chaque document est un petit article
fictif mais réaliste, couvrant les thématiques des mots-clés seed
(économie, agriculture, éducation, santé, sécurité) pour pouvoir tester
la similarité par mots-clés de bout en bout.
"""

SAMPLE_CORPUS = [
    {"title": "Hausse de la production céréalière", "url": "local://1",
     "text": "La campagne agricole s'annonce prometteuse cette année grâce "
             "à une pluviométrie favorable. Les producteurs de maïs et de "
             "sorgho anticipent une hausse significative des rendements."},
    {"title": "Le gouvernement annonce un plan de soutien à l'économie", "url": "local://2",
     "text": "Un nouveau plan de relance économique a été présenté pour "
             "soutenir les petites entreprises et stimuler l'investissement "
             "dans les secteurs clés de l'économie nationale."},
    {"title": "Campagne de vaccination dans les régions rurales", "url": "local://3",
     "text": "Le ministère de la santé lance une vaste campagne de "
             "vaccination infantile dans plusieurs régions du pays pour "
             "lutter contre la rougeole et la poliomyélite."},
    {"title": "Rentrée scolaire : les défis de l'éducation", "url": "local://4",
     "text": "La rentrée scolaire approche avec son lot de défis pour le "
             "système éducatif : manque d'enseignants, classes surchargées "
             "et besoin de nouvelles infrastructures scolaires."},
    {"title": "Renforcement des mesures de sécurité", "url": "local://5",
     "text": "Les autorités ont annoncé un renforcement des dispositifs de "
             "sécurité dans plusieurs localités pour protéger les "
             "populations civiles face aux menaces persistantes."},
    {"title": "Foire agricole régionale", "url": "local://6",
     "text": "La foire agricole régionale a réuni des centaines "
             "d'agriculteurs venus présenter leurs techniques de culture "
             "et échanger sur les innovations en matière d'irrigation."},
    {"title": "Croissance économique révisée à la hausse", "url": "local://7",
     "text": "Les prévisions de croissance économique ont été révisées à "
             "la hausse par les institutions financières, portées par le "
             "secteur des services et les exportations agricoles."},
    {"title": "Nouveau centre de santé communautaire", "url": "local://8",
     "text": "Un nouveau centre de santé communautaire a ouvert ses portes, "
             "offrant des soins de proximité pour les femmes enceintes et "
             "les enfants en bas âge dans une zone jusque-là mal desservie."},
    {"title": "Programme de bourses pour les étudiants", "url": "local://9",
     "text": "Un programme de bourses d'études a été lancé pour "
             "encourager la poursuite d'études supérieures dans les "
             "filières scientifiques et techniques, en lien avec l'éducation."},
    {"title": "Patrouilles conjointes pour la sécurité frontalière", "url": "local://10",
     "text": "Des patrouilles conjointes ont été mises en place le long de "
             "la frontière pour renforcer la sécurité et lutter contre les "
             "trafics transfrontaliers."},
    {"title": "Le prix des engrais impacte les agriculteurs", "url": "local://11",
     "text": "La hausse du prix des engrais pèse sur les petits "
             "agriculteurs, qui peinent à financer leurs intrants pour la "
             "prochaine saison agricole."},
    {"title": "Inflation et pouvoir d'achat des ménages", "url": "local://12",
     "text": "L'inflation continue d'éroder le pouvoir d'achat des ménages, "
             "poussant le gouvernement à envisager de nouvelles mesures "
             "économiques de soutien."},
]
