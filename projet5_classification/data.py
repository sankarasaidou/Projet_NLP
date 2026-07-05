# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Scraping ciblé
ÉTAPE 2 : Annotation manuelle
-----------------------------------
Le scraper (réutilisable depuis le projet 2 / lefaso.net, AIB) collecte
des articles ; on les étiquette manuellement par catégorie thématique.

Comme pour les projets précédents, on fournit un corpus de démo
hors-ligne déjà annoté, pour ne pas dépendre du réseau.
"""

import sys
from pathlib import Path

# Réutilise le scraper du projet 2 si présent à côté (sinon fallback corpus)
sys.path.append(str(Path(__file__).parent))

CATEGORIES = ["politique", "économie", "sport", "culture"]

# Corpus annoté : (texte, catégorie)
LABELED_CORPUS = [
    # politique
    ("Le président a présenté son nouveau gouvernement lors d'une allocution "
     "télévisée, annonçant un remaniement ministériel important.", "politique"),
    ("L'Assemblée nationale a adopté ce jeudi une loi sur la décentralisation "
     "administrative après de longs débats.", "politique"),
    ("Le parti au pouvoir a tenu son congrès annuel pour définir sa nouvelle "
     "feuille de route politique.", "politique"),
    ("Les élections municipales approchent, les candidats multiplient les "
     "meetings dans les principales villes du pays.", "politique"),
    ("Le ministre des affaires étrangères a rencontré son homologue lors "
     "d'un sommet diplomatique régional.", "politique"),
    ("L'opposition a dénoncé un projet de loi jugé liberticide devant le "
     "parlement.", "politique"),

    # économie
    ("La Banque centrale a annoncé une baisse de son taux directeur pour "
     "soutenir l'activité économique.", "économie"),
    ("Le prix du coton a fortement augmenté sur les marchés internationaux "
     "ce trimestre.", "économie"),
    ("Le gouvernement a présenté son budget annuel, marqué par une hausse "
     "des dépenses d'infrastructure.", "économie"),
    ("Les exportations agricoles ont connu une croissance de 12% sur "
     "l'année écoulée selon les douanes.", "économie"),
    ("Une nouvelle usine de transformation de céréales va créer plusieurs "
     "centaines d'emplois dans la région.", "économie"),
    ("L'inflation a atteint son plus haut niveau depuis cinq ans, "
     "inquiétant les ménages et les commerçants.", "économie"),

    # sport
    ("L'équipe nationale de football s'est qualifiée pour la phase finale "
     "de la compétition continentale après une victoire éclatante.", "sport"),
    ("Le champion national d'athlétisme a battu le record du 400 mètres "
     "lors des championnats régionaux.", "sport"),
    ("Le club local a remporté le derby face à son rival historique dans "
     "un stade comble.", "sport"),
    ("La fédération de basketball a annoncé le calendrier de la nouvelle "
     "saison du championnat national.", "sport"),
    ("Les jeunes talents du judo se sont illustrés lors du tournoi "
     "international organisé dans la capitale.", "sport"),
    ("Le sélectionneur a dévoilé la liste des joueurs convoqués pour le "
     "prochain match des éliminatoires.", "sport"),

    # culture
    ("Le festival international de musique a réuni des artistes venus de "
     "toute la sous-région pour une édition mémorable.", "culture"),
    ("Un nouveau musée dédié à l'art contemporain a ouvert ses portes dans "
     "le centre-ville.", "culture"),
    ("L'écrivain a présenté son dernier roman lors d'une séance de "
     "dédicace très suivie.", "culture"),
    ("La cérémonie de remise des prix du cinéma national a récompensé les "
     "meilleurs talents de l'année.", "culture"),
    ("Une exposition retraçant l'histoire des masques traditionnels attire "
     "de nombreux visiteurs.", "culture"),
    ("Le théâtre national a lancé sa nouvelle saison avec une pièce "
     "acclamée par la critique.", "culture"),
]

TEST_CORPUS = [
    ("Le chef de l'État a rencontré les leaders religieux pour discuter de "
     "la cohésion sociale.", "politique"),
    ("La monnaie nationale s'est légèrement dépréciée face au dollar cette "
     "semaine.", "économie"),
    ("Le sprinteur local a décroché la médaille d'or lors du meeting "
     "international d'athlétisme.", "sport"),
    ("La troupe de danse traditionnelle a représenté le pays lors d'un "
     "festival culturel à l'étranger.", "culture"),
]
