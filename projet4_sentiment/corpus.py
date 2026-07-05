# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Collecte de données
----------------------------------
Corpus francophone étiqueté (positif / négatif / neutre), simulant des
avis clients / commentaires. En conditions réelles on scraperait des
avis (Google Reviews, Trustpilot, réseaux sociaux) — voir projet 2 pour
un exemple de scraper réutilisable, à adapter à la source choisie.

On sépare TRAIN (entraînement des approches statistique/neuronale) et
TEST (jamais vu, pour l'évaluation comparative de l'étape 6).
"""

TRAIN_DATA = [
    ("Ce produit est excellent, je le recommande vivement !", "positif"),
    ("Un service client exceptionnel, très à l'écoute.", "positif"),
    ("Je suis très satisfait de mon achat, livraison rapide.", "positif"),
    ("Super qualité, dépasse largement mes attentes.", "positif"),
    ("Une expérience géniale du début à la fin.", "positif"),
    ("Le personnel était vraiment aimable et professionnel.", "positif"),
    ("Rapport qualité-prix imbattable, je rachèterai sans hésiter.", "positif"),
    ("Produit conforme à la description, rien à redire.", "positif"),

    ("Produit décevant, ne fonctionne pas comme annoncé.", "négatif"),
    ("Service client injoignable, très mauvaise expérience.", "négatif"),
    ("Je suis extrêmement déçu de la qualité reçue.", "négatif"),
    ("Livraison en retard et produit endommagé à l'arrivée.", "négatif"),
    ("Une perte de temps et d'argent, à éviter absolument.", "négatif"),
    ("Le personnel était désagréable et peu compétent.", "négatif"),
    ("Rapport qualité-prix catastrophique pour ce produit.", "négatif"),
    ("Produit non conforme à la description, très mécontent.", "négatif"),

    ("Le colis est arrivé aujourd'hui à l'adresse indiquée.", "neutre"),
    ("Le produit est disponible en trois coloris différents.", "neutre"),
    ("La commande a été passée le 3 du mois dernier.", "neutre"),
    ("Le magasin ouvre à 9h et ferme à 19h en semaine.", "neutre"),
    ("Le manuel d'utilisation est fourni avec l'appareil.", "neutre"),
    ("La commande contient deux articles différents.", "neutre"),
    ("Le produit pèse environ deux kilogrammes.", "neutre"),
    ("Le paiement peut se faire par carte ou en espèces.", "neutre"),
]

TEST_DATA = [
    ("Vraiment ravi de cet achat, je recommande à 100%.", "positif"),
    ("Un accueil chaleureux et un service impeccable.", "positif"),
    ("Ce n'est pas du tout ce à quoi je m'attendais, très déçu.", "négatif"),
    ("Article cassé à la réception, aucune réponse du vendeur.", "négatif"),
    ("Le produit est disponible en ligne et en magasin.", "neutre"),
    ("La livraison est prévue sous 48 heures ouvrées.", "neutre"),
]
