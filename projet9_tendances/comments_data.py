# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Collecte de données
----------------------------------
Commentaires horodatés simulant des réactions d'utilisateurs sous des
articles de lefaso.net, avec emojis, hashtags, mentions et langage
informel (comme dans la vraie vie des commentaires en ligne). Le
scraping réel réutiliserait `scraper.py` du projet 2, adapté pour cibler
la section commentaires des articles (souvent chargée en JS -> nécessite
Selenium/Playwright en conditions réelles, non couvert ici par souci de
simplicité).

Chaque commentaire a : texte, date (pour la détection de tendances
temporelles), et pseudo (anonymisé).
"""

from datetime import date

COMMENTS = [
    {"date": date(2026, 6, 1), "author": "user_001",
     "text": "Excellente nouvelle pour l'agriculture 👏👏 #Burkina #Agriculture merci au gouvernement !!"},
    {"date": date(2026, 6, 1), "author": "user_002",
     "text": "@ministere on attend des actes concrets, pas que des annonces 😒"},
    {"date": date(2026, 6, 2), "author": "user_003",
     "text": "C koi ce plan encore ?? on a djà entendu ça mille fois #déception"},
    {"date": date(2026, 6, 2), "author": "user_004",
     "text": "Le prix des engrais reste trop élevé pour nous les petits producteurs 😢"},
    {"date": date(2026, 6, 3), "author": "user_005",
     "text": "Franchement bravo, ça fait plaisir de voir du concret enfin 🙏 #Espoir"},
    {"date": date(2026, 6, 3), "author": "user_006",
     "text": "La date de mise en œuvre est prévue pour le mois prochain selon l'article."},
    {"date": date(2026, 6, 4), "author": "user_007",
     "text": "Encore des promesses en l'air, on verra bien mdr 🙄 #Politique"},
    {"date": date(2026, 6, 4), "author": "user_008",
     "text": "Le budget alloué est mentionné dans le communiqué officiel du ministère."},
    {"date": date(2026, 6, 5), "author": "user_009",
     "text": "Trop content pour les agriculteurs de ma région 🌾❤️ #Agriculture #Espoir"},
    {"date": date(2026, 6, 5), "author": "user_010",
     "text": "@lefaso vous pouvez vérifier ces chiffres svp, ça semble énorme"},
    {"date": date(2026, 6, 6), "author": "user_011",
     "text": "C'est une bonne chose mais il faut suivre l'exécution de près #Vigilance"},
    {"date": date(2026, 6, 6), "author": "user_012",
     "text": "Ces mesures concernent les régions les plus touchées par la sécheresse."},
    {"date": date(2026, 6, 7), "author": "user_013",
     "text": "Ras le bol de ces annonces sans lendemain 😡😡 #Colère #Agriculture"},
    {"date": date(2026, 6, 7), "author": "user_014",
     "text": "Super initiative, ça va aider beaucoup de familles 🙌 #Agriculture"},
    {"date": date(2026, 6, 8), "author": "user_015",
     "text": "Le comité de suivi sera mis en place dans les prochaines semaines."},
]
