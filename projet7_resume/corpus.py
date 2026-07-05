# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Corpus diversifié
--------------------------------
Trois types de documents, comme demandé : article de presse, document
académique, rapport technique. Chacun a un style et une structure
différents, ce qui permet de tester la robustesse des méthodes de
résumé (TextRank/TF-IDF/T5/mBART) sur des genres textuels variés.
"""

ARTICLE_PRESSE = """
Le gouvernement a annoncé ce mardi un plan national de soutien à
l'agriculture, doté d'un budget de plusieurs milliards de francs CFA.
Ce plan vise à moderniser les techniques de production, notamment par
la distribution de semences améliorées et d'engrais subventionnés aux
petits producteurs. Le ministre de l'agriculture a précisé que cette
initiative concernera prioritairement les régions les plus touchées
par les aléas climatiques des dernières années. Les organisations
paysannes ont accueilli favorablement l'annonce, tout en réclamant des
garanties sur la mise en œuvre concrète et rapide des mesures promises.
Plusieurs experts estiment que ce plan pourrait permettre une hausse
significative des rendements agricoles dès la prochaine campagne, à
condition que la distribution des intrants soit effective avant le
début de la saison des pluies. Le gouvernement a indiqué qu'un comité
de suivi serait mis en place pour évaluer l'impact du programme.
""".strip()

DOCUMENT_ACADEMIQUE = """
Cette étude examine l'impact de l'apprentissage par transfert sur la
performance des modèles de traitement du langage naturel appliqués aux
langues à faibles ressources. Nous montrons que le fine-tuning de
modèles pré-entraînés sur de grands corpus multilingues permet
d'obtenir des gains de performance significatifs, même lorsque le
corpus d'entraînement spécifique à la langue cible est de petite
taille. Nos expériences portent sur trois tâches : la classification
de texte, la reconnaissance d'entités nommées, et l'analyse de
sentiment. Les résultats indiquent une amélioration moyenne de 12
points de F1-score par rapport à un entraînement from scratch. Nous
discutons également des limites de cette approche, notamment le risque
de biais hérités du corpus d'entraînement multilingue, qui peuvent ne
pas être adaptés aux spécificités culturelles de la langue cible.
Enfin, nous proposons des pistes pour améliorer la qualité des
ressources disponibles pour les langues sous-représentées en NLP.
""".strip()

RAPPORT_TECHNIQUE = """
Ce rapport présente les résultats du déploiement pilote d'un système
de collecte de données par capteurs IoT dans le réseau d'adduction
d'eau potable de la ville. L'objectif principal était de détecter les
fuites et anomalies de pression en temps réel. Sur une période de test
de six mois, le système a permis d'identifier 14 fuites majeures,
réduisant les pertes en eau estimées de 18%. Les capteurs installés
transmettent des données toutes les cinq minutes via un réseau LoRaWAN,
avec une autonomie moyenne de batterie de huit mois. Le coût
d'installation par capteur s'élève à environ 45 000 francs CFA,
amorti en moins d'un an grâce aux économies réalisées sur les pertes
d'eau évitées. Le rapport recommande une extension du dispositif à
l'ensemble du réseau urbain d'ici deux ans, ainsi qu'une formation du
personnel technique à la maintenance des capteurs et à l'interprétation
des données collectées.
""".strip()

DOCUMENTS = {
    "Article de presse": ARTICLE_PRESSE,
    "Document académique": DOCUMENT_ACADEMIQUE,
    "Rapport technique": RAPPORT_TECHNIQUE,
}

# Résumés de référence écrits à la main, utilisés pour calculer ROUGE (étape 4)
REFERENCE_SUMMARIES = {
    "Article de presse": (
        "Le gouvernement lance un plan de soutien à l'agriculture avec "
        "semences améliorées et engrais subventionnés pour les petits "
        "producteurs, en priorité dans les régions touchées par le climat."
    ),
    "Document académique": (
        "L'apprentissage par transfert améliore significativement les "
        "performances NLP sur les langues à faibles ressources, avec un "
        "gain moyen de 12 points de F1-score, malgré un risque de biais "
        "culturels hérités des corpus multilingues."
    ),
    "Rapport technique": (
        "Le déploiement pilote de capteurs IoT a détecté 14 fuites "
        "majeures et réduit les pertes en eau de 18% en six mois, pour un "
        "coût amorti en moins d'un an ; une extension est recommandée."
    ),
}
