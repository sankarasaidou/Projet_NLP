# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Prétraitement des données
--------------------------------------
Point pédagogique clé : pour la NER, le texte donné au modèle DOIT rester
le texte original (ou presque). Si on supprime les stopwords ou qu'on
lemmatise avant de repérer les entités, les positions de caractères
(start/end) ne correspondent plus au texte d'origine, et l'EntityRuler /
le NER entraîné ne peuvent plus rien aligner correctement.

On propose donc DEUX fonctions bien séparées :

1. clean_for_ner(text)
   -> Nettoyage minimal (espaces multiples, caractères de contrôle) qui
      NE CHANGE PAS les positions de mots significatifs. C'est ce texte
      qu'on donne au pipeline spaCy (EntityRuler + NER + Regex).

2. clean_for_analysis(text, nlp)
   -> Nettoyage complet (suppression stopwords, ponctuation, lemmatisation)
      utile pour des tâches EN AVAL de la NER : nuages de mots, fréquences
      de termes, features pour un classifieur, etc. Ce texte ne doit
      JAMAIS être utilisé pour re-détecter des entités.
"""

import re


def clean_for_ner(text: str) -> str:
    """Nettoyage minimal et sûr pour la NER : ne touche pas au contenu,
    seulement aux espaces superflus."""
    text = text.replace("\u00a0", " ")  # espaces insécables
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_analysis(text: str, nlp) -> list[str]:
    """Nettoyage complet pour l'analyse lexicale : suppression des
    stopwords, de la ponctuation, et lemmatisation.

    Nécessite un objet `nlp` spaCy déjà chargé (avec un lemmatizer),
    par ex. `nlp = spacy.load("fr_core_news_sm")`.
    """
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return tokens


if __name__ == "__main__":
    sample = "Le   Machine Learning,  c'est fascinant !!"
    print("Avant :", repr(sample))
    print("clean_for_ner :", repr(clean_for_ner(sample)))
    # clean_for_analysis nécessite un modèle spaCy chargé, non exécuté ici.
