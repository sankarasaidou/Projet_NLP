# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Prétraitement des données
--------------------------------------
Nettoyage standard : tokenisation, suppression des stopwords, lemmatisation.

On essaie d'utiliser spaCy (fr_core_news_sm) pour une vraie lemmatisation
linguistique. S'il n'est pas installé/téléchargé, on retombe sur un
nettoyage plus simple (minuscule + regex + liste de stopwords français
maison) : moins précis, mais qui ne bloque jamais le reste du pipeline.
Ce genre de "graceful degradation" est une bonne pratique en prod.
"""

import re

_FALLBACK_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "au",
    "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "ce",
    "cette", "ces", "son", "sa", "ses", "leur", "leurs", "il", "elle",
    "ils", "elles", "on", "nous", "vous", "je", "tu", "qui", "que", "quoi",
    "dont", "où", "est", "sont", "a", "ont", "été", "être", "avoir", "se",
    "ne", "pas", "plus", "ou", "mais", "donc", "car", "comme", "d", "l",
    "s", "n", "c", "j", "y",
}

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

_nlp = None
_SPACY_AVAILABLE = False
try:
    import spacy
    _nlp = spacy.load("fr_core_news_sm")
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False


def preprocess(text: str) -> list[str]:
    """Retourne la liste des tokens nettoyés et lemmatisés d'un texte."""
    if _SPACY_AVAILABLE:
        doc = _nlp(text)
        return [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_stop and not tok.is_punct and not tok.is_space and tok.is_alpha
        ]
    # --- fallback sans spaCy ---
    tokens = _TOKEN_RE.findall(text.lower())
    return [tok for tok in tokens if tok not in _FALLBACK_STOPWORDS and len(tok) > 1]


def preprocess_corpus(documents: list[dict]) -> list[str]:
    """Applique preprocess() à une liste de documents {"text": ...} et
    retourne une liste de chaînes nettoyées (une par document), prêtes à
    être passées à un vectoriseur (CountVectorizer, TfidfVectorizer...)."""
    return [" ".join(preprocess(doc["text"])) for doc in documents]


if __name__ == "__main__":
    sample = "Les agriculteurs ont amélioré leurs rendements grâce à l'irrigation."
    print("spaCy disponible :", _SPACY_AVAILABLE)
    print("Tokens :", preprocess(sample))
