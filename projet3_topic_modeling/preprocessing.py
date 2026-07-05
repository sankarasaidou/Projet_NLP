# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Prétraitement des données
--------------------------------------
Suppression des stopwords, de la ponctuation, lemmatisation — avec
spaCy si disponible, sinon fallback simple (voir projet 2 pour la
même logique détaillée). On garde ici une version autonome pour que
ce projet soit indépendant.
"""

import re

_FALLBACK_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "au",
    "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "ce",
    "cette", "ces", "son", "sa", "ses", "leur", "leurs", "il", "elle",
    "ils", "elles", "on", "nous", "vous", "je", "tu", "qui", "que", "quoi",
    "dont", "où", "est", "sont", "a", "ont", "été", "être", "avoir", "se",
    "ne", "pas", "plus", "ou", "mais", "donc", "car", "comme", "d", "l",
    "s", "n", "c", "j", "y", "afin", "leur", "cette",
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
    if _SPACY_AVAILABLE:
        doc = _nlp(text)
        return [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_stop and not tok.is_punct and not tok.is_space and tok.is_alpha
        ]
    tokens = _TOKEN_RE.findall(text.lower())
    return [tok for tok in tokens if tok not in _FALLBACK_STOPWORDS and len(tok) > 2]


def preprocess_corpus(documents: list[str]) -> list[str]:
    """Retourne une liste de chaînes nettoyées, une par document."""
    return [" ".join(preprocess(doc)) for doc in documents]
