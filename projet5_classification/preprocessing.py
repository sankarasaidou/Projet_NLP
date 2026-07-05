# -*- coding: utf-8 -*-
"""Prétraitement (mêmes principes que les autres projets : fallback si
spaCy n'est pas disponible)."""

import re

_FALLBACK_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "au",
    "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "ce",
    "cette", "ces", "son", "sa", "ses", "leur", "leurs", "il", "elle",
    "ils", "elles", "on", "nous", "vous", "je", "tu", "qui", "que",
    "est", "sont", "a", "ont", "été", "être", "avoir", "se", "ne", "pas",
    "d", "l", "s", "n", "c", "j", "y", "plus", "lors", "après",
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


def preprocess(text: str) -> str:
    """Retourne le texte nettoyé sous forme de chaîne (tokens joints par
    des espaces), prêt pour un vectoriseur."""
    if _SPACY_AVAILABLE:
        doc = _nlp(text)
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_stop and not tok.is_punct and not tok.is_space and tok.is_alpha
        ]
    else:
        tokens = [
            tok for tok in _TOKEN_RE.findall(text.lower())
            if tok not in _FALLBACK_STOPWORDS and len(tok) > 1
        ]
    return " ".join(tokens)
