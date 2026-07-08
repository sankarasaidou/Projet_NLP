# -*- coding: utf-8 -*-
"""Prétraitement du texte.

Toujours séparer :
    - clean_text()      : nettoyage LÉGER (espaces), commun aux 3 approches,
                           préserve le texte pour l'approche neuronale qui a
                           besoin du texte quasi brut.
    - tokenize_lexical() : nettoyage COMPLET (stopwords, lemmatisation) pour
                           les approches lexicale et statistique (sacs de mots).

Ajout production : validate_text() rejette explicitement les entrées
invalides (vide, non textuelle, trop longue) avec un message clair,
plutôt que de laisser une exception cryptique remonter plus loin dans
le pipeline.
"""

import re

from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)

MAX_TEXT_LENGTH = 5000  # caractères ; protège contre un DoS applicatif accidentel

_FALLBACK_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "au",
    "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "ce",
    "cette", "ces", "son", "sa", "ses", "leur", "leurs", "il", "elle",
    "ils", "elles", "on", "nous", "vous", "je", "tu", "qui", "que",
    "est", "sont", "a", "ont", "été", "être", "avoir", "se", "ne",
    "d", "l", "s", "n", "c", "j", "y", "très", "plus", "pas",
}
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")

_nlp = None
_SPACY_AVAILABLE = False
try:
    import spacy
    _nlp = spacy.load("fr_core_news_sm")
    _SPACY_AVAILABLE = True
    logger.info("spaCy (fr_core_news_sm) chargé : lemmatisation complète activée.")
except Exception as exc:
    logger.warning(
        "spaCy non disponible (%s) : utilisation du fallback de tokenisation "
        "simple (sans lemmatisation).", exc,
    )


class InvalidTextError(ValueError):
    """Levée quand le texte fourni par l'utilisateur n'est pas exploitable."""


def validate_text(text) -> str:
    """Valide l'entrée utilisateur et lève une erreur explicite sinon.

    Ne jamais faire confiance à une entrée utilisateur brute en
    production : ce garde-fou évite des erreurs cryptiques plus loin
    dans le pipeline (ex. AttributeError sur None, TfidfVectorizer sur
    une liste vide, etc.)."""
    if text is None:
        raise InvalidTextError("Le texte fourni est vide (None).")
    if not isinstance(text, str):
        raise InvalidTextError(f"Le texte doit être une chaîne, reçu : {type(text).__name__}.")
    stripped = text.strip()
    if not stripped:
        raise InvalidTextError("Le texte fourni est vide ou ne contient que des espaces.")
    if len(stripped) > MAX_TEXT_LENGTH:
        raise InvalidTextError(
            f"Le texte est trop long ({len(stripped)} caractères, max {MAX_TEXT_LENGTH})."
        )
    return stripped


def clean_text(text: str) -> str:
    """Nettoyage léger commun aux 3 approches : espaces, casse basique.
    Ne modifie pas le contenu sémantique du texte."""
    text = validate_text(text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_lexical(text: str) -> list[str]:
    """Tokenisation + suppression stopwords + lemmatisation, pour les
    approches lexicale et statistique (PAS pour l'approche neuronale)."""
    text = clean_text(text)
    if _SPACY_AVAILABLE:
        doc = _nlp(text)
        return [
            tok.lemma_.lower()
            for tok in doc
            if not tok.is_stop and not tok.is_punct and not tok.is_space and tok.is_alpha
        ]
    tokens = _TOKEN_RE.findall(text.lower())
    return [tok for tok in tokens if tok not in _FALLBACK_STOPWORDS and len(tok) > 1]


def is_spacy_available() -> bool:
    return _SPACY_AVAILABLE
