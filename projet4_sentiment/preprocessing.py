# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Prétraitement unifié
------------------------------------
Pipeline standardisé et PARTAGÉ par les 3 approches (lexicale,
statistique, neuronale), comme demandé dans le cahier des charges.

Attention pédagogique : l'approche neuronale (CamemBERT/FlauBERT)
utilise en réalité SON PROPRE tokenizer (sous-mots / BPE) et n'a PAS
besoin qu'on supprime les stopwords au préalable (le modèle pré-entraîné
a appris sur du texte brut). On garde donc ici :
    - clean_text()      -> nettoyage léger commun (espaces, casse) que
                            les 3 approches utilisent avant tokenisation
    - tokenize_lexical() -> tokenisation + stopwords + lemmatisation,
                            pour les approches lexicale ET statistique
                            (qui reposent sur des sacs de mots / TF-IDF)
    - pour l'approche neuronale, on passe `clean_text(text)` directement
      au tokenizer du modèle (voir neural_approach.py)
"""

import re

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
except Exception:
    _SPACY_AVAILABLE = False


def clean_text(text: str) -> str:
    """Nettoyage léger commun aux 3 approches : espaces, casse basique."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_lexical(text: str) -> list[str]:
    """Tokenisation + suppression stopwords + lemmatisation, pour les
    approches lexicale et statistique (PAS pour l'approche neuronale,
    qui utilise son propre tokenizer sur le texte brut)."""
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
