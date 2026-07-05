# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Prétraitement avancé
------------------------------------
- normalize() : minuscule, suppression ponctuation/accents-insensible
  aux variantes typographiques (utile car un plagiat reformulé peut
  changer la ponctuation sans changer le fond).
- ngrams() : n-grammes de mots consécutifs (utile pour Jaccard/Cosinus
  sur des séquences plutôt que des mots isolés -> détecte mieux les
  passages copiés mot-à-mot).
- skip_grams() : paires de mots à distance k (utile pour capter des
  similarités même quand quelques mots ont été insérés/supprimés entre
  eux, ce qui est un signe classique de paraphrase légère).
"""

import re
import unicodedata


def normalize(text: str) -> str:
    """Minuscule + suppression accents + ponctuation -> espace unique."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))  # enlève les accents
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def ngrams(tokens: list[str], n: int = 3) -> list[tuple]:
    """N-grammes de mots consécutifs."""
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def skip_grams(tokens: list[str], k: int = 2) -> list[tuple]:
    """Paires de mots (w_i, w_{i+1+k}) espacés de k mots -> capte des
    similarités robustes à de petites insertions/suppressions."""
    pairs = []
    for i in range(len(tokens)):
        j = i + 1 + k
        if j < len(tokens):
            pairs.append((tokens[i], tokens[j]))
    return pairs


if __name__ == "__main__":
    sample = "L'IA transforme la société, notamment à travers l'automatisation."
    tokens = tokenize(sample)
    print("Tokens :", tokens)
    print("Trigrammes :", ngrams(tokens, 3)[:3])
    print("Skip-grammes (k=2) :", skip_grams(tokens, 2)[:3])
