# -*- coding: utf-8 -*-
"""Approche lexicale : dictionnaire de polarité + règles (négation,
intensificateurs, emojis). Le lexique est chargé depuis data/lexicon_fr.csv
(voir data_loader.py), pas codé en dur -> remplaçable sans toucher au code.
"""

import re
from functools import lru_cache

from sentiment_analysis.config import NEUTRAL_THRESHOLD
from sentiment_analysis.data_loader import load_lexicon
from sentiment_analysis.preprocessing import clean_text
from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)

NEGATION_WORDS = {"pas", "ne", "jamais", "aucun", "aucune", "sans", "ni"}
INTENSIFIERS = {"très": 1.5, "vraiment": 1.5, "extrêmement": 2.0, "totalement": 1.8, "franchement": 1.3}

POSITIVE_EMOJIS = {"👏", "🙏", "❤️", "🌾", "🙌", "😊", "👍", "😍", "🎉"}
NEGATIVE_EMOJIS = {"😢", "😡", "😒", "🙄", "😞", "👎", "😠"}
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


@lru_cache(maxsize=1)
def _get_lexicon() -> dict[str, float]:
    """Chargé une seule fois par process (le fichier ne change pas en
    cours d'exécution), mis en cache pour éviter de relire le disque à
    chaque appel."""
    return load_lexicon()


class LexicalSentimentAnalyzer:
    """Approche lexicale, 100% interprétable : chaque décision peut être
    justifiée mot par mot (utile pour de la modération ou de l'audit)."""

    def __init__(self, lexicon: dict[str, float] | None = None, neutral_threshold: float = NEUTRAL_THRESHOLD):
        self.lexicon = lexicon if lexicon is not None else _get_lexicon()
        self.neutral_threshold = neutral_threshold

    def analyze(self, text: str) -> dict:
        text_clean = clean_text(text)
        emojis = _EMOJI_RE.findall(text_clean)
        words = _EMOJI_RE.sub(" ", text_clean).lower().split()

        score = 0.0
        details = []
        for i, word in enumerate(words):
            word_stripped = word.strip(".,!?;:\"'()")
            if word_stripped not in self.lexicon:
                continue

            base_polarity = self.lexicon[word_stripped]
            multiplier = 1.0
            negated = False

            window = words[max(0, i - 3):i]
            for w in window:
                w_stripped = w.strip(".,!?;:\"'()")
                if w_stripped in NEGATION_WORDS:
                    negated = True
                if w_stripped in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[w_stripped]

            polarity = base_polarity * multiplier * (-1 if negated else 1)
            score += polarity
            details.append({"word": word_stripped, "polarity": round(polarity, 2), "negated": negated})

        for emoji in emojis:
            if emoji in POSITIVE_EMOJIS:
                score += 1.5
                details.append({"word": emoji, "polarity": 1.5, "negated": False})
            elif emoji in NEGATIVE_EMOJIS:
                score -= 1.5
                details.append({"word": emoji, "polarity": -1.5, "negated": False})

        if score > self.neutral_threshold:
            label = "positif"
        elif score < -self.neutral_threshold:
            label = "négatif"
        else:
            label = "neutre"

        return {"label": label, "score": round(score, 2), "details": details}
