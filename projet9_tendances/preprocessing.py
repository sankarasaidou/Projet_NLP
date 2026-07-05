# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : Nettoyage spécialisé
------------------------------------
Les commentaires de réseaux sociaux ont des spécificités que le
prétraitement "classique" (projets 2-5) ne gère pas :

    - emojis        : à extraire séparément (ils portent une forte
                       charge de sentiment : 😢 😡 🙏 ne sont pas du bruit)
    - hashtags       : à extraire comme mots-clés thématiques à part
                       entière (utiles pour la détection de tendances)
    - mentions (@x)  : à retirer du texte analysé (bruit pour le NLP)
                       mais à garder en métadonnée si besoin (qui est
                       interpellé)
    - langage informel : normalisation de quelques abréviations
                       fréquentes en français des réseaux ("c" -> "c'est",
                       "mdr" -> reste tel quel car porteur de sens, etc.)
"""

import re

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # emojis divers, symboles, pictogrammes
    "\U00002600-\U000027BF"  # symboles divers / dingbats
    "]+"
)
HASHTAG_PATTERN = re.compile(r"#(\w+)")
MENTION_PATTERN = re.compile(r"@(\w+)")

# Normalisation de quelques abréviations informelles courantes (à
# étendre selon le corpus réel observé).
INFORMAL_NORMALIZATION = {
    r"\bc\b": "c'est",
    r"\bkoi\b": "quoi",
    r"\bdjà\b": "déjà",
    r"\bpk\b": "pourquoi",
    r"\bstp\b": "s'il te plaît",
    r"\bsvp\b": "s'il vous plaît",
}


def extract_emojis(text: str) -> list[str]:
    return EMOJI_PATTERN.findall(text)


def extract_hashtags(text: str) -> list[str]:
    return HASHTAG_PATTERN.findall(text)


def extract_mentions(text: str) -> list[str]:
    return MENTION_PATTERN.findall(text)


def clean_comment(text: str) -> dict:
    """Retourne un dict avec le texte nettoyé (pour analyse NLP) et les
    métadonnées extraites séparément (emojis, hashtags, mentions)."""
    emojis = extract_emojis(text)
    hashtags = extract_hashtags(text)
    mentions = extract_mentions(text)

    clean_text = EMOJI_PATTERN.sub(" ", text)
    clean_text = HASHTAG_PATTERN.sub(" ", clean_text)
    clean_text = MENTION_PATTERN.sub(" ", clean_text)

    clean_text_lower = clean_text.lower()
    for pattern, replacement in INFORMAL_NORMALIZATION.items():
        clean_text_lower = re.sub(pattern, replacement, clean_text_lower)

    clean_text_lower = re.sub(r"[^\w\sàâäéèêëïîôöùûüç']", " ", clean_text_lower)
    clean_text_lower = re.sub(r"\s+", " ", clean_text_lower).strip()

    return {
        "clean_text": clean_text_lower,
        "emojis": emojis,
        "hashtags": hashtags,
        "mentions": mentions,
    }


if __name__ == "__main__":
    sample = "C koi ce plan encore ?? on a djà entendu ça mille fois #déception 😒"
    result = clean_comment(sample)
    print(result)
