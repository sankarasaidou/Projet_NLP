# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Approche lexicale
--------------------------------
On utilise un petit dictionnaire de polarité français (mots positifs /
négatifs, avec un poids) et des règles simples :

    - Négation : "pas", "jamais", "aucun" avant un mot inverse sa polarité
      dans une fenêtre de 3 mots (ex. "pas satisfait" -> polarité inversée).
    - Intensificateurs : "très", "vraiment", "extrêmement" multiplient le
      poids du mot suivant.

Score final = somme des polarités pondérées. On classe en "positif" si
score > seuil, "négatif" si score < -seuil, sinon "neutre".

C'est une approche 100% interprétable (chaque décision s'explique), mais
qui ne comprend pas le contexte fin ni les tournures complexes (ironie,
double négation...).
"""

from preprocessing import clean_text

POLARITY_LEXICON = {
    # positifs
    "excellent": 2.0, "recommande": 1.5, "exceptionnel": 2.0, "satisfait": 1.5,
    "super": 1.5, "génial": 2.0, "aimable": 1.0, "professionnel": 1.0,
    "imbattable": 1.5, "conforme": 0.5, "ravi": 2.0, "chaleureux": 1.0,
    "impeccable": 2.0, "rapide": 0.5, "qualité": 0.5,
    # négatifs
    "décevant": -2.0, "déçu": -2.0, "injoignable": -1.5, "mauvaise": -1.5,
    "endommagé": -1.5, "perte": -1.0, "désagréable": -1.5, "catastrophique": -2.5,
    "cassé": -1.5, "mécontent": -1.5, "retard": -1.0,
}

NEGATION_WORDS = {"pas", "ne", "jamais", "aucun", "aucune", "sans"}
INTENSIFIERS = {"très": 1.5, "vraiment": 1.5, "extrêmement": 2.0, "totalement": 1.8}

NEUTRAL_THRESHOLD = 0.5


def analyze_lexical(text: str) -> dict:
    """Retourne {"label": ..., "score": ..., "details": [...]}"""
    text_clean = clean_text(text).lower()
    words = text_clean.split()

    score = 0.0
    details = []
    for i, word in enumerate(words):
        word_stripped = word.strip(".,!?;:\"'")
        if word_stripped not in POLARITY_LEXICON:
            continue

        base_polarity = POLARITY_LEXICON[word_stripped]
        multiplier = 1.0
        negated = False

        # on regarde les 3 mots précédents pour négation / intensificateur
        window = words[max(0, i - 3):i]
        for w in window:
            w_stripped = w.strip(".,!?;:\"'")
            if w_stripped in NEGATION_WORDS:
                negated = True
            if w_stripped in INTENSIFIERS:
                multiplier *= INTENSIFIERS[w_stripped]

        polarity = base_polarity * multiplier * (-1 if negated else 1)
        score += polarity
        details.append({"word": word_stripped, "polarity": polarity, "negated": negated})

    if score > NEUTRAL_THRESHOLD:
        label = "positif"
    elif score < -NEUTRAL_THRESHOLD:
        label = "négatif"
    else:
        label = "neutre"

    return {"label": label, "score": score, "details": details}


if __name__ == "__main__":
    tests = [
        "Ce produit est excellent, je le recommande !",
        "Je ne suis pas satisfait du tout, c'est décevant.",
        "Le colis est arrivé aujourd'hui.",
    ]
    for t in tests:
        result = analyze_lexical(t)
        print(f"{t!r} -> {result['label']} (score={result['score']:.2f})")
