# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Métriques d'évaluation
--------------------------------------
- ROUGE-N / ROUGE-L : implémentés nous-mêmes (sans dépendre du paquet
  `rouge-score`, pour garantir que ça tourne partout) :
    - ROUGE-1 : recoupement d'unigrammes entre résumé généré et
      résumé de référence.
    - ROUGE-2 : recoupement de bigrammes (capte un peu de fluidité/ordre).
    - ROUGE-L : plus longue sous-séquence commune (capte l'ordre des
      mots partagés, plus tolérant aux réordonnancements).
- coherence_score() : heuristique simple de cohérence intra-résumé
  (similarité moyenne entre phrases consécutives du résumé -> un résumé
  cohérent ne devrait pas sauter d'un sujet à un autre sans transition).
- L'évaluation humaine reste, comme souvent en NLG, la référence
  ultime : on fournit une fonction `human_eval_template()` pour
  structurer une grille de notation manuelle (clarté, fidélité,
  concision) plutôt que de simuler une évaluation humaine.
"""

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n: int = 1) -> dict:
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    cand_grams, ref_grams = _ngrams(cand_tokens, n), _ngrams(ref_tokens, n)

    overlap = sum((cand_grams & ref_grams).values())
    precision = overlap / sum(cand_grams.values()) if sum(cand_grams.values()) else 0.0
    recall = overlap / sum(ref_grams.values()) if sum(ref_grams.values()) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(candidate: str, reference: str) -> dict:
    cand_tokens, ref_tokens = _tokenize(candidate), _tokenize(reference)
    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_scores(candidate: str, reference: str) -> dict:
    return {
        "ROUGE-1": rouge_n(candidate, reference, 1),
        "ROUGE-2": rouge_n(candidate, reference, 2),
        "ROUGE-L": rouge_l(candidate, reference),
    }


def coherence_score(summary: str) -> float:
    """Similarité moyenne (Jaccard sur mots) entre phrases consécutives
    du résumé -> proxy simple de fluidité thématique."""
    from extractive import split_sentences

    sentences = split_sentences(summary)
    if len(sentences) < 2:
        return 1.0

    scores = []
    for s1, s2 in zip(sentences, sentences[1:]):
        w1, w2 = set(_tokenize(s1)), set(_tokenize(s2))
        union = w1 | w2
        scores.append(len(w1 & w2) / len(union) if union else 0.0)
    return sum(scores) / len(scores)


def human_eval_template() -> dict:
    """Grille d'évaluation humaine à remplir manuellement (1 à 5) :
    clarté, fidélité au texte source, concision."""
    return {"clarté": None, "fidélité": None, "concision": None}


if __name__ == "__main__":
    from corpus import DOCUMENTS, REFERENCE_SUMMARIES
    from extractive import summarize_textrank

    text = DOCUMENTS["Article de presse"]
    reference = REFERENCE_SUMMARIES["Article de presse"]
    generated = summarize_textrank(text, n_sentences=2)

    scores = rouge_scores(generated, reference)
    for name, s in scores.items():
        print(f"{name}: précision={s['precision']:.2f} rappel={s['recall']:.2f} f1={s['f1']:.2f}")
    print(f"Cohérence intra-résumé : {coherence_score(generated):.2f}")
