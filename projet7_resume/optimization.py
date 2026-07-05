# -*- coding: utf-8 -*-
"""
ÉTAPE 5 : Interface adaptative (longueur/style) — logique métier, l'UI
           elle-même est dans app.py.
ÉTAPE 6 : Optimisation — comparaison des approches + hybridation
--------------------------------------------------------------------
- summarize_adaptive() : point d'entrée unique qui adapte le résumé
  selon le "style" demandé (court/moyen/long) en ajustant le nombre de
  phrases (extractif) ou max_length/min_length (abstractif).

- summarize_hybrid() : HYBRIDATION simple et efficace en pratique :
  on utilise d'abord un résumé EXTRACTIF (TextRank) pour réduire le
  texte à ses phrases les plus importantes, PUIS on passe ce texte
  réduit à un modèle ABSTRACTIF pour le reformuler et le fluidifier.
  Avantage : le modèle abstractif travaille sur un texte plus court et
  déjà filtré, ce qui réduit les hallucinations et accélère l'inférence
  par rapport à un résumé abstractif du document complet.

- compare_all_methods() : fait tourner toutes les méthodes disponibles
  sur un document et calcule leur score ROUGE par rapport à la
  référence, pour comparaison directe.
"""

from extractive import EXTRACTIVE_METHODS, summarize_textrank
from evaluation import rouge_scores, coherence_score

STYLE_TO_N_SENTENCES = {"Court": 2, "Moyen": 4, "Long": 6}
STYLE_TO_LENGTH = {"Court": (20, 60), "Moyen": (40, 100), "Long": (80, 180)}


def summarize_adaptive(text: str, method: str, style: str = "Moyen") -> str:
    if method in EXTRACTIVE_METHODS:
        n_sentences = STYLE_TO_N_SENTENCES.get(style, 4)
        return EXTRACTIVE_METHODS[method](text, n_sentences=n_sentences)

    from abstractive import ABSTRACTIVE_METHODS
    if method in ABSTRACTIVE_METHODS:
        min_len, max_len = STYLE_TO_LENGTH.get(style, (40, 100))
        return ABSTRACTIVE_METHODS[method](text, max_length=max_len, min_length=min_len)

    raise ValueError(f"Méthode inconnue : {method}")


def summarize_hybrid(text: str, style: str = "Moyen", abstractive_model: str = "T5 (français)") -> str:
    """Étape 1 : réduction extractive (TextRank) -> étape 2 : reformulation abstractive."""
    n_sentences = STYLE_TO_N_SENTENCES.get(style, 4) + 2  # on garde un peu plus de matière avant reformulation
    extracted = summarize_textrank(text, n_sentences=n_sentences)

    from abstractive import summarize_abstractive
    min_len, max_len = STYLE_TO_LENGTH.get(style, (40, 100))
    return summarize_abstractive(extracted, model_name=abstractive_model, max_length=max_len, min_length=min_len)


def compare_all_methods(text: str, reference: str, style: str = "Moyen") -> dict:
    comparison = {}
    for method_name in EXTRACTIVE_METHODS:
        summary = summarize_adaptive(text, method_name, style)
        comparison[method_name] = {
            "summary": summary,
            "rouge": rouge_scores(summary, reference),
            "coherence": coherence_score(summary),
        }

    try:
        from abstractive import ABSTRACTIVE_METHODS
        for method_name in ABSTRACTIVE_METHODS:
            summary = summarize_adaptive(text, method_name, style)
            comparison[method_name] = {
                "summary": summary,
                "rouge": rouge_scores(summary, reference),
                "coherence": coherence_score(summary),
            }
    except ImportError as e:
        comparison["Approches abstractives"] = {"error": str(e)}

    return comparison


if __name__ == "__main__":
    from corpus import DOCUMENTS, REFERENCE_SUMMARIES

    text = DOCUMENTS["Article de presse"]
    reference = REFERENCE_SUMMARIES["Article de presse"]

    comparison = compare_all_methods(text, reference, style="Court")
    for method, result in comparison.items():
        if "error" in result:
            print(f"{method}: non disponible ({result['error']})")
            continue
        f1 = result["rouge"]["ROUGE-1"]["f1"]
        print(f"{method:<20} ROUGE-1 F1={f1:.2f}  cohérence={result['coherence']:.2f}")
        print(f"   -> {result['summary']}")
