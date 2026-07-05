# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Interprétation des résultats
ÉTAPE 5 : Comparaison des techniques
------------------------------------------
- interpret_topics() : donne un "thème général" heuristique à chaque
  topic à partir de ses mots-clés (recherche de mots-clés caractéristiques
  d'un thème connu). En conditions réelles, cette interprétation reste
  humaine ; ici on montre une heuristique simple utile pour un premier
  tri automatique.

- topic_coherence() : mesure de cohérence simple (PMI moyenne entre paires
  de mots-clés du topic, calculée sur le corpus). Une cohérence élevée
  signifie que les mots-clés d'un topic apparaissent souvent ensemble
  dans les mêmes documents (signe d'un topic "propre").
"""

import itertools
import math

from preprocessing import preprocess_corpus

# Heuristique simple : mots-clés caractéristiques de thèmes connus dans
# notre corpus de démo (à adapter si le corpus change de domaine).
_THEME_HINTS = {
    "Agriculture": {"agricole", "agriculteur", "agriculteurs", "culture", "production",
                    "récolte", "irrigation", "rendement", "engrais", "saison"},
    "Santé": {"santé", "vaccination", "médicament", "médicaments", "paludisme",
              "hôpital", "centre", "sanitaire"},
    "Économie": {"économique", "économie", "croissance", "inflation", "investissement",
                 "entreprise", "entreprises", "banque", "taux"},
    "Éducation": {"éducatif", "enseignant", "enseignants", "scolaire", "école",
                  "élève", "bourse", "classe", "classes"},
    "Sécurité": {"sécurité", "patrouille", "patrouilles", "frontière", "menace",
                 "menaces", "dispositif", "dispositifs"},
}


def interpret_topics(topics: list[dict]) -> list[dict]:
    """Ajoute un champ 'theme_probable' à chaque topic, en comptant les
    recoupements entre ses mots-clés et les thèmes connus."""
    enriched = []
    for topic in topics:
        words = set(topic["top_words"])
        best_theme, best_score = "Non identifié", 0
        for theme, hints in _THEME_HINTS.items():
            score = len(words & hints)
            if score > best_score:
                best_theme, best_score = theme, score
        enriched.append({**topic, "theme_probable": best_theme})
    return enriched


def topic_coherence(topic_words: list[str], documents: list[str]) -> float:
    """Cohérence simplifiée façon UMass : moyenne des log( (co-occurrence
    + 1) / occurrence_du_mot_le_plus_fréquent ) sur toutes les paires de
    mots-clés du topic. Plus la valeur est proche de 0 (ou positive), plus
    les mots co-apparaissent souvent ; plus elle est très négative, moins
    les mots sont liés dans le corpus."""
    clean_docs = [set(doc.split()) for doc in preprocess_corpus(documents)]

    def doc_freq(word):
        return sum(1 for doc in clean_docs if word in doc)

    def co_doc_freq(w1, w2):
        return sum(1 for doc in clean_docs if w1 in doc and w2 in doc)

    scores = []
    for w1, w2 in itertools.combinations(topic_words, 2):
        d1 = doc_freq(w1)
        if d1 == 0:
            continue
        co = co_doc_freq(w1, w2)
        scores.append(math.log((co + 1) / d1))

    return sum(scores) / len(scores) if scores else float("-inf")


def compare_methods_coherence(results_by_method: dict, documents: list[str]) -> dict:
    """Pour chaque méthode (LDA, NMF, BERTopic), calcule la cohérence
    moyenne sur tous ses topics -> permet de comparer objectivement les
    techniques entre elles, en plus de la lecture humaine des mots-clés."""
    comparison = {}
    for method, result in results_by_method.items():
        topic_coherences = [
            topic_coherence(topic["top_words"], documents) for topic in result["topics"]
        ]
        avg = sum(topic_coherences) / len(topic_coherences) if topic_coherences else float("-inf")
        comparison[method] = {"avg_coherence": avg, "per_topic": topic_coherences}
    return comparison


if __name__ == "__main__":
    from corpus import CORPUS
    from topic_models import run_lda, run_nmf

    results = {"LDA": run_lda(CORPUS, n_topics=5), "NMF": run_nmf(CORPUS, n_topics=5)}
    for method, result in results.items():
        print(f"\n=== {method} — interprétation ===")
        for topic in interpret_topics(result["topics"]):
            print(f"  Topic {topic['id']} ({topic['theme_probable']}): {', '.join(topic['top_words'][:5])}")

    coherence = compare_methods_coherence(results, CORPUS)
    print("\n=== Cohérence moyenne par méthode ===")
    for method, c in coherence.items():
        print(f"  {method}: {c['avg_coherence']:.3f}")
