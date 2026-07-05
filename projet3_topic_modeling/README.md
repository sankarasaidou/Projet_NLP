# Projet 3 — Modélisation de sujets (LDA / NMF / BERTopic)

## Installation
```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm   # optionnel mais recommandé
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Constitution du corpus | `corpus.py` |
| 2. Prétraitement | `preprocessing.py` |
| 3. LDA / NMF / BERTopic | `topic_models.py` |
| 4. Interprétation des résultats | `interpretation.py` (`interpret_topics`) |
| 5. Comparaison des techniques | `interpretation.py` (`compare_methods_coherence`) |
| Interface Streamlit | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Résultats obtenus en test (corpus de démo, 20 documents / 5 thèmes)

LDA et NMF ont bien été exécutés et testés dans cet environnement
(scikit-learn est disponible). Résultat concret observé :

```
LDA — cohérence moyenne : -0.200
NMF — cohérence moyenne :  0.087
```

**NMF obtient une meilleure cohérence que LDA sur ce petit corpus.**
C'est un résultat attendu et à documenter dans un rapport : LDA est un
modèle probabiliste bayésien qui a besoin de **beaucoup de documents**
pour bien estimer ses distributions ; sur un corpus de 20 documents, NMF
(factorisation matricielle déterministe) est souvent plus stable. Sur un
corpus de plusieurs centaines/milliers d'articles réels, l'écart se
réduit généralement, voire s'inverse.

BERTopic n'a pas pu être testé dans cet environnement (pas d'accès
réseau pour installer le paquet), mais le code est fourni et suit
exactement la même interface (`{"topics": ..., "doc_topic_assignment": ...}`)
que LDA/NMF pour rester interchangeable dans `app.py`.

## Avantages / inconvénients de chaque méthode (à discuter dans le rapport)

| Méthode | Avantages | Inconvénients |
|---|---|---|
| **LDA** | Modèle probabiliste bien établi, chaque document est un mélange de topics (pas une assignation dure) | Sensible à la taille du corpus, nécessite de fixer `n_topics` à l'avance, hyperparamètres à régler (alpha, beta) |
| **NMF** | Résultats souvent plus interprétables sur petits corpus, déterministe | Ne modélise pas d'incertitude probabiliste, nécessite aussi de fixer `n_topics` |
| **BERTopic** | Détermine automatiquement le nombre de topics, capture le contexte sémantique (embeddings) | Plus lourd/lent, `min_topic_size` sensible sur petits corpus, dépendance à un modèle pré-entraîné |

## Limites
- Corpus de démo volontairement petit et "propre" (5 thèmes bien
  distincts) pour valider la mécanique ; un corpus réel scrapé sera plus
  bruité et demandera plus de nettoyage et un plus grand nombre de
  documents pour des topics stables.
- `interpret_topics()` utilise une heuristique simple (recoupement de
  mots-clés) pour proposer un thème probable : à affiner manuellement,
  l'interprétation reste en dernier ressort une tâche humaine.
