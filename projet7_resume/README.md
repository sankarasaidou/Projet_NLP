# Projet 7 — Générateur de résumés automatiques multi-techniques

## Installation
```bash
pip install -r requirements.txt   # transformers/torch nécessaires pour l'abstractif
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Corpus diversifié | `corpus.py` |
| 2. Résumé extractif (TextRank, TF-IDF) | `extractive.py` |
| 3. Résumé abstractif (T5, mBART) | `abstractive.py` |
| 4. Métriques d'évaluation (ROUGE, cohérence) | `evaluation.py` |
| 5. Interface adaptative | `optimization.py` (`summarize_adaptive`) + `app.py` |
| 6. Optimisation (comparaison, hybridation) | `optimization.py` (`compare_all_methods`, `summarize_hybrid`) |
| Interface Streamlit | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Résultats réels obtenus en test dans cet environnement

ROUGE et cohérence, TextRank vs scoring TF-IDF, sur l'article de presse
(résumé "Court", comparé à un résumé de référence écrit à la main) :
```
TextRank        ROUGE-1 F1 = 0.23   cohérence = 0.13
Scoring TF-IDF  ROUGE-1 F1 = 0.18   cohérence = 0.04
```

**TextRank surpasse le scoring TF-IDF simple sur les deux métriques**
ici : en tenant compte des relations entre phrases (graphe de
similarité), TextRank sélectionne des phrases plus centrales au
document, alors que le scoring TF-IDF favorise des phrases contenant
des mots rares, parfois peu représentatives du sujet principal.

Les approches abstractives (T5, mBART) n'ont pas pu être testées ici
(transformers/torch non installés, pas d'accès réseau dans ce bac à
sable), mais le code suit l'API standard Hugging Face
`pipeline("summarization", ...)` et est directement exécutable dans un
environnement avec ces paquets.

## ROUGE : implémentation maison, pourquoi ?

Plutôt que de dépendre du paquet `rouge-score` (pas toujours installé),
ROUGE-1, ROUGE-2 et ROUGE-L sont réimplémentés à la main dans
`evaluation.py` (recoupement de n-grammes / plus longue sous-séquence
commune). Ça garantit que l'évaluation fonctionne dans n'importe quel
environnement Python standard, sans dépendance supplémentaire.

## Hybridation (étape 6)

`summarize_hybrid()` illustre une stratégie utilisée en production :
1. Réduction extractive (TextRank) du document à ses phrases clés.
2. Reformulation abstractive (T5/mBART) de ce texte déjà réduit.

Avantage concret : le modèle abstractif travaille sur un texte plus
court et pré-filtré, ce qui réduit le risque d'hallucination et accélère
l'inférence, comparé à un résumé abstractif directement sur le
document complet.

## Limites de cette version
- Corpus de démo à 3 documents (un par type demandé) : suffisant pour
  valider la mécanique, mais un vrai projet testerait sur des dizaines
  de documents par catégorie.
- Les résumés de référence (`REFERENCE_SUMMARIES`) sont écrits à la
  main pour cette démo ; pour une évaluation ROUGE représentative, il
  faudrait des résumés de référence produits par des humains sur un
  échantillon plus large.
- L'évaluation humaine (`human_eval_template()`) n'est qu'un squelette
  de grille de notation : à faire remplir par de vrais évaluateurs
  humains, jamais à simuler automatiquement.
