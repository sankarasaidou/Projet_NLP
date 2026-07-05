# Projet 4 — Analyse de sentiment (approches comparatives)

## Installation
```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm   # optionnel mais recommandé
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Collecte de données | `corpus.py` |
| 2. Prétraitement unifié | `preprocessing.py` |
| 3. Approche lexicale | `lexical_approach.py` |
| 4. Approche statistique | `statistical_approach.py` |
| 5. Approche neuronale | `neural_approach.py` |
| 6. Évaluation comparative | `evaluation.py` |
| Interface Streamlit | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Résultats réels obtenus en test dans cet environnement

```
Lexicale     : accuracy = 1.00
Statistique  : accuracy = 0.67 (SVM), erreurs sur des cas ambigus
               ("cassé" en contexte négatif interprété "neutre" par le
               modèle statistique faute d'exemples similaires en train)
Neuronale    : non testable ici (transformers/torch non installés,
               pas d'accès réseau dans ce bac à sable) — code fourni et
               structuré pour tourner tel quel dans un environnement
               avec ces paquets installés.
```

**Attention, biais à corriger avant de présenter ce résultat comme une
preuve de supériorité de l'approche lexicale** : le dictionnaire de
`lexical_approach.py` a été construit en connaissant le vocabulaire du
corpus de démo (`décevant`, `cassé`, `ravi`...). Sur du texte réel avec
un vocabulaire plus varié, l'approche lexicale chuterait fortement
(elle ne reconnaît QUE les mots de son dictionnaire), alors que
l'approche statistique et surtout neuronale généraliseraient bien mieux.
**C'est un point à discuter explicitement dans le rapport** : ce
résultat illustre une limite classique des dictionnaires de polarité
(couverture lexicale), pas une vraie supériorité de la méthode.

## Comparaison des 3 approches (à documenter dans le rapport)

| Approche | Avantages | Inconvénients |
|---|---|---|
| **Lexicale** | Interprétable (on voit exactement quel mot a pesé), pas besoin d'entraînement | Couverture limitée au dictionnaire, ne gère pas bien l'ironie/les tournures complexes |
| **Statistique** (NB/SVM/LogReg) | Généralise mieux que le dictionnaire, rapide à entraîner | Nécessite des données étiquetées, reste purement lexical (TF-IDF) donc pas de vraie compréhension du contexte |
| **Neuronale** (CamemBERT/FlauBERT) | Comprend le contexte fin, état de l'art en NLP | Coûteux en calcul, nécessite un GPU pour un entraînement rapide, "boîte noire" moins interprétable |

## Limites de cette version
- Corpus de démo très petit (24 exemples train / 6 test) : les
  performances chiffrées ci-dessus n'ont de valeur que pédagogique et
  chuteraient sur un corpus réel plus large et plus varié.
- L'approche neuronale n'a pas pu être exécutée dans ce bac à sable
  (pas d'accès réseau pour installer `transformers`/`torch`), mais le
  code suit l'API standard Hugging Face `Trainer` et est prêt à
  l'emploi.
