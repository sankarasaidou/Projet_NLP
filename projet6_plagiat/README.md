# Projet 6 — Détecteur de plagiat (mesures de similarité)

## Installation
```bash
pip install -r requirements.txt
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Constitution du corpus (cas connus) | `data.py` |
| 2. Prétraitement avancé (n-grammes, skip-grammes) | `preprocessing.py` |
| 3. Métriques de similarité | `similarity_metrics.py` |
| 4. Optimisation des seuils | `threshold_optimization.py` |
| 5. Visualisation | `visualization.py` |
| 6. Validation | `validation.py` |
| Interface Streamlit | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Résultats réels obtenus en test (tout a été exécuté dans cet environnement)

Seuils optimaux trouvés empiriquement (F1-score sur les 6 cas connus) :
```
Cosinus (n-grammes)    seuil = 0.05   F1 = 0.80
Jaccard (n-grammes)    seuil = 0.05   F1 = 0.80
Levenshtein            seuil = 0.30   F1 = 0.80
```

Validation finale (accuracy = 0.83 pour les 3 métriques, mais **chacune
se trompe sur un cas différent** — c'est le résultat le plus important
de ce projet) :

| Métrique | Cas manqué |
|---|---|
| Cosinus / Jaccard (n-grammes) | La **paraphrase profonde** (`doc_plagiat_reformule_1`, vocabulaire très différent) — les n-grammes de mots ne se recoupent presque plus après reformulation. |
| Levenshtein | Le **plagiat partiel réorganisé** (`doc_plagiat_partiel_2`, phrases dans un ordre différent) — la distance d'édition caractère par caractère explose dès que l'ordre des phrases change, même si le contenu est repris. |

**Conclusion à mettre dans le rapport** : aucune métrique seule ne
détecte tous les types de plagiat. Un vrai détecteur de plagiat
combine plusieurs métriques (vote majoritaire ou score pondéré), et
pour les paraphrases profondes, il faudrait ajouter une métrique
**sémantique** (similarité cosinus sur des embeddings de phrase type
BERT/Sentence-Transformers) qui, contrairement aux n-grammes de mots,
capture le sens plutôt que la forme de surface.

## Visualisations générées
- **Heatmap** : matrice de similarité entre tous les documents, générée
  et vérifiée visuellement dans ce test (fichier PNG produit avec
  succès).
- **Dendrogramme** : regroupement hiérarchique des documents par
  similarité, pour repérer visuellement des "grappes" suspectes.

## Limites de cette version
- Corpus de cas connus volontairement petit (6 paires) pour rester
  vérifiable à la main ; un vrai jeu de validation académique
  utiliserait des centaines de paires (par ex. corpus PAN Plagiarism
  Detection).
- Les seuils optimisés ici sont propres à ce petit jeu de test : à
  ré-optimiser sur un corpus de validation plus représentatif avant
  tout déploiement réel.
