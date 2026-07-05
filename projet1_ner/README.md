# Projet 1 — NER spécifique au domaine de l'IA

Pipeline spaCy hybride (EntityRuler + NER entraîné + couche Regex) pour
reconnaître des entités propres au domaine de l'intelligence artificielle,
avec interface Streamlit.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm  # optionnel, non utilisé par ce pipeline "blank"
```

> Le pipeline utilise `spacy.blank("fr")` (tokenizer français seul, sans
> modèle pré-entraîné) car on construit notre propre NER de A à Z. C'est
> volontaire : on n'a pas besoin d'un modèle pré-entraîné généraliste
> pour ce projet, seulement du tokenizer et de nos propres composants.

## Lancer le projet, étape par étape

| Étape du cahier des charges | Fichier | Commande |
|---|---|---|
| 1. Collecte des données | `data/corpus.py` | — (données statiques) |
| 2. Prétraitement | `preprocessing.py` | `python preprocessing.py` |
| 3. Motifs Regex | `patterns.py` | — (importé par train_ner.py) |
| 4. Annotation manuelle | `data/annotations.py` | — (importé par train_ner.py) |
| 5. EntityRuler + entraînement NER | `train_ner.py` | `python train_ner.py` |
| 6. Évaluation | `evaluate.py` | `python evaluate.py` |
| Interface de test | `app.py` | `streamlit run app.py` |

L'ordre logique d'exécution : `train_ner.py` (crée `model/domain_ner/`)
puis `evaluate.py`, ou directement `streamlit run app.py` qui entraîne
automatiquement le modèle au premier lancement s'il n'existe pas encore
(voir `load_model()` dans `app.py`).

## Architecture du pipeline

```
texte brut
    │
    ▼
[EntityRuler]   -> termes lexicaux connus (TECHNIQUE, FRAMEWORK, DATASET, METRIC)
    │
    ▼
[ner entraîné]  -> généralise à des formulations non listées explicitement
    │
    ▼
[regex_entity_component] -> ajoute DATE / VERSION / CODE_PROJET par regex
    │
    ▼
doc.ents (fusion sans chevauchement, priorité aux couches précédentes)
```

## Pourquoi séparer entités lexicales / entraînées et entités structurées ?

- **EntityRuler** (dictionnaire de termes) : parfait pour du vocabulaire
  fermé et stable ("TensorFlow", "PyTorch"...). Rapide, 100% précis sur
  ce qu'il connaît, mais ne généralise à rien d'autre.
- **NER entraîné** : nécessaire pour reconnaître des variantes non
  listées ("un réseau profond", "de l'apprentissage automatique"...).
  Demande des exemples annotés (`data/annotations.py`) et un
  entraînement (`train_ner.py`).
- **Regex** : idéal pour des formats structurés et prévisibles (dates,
  numéros de version, codes projet). Un modèle statistique serait
  overkill et moins fiable qu'un simple pattern.

## Affiner le modèle (étape 6, "raffiner")

Si l'évaluation (`python evaluate.py`) montre un label avec un mauvais
rappel ou une mauvaise précision :
- **Rappel faible** → ajouter plus d'exemples annotés dans
  `data/annotations.py` pour ce label, ou enrichir les listes de termes
  dans `patterns.py` (`_TECHNIQUES`, `_FRAMEWORKS`, etc.).
- **Précision faible** → les regex sont probablement trop permissives
  (ex. `VERSION` capture un faux positif) : resserrer le pattern dans
  `patterns.py`.
- Toujours ré-entraîner (`python train_ner.py`) après avoir modifié les
  annotations ou les patterns, puis relancer `evaluate.py`.

## Limites de cette version pédagogique

- Corpus volontairement petit (15 exemples train / 4 test) pour rester
  lisible : en conditions réelles il faudrait des centaines d'exemples
  annotés, idéalement avec un outil comme Prodigy ou Label Studio.
- Les regex `DATE`/`VERSION`/`CODE_PROJET` sont volontairement simples ;
  à étendre selon les formats réellement rencontrés dans le corpus final.
