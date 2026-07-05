# Projet 2 — Comparaison des techniques de vectorisation

## Installation

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm   # optionnel mais recommandé
```

## Étapes du cahier des charges -> fichiers

| Étape | Fichier |
|---|---|
| 1. Constitution du corpus (scraping + mots-clés seed) | `scraper.py`, `corpus_sample.py` |
| 2. Prétraitement | `preprocessing.py` |
| 3. Vectorisation (BoW, TF-IDF, Word2Vec, BERT) | `vectorizers.py` |
| 4. Calcul de similarité | `similarity.py` |
| 5. Évaluation et comparaison | `similarity.py` (`compare_techniques`) |
| Interface Streamlit | `app.py` |

## Lancer

```bash
streamlit run app.py
```

Par défaut l'appli utilise le **corpus de démo hors-ligne** (`corpus_sample.py`),
utile si le scraping réseau n'est pas disponible ou que lefaso.net change de
structure HTML. Bascule possible vers le scraping réel dans la barre latérale.

## Point important : le fallback de prétraitement

`preprocessing.py` utilise spaCy (`fr_core_news_sm`) pour lemmatiser
correctement le français ("agriculture" / "agricole" / "agriculteurs" ->
même lemme). **Si spaCy n'est pas installé**, un fallback plus simple
prend le relai (minuscule + suppression de stopwords), mais il ne relie
PAS les formes fléchies entre elles. Conséquence concrète observée en
test : chercher "agriculture" ne retrouve aucun document contenant
"agricole" sans spaCy, alors que chercher "agricole" directement
fonctionne bien. **Installez spaCy pour un vrai comportement de
production.**

## Comparaison des 4 techniques (ce qu'on observe typiquement)

| Technique | Force | Faiblesse |
|---|---|---|
| **BoW** | Simple, rapide, interprétable | Ignore l'importance relative des mots, sensible aux mots fréquents non discriminants |
| **TF-IDF** | Pondère les mots rares/discriminants | Toujours purement lexical : ne capture pas les synonymes |
| **Word2Vec** | Capture des relations sémantiques (moyenne de vecteurs de mots) | Nécessite un corpus d'entraînement assez grand pour être fiable ; ici le corpus est petit, les vecteurs seront peu robustes |
| **BERT (sentence-transformers)** | Comprend le contexte, capture des synonymes et paraphrases | Plus lourd/lent, nécessite un modèle pré-entraîné téléchargé |

Sur un **petit corpus** (comme notre démo à 12 documents), BoW/TF-IDF
donnent souvent des résultats aussi bons voire meilleurs que Word2Vec
entraîné from scratch, car Word2Vec a besoin de beaucoup de données pour
apprendre de bons vecteurs. BERT pré-entraîné, lui, reste pertinent même
sur un petit corpus car il n'a pas besoin d'être ré-entraîné.

## Limites de cette version

- Corpus de démo volontairement petit (12 articles).
- Word2Vec et BERT nécessitent `gensim` / `sentence-transformers`
  (non testés dans cet environnement sans accès réseau, mais code
  fonctionnel et structuré de façon identique aux autres méthodes).
