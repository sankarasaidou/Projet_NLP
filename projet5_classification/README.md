# Projet 5 — Classification automatique d'articles de presse

## Installation
```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm   # optionnel mais recommandé
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Scraping ciblé | `data.py` (utiliser `scraper.py` du projet 2 pour un vrai scraping lefaso.net/AIB) |
| 2. Annotation manuelle | `data.py` (`LABELED_CORPUS`) |
| 3. Vectorisation multiple | `classification.py` (`VECTORIZERS`) |
| 4. Classification comparative | `classification.py` (`train_and_predict`) |
| 5. Optimisation | `optimization.py` |
| 6. Déploiement | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Résultats réels obtenus en test

Comparaison BoW/TF-IDF x Naive Bayes/SVM/MLP (testé réellement) :
```
BoW    + Naive Bayes                accuracy = 0.75
BoW    + SVM                        accuracy = 0.50
BoW    + Réseau de neurones (MLP)   accuracy = 0.75
TF-IDF + Naive Bayes                accuracy = 0.50
TF-IDF + SVM                        accuracy = 0.50
TF-IDF + Réseau de neurones (MLP)   accuracy = 0.75
```

Après optimisation par GridSearchCV (`optimization.py`), le MLP est
sélectionné comme meilleur modèle (accuracy test = 0.75), malgré une
accuracy en validation croisée plus faible (0.17-0.21) — **ce dernier
point s'explique par la taille minuscule du jeu d'entraînement (24
exemples, 4 catégories)** : en 3-fold CV, chaque fold ne contient que 2
exemples par catégorie, ce qui rend la validation croisée peu fiable ici.
Sur un corpus réel de plusieurs centaines/milliers d'articles, ces
scores de CV deviendraient beaucoup plus représentatifs.

Un article de test réel a été soumis au pipeline de déploiement (TF-IDF
+ SVM) et correctement classé en `sport` avec 40% de confiance — un
score de confiance modeste qui reflète honnêtement le peu de données
d'entraînement disponibles.

## Pourquoi Naive Bayes est exclu de Word2Vec/BERT

`MultinomialNB` suppose des features non négatives (des comptes ou des
fréquences). Les embeddings Word2Vec/BERT contiennent des valeurs
négatives : les combiner ferait planter le modèle. C'est documenté et
bloqué explicitement dans `classification.py` (`_INCOMPATIBLE`) plutôt
que de laisser un plantage silencieux — l'interface Streamlit filtre
elle aussi automatiquement les combinaisons incompatibles.

## Limites de cette version
- Corpus minuscule (24 exemples / 4 test) : purement pédagogique, à
  remplacer par un vrai scraping massif (`scraper.py` du projet 2,
  adaptable) pour des résultats représentatifs.
- Word2Vec et BERT non testés ici (paquets non installés, pas d'accès
  réseau dans ce bac à sable), mais code fonctionnel et suit la même
  interface que BoW/TF-IDF.
