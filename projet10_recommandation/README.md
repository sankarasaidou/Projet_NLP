# Projet 10 — Système de recommandation de contenu textuel

## Installation
```bash
pip install -r requirements.txt
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Corpus de contenu | `data.py` |
| 2. Profiling utilisateur | `profiling.py` (`build_user_profile`) |
| 3. Analyse de contenu | `profiling.py` (`build_content_vectors`) |
| 4. Similarité de contenu | `recommend.py` (cosinus) |
| 5. Algorithmes de recommandation | `recommend.py` (content-based, collaboratif, hybride) |
| 6. Évaluation | `evaluation.py` |
| Interface Streamlit | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Résultats réels obtenus en test

Le profil de `user_A` (historique 100% agriculture) est bien dominé par
des termes comme `agriculteurs`, `agricole` — validé concrètement en
inspectant les poids TF-IDF du profil construit.

**Évaluation leave-one-out (Précision@3 / Rappel@3), exécutée
réellement :**
```
user_A: held-out = 'Le prix des engrais...' -> ✅ trouvé
user_B: held-out = 'Nouveau centre de santé...' -> ❌ manqué
user_C: held-out = "Inflation et pouvoir d'achat" -> ❌ manqué
user_D: held-out = "Programme de bourses d'études" -> ❌ manqué

Précision@3 moyenne : 0.08
Rappel@3 moyenne    : 0.25
```

**Ce résultat est honnête et attendu, pas un bug** : avec seulement 12
articles au total et 4 utilisateurs, (a) chaque profil utilisateur
repose sur très peu de signal, (b) le leave-one-out ne retire qu'UN
SEUL article par utilisateur ce qui rend la métrique très binaire (0%
ou 100% par utilisateur), et (c) certains thèmes (santé, éducation)
n'ont que 2-3 articles dans tout le corpus, donc "deviner" le bon
article held-out parmi peu d'alternatives proches est difficile.
**Sur un corpus réel de centaines/milliers d'articles et des dizaines
d'utilisateurs, ces métriques seraient bien plus significatives.**

**Filtrage collaboratif — limite observée** : avec seulement 4
utilisateurs, la matrice utilisateur x article est très éparse, et le
filtrage collaboratif recommande parfois les mêmes articles à des
utilisateurs différents (ex. "Foire agricole régionale" recommandé à
la fois à `user_C` et `user_D`, alors que leurs profils sont pourtant
différents) — c'est le classique problème de **données éparses /
cold-start** du filtrage collaboratif sur petits jeux de données.

## Pourquoi une approche hybride (étape 5) ?

Le content-based et le filtrage collaboratif ont des forces
complémentaires :
- **Content-based** fonctionne même pour un nouvel article jamais noté
  par personne (pas de "cold-start article"), mais reste enfermé dans
  les thèmes déjà lus par l'utilisateur (peu de sérendipité).
- **Filtrage collaboratif** peut découvrir des articles inattendus
  (recommandés par des utilisateurs aux goûts proches), mais échoue
  totalement sur un nouvel article sans aucune note ("cold-start
  article"), et nécessite beaucoup d'utilisateurs pour être fiable.

`hybrid_recommend()` combine les deux scores (pondération réglable dans
l'interface Streamlit) pour atténuer les faiblesses de chaque approche.

## Limites de cette version
- Corpus minuscule (12 articles, 4 utilisateurs) : purement
  pédagogique. Les scores de précision/rappel n'ont de valeur
  qu'illustrative de la MÉTHODOLOGIE d'évaluation, pas de la
  performance réelle du système.
- Le filtrage collaboratif implémenté est "item-based" simple ; des
  approches plus robustes (factorisation matricielle SVD, ALS) seraient
  nécessaires en production avec plus de données.
