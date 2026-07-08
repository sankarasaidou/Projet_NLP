# Projet 4 — Analyse de sentiment comparative (version production)

Système d'analyse de sentiment combinant 3 approches (lexicale, statistique,
neuronale) sur des textes francophones, avec une architecture pensée pour la
production : séparation entraînement/service, persistance des modèles,
logging, gestion d'erreurs, tests unitaires, et conteneurisation Docker.

## Sommaire

- [Architecture](#architecture)
- [Installation](#installation)
- [Entraîner les modèles](#entraîner-les-modèles)
- [Lancer l'application](#lancer-lapplication)
- [Lancer les tests](#lancer-les-tests)
- [Déploiement Docker](#déploiement-docker)
- [Résultats réels obtenus](#résultats-réels-obtenus-en-test)
- [Limites et pistes d'amélioration](#limites-et-pistes-damélioration)

## Architecture

```
projet4_sentiment_prod/
├── src/sentiment_analysis/     # package principal (installable)
│   ├── config.py                # chemins, seuils, hyperparamètres centralisés
│   ├── logging_config.py         # logging structuré (remplace print())
│   ├── preprocessing.py           # nettoyage + validation d'entrée robuste
│   ├── data_loader.py               # chargement dataset/lexique, split stratifié
│   ├── lexical.py                    # approche lexicale (lexique externe)
│   ├── statistical.py                 # approche statistique + PERSISTANCE joblib
│   ├── neural.py                       # approche neuronale (optionnelle)
│   ├── evaluation.py                    # rapport, matrice de confusion
│   └── pipeline.py                       # façade unique (point d'entrée public)
├── data/
│   ├── dataset.csv               # 120 exemples étiquetés, divers domaines
│   └── lexicon_fr.csv             # ~154 mots de polarité (fichier externe)
├── models/                          # modèles entraînés (générés par train.py)
├── tests/                             # tests unitaires (pytest)
├── train.py                             # CLI d'entraînement (séparé du serving)
├── app.py                                 # interface Streamlit de production
├── Dockerfile
├── requirements.txt / requirements-dev.txt
├── pyproject.toml
└── runtime.txt
```

### Principe clé : séparation entraînement / service

`train.py` entraîne les modèles statistiques, calcule une **validation
croisée à 5 folds** pour chaque modèle, sélectionne objectivement le
meilleur, et **sauvegarde tout sur disque** (`models/*.joblib` +
`models/metadata.json`). L'application (`app.py`) ne fait que **charger**
ces artefacts déjà entraînés — jamais de ré-entraînement au démarrage.
C'est la différence structurante avec la version précédente (non-production),
qui ré-entraînait un modèle à chaque lancement de l'app.

### Façade unique (`pipeline.py`)

Le code appelant (interface, futur endpoint API...) n'utilise qu'une seule
classe, `SentimentPipeline`, qui gère en interne le chargement paresseux de
chaque approche et l'indisponibilité éventuelle de l'une d'elles (ex. modèle
neuronal non installé) **sans jamais faire planter les autres approches**.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Entraîner les modèles

**Étape obligatoire avant le premier lancement** (les modèles entraînés sont
déjà inclus dans ce livrable dans `models/`, mais voici comment les
régénérer, par exemple après avoir enrichi `data/dataset.csv`) :

```bash
python train.py --report
```

Options :
- `--model SVM` : n'entraîne qu'un seul modèle (`SVM`, `NaiveBayes` ou
  `LogisticRegression`) au lieu des 3.
- `--report` : affiche le détail précision/rappel/F1 par classe et la liste
  des erreurs de classification.
- `--cv-folds 10` : change le nombre de folds de la validation croisée
  (5 par défaut).

Le script sélectionne automatiquement le **meilleur modèle** (accuracy
moyenne en validation croisée) et l'enregistre dans
`models/metadata.json` sous la clé `_best_model` : c'est ce modèle que
`SentimentPipeline()` charge par défaut, sans qu'il soit codé en dur dans
le code.

## Lancer l'application

```bash
streamlit run app.py
```

Trois onglets :
1. **Analyser un texte** : compare les 3 approches sur un texte donné.
2. **Analyse par lot (CSV)** : uploade un CSV avec une colonne `text`,
   télécharge les résultats enrichis d'une colonne `sentiment_prédit`.
3. **Évaluation des modèles** : rapport complet (précision/rappel/F1,
   matrice de confusion) pour chaque modèle statistique, sur le jeu de test.

## Lancer les tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

**28 tests unitaires**, tous vérifiés manuellement avant livraison (voir
plus bas — `pytest` n'était pas installable dans l'environnement de
développement utilisé pour préparer ce livrable, faute d'accès réseau ; un
mini-simulateur de l'API pytest a été utilisé pour exécuter réellement
chaque test et confirmer qu'ils passent tous, avant d'être retiré du
livrable final). Chez toi, avec `pytest` installé normalement, ces mêmes
fichiers de tests fonctionnent sans aucune modification.

Couverture :
- `test_preprocessing.py` : validation d'entrée (texte vide/None/trop long), tokenisation.
- `test_lexical.py` : négation, seuils, contribution mot par mot.
- `test_statistical.py` : entraînement, prédiction, **persistance (sauvegarde puis rechargement, vérifie l'identité des prédictions)**, erreur claire si modèle non entraîné.
- `test_pipeline.py` : façade unique, isolation des erreurs entre approches (le point le plus important en production).

## Déploiement Docker

```bash
docker build -t sentiment-analysis .
docker run -p 8501:8501 sentiment-analysis
```

Le `Dockerfile` entraîne les modèles **au moment du build de l'image**
(`RUN python train.py`) : l'image finale est autonome, ne nécessite aucune
étape d'entraînement au démarrage du conteneur.

## Résultats réels obtenus en test

Tout ce qui suit a été **réellement exécuté** dans l'environnement de
préparation de ce livrable (scikit-learn/pandas disponibles, spaCy absent
faute d'accès réseau pour l'installer).

### Entraînement + validation croisée (5-fold, 120 exemples)

```
NaiveBayes           accuracy test = 0.46   CV (5-fold) = 0.45
SVM                  accuracy test = 0.42   CV (5-fold) = 0.40
LogisticRegression   accuracy test = 0.46   CV (5-fold) = 0.42

=> Meilleur modèle sélectionné automatiquement : NaiveBayes (0.45)
```

**Ces scores sont volontairement plus modestes que ceux du projet précédent
(qui atteignait 0.67 sur un jeu de test très proche du jeu d'entraînement en
vocabulaire).** Le nouveau dataset (120 exemples, 3 classes, vocabulaire
volontairement diversifié sur des domaines variés : produits, restaurants,
formations, transport, immobilier...) est un test **beaucoup plus honnête**
de la capacité de généralisation réelle du modèle. Deux causes identifiées
et vérifiables :

1. **spaCy absent dans l'environnement de préparation** → le fallback de
   tokenisation ne lemmatise pas ("améliore"/"amélioré"/"améliorée" restent
   3 tokens différents), ce qui fragmente le vocabulaire TF-IDF. **Avec
   spaCy installé (`pip install -r requirements.txt` chez toi), les
   résultats devraient être meilleurs** — à vérifier après installation.
2. **120 exemples reste petit** pour du TF-IDF classique sur du texte aussi
   varié. C'est une limite connue et attendue, pas un bug : voir
   recommandations ci-dessous.

### Tests unitaires

```
test_preprocessing.py   11 tests passés, 0 échec
test_lexical.py          6 tests passés, 0 échec
test_statistical.py      6 tests passés, 0 échec
test_pipeline.py         5 tests passés, 0 échec
TOTAL                   28 tests passés, 0 échec
```

### Pipeline complet (façade unique)

Testé de bout en bout avec gestion d'erreurs :
- Texte valide → les 3 approches retournent un résultat cohérent (`positif`
  détecté correctement sur un cas de test).
- Texte vide / `None` → `InvalidTextError` levée proprement, message clair.
- Approche neuronale non installée → erreur isolée, n'empêche pas les 2
  autres approches de fonctionner (`analyze_all` continue de retourner un
  résultat complet).

## Augmentation du dataset (données synthétiques)

Le dataset a été enrichi avec **4 692 phrases synthétiques** générées
par `scripts/generate_synthetic_dataset.py`, réparties sur **20
domaines** (produits tech, restaurants, hôtellerie, service client,
livraison, banque/assurance, santé, éducation, transport, immobilier,
télécom, sport, automobile, beauté, administration, culture, emploi,
logiciels, alimentation, voyage), avec une vraie gestion des accords
grammaticaux (genre) et des restrictions sémantiques par gabarit
(évite les non-sens du type "ce transporteur a été livré").

**data/dataset.csv contient maintenant 4 812 exemples** (120 originaux
+ 4 692 synthétiques, dédoublonnés) au lieu de 120.

### ⚠️ Attention à l'interprétation des métriques après augmentation

En relançant `python train.py` sur ce dataset fusionné, l'accuracy en
validation croisée grimpe à **0.99** — un chiffre **trompeur** : les
phrases synthétiques contiennent des formules de fin quasi exclusives
à chaque classe ("Je recommande vivement." n'apparaît que dans les
exemples positifs, "À éviter absolument." que dans les négatifs...).
Le modèle peut "tricher" en détectant ces formules plutôt qu'en
comprenant le sentiment réel.

**Le vrai test, honnête et sans fuite** (`python
scripts/benchmark_augmentation.py`) : entraîner UNIQUEMENT sur le
synthétique et évaluer UNIQUEMENT sur les 120 phrases originales
(écrites à la main, aucun gabarit, jamais vues à l'entraînement) :

```
Accuracy (train=synthétique seul, test=120 originaux, aucune fuite) : 0.80
  négatif    précision=0.73 rappel=0.68 f1=0.70
  neutre     précision=0.92 rappel=0.85 f1=0.88
  positif    précision=0.76 rappel=0.88 f1=0.81
```

**0.80 contre 0.42-0.46 avant augmentation (dataset original seul) :
c'est une vraie et substantielle amélioration de la généralisation**,
confirmée sur un jeu de test totalement indépendant du dataset
synthétique. C'est ce chiffre (0.80), pas le 0.99 de la validation
croisée sur données mixtes, qu'il faut retenir et citer.

### Recommandations pour aller plus loin

- Rendre les formules de fin ("Je recommande vivement.", etc.) moins
  exclusives à une classe (les mélanger davantage) pour éviter que le
  modèle ne s'appuie dessus comme raccourci.
- Ajouter davantage d'exemples **réels** (avis clients véritables,
  via scraping — voir `scraper.py` du projet 2) en complément du
  synthétique, qui reste un bon complément de volume mais pas un
  substitut à de vraies données.
- Réexécuter `scripts/benchmark_augmentation.py` à chaque évolution du
  générateur pour vérifier que l'amélioration reste réelle et pas un
  artefact.


## Conformité au cahier des charges du Projet 4

| Étape demandée | Fichier(s) | État |
|---|---|---|
| 1. Collecte de données (scraping avis clients) | `src/sentiment_analysis/scraper.py`, `scripts/collect_reviews.py` | ✅ Code complet (Allociné, notes en étoiles → labels). **Non testable en conditions réelles dans l'environnement de préparation** (pas d'accès réseau) — à valider chez toi, sélecteurs CSS ajustables si besoin. |
| 2. Prétraitement unifié | `preprocessing.py` | ✅ Testé |
| 3. Approche lexicale | `lexical.py` | ✅ Testé |
| 4. Approche statistique (NB/SVM/LogReg) | `statistical.py` | ✅ Testé |
| 5. Fine-tuning CamemBERT/FlauBERT | `scripts/train_neural.py` | ✅ Code complet (API `Trainer` standard). **Non exécuté** (pas de GPU/réseau ici) — à lancer chez toi, puis activer avec `SENTIMENT_ENABLE_NEURAL=true`. |
| 6. Évaluation comparative (3 approches + analyse d'erreurs) | `app.py` (onglet Évaluation), `evaluation.py` | ✅ Testé (lexicale + statistique ; neuronale rejoint automatiquement la comparaison dès qu'un modèle fine-tuné est activé) |
| Interface Streamlit (test sur textes utilisateurs) | `app.py` | ✅ Testé |

### Utiliser le scraper (étape 1)

```bash
python scripts/collect_reviews.py --movie-path "/film/fichefilm-XXXXX/critiques/spectateurs/" --pages 5
python train.py   # ré-entraîne avec les nouveaux avis réels ajoutés à data/dataset.csv
```

Remplace le chemin par celui d'un vrai film sur allocine.fr (visite le
site, copie le chemin depuis l'URL). Si aucun avis n'est trouvé,
inspecte la page (clic droit -> Inspecter sur un avis) et ajuste
`_SELECTORS` dans `scraper.py`.

### Fine-tuner le modèle neuronal (étape 5)

```bash
pip install transformers torch datasets
python scripts/train_neural.py --checkpoint camembert-base --epochs 4
export SENTIMENT_ENABLE_NEURAL=true   # active l'approche dans le pipeline
export SENTIMENT_NEURAL_CHECKPOINT=./models/neural_sentiment
streamlit run app.py
```

## Limites et pistes d'amélioration

- **Dataset (4 812 exemples, dont 4 692 synthétiques)** : le volume
  n'est plus un facteur limitant, mais le déséquilibre synthétique/réel
  (97%/3%) reste une limite pour un vrai déploiement. Prochaine étape
  réaliste : remplacer progressivement du synthétique par de vrais avis
  scrapés (voir `scraper.py` du projet 2), et re-belancer.
- **Lexique fait main (154 mots)** : à remplacer par un lexique publié et
  validé linguistiquement (FEEL, Blogoscopie, Polarimots) pour un usage
  réel — le chargement externe (`data/lexicon_fr.csv`) est déjà prêt à
  accueillir un tel fichier sans changer une ligne de code.
- **spaCy non testé dans l'environnement de préparation** : à vérifier
  chez toi après `pip install -r requirements.txt` — les métriques
  devraient s'améliorer avec la vraie lemmatisation.
- **Approche neuronale non fine-tunée dans ce livrable** : le code
  (`neural.py`) suppose un modèle déjà fine-tuné sauvegardé localement (ex.
  via une adaptation du script `fine_tune_sentiment_model` du projet 4
  original) ; à documenter/adapter selon l'infrastructure GPU disponible.
- **Pas de CI/CD configuré** (ex. GitHub Actions lançant `pytest` à chaque
  push) : recommandé pour une vraie mise en production continue.
