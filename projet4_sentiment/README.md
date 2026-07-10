# Tonalité — Analyse de sentiment comparative

Système d'analyse de sentiment combinant trois approches (lexicale,
statistique, neuronale) sur des textes francophones. Architecture pensée
pour la production : séparation entraînement/service, persistance des
modèles, logging, gestion d'erreurs, tests unitaires, conteneurisation
Docker.

## Sommaire

- [Architecture](#architecture)
- [Installation](#installation)
- [Entraîner les modèles](#entraîner-les-modèles)
- [Lancer l'application](#lancer-lapplication)
- [Lancer les tests](#lancer-les-tests)
- [Déploiement Docker](#déploiement-docker)
- [Collecte de données réelles](#collecte-de-données-réelles)
- [Approche neuronale](#approche-neuronale)
- [Augmentation du dataset](#augmentation-du-dataset)
- [Résultats](#résultats)
- [Limites et pistes d'amélioration](#limites-et-pistes-damélioration)

## Architecture

```
projet4_sentiment_prod/
├── src/sentiment_analysis/     # package principal (installable)
│   ├── config.py                # chemins, seuils, hyperparamètres centralisés
│   ├── logging_config.py         # logging structuré
│   ├── preprocessing.py           # nettoyage + validation d'entrée
│   ├── data_loader.py               # chargement dataset/lexique, split stratifié
│   ├── lexical.py                    # approche lexicale (lexique externe)
│   ├── statistical.py                 # approche statistique + persistance joblib
│   ├── neural.py                       # approche neuronale (optionnelle)
│   ├── scraper.py                       # collecte d'avis réels (Allociné)
│   ├── evaluation.py                     # rapport, matrice de confusion
│   ├── ui_components.py                   # design system de l'interface
│   └── pipeline.py                         # façade unique (point d'entrée)
├── data/
│   ├── dataset.csv               # exemples étiquetés, domaines variés
│   └── lexicon_fr.csv             # lexique de polarité (fichier externe)
├── models/                          # modèles entraînés (générés par train.py)
├── tests/                             # tests unitaires (pytest)
├── scripts/
│   ├── collect_reviews.py               # CLI de scraping
│   ├── train_neural.py                   # fine-tuning CamemBERT/FlauBERT
│   ├── generate_synthetic_dataset.py      # génération de données d'entraînement
│   └── benchmark_augmentation.py           # validation de l'apport du synthétique
├── train.py                             # CLI d'entraînement (séparé du serving)
├── app.py                                 # interface Streamlit
├── Dockerfile
├── requirements.txt / requirements-dev.txt
├── pyproject.toml
└── runtime.txt
```

### Séparation entraînement / service

`train.py` entraîne les modèles statistiques, calcule une validation
croisée à 5 folds pour chaque modèle, sélectionne le meilleur, et
sauvegarde tout sur disque (`models/*.joblib` + `models/metadata.json`).
L'application (`app.py`) charge ces artefacts déjà entraînés au
démarrage plutôt que de réentraîner à chaque lancement.

Si aucun modèle compatible n'est disponible au démarrage (première
exécution, ou modèle sauvegardé avec une version de scikit-learn
différente de celle installée), l'application entraîne automatiquement
un nouveau modèle avant de servir la première requête — voir
[Limites](#limites-et-pistes-damélioration) pour le détail de ce
mécanisme.

### Façade unique (`pipeline.py`)

Le code appelant (interface, futur endpoint API...) n'utilise qu'une
seule classe, `SentimentPipeline`, qui gère le chargement de chaque
approche et l'indisponibilité éventuelle de l'une d'elles (ex. modèle
neuronal non installé) sans faire planter les autres.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Entraîner les modèles

Des modèles pré-entraînés sont inclus dans `models/`. Pour les
régénérer (par exemple après avoir modifié `data/dataset.csv`) :

```bash
python train.py --report
```

Options :
- `--model SVM` : n'entraîne qu'un seul modèle (`SVM`, `NaiveBayes` ou
  `LogisticRegression`) au lieu des trois.
- `--report` : affiche le détail précision/rappel/F1 par classe et la
  liste des erreurs de classification.
- `--cv-folds 10` : change le nombre de folds de la validation croisée
  (5 par défaut).

Le script sélectionne automatiquement le meilleur modèle (accuracy
moyenne en validation croisée) et l'enregistre dans
`models/metadata.json` sous la clé `_best_model` : c'est ce modèle que
`SentimentPipeline()` charge par défaut.

## Lancer l'application

```bash
streamlit run app.py
```

Trois onglets :
1. **Analyser un texte** : compare les trois approches sur un texte donné.
2. **Analyse par lot** : dépose un CSV avec une colonne `text`,
   télécharge les résultats enrichis d'une colonne `sentiment_prédit`.
3. **Évaluation** : compare les trois approches entre elles (accuracy,
   précision/rappel/F1, analyse d'erreurs) et les trois modèles
   statistiques entre eux (matrice de confusion), sur le jeu de test.

## Lancer les tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

28 tests unitaires :
- `test_preprocessing.py` : validation d'entrée (texte vide/None/trop long), tokenisation.
- `test_lexical.py` : négation, seuils, contribution mot par mot.
- `test_statistical.py` : entraînement, prédiction, persistance (sauvegarde puis rechargement, identité des prédictions), erreur claire si modèle non entraîné.
- `test_pipeline.py` : façade unique, isolation des erreurs entre approches.

## Déploiement Docker

```bash
docker build -t sentiment-analysis .
docker run -p 8501:8501 sentiment-analysis
```

Le `Dockerfile` entraîne les modèles au moment du build de l'image
(`RUN python train.py`) : l'image finale est autonome et ne nécessite
aucune étape d'entraînement au démarrage du conteneur.

## Collecte de données réelles

`src/sentiment_analysis/scraper.py` collecte des avis spectateurs sur
Allociné : chaque avis est accompagné d'une note en étoiles (0,5 à 5)
convertie automatiquement en label de sentiment (≤2 → négatif, ≥4 →
positif, entre les deux → neutre). C'est la même technique utilisée
pour construire plusieurs datasets publics de sentiment en français.

```bash
python scripts/collect_reviews.py --movie-path "/film/fichefilm-XXXXX/critiques/spectateurs/" --pages 5
python train.py   # réentraîne avec les nouveaux avis ajoutés à data/dataset.csv
```

Remplacer le chemin par celui d'un film sur allocine.fr (visiter le
site, copier le chemin depuis l'URL). Les sélecteurs CSS utilisés pour
extraire le texte et la note sont centralisés dans `_SELECTORS` en tête
de `scraper.py` : si le site fait évoluer sa structure HTML, c'est le
premier endroit à ajuster (inspecter la page avec les outils de
développement du navigateur pour retrouver les bons sélecteurs).

## Approche neuronale

`neural.py` utilise un modèle CamemBERT déjà fine-tuné pour l'analyse
de sentiment en français
(`cmarkea/distilcamembert-base-sentiment`, sur une échelle de 1 à 5
étoiles), convertie vers le schéma à 3 classes du projet. Aucune étape
de fine-tuning n'est nécessaire pour l'utiliser : installer les
dépendances suffit à l'activer.

```bash
pip install transformers torch
streamlit run app.py
```

Pour un modèle adapté au vocabulaire propre du projet plutôt qu'au
modèle générique par défaut, `scripts/train_neural.py` permet de
fine-tuner sur `data/dataset.csv` (API `Trainer` standard de Hugging
Face) et de substituer le checkpoint obtenu via la variable
d'environnement `SENTIMENT_NEURAL_CHECKPOINT` :

```bash
pip install transformers torch datasets
python scripts/train_neural.py --checkpoint camembert-base --epochs 4
export SENTIMENT_NEURAL_CHECKPOINT=./models/neural_sentiment
streamlit run app.py
```

## Augmentation du dataset

Le dataset combine des exemples écrits à la main et des phrases
générées par `scripts/generate_synthetic_dataset.py`, réparties sur une
vingtaine de domaines (produits tech, restaurants, hôtellerie, service
client, livraison, banque/assurance, santé, éducation, transport,
immobilier, télécom, sport, automobile, beauté, administration,
culture, emploi, logiciels, alimentation, voyage), avec gestion des
accords grammaticaux et restrictions sémantiques par gabarit pour
éviter les combinaisons incohérentes.

### Interpréter les métriques après augmentation

En entraînant sur le dataset combiné, l'accuracy en validation croisée
grimpe autour de 0.99 — un chiffre à interpréter avec prudence : les
phrases générées contiennent des formules de conclusion largement
exclusives à leur classe ("Je recommande vivement." n'apparaît que
dans les exemples positifs, "À éviter absolument." que dans les
négatifs), ce qui facilite artificiellement la tâche de classification.

`scripts/benchmark_augmentation.py` mesure l'apport réel de
l'augmentation sans cette fuite : entraînement uniquement sur les
données synthétiques, évaluation uniquement sur les exemples écrits à
la main (jamais vus à l'entraînement) :

```
Accuracy (train=synthétique seul, test=exemples manuscrits) : 0.80
  négatif    précision=0.73 rappel=0.68 f1=0.70
  neutre     précision=0.92 rappel=0.85 f1=0.88
  positif    précision=0.76 rappel=0.88 f1=0.81
```

0.80 contre 0.42-0.46 sur le dataset manuscrit seul (avant
augmentation) : c'est ce chiffre, mesuré sans fuite entre train et
test, qui reflète l'apport réel de l'augmentation — pas le 0.99 obtenu
par simple validation croisée sur le mélange.

## Résultats

Validation croisée à 5 folds sur le dataset manuscrit initial (avant
augmentation par données synthétiques) :

```
NaiveBayes           accuracy test = 0.46   CV (5-fold) = 0.45
SVM                  accuracy test = 0.42   CV (5-fold) = 0.40
LogisticRegression   accuracy test = 0.46   CV (5-fold) = 0.42
```

Ces scores modestes reflètent la petite taille du dataset initial (120
exemples) et l'absence de lemmatisation avancée sans spaCy installé
(le fallback de tokenisation ne relie pas "améliore" / "amélioré" /
"améliorée", ce qui fragmente le vocabulaire TF-IDF). Installer spaCy
et `fr_core_news_sm` améliore la qualité du prétraitement partagé par
les approches lexicale et statistique.

## Limites et pistes d'amélioration

- **Ré-entraînement automatique au démarrage** : si aucun modèle
  compatible n'est trouvé (absent, ou sérialisé avec une version de
  scikit-learn différente de celle installée), l'application entraîne
  un nouveau modèle avant de répondre à la première requête. Premier
  démarrage plus lent dans ce cas ; les démarrages suivants utilisent
  le modèle mis en cache. Pour éviter tout aléa de compatibilité entre
  environnements, entraîner localement et committer les artefacts
  `models/*.joblib` correspondant à la version de scikit-learn utilisée
  en déploiement.
- **Dataset synthétique majoritaire** : le déséquilibre entre exemples
  manuscrits et générés reste une limite pour un déploiement réel.
  Prochaine étape : remplacer progressivement le synthétique par des
  avis collectés via `scraper.py` et rééquilibrer.
- **Lexique fait main (154 mots)** : à remplacer par un lexique publié
  et validé linguistiquement (FEEL, Blogoscopie, Polarimots) pour un
  usage réel — le chargement externe (`data/lexicon_fr.csv`) accueille
  un tel fichier sans changement de code.
- **Approche neuronale** : fonctionne dès l'installation de
  `transformers`/`torch` grâce à un modèle déjà entraîné pour le
  sentiment ; le fine-tuning (`scripts/train_neural.py`) reste une
  option pour affiner sur le vocabulaire propre du projet, pas un
  prérequis (voir [Approche neuronale](#approche-neuronale)).
- **Pas de CI/CD configuré** (ex. GitHub Actions lançant `pytest` à
  chaque push) : recommandé pour une mise en production continue.
