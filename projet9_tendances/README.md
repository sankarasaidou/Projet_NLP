# Projet 9 — Analyseur de tendances sur les commentaires (lefaso.net)

## Installation
```bash
pip install -r requirements.txt
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Collecte de données | `comments_data.py` |
| 2. Nettoyage spécialisé | `preprocessing.py` |
| 3. Détection de tendances | `trends.py` |
| 4. Analyse de sentiment (+ neutre/libre) | `sentiment.py` |
| 5. Visualisations | `visualization.py` |
| 6. Alertes automatiques | `trends.py` (`detect_spikes`) + `app.py` |
| Interface Streamlit | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Interprétation du sujet : "neutres et/ou libres"

Le sujet demande de voir si les commentaires sont **neutres** et/ou
**libres**. On a interprété ça comme une classification complémentaire
au sentiment classique : un commentaire est **"neutre"** s'il est
purement factuel (ex. « Le budget alloué est mentionné dans le
communiqué officiel. »), et **"libre"** s'il exprime une opinion
personnelle (marqueurs : pronoms je/on, ponctuation expressive, emoji,
hashtag). **Testé réellement** : les 4 commentaires factuels du corpus
de démo sont bien classés "neutre", tous les autres (avec émotion,
hashtag ou exclamation) sont classés "libre".

## Résultats réels obtenus en test

- **Nettoyage spécialisé** validé sur un cas réel : `"C koi ce plan
  encore ?? ... #déception 😒"` → texte nettoyé
  `"c'est quoi ce plan encore..."`, hashtag `déception` et emoji `😒`
  correctement extraits séparément.
- **Tendances** : hashtag le plus mentionné = `#agriculture` (4
  occurrences), avec un **pic réel détecté** le 7 juin 2026 (2
  mentions contre 1 en moyenne les autres jours) → alerte automatique
  déclenchée avec succès dans `detect_spikes()`.
- **Sentiment** : les commentaires positifs/négatifs/neutres sont
  correctement discriminés, y compris grâce aux emojis (souvent plus
  fiables que le texte seul sur des commentaires courts).

## Limites de cette version
- Corpus de démo (15 commentaires sur 8 jours) : volontairement petit
  pour vérifier chaque résultat à la main ; un vrai monitoring
  utiliserait un flux continu de commentaires scrapés.
- Le scraping réel des commentaires de lefaso.net nécessiterait
  probablement Selenium/Playwright (contenu souvent chargé en
  JavaScript), non couvert ici — `scraper.py` du projet 2 couvre le
  scraping d'articles statiques, à adapter.
- Le dictionnaire de polarité et les règles de subjectivité sont
  volontairement simples (mêmes limites que le projet 4 : couverture
  lexicale restreinte, à enrichir pour un usage réel).
- `wordcloud` est optionnel : sans lui, un graphique en barres des
  termes les plus fréquents est utilisé en remplacement (testé et
  fonctionnel).
