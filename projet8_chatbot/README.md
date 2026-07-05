# Projet 8 — Chatbot FAQ intelligent (NER + intentions) — UV-BF

## Installation
```bash
pip install -r requirements.txt
```

## Étapes -> fichiers
| Étape | Fichier |
|---|---|
| 1. Base de FAQ | `faq_data.py` |
| 2. NER personnalisé | `ner.py` |
| 3. Classification d'intentions | `intent_classifier.py` |
| 4. Matching intelligent | `chatbot.py` |
| 5. Génération de réponses | `chatbot.py` (`_fill_template`) |
| 6. Apprentissage continu | `feedback.py` |
| Interface Streamlit | `app.py` |

## Lancer
```bash
streamlit run app.py
```

## Résultats réels obtenus en test

Le chatbot a été testé de bout en bout (NER + intention + matching +
génération) sur 3 questions non vues à l'entraînement :

```
Q: Comment accéder au cours de NLP ?
  Intention détectée : cours (confiance=0.26)
  Entités : [{'text': 'NLP', 'label': 'MATIERE'}]
  Réponse : Les cours de NLP sont disponibles sur la plateforme
            pédagogique, dans la rubrique 'Mes cours'.

Q: Je veux m'inscrire, quels documents faut-il ?
  Intention détectée : inscription (confiance=0.30)
  Réponse : Il vous faut : diplôme ou attestation, relevés de notes...

Q: Ma vidéo de cours d'informatique ne se charge pas.
  Intention détectée : technique (confiance=0.35)
  Entités : [{'text': 'informatique', 'label': 'MATIERE'}]
  Réponse : Essayez de rafraîchir la page ou de changer de navigateur...
```

Les 3 cas sont correctement traités. **Un cas d'erreur réel a aussi été
observé** en testant `intent_classifier.py` isolément : « Le cours de
statistiques ne s'affiche pas. » a été classé `administratif` au lieu de
`technique`. C'est un point à documenter dans le rapport : avec
seulement 15 questions-types d'entraînement (3 par intention), le
classifieur confond parfois des intentions proches sémantiquement
(problème technique lié à un cours vs démarche administrative). La
solution : enrichir `faq_data.py` avec plus d'exemples par intention,
ou utiliser le mécanisme de feedback (`feedback.py`) pour corriger ces
cas au fil de l'usage réel.

## Pourquoi masquer les entités avant classification d'intention ?

`intent_classifier.py` remplace "NLP", "certificat de scolarité", etc.
par un placeholder générique avant classification. Sans ça, le
classifieur apprendrait à associer chaque matière/document à une
intention selon les hasards du petit jeu d'entraînement (ex. si "NLP"
n'apparaît que dans des questions "cours", le modèle pourrait
apprendre à tort que voir "NLP" = intention "cours", même dans une
question technique sur NLP). Le masquage force le modèle à se
concentrer sur la STRUCTURE de la question, pas sur l'entité mentionnée.

## Apprentissage continu (étape 6)

Le mécanisme de feedback est fonctionnel dans l'app Streamlit : chaque
réponse peut être notée 👍/👎. Les cas négatifs sont listés dans
l'onglet "Tableau de bord" pour être examinés (et potentiellement
ajoutés à la base d'entraînement via `feedback.retrain_with_correction()`).

## Limites de cette version
- FAQ très réduite (15 entrées, 5 intentions) : à étendre avec les
  vraies FAQ de l'UV-BF pour un usage réel.
- Le stockage du feedback est en mémoire (liste Python / session
  Streamlit) : à faire persister en base de données pour un vrai
  déploiement.
- Le "matching intelligent" utilise TF-IDF + cosinus, simple mais
  efficace sur un petit corpus fermé ; sur une FAQ plus large, des
  embeddings de phrase (Sentence-BERT) amélioreraient la robustesse aux
  reformulations.
