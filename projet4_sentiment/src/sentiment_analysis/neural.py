# -*- coding: utf-8 -*-
"""Approche neuronale : classification de sentiment par un modèle
CamemBERT déjà fine-tuné pour cette tâche.

Contrairement à un modèle de base (`camembert-base`) qui n'a jamais vu
de tâche de classification de sentiment et donnerait des prédictions
sans valeur, ce module charge un modèle déjà spécialisé pour l'analyse
de sentiment en français, publié sur le Hub Hugging Face. Aucune étape
de fine-tuning n'est nécessaire pour que l'approche fonctionne :
installer `transformers` et `torch` suffit à l'activer.

Le modèle par défaut (`cmarkea/distilcamembert-base-sentiment`) classe
le texte sur une échelle de 1 à 5 étoiles ; on convertit cette échelle
vers le schéma à 3 classes du projet (négatif / neutre / positif) avec
le même seuillage que celui utilisé pour la collecte d'avis notés
(voir scraper.py) :
    1-2 étoiles -> négatif
    3 étoiles   -> neutre
    4-5 étoiles -> positif

Pour un besoin plus spécifique (vocabulaire de domaine particulier),
`scripts/train_neural.py` permet de fine-tuner un modèle sur les
données propres au projet (`data/dataset.csv`) et de le substituer au
modèle par défaut via la variable d'environnement
`SENTIMENT_NEURAL_CHECKPOINT`.
"""

import re
from functools import lru_cache

from sentiment_analysis.config import NEURAL_MODEL_CHECKPOINT
from sentiment_analysis.preprocessing import clean_text
from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)

_STAR_PATTERN = re.compile(r"(\d)")


class NeuralNotAvailableError(RuntimeError):
    """Levée quand on tente d'utiliser l'approche neuronale sans les
    dépendances requises (transformers, torch), ou quand le modèle ne
    peut pas être chargé (réseau indisponible, checkpoint introuvable)."""


@lru_cache(maxsize=1)
def is_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _label_to_sentiment(raw_label: str) -> str:
    """Convertit le label natif du modèle vers le schéma à 3 classes du
    projet. Gère à la fois un format 'étoiles' ('1 star', '5 stars',
    'LABEL_4'...) et un format binaire positif/négatif, pour rester
    compatible avec plusieurs modèles de sentiment publiés en français."""
    label_lower = raw_label.lower()

    if "pos" in label_lower:
        return "positif"
    if "neg" in label_lower:
        return "négatif"
    if "neutr" in label_lower:
        return "neutre"

    if "star" in label_lower or "étoile" in label_lower:
        match = _STAR_PATTERN.search(label_lower)
        if match:
            stars = int(match.group(1))
            if stars <= 2:
                return "négatif"
            if stars >= 4:
                return "positif"
            return "neutre"

    logger.warning("Label neuronal non reconnu (%r), retour 'neutre' par défaut.", raw_label)
    return "neutre"


class NeuralSentimentClassifier:
    """Classifieur neuronal. Chargement paresseux : le modèle n'est
    chargé en mémoire qu'au premier appel à predict()."""

    def __init__(self, model_path: str = NEURAL_MODEL_CHECKPOINT):
        if not is_available():
            raise NeuralNotAvailableError(
                "L'approche neuronale nécessite `transformers` et `torch` "
                "(non installés) : `pip install transformers torch`"
            )
        self.model_path = model_path
        self._classifier = None

    def _ensure_loaded(self):
        if self._classifier is not None:
            return
        from transformers import pipeline as hf_pipeline

        logger.info("Chargement du modèle neuronal '%s'...", self.model_path)
        try:
            self._classifier = hf_pipeline(
                "text-classification", model=self.model_path, top_k=1,
            )
        except Exception as e:
            raise NeuralNotAvailableError(
                f"Impossible de charger le modèle neuronal '{self.model_path}' : {e}"
            ) from e
        logger.info("Modèle neuronal chargé.")

    def predict(self, texts: list[str]) -> list[dict]:
        self._ensure_loaded()
        clean_texts = [clean_text(t) for t in texts]

        raw_results = self._classifier(clean_texts, truncation=True, max_length=128)

        results = []
        for r in raw_results:
            # top_k=1 retourne soit un dict, soit une liste d'un seul dict
            # selon la version de transformers installée.
            best = r[0] if isinstance(r, list) else r
            results.append({
                "label": _label_to_sentiment(best["label"]),
                "confidence": float(best["score"]),
            })
        return results
