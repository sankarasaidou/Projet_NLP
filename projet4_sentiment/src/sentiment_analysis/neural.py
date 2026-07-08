# -*- coding: utf-8 -*-
"""Approche neuronale : fine-tuning / inférence CamemBERT pour la
classification de sentiment à 3 classes.

Dépendance lourde et OPTIONNELLE (transformers + torch) : le reste de
l'application (lexicale + statistique + interface) doit fonctionner
parfaitement même si ces paquets ne sont pas installés. On centralise
donc ici la détection de disponibilité (`is_available()`), pour que le
reste du code (pipeline.py, app.py) n'ait qu'à vérifier un booléen au
lieu de dupliquer des try/except partout.
"""

from functools import lru_cache

from sentiment_analysis.config import NEURAL_MODEL_CHECKPOINT, LABELS
from sentiment_analysis.preprocessing import clean_text
from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


class NeuralNotAvailableError(RuntimeError):
    """Levée quand on tente d'utiliser l'approche neuronale sans les
    dépendances requises (transformers, torch)."""


@lru_cache(maxsize=1)
def is_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


class NeuralSentimentClassifier:
    """Classifieur neuronal basé sur un modèle CamemBERT pré-entraîné
    (fine-tuné au préalable, voir train.py --with-neural). Chargement
    paresseux : le modèle n'est chargé en mémoire qu'au premier appel."""

    def __init__(self, model_path: str = NEURAL_MODEL_CHECKPOINT):
        if not is_available():
            raise NeuralNotAvailableError(
                "L'approche neuronale nécessite `transformers` et `torch` "
                "(non installés dans cet environnement) : "
                "`pip install transformers torch`"
            )
        self.model_path = model_path
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info("Chargement du modèle neuronal depuis %s...", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        )
        self._model.eval()
        self._torch = torch
        logger.info("Modèle neuronal chargé.")

    def predict(self, texts: list[str]) -> list[dict]:
        self._ensure_loaded()
        clean_texts = [clean_text(t) for t in texts]
        inputs = self._tokenizer(
            clean_texts, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = self._torch.softmax(logits, dim=-1)

        results = []
        for probs in probabilities:
            pred_id = int(probs.argmax())
            results.append({"label": ID2LABEL[pred_id], "confidence": float(probs[pred_id])})
        return results
