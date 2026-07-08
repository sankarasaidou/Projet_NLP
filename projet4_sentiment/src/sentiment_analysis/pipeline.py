# -*- coding: utf-8 -*-
"""Façade unique du système d'analyse de sentiment.

En production, le code appelant (interface Streamlit, endpoint API...)
ne devrait jamais avoir à connaître les détails de chaque approche : il
appelle `SentimentPipeline`, qui gère le chargement, le cache et les
erreurs de façon centralisée. C'est le seul point d'entrée public
recommandé du package.
"""

from sentiment_analysis.config import DEFAULT_STATISTICAL_MODEL
from sentiment_analysis.lexical import LexicalSentimentAnalyzer
from sentiment_analysis.statistical import StatisticalSentimentClassifier, ModelNotTrainedError, get_best_model_name
from sentiment_analysis.neural import NeuralSentimentClassifier, NeuralNotAvailableError, is_available as neural_is_available
from sentiment_analysis.preprocessing import validate_text, InvalidTextError
from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)


class SentimentPipeline:
    """Point d'entrée unique. Charge chaque approche paresseusement (au
    premier besoin) et met en cache l'instance pour les appels suivants."""

    def __init__(self, statistical_model_name: str | None = None):
        # Si aucun modèle n'est explicitement demandé, on utilise le
        # meilleur modèle déterminé par validation croisée lors du
        # dernier entraînement (train.py), plutôt qu'un choix arbitraire.
        self.statistical_model_name = statistical_model_name or get_best_model_name(DEFAULT_STATISTICAL_MODEL)
        self._lexical: LexicalSentimentAnalyzer | None = None
        self._statistical: StatisticalSentimentClassifier | None = None
        self._neural: NeuralSentimentClassifier | None = None

    @property
    def lexical(self) -> LexicalSentimentAnalyzer:
        if self._lexical is None:
            self._lexical = LexicalSentimentAnalyzer()
        return self._lexical

    @property
    def statistical(self) -> StatisticalSentimentClassifier:
        if self._statistical is None:
            self._statistical = StatisticalSentimentClassifier.load(self.statistical_model_name)
        return self._statistical

    @property
    def neural(self) -> NeuralSentimentClassifier:
        if self._neural is None:
            self._neural = NeuralSentimentClassifier()
        return self._neural

    def analyze_lexical(self, text: str) -> dict:
        text = validate_text(text)
        return self.lexical.analyze(text)

    def analyze_statistical(self, text: str) -> dict:
        text = validate_text(text)
        try:
            prediction = self.statistical.predict([text])[0]
            proba = self.statistical.predict_proba([text])[0]
            confidence = float(max(proba))
            return {"label": prediction, "confidence": confidence, "model": self.statistical_model_name}
        except ModelNotTrainedError as e:
            logger.error("Modèle statistique non disponible : %s", e)
            raise

    def analyze_neural(self, text: str) -> dict:
        text = validate_text(text)
        if not neural_is_available():
            raise NeuralNotAvailableError(
                "L'approche neuronale nécessite `transformers` et `torch` (non installés)."
            )
        result = self.neural.predict([text])[0]
        return result

    def analyze_all(self, text: str) -> dict:
        """Exécute les 3 approches et retourne un résultat consolidé,
        en gérant individuellement l'indisponibilité de chacune (une
        approche indisponible ne doit jamais faire planter les autres)."""
        text = validate_text(text)
        results = {}

        try:
            results["lexicale"] = self.analyze_lexical(text)
        except Exception as e:
            logger.exception("Erreur approche lexicale")
            results["lexicale"] = {"error": str(e)}

        try:
            results["statistique"] = self.analyze_statistical(text)
        except Exception as e:
            results["statistique"] = {"error": str(e)}

        try:
            results["neuronale"] = self.analyze_neural(text)
        except Exception as e:
            results["neuronale"] = {"error": str(e)}

        return results


__all__ = [
    "SentimentPipeline",
    "InvalidTextError",
    "ModelNotTrainedError",
    "NeuralNotAvailableError",
]
