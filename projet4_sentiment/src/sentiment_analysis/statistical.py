# -*- coding: utf-8 -*-
"""Approche statistique : TF-IDF + classifieur (SVM / Naive Bayes /
Régression logistique).

DIFFÉRENCE CLÉ avec la version précédente (non-production) : le modèle
est entraîné UNE SEULE FOIS via `train.py`, puis SÉRIALISÉ sur disque
(joblib). L'application (Streamlit ou API) ne fait que CHARGER le
modèle déjà entraîné -> démarrage quasi instantané, comportement
reproductible, et on peut versionner/auditer le modèle exact utilisé en
production (fichier .joblib + métadonnées horodatées).
"""

import json
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment_analysis.config import STATISTICAL_MODEL_PATH, METADATA_PATH, RANDOM_STATE
from sentiment_analysis.preprocessing import tokenize_lexical
from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)

MODEL_FACTORIES = {
    "NaiveBayes": lambda: MultinomialNB(),
    "SVM": lambda: SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
}


class ModelNotTrainedError(RuntimeError):
    """Levée quand on essaie de charger un modèle qui n'a jamais été entraîné."""


def get_best_model_name(default: str = "SVM") -> str:
    """Retourne le nom du meilleur modèle selon la validation croisée
    enregistrée par train.py (metadata.json -> "_best_model"), ou
    `default` si aucun entraînement comparatif n'a encore été fait."""
    if not METADATA_PATH.exists():
        return default
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return metadata.get("_best_model", default)


def _tokenized_text(text: str) -> str:
    return " ".join(tokenize_lexical(text))


class StatisticalSentimentClassifier:
    def __init__(self, model_name: str = "SVM"):
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Modèle inconnu : {model_name}. Choix : {list(MODEL_FACTORIES)}")
        self.model_name = model_name
        self.pipeline: Pipeline | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "StatisticalSentimentClassifier":
        logger.info("Entraînement du modèle statistique '%s' sur %d exemples...", self.model_name, len(texts))
        clean_texts = [_tokenized_text(t) for t in texts]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MODEL_FACTORIES[self.model_name]()),
        ])
        self.pipeline.fit(clean_texts, labels)
        logger.info("Entraînement terminé.")
        return self

    def predict(self, texts: list[str]) -> list[str]:
        self._check_fitted()
        clean_texts = [_tokenized_text(t) for t in texts]
        return list(self.pipeline.predict(clean_texts))

    def predict_proba(self, texts: list[str]):
        self._check_fitted()
        clean_texts = [_tokenized_text(t) for t in texts]
        return self.pipeline.predict_proba(clean_texts)

    def _check_fitted(self):
        if self.pipeline is None:
            raise ModelNotTrainedError(
                "Ce modèle n'a pas été entraîné (fit()) ni chargé (load())."
            )

    # --- Persistance ---
    # Chaque modèle (SVM, NaiveBayes, LogisticRegression) est sauvegardé
    # dans un fichier distinct (suffixe par nom de modèle) : entraîner
    # plusieurs modèles pour les comparer ne doit jamais écraser les
    # autres modèles déjà entraînés.
    def _model_path(self, base_path=STATISTICAL_MODEL_PATH):
        return base_path.with_name(f"{base_path.stem}_{self.model_name}{base_path.suffix}")

    def save(self, base_path=STATISTICAL_MODEL_PATH, metadata_path=METADATA_PATH) -> None:
        self._check_fitted()
        path = self._model_path(base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model_name": self.model_name, "pipeline": self.pipeline, "sklearn_version": sklearn.__version__},
            path,
        )

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[self.model_name] = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "path": str(path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Modèle '%s' sauvegardé dans %s", self.model_name, path)

    @classmethod
    def load(cls, model_name: str = "SVM", base_path=STATISTICAL_MODEL_PATH) -> "StatisticalSentimentClassifier":
        instance = cls(model_name=model_name)
        path = instance._model_path(base_path)
        if not path.exists():
            raise ModelNotTrainedError(
                f"Aucun modèle '{model_name}' entraîné trouvé à {path}. "
                "Lance d'abord `python train.py` pour entraîner et sauvegarder les modèles."
            )
        payload = joblib.load(path)
        instance.pipeline = payload["pipeline"]

        saved_version = payload.get("sklearn_version")
        if saved_version and saved_version != sklearn.__version__:
            logger.warning(
                "Modèle '%s' sauvegardé avec scikit-learn %s, environnement actuel %s : "
                "test de compatibilité...", model_name, saved_version, sklearn.__version__,
            )

        # Un modèle sérialisé avec une version de scikit-learn différente
        # peut se charger sans erreur mais planter à la première prédiction
        # (incompatibilité binaire interne, en particulier pour SVC). On le
        # détecte ici avec une prédiction test plutôt que de laisser
        # planter la première vraie requête utilisateur.
        try:
            instance.pipeline.predict(["test de compatibilité"])
        except Exception as e:
            raise ModelNotTrainedError(
                f"Le modèle '{model_name}' sauvegardé est incompatible avec la version "
                f"de scikit-learn installée ici ({sklearn.__version__}, modèle sauvegardé "
                f"avec {saved_version or 'version inconnue'}). Relance `python train.py` "
                f"pour le ré-entraîner dans cet environnement."
            ) from e

        logger.info("Modèle '%s' chargé depuis %s", instance.model_name, path)
        return instance
