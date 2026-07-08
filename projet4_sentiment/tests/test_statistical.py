# -*- coding: utf-8 -*-
import pytest

from sentiment_analysis.statistical import StatisticalSentimentClassifier, ModelNotTrainedError

TRAIN_TEXTS = [
    "Excellent produit, je suis ravi", "Service exceptionnel, merci beaucoup",
    "Produit décevant, je suis déçu", "Service catastrophique, à éviter",
    "Le colis est arrivé aujourd'hui", "La commande contient deux articles",
] * 5  # répété pour avoir assez d'exemples par classe pour TF-IDF/CV
TRAIN_LABELS = ["positif", "positif", "négatif", "négatif", "neutre", "neutre"] * 5


class TestStatisticalSentimentClassifier:
    def test_invalid_model_name_raises(self):
        with pytest.raises(ValueError):
            StatisticalSentimentClassifier("modele_inexistant")

    def test_predict_before_fit_raises(self):
        classifier = StatisticalSentimentClassifier("SVM")
        with pytest.raises(ModelNotTrainedError):
            classifier.predict(["un texte"])

    def test_fit_and_predict(self):
        classifier = StatisticalSentimentClassifier("SVM").fit(TRAIN_TEXTS, TRAIN_LABELS)
        predictions = classifier.predict(["Je suis très content, excellent service"])
        assert predictions[0] in {"positif", "négatif", "neutre"}

    def test_predict_proba_sums_to_one(self):
        classifier = StatisticalSentimentClassifier("SVM").fit(TRAIN_TEXTS, TRAIN_LABELS)
        proba = classifier.predict_proba(["Un texte quelconque"])
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_save_and_load_roundtrip(self, tmp_path):
        classifier = StatisticalSentimentClassifier("NaiveBayes").fit(TRAIN_TEXTS, TRAIN_LABELS)
        model_path = tmp_path / "model.joblib"
        metadata_path = tmp_path / "metadata.json"
        classifier.save(base_path=model_path, metadata_path=metadata_path)

        reloaded = StatisticalSentimentClassifier.load(model_name="NaiveBayes", base_path=model_path)
        original_preds = classifier.predict(["Je suis content"])
        reloaded_preds = reloaded.predict(["Je suis content"])
        assert original_preds == reloaded_preds

    def test_load_missing_model_raises_clear_error(self, tmp_path):
        with pytest.raises(ModelNotTrainedError):
            StatisticalSentimentClassifier.load(model_name="SVM", base_path=tmp_path / "inexistant.joblib")
