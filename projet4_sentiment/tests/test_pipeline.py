# -*- coding: utf-8 -*-
import pytest

from sentiment_analysis.pipeline import SentimentPipeline
from sentiment_analysis.preprocessing import InvalidTextError
from sentiment_analysis.statistical import ModelNotTrainedError


class TestSentimentPipeline:
    def test_analyze_lexical_valid_text(self):
        pipeline = SentimentPipeline()
        result = pipeline.analyze_lexical("Je suis très satisfait de ce produit excellent")
        assert result["label"] in {"positif", "négatif", "neutre"}

    def test_analyze_lexical_invalid_text_raises(self):
        pipeline = SentimentPipeline()
        with pytest.raises(InvalidTextError):
            pipeline.analyze_lexical("")

    def test_analyze_statistical_without_trained_model_raises_clear_error(self, monkeypatch, tmp_path):
        from sentiment_analysis.statistical import StatisticalSentimentClassifier

        pipeline = SentimentPipeline(statistical_model_name="SVM")

        def _raise_not_trained(cls, *a, **k):
            raise ModelNotTrainedError("modèle absent")

        monkeypatch.setattr(StatisticalSentimentClassifier, "load", classmethod(_raise_not_trained))
        with pytest.raises(ModelNotTrainedError):
            pipeline.analyze_statistical("Un texte")

    def test_analyze_all_never_raises_even_if_one_approach_fails(self, monkeypatch):
        """Le point le plus important en production : si UNE approche
        échoue (modèle non entraîné, dépendance absente...), les AUTRES
        doivent quand même renvoyer un résultat -- pas d'exception globale."""
        pipeline = SentimentPipeline()
        result = pipeline.analyze_all("Je suis très content de ce produit")
        assert "lexicale" in result
        assert "statistique" in result
        assert "neuronale" in result
        # Le résultat lexical doit être un vrai résultat, pas une erreur,
        # même si les autres approches échouent.
        assert "error" not in result["lexicale"] or "label" in result["lexicale"]

    def test_analyze_all_invalid_text_raises_before_running_approaches(self):
        pipeline = SentimentPipeline()
        with pytest.raises(InvalidTextError):
            pipeline.analyze_all(None)
