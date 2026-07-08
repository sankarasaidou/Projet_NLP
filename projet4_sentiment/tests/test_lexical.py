# -*- coding: utf-8 -*-
from sentiment_analysis.lexical import LexicalSentimentAnalyzer


class TestLexicalSentimentAnalyzer:
    def test_positive_text(self, sample_lexicon):
        analyzer = LexicalSentimentAnalyzer(lexicon=sample_lexicon)
        result = analyzer.analyze("Je suis très content de ce produit excellent")
        assert result["label"] == "positif"
        assert result["score"] > 0

    def test_negative_text(self, sample_lexicon):
        analyzer = LexicalSentimentAnalyzer(lexicon=sample_lexicon)
        result = analyzer.analyze("Ce produit est vraiment mauvais et nul")
        assert result["label"] == "négatif"
        assert result["score"] < 0

    def test_neutral_text_no_lexicon_words(self, sample_lexicon):
        analyzer = LexicalSentimentAnalyzer(lexicon=sample_lexicon)
        result = analyzer.analyze("Le colis est arrivé aujourd'hui à midi")
        assert result["label"] == "neutre"
        assert result["score"] == 0

    def test_negation_inverts_polarity(self, sample_lexicon):
        analyzer = LexicalSentimentAnalyzer(lexicon=sample_lexicon)
        without_negation = analyzer.analyze("Je suis satisfait")
        with_negation = analyzer.analyze("Je ne suis pas satisfait")
        assert without_negation["score"] > 0
        assert with_negation["score"] < without_negation["score"]

    def test_details_contain_contributing_words(self, sample_lexicon):
        analyzer = LexicalSentimentAnalyzer(lexicon=sample_lexicon)
        result = analyzer.analyze("Ce produit est excellent")
        words_found = [d["word"] for d in result["details"]]
        assert "excellent" in words_found

    def test_custom_threshold_changes_classification(self, sample_lexicon):
        # Un score faiblement positif peut devenir "neutre" avec un seuil plus élevé
        analyzer_low = LexicalSentimentAnalyzer(lexicon=sample_lexicon, neutral_threshold=0.1)
        analyzer_high = LexicalSentimentAnalyzer(lexicon=sample_lexicon, neutral_threshold=5.0)
        text = "Je suis content"  # score = 1.0 avec ce lexique
        assert analyzer_low.analyze(text)["label"] == "positif"
        assert analyzer_high.analyze(text)["label"] == "neutre"
