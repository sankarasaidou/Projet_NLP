# -*- coding: utf-8 -*-
from sentiment_analysis.neural import _label_to_sentiment


class TestLabelToSentiment:
    def test_star_ratings_low_are_negative(self):
        assert _label_to_sentiment("1 star") == "négatif"
        assert _label_to_sentiment("2 stars") == "négatif"

    def test_star_rating_middle_is_neutral(self):
        assert _label_to_sentiment("3 stars") == "neutre"

    def test_star_ratings_high_are_positive(self):
        assert _label_to_sentiment("4 stars") == "positif"
        assert _label_to_sentiment("5 stars") == "positif"

    def test_binary_labels(self):
        assert _label_to_sentiment("POSITIVE") == "positif"
        assert _label_to_sentiment("NEGATIVE") == "négatif"
        assert _label_to_sentiment("positive") == "positif"
        assert _label_to_sentiment("negative") == "négatif"

    def test_explicit_neutral_label(self):
        assert _label_to_sentiment("neutral") == "neutre"

    def test_unrecognized_generic_label_defaults_to_neutral(self):
        # Format 'LABEL_N' générique (sans mot 'star' ni polarité
        # explicite) : ambigu, ne doit jamais être mal interprété comme
        # un nombre d'étoiles -> repli sûr sur 'neutre'.
        assert _label_to_sentiment("LABEL_0") == "neutre"
        assert _label_to_sentiment("LABEL_4") == "neutre"
