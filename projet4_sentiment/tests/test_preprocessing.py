# -*- coding: utf-8 -*-
import pytest

from sentiment_analysis.preprocessing import (
    clean_text, tokenize_lexical, validate_text, InvalidTextError, MAX_TEXT_LENGTH,
)


class TestValidateText:
    def test_valid_text_returns_stripped(self):
        assert validate_text("  bonjour  ") == "bonjour"

    def test_none_raises(self):
        with pytest.raises(InvalidTextError):
            validate_text(None)

    def test_empty_string_raises(self):
        with pytest.raises(InvalidTextError):
            validate_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(InvalidTextError):
            validate_text("   \n\t  ")

    def test_non_string_raises(self):
        with pytest.raises(InvalidTextError):
            validate_text(12345)

    def test_too_long_raises(self):
        with pytest.raises(InvalidTextError):
            validate_text("a" * (MAX_TEXT_LENGTH + 1))


class TestCleanText:
    def test_collapses_multiple_spaces(self):
        assert clean_text("Un    texte   avec   des espaces") == "Un texte avec des espaces"

    def test_strips_leading_trailing_spaces(self):
        assert clean_text("   texte   ") == "texte"


class TestTokenizeLexical:
    def test_returns_lowercase_tokens(self):
        tokens = tokenize_lexical("Ce Produit Est Excellent")
        assert all(t == t.lower() for t in tokens)

    def test_removes_common_stopwords(self):
        tokens = tokenize_lexical("Le produit est vraiment excellent")
        assert "le" not in tokens
        assert "est" not in tokens

    def test_keeps_meaningful_words(self):
        tokens = tokenize_lexical("Le produit est vraiment excellent")
        assert "produit" in tokens or "excellent" in tokens
