# -*- coding: utf-8 -*-
"""Fixtures partagées par tous les tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture(scope="session")
def sample_lexicon():
    return {
        "excellent": 2.0, "content": 1.0, "satisfait": 1.5,
        "mauvais": -1.5, "déçu": -2.0, "nul": -1.8,
    }
