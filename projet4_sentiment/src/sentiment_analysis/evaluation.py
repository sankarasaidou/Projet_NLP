# -*- coding: utf-8 -*-
"""Évaluation des approches : accuracy, précision/rappel/F1 par classe,
matrice de confusion. Sert à la fois pour le rapport de `train.py` et
pour l'onglet "Évaluation" de l'interface Streamlit.
"""

from collections import defaultdict

from sentiment_analysis.config import LABELS
from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)


def classification_report(predictions: list[str], gold: list[str]) -> dict:
    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for pred, true in zip(predictions, gold):
        if pred == true:
            stats[true]["tp"] += 1
        else:
            stats[pred]["fp"] += 1
            stats[true]["fn"] += 1

    report = {}
    for label, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    accuracy = sum(p == g for p, g in zip(predictions, gold)) / len(gold) if gold else 0.0
    return {"accuracy": accuracy, "per_label": report}


def confusion_matrix(predictions: list[str], gold: list[str]) -> dict:
    """Retourne une matrice de confusion sous forme de dict imbriqué :
    matrix[label_reel][label_predit] = nombre de cas."""
    matrix = {true_label: {pred_label: 0 for pred_label in LABELS} for true_label in LABELS}
    for pred, true in zip(predictions, gold):
        matrix[true][pred] += 1
    return matrix


def error_analysis(texts: list[str], predictions: list[str], gold: list[str]) -> list[dict]:
    """Liste les cas mal classés, pour inspection manuelle (indispensable
    en production pour comprendre où le modèle échoue réellement)."""
    return [
        {"text": t, "predicted": p, "expected": g}
        for t, p, g in zip(texts, predictions, gold) if p != g
    ]


def compare_approaches(results_by_approach: dict[str, tuple[list[str], list[str]]]) -> dict:
    """results_by_approach: {"Lexicale": (predictions, gold), "SVM": (...), ...}"""
    comparison = {}
    for approach, (predictions, gold) in results_by_approach.items():
        report = classification_report(predictions, gold)
        comparison[approach] = report
        logger.info("%s : accuracy=%.2f", approach, report["accuracy"])
    return comparison
