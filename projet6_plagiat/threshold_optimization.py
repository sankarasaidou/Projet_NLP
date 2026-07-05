# -*- coding: utf-8 -*-
"""
ÉTAPE 4 : Optimisation des seuils
--------------------------------------
Pour chaque métrique, on teste une grille de seuils (0.05 à 0.95) sur les
cas connus (data.py -> KNOWN_CASES) et on garde le seuil qui maximise le
F1-score de détection de plagiat (compromis précision/rappel). C'est une
détermination EMPIRIQUE comme demandé dans le sujet : pas de seuil choisi
arbitrairement, mais optimisé sur des exemples annotés.
"""

import numpy as np

from data import DOCUMENTS, KNOWN_CASES
from similarity_metrics import METRICS


def _f1_at_threshold(scores, labels, threshold):
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def find_optimal_thresholds(known_cases=None, step=0.05):
    known_cases = known_cases or KNOWN_CASES
    labels = [expected for _, _, expected in known_cases]

    best_thresholds = {}
    for metric_name, metric_fn in METRICS.items():
        scores = [metric_fn(DOCUMENTS[a], DOCUMENTS[b]) for a, b, _ in known_cases]

        best_threshold, best_f1, best_stats = 0.5, -1, None
        for threshold in np.arange(0.05, 1.0, step):
            stats = _f1_at_threshold(scores, labels, threshold)
            if stats["f1"] > best_f1:
                best_threshold, best_f1, best_stats = threshold, stats["f1"], stats

        best_thresholds[metric_name] = {
            "threshold": round(float(best_threshold), 2),
            "f1": best_f1,
            "stats": best_stats,
            "scores": scores,
        }
    return best_thresholds


if __name__ == "__main__":
    thresholds = find_optimal_thresholds()
    for metric, info in thresholds.items():
        print(f"{metric:<22} seuil optimal = {info['threshold']:.2f}  F1 = {info['f1']:.2f}  "
              f"(TP={info['stats']['tp']} FP={info['stats']['fp']} FN={info['stats']['fn']} TN={info['stats']['tn']})")
