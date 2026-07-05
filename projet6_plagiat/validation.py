# -*- coding: utf-8 -*-
"""
ÉTAPE 6 : Validation
-------------------------
Applique les seuils optimisés (étape 4) sur les cas connus (étape 1) et
rapporte, pour chaque métrique, si chaque cas est correctement détecté.
C'est la validation finale demandée : "tests sur des cas connus de
plagiat et de contenus originaux".
"""

from data import DOCUMENTS, KNOWN_CASES
from similarity_metrics import METRICS
from threshold_optimization import find_optimal_thresholds


def validate(known_cases=None):
    known_cases = known_cases or KNOWN_CASES
    thresholds = find_optimal_thresholds(known_cases)

    report = {}
    for metric_name, metric_fn in METRICS.items():
        threshold = thresholds[metric_name]["threshold"]
        case_results = []
        for doc_a, doc_b, expected in known_cases:
            score = metric_fn(DOCUMENTS[doc_a], DOCUMENTS[doc_b])
            predicted = score >= threshold
            case_results.append({
                "doc_a": doc_a, "doc_b": doc_b,
                "score": round(score, 3), "expected": expected,
                "predicted": predicted, "correct": predicted == expected,
            })
        report[metric_name] = {
            "threshold": threshold,
            "cases": case_results,
            "accuracy": sum(c["correct"] for c in case_results) / len(case_results),
        }
    return report


if __name__ == "__main__":
    report = validate()
    for metric_name, info in report.items():
        print(f"\n=== {metric_name} (seuil = {info['threshold']:.2f}) — accuracy = {info['accuracy']:.2f} ===")
        for case in info["cases"]:
            status = "✅" if case["correct"] else "❌"
            print(f"  {status} {case['doc_a']} vs {case['doc_b']}: score={case['score']:.3f} "
                  f"prédit={case['predicted']} attendu={case['expected']}")
