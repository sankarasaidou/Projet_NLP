# -*- coding: utf-8 -*-
"""
ÉTAPE 6 : Évaluation comparative
--------------------------------------
Compare les 3 approches (lexicale, statistique, neuronale) sur le même
jeu de test, avec les mêmes métriques : accuracy, précision/rappel/F1
par classe, et une analyse d'erreurs (exemples mal classés).
"""

from collections import defaultdict

from lexical_approach import analyze_lexical
from statistical_approach import StatisticalSentimentClassifier


def _prf_by_label(predictions, gold, labels):
    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for pred, true in zip(predictions, gold):
        if pred == true:
            stats[true]["tp"] += 1
        else:
            stats[pred]["fp"] += 1
            stats[true]["fn"] += 1

    report = {}
    for label, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        report[label] = {"precision": p, "recall": r, "f1": f1}
    return report


def evaluate_all(train_data, test_data, statistical_model_name="SVM"):
    test_texts = [t for t, _ in test_data]
    gold_labels = [l for _, l in test_data]
    labels = sorted(set(gold_labels))

    # --- Lexicale (pas d'entraînement nécessaire) ---
    lexical_preds = [analyze_lexical(t)["label"] for t in test_texts]

    # --- Statistique (entraînée sur train_data) ---
    train_texts = [t for t, _ in train_data]
    train_labels = [l for _, l in train_data]
    clf = StatisticalSentimentClassifier(statistical_model_name).fit(train_texts, train_labels)
    statistical_preds = clf.predict(test_texts)

    results = {}
    for approach_name, preds in [("Lexicale", lexical_preds), ("Statistique", statistical_preds)]:
        accuracy = sum(p == g for p, g in zip(preds, gold_labels)) / len(gold_labels)
        report = _prf_by_label(preds, gold_labels, labels)
        errors = [
            {"text": t, "gold": g, "predicted": p}
            for t, g, p in zip(test_texts, gold_labels, preds) if g != p
        ]
        results[approach_name] = {"accuracy": accuracy, "report": report, "errors": errors, "predictions": preds}

    # --- Neuronale : nécessite transformers/torch, souvent absent -> optionnel ---
    try:
        from neural_approach import predict_neural
        neural_preds = predict_neural(test_texts)
        accuracy = sum(p == g for p, g in zip(neural_preds, gold_labels)) / len(gold_labels)
        report = _prf_by_label(neural_preds, gold_labels, labels)
        errors = [
            {"text": t, "gold": g, "predicted": p}
            for t, g, p in zip(test_texts, gold_labels, neural_preds) if g != p
        ]
        results["Neuronale"] = {"accuracy": accuracy, "report": report, "errors": errors, "predictions": neural_preds}
    except Exception as e:
        results["Neuronale"] = {"error": str(e)}

    return results


if __name__ == "__main__":
    from corpus import TRAIN_DATA, TEST_DATA

    results = evaluate_all(TRAIN_DATA, TEST_DATA)
    for approach, r in results.items():
        print(f"\n=== {approach} ===")
        if "error" in r:
            print(f"  Non disponible : {r['error']}")
            continue
        print(f"  Accuracy : {r['accuracy']:.2f}")
        for label, m in r["report"].items():
            print(f"  {label:<10} précision={m['precision']:.2f} rappel={m['recall']:.2f} f1={m['f1']:.2f}")
        if r["errors"]:
            print("  Erreurs :")
            for e in r["errors"]:
                print(f"    {e['text']!r} -> prédit {e['predicted']}, attendu {e['gold']}")
