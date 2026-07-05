# -*- coding: utf-8 -*-
"""
ÉTAPE 6 : Évaluation du modèle
----------------------------------
On charge le pipeline entraîné, on l'exécute sur CORPUS_TEST (jamais vu
à l'entraînement) et on compare les entités prédites aux entités gold
(vérité terrain annotée à la main dans data/corpus.py).

Métriques calculées, par label ET globalement :
    - Précision = TP / (TP + FP)
    - Rappel    = TP / (TP + FN)
    - F1-score  = moyenne harmonique de précision et rappel

Une correspondance est comptée comme "vraie" (TP) seulement si le span
(start, end) ET le label prédits sont EXACTEMENT identiques au gold.
C'est la mesure la plus stricte (spaCy Scorer utilise le même principe).
"""

from collections import defaultdict

import spacy

from train_ner import MODEL_DIR, regex_entity_component  # noqa: F401 (enregistre le composant)
from data.corpus import CORPUS_TEST


def evaluate(nlp=None):
    if nlp is None:
        nlp = spacy.load(MODEL_DIR)

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for text, gold_entities in CORPUS_TEST:
        doc = nlp(text)
        predicted = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}
        gold = {(start, end, label) for start, end, label in gold_entities}

        for ent in predicted & gold:
            stats[ent[2]]["tp"] += 1
        for ent in predicted - gold:
            stats[ent[2]]["fp"] += 1
        for ent in gold - predicted:
            stats[ent[2]]["fn"] += 1

    print(f"{'LABEL':<15}{'Précision':>12}{'Rappel':>10}{'F1-score':>10}{'Support':>10}")
    print("-" * 57)

    total_tp = total_fp = total_fn = 0
    for label in sorted(stats):
        tp, fp, fn = stats[label]["tp"], stats[label]["fp"], stats[label]["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = tp + fn
        print(f"{label:<15}{precision:>12.2f}{recall:>10.2f}{f1:>10.2f}{support:>10}")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print("-" * 57)
    print(f"{'GLOBAL':<15}{precision:>12.2f}{recall:>10.2f}{f1:>10.2f}{total_tp + total_fn:>10}")

    return {"precision": precision, "recall": recall, "f1": f1, "by_label": dict(stats)}


if __name__ == "__main__":
    evaluate()
