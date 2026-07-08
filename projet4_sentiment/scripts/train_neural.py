#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ÉTAPE 5 : Fine-tuning de modèles pré-entraînés (CamemBERT / FlauBERT)
-----------------------------------------------------------------------------
Ce script effectue le VRAI fine-tuning (contrairement à
`src/sentiment_analysis/neural.py` qui ne fait que de l'inférence sur un
modèle déjà fine-tuné). À exécuter une fois pour produire le modèle que
`neural.py` chargera ensuite.

Usage :
    python scripts/train_neural.py --checkpoint camembert-base --epochs 4
    python scripts/train_neural.py --checkpoint flaubert/flaubert_base_cased

Nécessite `transformers`, `torch`, `datasets` (voir requirements.txt,
section optionnelle). Idéalement sur GPU (un fine-tuning BERT sur CPU
est possible mais lent : compter plusieurs dizaines de minutes selon la
taille du dataset).

IMPORTANT — Honnêteté sur les limites de ce livrable : ce script n'a
PAS pu être exécuté dans l'environnement de préparation (pas de GPU, et
`transformers`/`torch` non installables faute d'accès réseau dans ce
bac à sable). Le code suit strictement l'API standard `Trainer` de
Hugging Face et a été relu attentivement, mais **tu dois le lancer et
le vérifier toi-même** avant de considérer l'approche neuronale comme
opérationnelle. Une fois l'entraînement terminé, active l'approche avec
la variable d'environnement `SENTIMENT_ENABLE_NEURAL=true` (voir
config.py) -- elle reste désactivée par défaut tant qu'aucun modèle
fine-tuné n'existe, pour éviter de charger silencieusement un modèle
non entraîné pour cette tâche.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analysis.config import DATASET_PATH, MODELS_DIR, TEST_SIZE, RANDOM_STATE, LABELS
from sentiment_analysis.preprocessing import clean_text
from sentiment_analysis.logging_config import get_logger

logger = get_logger("train_neural")

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

CHECKPOINTS = {
    "camembert-base": "camembert-base",
    "flaubert-base": "flaubert/flaubert_base_cased",
}


def load_dataset_for_training():
    """Charge data/dataset.csv et fait un split stratifié train/test,
    cohérent avec celui utilisé pour les modèles statistiques
    (data_loader.py) pour pouvoir comparer les approches équitablement."""
    from sklearn.model_selection import train_test_split

    texts, labels = [], []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(clean_text(row["text"]))
            labels.append(row["label"])

    return train_test_split(texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)


def fine_tune(checkpoint: str, epochs: int, batch_size: int, output_dir: Path):
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer,
        )
    except ImportError as e:
        raise ImportError(
            "Ce script nécessite `transformers`, `torch` et `datasets` : "
            "`pip install transformers torch datasets`"
        ) from e

    train_texts, test_texts, train_labels, test_labels = load_dataset_for_training()
    logger.info("Dataset chargé : %d exemples train, %d exemples test.", len(train_texts), len(test_texts))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    def to_hf_dataset(texts, labels):
        return Dataset.from_dict({"text": texts, "label": [LABEL2ID[l] for l in labels]})

    train_ds = to_hf_dataset(train_texts, train_labels)
    test_ds = to_hf_dataset(test_texts, test_labels)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Début du fine-tuning de %s (%d époques)...", checkpoint, epochs)
    trainer.train()

    metrics = trainer.evaluate()
    logger.info("Résultat final sur le jeu de test : %s", metrics)

    final_dir = output_dir / "neural_sentiment"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Modèle fine-tuné sauvegardé dans %s", final_dir)
    logger.info(
        "Active maintenant l'approche neuronale avec la variable d'environnement "
        "SENTIMENT_ENABLE_NEURAL=true et SENTIMENT_NEURAL_CHECKPOINT=%s", final_dir,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CamemBERT ou FlauBERT pour la classification de sentiment.")
    parser.add_argument("--checkpoint", choices=list(CHECKPOINTS), default="camembert-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    fine_tune(
        checkpoint=CHECKPOINTS[args.checkpoint],
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=MODELS_DIR,
    )


if __name__ == "__main__":
    main()
