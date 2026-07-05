# -*- coding: utf-8 -*-
"""
ÉTAPE 5 : Approche neuronale
---------------------------------
Fine-tuning d'un modèle pré-entraîné français (CamemBERT ou FlauBERT)
pour la classification de sentiment à 3 classes.

Nécessite `transformers` + `torch` (paquets lourds, non installés dans
ce bac à sable sans accès réseau). Le code suit l'API standard
`Trainer` de Hugging Face, directement réutilisable tel quel dans un
environnement avec GPU/CPU suffisant.

Point important : contrairement aux approches lexicale/statistique, on
NE prétraite PAS le texte avec tokenize_lexical() (pas de suppression de
stopwords ni lemmatisation) : le tokenizer du modèle pré-entraîné a été
entraîné sur du texte brut, et retirer des mots dégraderait sa
compréhension du contexte. On utilise seulement clean_text() (espaces).
"""

from preprocessing import clean_text

MODEL_CHECKPOINTS = {
    "CamemBERT": "camembert-base",
    "FlauBERT": "flaubert/flaubert_base_cased",
}

LABEL2ID = {"négatif": 0, "neutre": 1, "positif": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def fine_tune_sentiment_model(train_data, test_data, checkpoint_name="CamemBERT",
                               output_dir="./model/neural_sentiment", epochs=4):
    """Fine-tune un modèle pré-entraîné sur nos données de sentiment.

    train_data / test_data : liste de tuples (texte, label_str), comme
    dans corpus.py (TRAIN_DATA / TEST_DATA).
    """
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer,
        )
    except ImportError as e:
        raise ImportError(
            "Ce module nécessite `transformers`, `torch` et `datasets` : "
            "`pip install transformers torch datasets`"
        ) from e

    checkpoint = MODEL_CHECKPOINTS[checkpoint_name]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def to_hf_dataset(data):
        texts = [clean_text(t) for t, _ in data]
        labels = [LABEL2ID[label] for _, label in data]
        return Dataset.from_dict({"text": texts, "label": labels})

    train_ds = to_hf_dataset(train_data)
    test_ds = to_hf_dataset(test_data)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=5,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


def predict_neural(texts: list[str], model_dir="./model/neural_sentiment") -> list[str]:
    """Charge un modèle fine-tuné et prédit le sentiment de nouveaux textes."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        raise ImportError("`pip install transformers torch`") from e

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer([clean_text(t) for t in texts], return_tensors="pt",
                        truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = logits.argmax(dim=-1).tolist()
    return [ID2LABEL[i] for i in predicted_ids]


if __name__ == "__main__":
    from corpus import TRAIN_DATA, TEST_DATA
    try:
        fine_tune_sentiment_model(TRAIN_DATA, TEST_DATA, checkpoint_name="CamemBERT", epochs=2)
    except ImportError as e:
        print(f"Non exécuté ici : {e}")
