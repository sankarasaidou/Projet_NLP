# -*- coding: utf-8 -*-
"""
ÉTAPE 5 : Ajouter la couche EntityRuler de spaCy personnalisé
------------------------------------------------------------------
On construit un pipeline spaCy hybride à trois couches.
"""

# ==============================================================================
# 1. CORRECTIF DE CHEMINS LOCAL (Placé STRICTEMENT en première position)
# ==============================================================================
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

# ==============================================================================
# 2. AUTRES IMPORTS STANDARD ET DE SPACY
# ==============================================================================
import random
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.training import Example
from spacy.util import filter_spans

# Imports locaux désormais sécurisés (grâce au bloc sys.path ci-dessus)
from patterns import PHRASE_PATTERNS, REGEX_PATTERNS
from data.annotations import TRAIN_DATA, ENTITY_LABELS

MODEL_DIR = Path(__file__).parent / "model" / "domain_ner"


@Language.component("regex_entity_component")
def regex_entity_component(doc):
    """Composant spaCy personnalisé : scanne doc.text avec nos regex et
    ajoute les correspondances comme entités, sans écraser celles déjà
    posées par l'EntityRuler ou le NER entraîné."""
    new_spans = []
    for label, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(doc.text):
            span = doc.char_span(
                match.start(), match.end(), label=label, alignment_mode="expand"
            )
            if span is not None:
                new_spans.append(span)

    # filter_spans priorise les spans déjà présents (doc.ents) sur les
    # nouveaux, et enlève les chevauchements entre nouveaux spans.
    all_spans = filter_spans(list(doc.ents) + new_spans)
    doc.ents = all_spans
    return doc


def build_pipeline():
    """Construit le pipeline complet : EntityRuler + NER (vide, à
    entraîner) + composant regex."""
    nlp = spacy.blank("fr")

    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(PHRASE_PATTERNS)

    ner = nlp.add_pipe("ner")
    for label in ENTITY_LABELS:
        ner.add_label(label)

    # Le composant regex doit tourner APRÈS le ner pour ne pas être
    # écrasé par lui, mais il respecte déjà les entités existantes.
    nlp.add_pipe("regex_entity_component", last=True)

    return nlp


def train(n_iter: int = 30, seed: int = 42):
    nlp = build_pipeline()

    # On entraîne UNIQUEMENT le composant "ner" : l'EntityRuler et le
    # composant regex n'ont pas de poids à apprendre.
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    examples = []
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, annotations))

    random.seed(seed)
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize(lambda: examples)
        for i in range(n_iter):
            random.shuffle(examples)
            losses = {}
            batches = spacy.util.minibatch(examples, size=4)
            for batch in batches:
                nlp.update(batch, sgd=optimizer, losses=losses, drop=0.2)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Itération {i + 1:>2}/{n_iter} - pertes NER : {losses.get('ner', 0):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"\nModèle sauvegardé dans : {MODEL_DIR}")
    return nlp


if __name__ == "__main__":
    train()