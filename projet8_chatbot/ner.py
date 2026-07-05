# -*- coding: utf-8 -*-
"""
ÉTAPE 2 : NER personnalisé
--------------------------------
Détection d'entités spécifiques au domaine étudiant UV-BF : essentiellement
la MATIERE mentionnée dans la question (ex. "cours de NLP" -> MATIERE=NLP),
mais aussi SERVICE (scolarité, support technique) et DOCUMENT (attestation,
certificat).

Approche : simple correspondance de vocabulaire (comme un EntityRuler),
suffisante ici car le vocabulaire du domaine est fermé et connu à
l'avance (liste de matières, services, documents proposés par l'UV-BF).
Pas besoin d'un modèle statistique complexe pour ce cas d'usage fermé.
"""

import re

from faq_data import MATIERES

SERVICES = ["scolarité", "support technique", "service technique"]
DOCUMENTS = ["certificat de scolarité", "attestation de réussite", "diplôme", "relevé de notes"]

_ENTITY_VOCAB = (
    [(m, "MATIERE") for m in MATIERES]
    + [(s, "SERVICE") for s in SERVICES]
    + [(d, "DOCUMENT") for d in DOCUMENTS]
)
# Trier par longueur décroissante pour matcher les expressions les plus
# longues en priorité (ex. "certificat de scolarité" avant "scolarité").
_ENTITY_VOCAB.sort(key=lambda pair: len(pair[0]), reverse=True)


def extract_entities(text: str) -> list[dict]:
    """Retourne la liste des entités détectées : [{"text":..., "label":...}]"""
    text_lower = text.lower()
    entities = []
    covered_spans = []

    for term, label in _ENTITY_VOCAB:
        for match in re.finditer(re.escape(term.lower()), text_lower):
            start, end = match.span()
            if any(not (end <= s or start >= e) for s, e in covered_spans):
                continue  # évite les chevauchements (ex. "scolarité" dans "certificat de scolarité")
            covered_spans.append((start, end))
            entities.append({"text": text[start:end], "label": label})

    return entities


def get_matiere(text: str) -> str | None:
    """Raccourci : retourne la première MATIERE détectée, ou None."""
    for ent in extract_entities(text):
        if ent["label"] == "MATIERE":
            return ent["text"]
    return None


if __name__ == "__main__":
    tests = [
        "Comment accéder aux cours de NLP ?",
        "Où trouver mon certificat de scolarité ?",
        "Quand aura lieu l'examen de bases de données ?",
    ]
    for t in tests:
        print(f"{t!r} -> {extract_entities(t)}")
