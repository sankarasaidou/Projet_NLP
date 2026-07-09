#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script CLI — Collecte d'avis réels.
-------------------------------------------------------------------
Lance le scraper (src/sentiment_analysis/scraper.py) sur une liste de
films Allociné et ajoute les avis collectés à data/dataset.csv.

Usage :
    python scripts/collect_reviews.py --movie-path "/film/fichefilm-XXXXX/critiques/spectateurs/" --pages 5
    python scripts/collect_reviews.py --movies-file movies.txt --pages 3

Nécessite un accès réseau.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment_analysis.scraper import scrape_multiple_movies
from sentiment_analysis.config import DATASET_PATH
from sentiment_analysis.logging_config import get_logger

logger = get_logger("collect_reviews")


def append_to_dataset(reviews, dataset_path=DATASET_PATH):
    existing_rows = []
    if dataset_path.exists():
        with open(dataset_path, encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    existing_texts = {row["text"] for row in existing_rows}
    new_rows = [
        {"text": r.text, "label": r.label}
        for r in reviews
        if r.text not in existing_texts
    ]

    all_rows = existing_rows + new_rows
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(
        "%d nouveaux avis réels ajoutés (%d doublons ignorés). Dataset total : %d exemples.",
        len(new_rows), len(reviews) - len(new_rows), len(all_rows),
    )


def main():
    parser = argparse.ArgumentParser(description="Collecte des avis réels et les ajoute au dataset.")
    parser.add_argument("--movie-path", help="Chemin Allociné d'un seul film à scraper.")
    parser.add_argument("--movies-file", help="Fichier texte listant un chemin de film par ligne.")
    parser.add_argument("--pages", type=int, default=3, help="Nombre de pages d'avis par film.")
    args = parser.parse_args()

    movie_paths = []
    if args.movie_path:
        movie_paths.append(args.movie_path)
    if args.movies_file:
        movie_paths.extend(Path(args.movies_file).read_text(encoding="utf-8").splitlines())

    if not movie_paths:
        parser.error("Fournir --movie-path ou --movies-file (voir --help).")

    reviews = scrape_multiple_movies(movie_paths, max_pages_per_movie=args.pages)
    if not reviews:
        logger.warning(
            "Aucun avis collecté. Vérifie ta connexion réseau et les sélecteurs "
            "CSS dans src/sentiment_analysis/scraper.py (_SELECTORS)."
        )
        return

    append_to_dataset(reviews)
    logger.info("Pense à relancer `python train.py` pour ré-entraîner sur les nouvelles données.")


if __name__ == "__main__":
    main()
