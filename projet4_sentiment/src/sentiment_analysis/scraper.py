# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Collecte de données — scraping d'avis clients réels
-------------------------------------------------------------------
Cible : Allociné (avis spectateurs sur des films), choisi pour 3 raisons :
    1. Avis en français, volume important, renouvelés en continu.
    2. Chaque avis est accompagné d'une NOTE EN ÉTOILES (0,5 à 5) qui sert
       de label de sentiment quasi gratuit (weak supervision) :
           note <= 2       -> négatif
           2 < note < 4     -> neutre
           note >= 4        -> positif
       C'est la même technique utilisée pour construire le dataset public
       "allocine" largement utilisé en recherche NLP française.
    3. Structure HTML relativement stable et pas de JavaScript obligatoire
       pour afficher les avis (contrairement à beaucoup de sites d'avis).

IMPORTANT — Honnêteté sur les limites de ce livrable :
    Ce scraper n'a PAS pu être testé en conditions réelles dans
    l'environnement de préparation de ce projet (bac à sable sans accès
    réseau sortant). Le code suit une structure HTML standard observée
    sur ce type de site, mais les sélecteurs CSS peuvent avoir changé
    depuis. **Teste-le chez toi avec un vrai accès réseau, et ajuste les
    sélecteurs dans `_SELECTORS` ci-dessous si besoin** (inspecte la page
    avec les outils de développement du navigateur : clic droit ->
    Inspecter sur un avis).

Cadre légal : ce scraping est fait dans un cadre pédagogique, à volume
raisonnable (quelques centaines d'avis, avec délai entre requêtes). Pour
un usage en production réelle, vérifie les conditions d'utilisation
(CGU) et le fichier robots.txt du site cible avant tout scraping à plus
grande échelle.
"""

import time
import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from sentiment_analysis.logging_config import get_logger

logger = get_logger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; ProjetNLP-M1FDIA/1.0; educational use)"
BASE_URL = "https://www.allocine.fr"

# Centralise les sélecteurs CSS ici : c'est le PREMIER endroit à vérifier
# et corriger si le scraping ne remonte aucun résultat (la structure du
# site a probablement changé depuis l'écriture de ce script).
_SELECTORS = {
    "review_card": "div.review-card, div.js-review-card",
    "review_text": "div.content-txt, p.content-txt",
    "rating_stars": "span.stareval-note, div.stareval-note",
}


@dataclass
class ScrapedReview:
    text: str
    rating: float
    label: str
    source_url: str


def rating_to_label(rating: float) -> str:
    """Convertit une note en étoiles (0.5 à 5) en label de sentiment."""
    if rating <= 2.0:
        return "négatif"
    if rating >= 4.0:
        return "positif"
    return "neutre"


def _parse_rating(raw_text: str) -> float | None:
    """Extrait une note numérique depuis le texte brut de l'élément note
    (le format exact varie selon les pages, d'où une regex tolérante)."""
    match = re.search(r"(\d+[.,]?\d?)", raw_text.replace(",", "."))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


def scrape_allocine_reviews(movie_path: str, max_pages: int = 5, delay: float = 1.5) -> list[ScrapedReview]:
    """Scrape les avis spectateurs d'un film Allociné.

    Args:
        movie_path : chemin du film sur Allociné, ex.
            "/film/fichefilm-12345/critiques/spectateurs/" (à récupérer
            depuis l'URL réelle de la page du film visée).
        max_pages : nombre de pages d'avis à parcourir.
        delay : délai en secondes entre deux requêtes (politesse envers
            le serveur, évite d'être bloqué).

    Returns:
        Liste de ScrapedReview (texte, note, label déduit).
    """
    headers = {"User-Agent": USER_AGENT}
    reviews: list[ScrapedReview] = []

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}{movie_path}?page={page}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Échec de la requête sur %s : %s", url, e)
            break

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select(_SELECTORS["review_card"])
        if not cards:
            logger.info(
                "Aucun avis trouvé page %d — soit la dernière page est "
                "atteinte, soit les sélecteurs CSS doivent être ajustés "
                "(voir _SELECTORS en haut du fichier).", page,
            )
            break

        for card in cards:
            text_el = card.select_one(_SELECTORS["review_text"])
            rating_el = card.select_one(_SELECTORS["rating_stars"])
            if text_el is None or rating_el is None:
                continue

            text = text_el.get_text(" ", strip=True)
            rating = _parse_rating(rating_el.get_text(" ", strip=True))
            if not text or rating is None:
                continue

            reviews.append(ScrapedReview(
                text=text, rating=rating, label=rating_to_label(rating), source_url=url,
            ))

        time.sleep(delay)  # politesse : évite de surcharger le serveur cible

    logger.info("Scraping terminé : %d avis collectés depuis %s", len(reviews), movie_path)
    return reviews


def scrape_multiple_movies(movie_paths: list[str], max_pages_per_movie: int = 3) -> list[ScrapedReview]:
    """Scrape plusieurs films pour diversifier le vocabulaire collecté
    (des films de genres différents produisent des avis lexicalement
    variés, utile pour l'entraînement)."""
    all_reviews = []
    for path in movie_paths:
        all_reviews.extend(scrape_allocine_reviews(path, max_pages=max_pages_per_movie))
    return all_reviews


if __name__ == "__main__":
    # Exemple d'usage : à adapter avec de vrais chemins de films Allociné
    # (visite allocine.fr, cherche un film, copie le chemin de son URL).
    EXAMPLE_MOVIE_PATHS = [
        "/film/fichefilm-29584/critiques/spectateurs/",  # exemple, à remplacer
    ]
    try:
        reviews = scrape_multiple_movies(EXAMPLE_MOVIE_PATHS, max_pages_per_movie=2)
        for r in reviews[:5]:
            print(f"[{r.label}] ({r.rating}/5) {r.text[:80]}...")
    except Exception as e:
        print(f"Scraping non exécutable dans cet environnement : {e}")
