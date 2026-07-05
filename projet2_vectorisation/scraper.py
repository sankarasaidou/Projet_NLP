# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Constitution du corpus thématique
-----------------------------------------------
Deux choses sont demandées ici :
(a) collecter automatiquement des articles en ligne (web scraping)
(b) documenter la technique utilisée pour constituer les MOTS-CLÉS de
    collecte (c.-à-d. comment on choisit quoi scraper).

(a) Scraping
------------
On cible ici lefaso.net (site d'actualités burkinabè cité dans le sujet).
Le scraping réel nécessite un accès réseau ; comme ce n'est pas toujours
disponible (sandbox, pare-feu, site indisponible...), on fournit un
FALLBACK : un petit corpus local pré-enregistré, pour que le reste du
pipeline (vectorisation, similarité, Streamlit) soit testable hors-ligne.

(b) Mots-clés de collecte
--------------------------
Technique utilisée : on part d'une liste de MOTS-CLÉS SEED (ex. "économie",
"agriculture", "éducation") représentant les thématiques ciblées, on les
utilise comme requêtes de recherche interne au site ou comme filtres sur
les catégories du site, puis on élargit avec les mots-clés qui reviennent
le plus souvent dans les résultats obtenus (co-occurrence) : c'est une
approche "seed + expansion" très classique en veille documentaire.
"""

import time
import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; ProjetNLP-UVBF/1.0)"

# Mots-clés "seed" utilisés pour orienter la collecte (voir docstring ci-dessus)
SEED_KEYWORDS = ["économie", "agriculture", "éducation", "santé", "sécurité"]


def scrape_lefaso(max_articles: int = 20, delay: float = 1.0) -> list[dict]:
    """Scrape les derniers articles de la page d'accueil de lefaso.net.

    Retourne une liste de dicts : {"title": str, "url": str, "text": str}.
    Nécessite un accès réseau ; peut échouer si la structure HTML du site
    change (le scraping web est par nature fragile -> toujours prévoir
    un fallback et des try/except).
    """
    base_url = "https://lefaso.net/"
    headers = {"User-Agent": USER_AGENT}

    articles = []
    resp = requests.get(base_url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = soup.select("a[href*='/spip.php?article']")[:max_articles]
    seen_urls = set()

    for link in links:
        url = link.get("href")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        if not url.startswith("http"):
            url = base_url.rstrip("/") + "/" + url.lstrip("/")

        try:
            article_resp = requests.get(url, headers=headers, timeout=10)
            article_resp.raise_for_status()
            article_soup = BeautifulSoup(article_resp.text, "html.parser")

            title_tag = article_soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else url

            body = article_soup.find("div", {"id": "art_texte"}) or article_soup.find("article")
            text = body.get_text(" ", strip=True) if body else ""

            if text:
                articles.append({"title": title, "url": url, "text": text})
        except requests.RequestException:
            continue  # on ignore un article qui échoue, on continue les autres

        time.sleep(delay)  # politesse envers le serveur cible

    return articles


if __name__ == "__main__":
    try:
        results = scrape_lefaso(max_articles=10)
        print(f"{len(results)} articles scrapés.")
    except requests.RequestException as e:
        print(f"Scraping indisponible ({e}). Utiliser corpus_sample.py en fallback.")
