# -*- coding: utf-8 -*-
"""
ÉTAPE 5 : Visualisations
----------------------------
- plot_daily_activity() : nombre de commentaires par jour + répartition
  du sentiment (positif/négatif/neutre) empilée -> vue d'ensemble de
  l'évolution de l'opinion publique dans le temps.
- plot_wordcloud() : nuage de mots des termes les plus fréquents
  (implémenté nous-mêmes en positionnement simple si le paquet
  `wordcloud` n'est pas installé, pour ne jamais bloquer l'affichage).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import clean_comment
from sentiment import full_analysis
from trends import top_terms_overall


def plot_daily_activity(comments: list[dict], output_path: str = "/tmp/daily_activity.png"):
    records = []
    for c in comments:
        analysis = full_analysis(c["text"])
        records.append({"date": c["date"], "sentiment": analysis["sentiment"]})

    df = pd.DataFrame(records)
    pivot = df.groupby(["date", "sentiment"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["positif", "neutre", "négatif"], fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color={"positif": "#4caf50", "neutre": "#9e9e9e", "négatif": "#f44336"})
    ax.set_title("Activité quotidienne des commentaires par sentiment")
    ax.set_xlabel("Date")
    ax.set_ylabel("Nombre de commentaires")
    ax.legend(title="Sentiment")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_wordcloud(comments: list[dict], output_path: str = "/tmp/wordcloud.png", term_type: str = "hashtag"):
    try:
        from wordcloud import WordCloud
        top_terms = dict(top_terms_overall(comments, term_type, top_n=30))
        wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis")
        wc.generate_from_frequencies(top_terms)
        wc.to_file(output_path)
        return output_path
    except ImportError:
        # Fallback : simple bar chart si le paquet `wordcloud` n'est pas installé
        return plot_top_terms_bar(comments, output_path, term_type)


def plot_top_terms_bar(comments: list[dict], output_path: str = "/tmp/top_terms.png", term_type: str = "hashtag"):
    top_terms = top_terms_overall(comments, term_type, top_n=10)
    if not top_terms:
        return None
    terms, counts = zip(*top_terms)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(terms[::-1], counts[::-1], color="#1976d2")
    ax.set_title(f"Top {term_type}s les plus mentionnés")
    ax.set_xlabel("Fréquence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    from comments_data import COMMENTS

    p1 = plot_daily_activity(COMMENTS, "daily_activity_demo.png")
    p2 = plot_top_terms_bar(COMMENTS, "top_terms_demo.png")
    print(f"Générés : {p1}, {p2}")
