# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Détection de tendances
--------------------------------------
Analyse de fréquence TEMPORELLE des mots-clés et hashtags : pour chaque
jour, on compte les occurrences des termes les plus fréquents, ce qui
permet de voir un mot-clé "monter" en popularité au fil du temps (signe
de tendance émergente) ou un pic ponctuel (signe d'un événement précis).
"""

from collections import Counter, defaultdict

import pandas as pd

from preprocessing import clean_comment


def build_trend_dataframe(comments: list[dict]) -> pd.DataFrame:
    """Construit un DataFrame (date, mot/hashtag, fréquence) à partir
    des commentaires bruts."""
    records = []
    for comment in comments:
        cleaned = clean_comment(comment["text"])
        words = cleaned["clean_text"].split()
        for word in words:
            if len(word) > 3:  # on ignore les mots très courts peu informatifs
                records.append({"date": comment["date"], "term": word, "type": "mot"})
        for hashtag in cleaned["hashtags"]:
            records.append({"date": comment["date"], "term": hashtag.lower(), "type": "hashtag"})

    return pd.DataFrame(records)


def top_terms_overall(comments: list[dict], term_type: str = "hashtag", top_n: int = 10) -> list[tuple]:
    df = build_trend_dataframe(comments)
    subset = df[df["type"] == term_type]
    return Counter(subset["term"]).most_common(top_n)


def daily_term_frequency(comments: list[dict], term: str) -> pd.Series:
    """Fréquence quotidienne d'un terme (mot ou hashtag) donné -> série
    temporelle utilisable pour un graphique de tendance."""
    df = build_trend_dataframe(comments)
    term_df = df[df["term"] == term.lower()]
    if term_df.empty:
        return pd.Series(dtype=int)
    return term_df.groupby("date").size()


def detect_spikes(comments: list[dict], term: str, spike_multiplier: float = 2.0) -> list[dict]:
    """ÉTAPE 6 (Alertes automatiques) : détecte les jours où la
    fréquence d'un terme dépasse `spike_multiplier` fois la moyenne des
    autres jours -> signal de pic de mention à surveiller."""
    series = daily_term_frequency(comments, term)
    if series.empty or len(series) < 2:
        return []

    alerts = []
    for current_date in series.index:
        other_days = series.drop(index=current_date)
        baseline = other_days.mean() if not other_days.empty else 0
        if baseline > 0 and series[current_date] >= spike_multiplier * baseline:
            alerts.append({
                "date": current_date, "term": term,
                "count": int(series[current_date]), "baseline": round(baseline, 2),
            })
    return alerts


if __name__ == "__main__":
    from comments_data import COMMENTS

    print("Top hashtags :", top_terms_overall(COMMENTS, "hashtag"))
    print("\nFréquence quotidienne de 'agriculture' :")
    print(daily_term_frequency(COMMENTS, "agriculture"))
