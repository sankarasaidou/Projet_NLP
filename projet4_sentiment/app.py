# -*- coding: utf-8 -*-
"""Interface Streamlit de production — "Tonalité", analyse de sentiment
comparative.

Identité visuelle : voir src/sentiment_analysis/ui_components.py pour le
design system (couleurs, typographie, composant jauge tri-zone signature).

Lancement : streamlit run app.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st

from sentiment_analysis.config import METADATA_PATH, LABELS
from sentiment_analysis.pipeline import SentimentPipeline
from sentiment_analysis.preprocessing import InvalidTextError, is_spacy_available
from sentiment_analysis.statistical import ModelNotTrainedError, MODEL_FACTORIES
from sentiment_analysis.neural import is_available as neural_is_available, NeuralNotAvailableError
from sentiment_analysis.data_loader import load_dataset
from sentiment_analysis.evaluation import classification_report, confusion_matrix, compare_approaches, error_analysis
from sentiment_analysis.lexical import LexicalSentimentAnalyzer
from sentiment_analysis.ui_components import (
    inject_design_system, render_header, result_card, contribution_chip_html, COLORS,
)

st.set_page_config(page_title="Tonalité — Analyse de sentiment", layout="wide", page_icon="◐")
inject_design_system()

# --- Chargement du pipeline (modèles déjà entraînés, jamais réentraînés ici) ---


@st.cache_resource(show_spinner="Chargement des modèles...")
def load_pipeline() -> SentimentPipeline:
    pipeline = SentimentPipeline()
    try:
        _ = pipeline.statistical  # force le chargement (paresseux par défaut) pour valider maintenant
    except ModelNotTrainedError:
        # Aucun modèle exploitable (absent, ou sauvegardé avec une version
        # de scikit-learn incompatible avec celle installée ici) : on
        # entraîne à la volée plutôt que de bloquer l'utilisateur.
        from train import train_all
        train_all()
        pipeline = SentimentPipeline()
        _ = pipeline.statistical  # revalide après entraînement
    return pipeline


pipeline = load_pipeline()

render_header(pipeline.statistical_model_name, is_spacy_available(), neural_is_available())

if not is_spacy_available():
    st.markdown(
        f"""
        <div class="tn-card" style="border-color:{COLORS['neutre']}40; background:{COLORS['neutre_soft']}30;">
            <span style="color:{COLORS['neutre']}; font-weight:600;">Mode simplifié —</span>
            spaCy n'est pas installé : la lemmatisation utilise un repli automatique.
            Installe spaCy et <code>fr_core_news_sm</code> pour de meilleurs résultats (voir README).
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Panneau de configuration (sidebar) ---
with st.sidebar:
    st.markdown("#### Configuration")
    statistical_model_choice = st.selectbox(
        "Modèle statistique",
        list(MODEL_FACTORIES.keys()),
        index=list(MODEL_FACTORIES.keys()).index(pipeline.statistical_model_name),
        help="Présélectionné : le modèle ayant obtenu la meilleure accuracy "
             "en validation croisée lors du dernier entraînement.",
    )
    if statistical_model_choice != pipeline.statistical_model_name:
        pipeline = SentimentPipeline(statistical_model_name=statistical_model_choice)

    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        cv_scores = metadata.get("_cv_scores", {})
        if cv_scores:
            st.markdown("###### Validation croisée (5-fold)")
            for name, score in cv_scores.items():
                is_best = name == metadata.get("_best_model")
                marker = " — meilleur" if is_best else ""
                st.markdown(
                    f'<div class="tn-mono" style="font-size:0.82rem; color:{COLORS["ink_muted"]};">'
                    f'{name}{marker} : {score:.2f}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("###### État de l'approche neuronale")
    if neural_is_available():
        st.markdown(
            f'<span class="tn-pill on">transformers/torch détectés</span>', unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="tn-pill off">non installés</span>', unsafe_allow_html=True,
        )
        st.caption("`pip install transformers torch` suffit pour l'activer (modèle déjà entraîné pour le sentiment).")

tab_single, tab_batch, tab_eval = st.tabs(["Analyser un texte", "Analyse par lot", "Évaluation"])

# ==============================================================================
# ONGLET 1 — Analyse d'un texte unique
# ==============================================================================
with tab_single:
    text = st.text_area(
        "Texte à analyser",
        value="Je suis très satisfait, le service était excellent et rapide.",
        height=100,
        label_visibility="collapsed",
        placeholder="Colle ou écris un avis, un commentaire, un extrait d'article...",
    )

    analyze_clicked = st.button("Analyser", type="primary")

    if analyze_clicked:
        try:
            result = pipeline.analyze_all(text)
        except InvalidTextError as e:
            st.markdown(
                f'<div class="tn-card" style="border-color:{COLORS["négatif"]}40;">'
                f'<span style="color:{COLORS["négatif"]}; font-weight:600;">Texte invalide —</span> {e}</div>',
                unsafe_allow_html=True,
            )
            result = None

        if result:
            col1, col2, col3 = st.columns(3)

            with col1:
                r = result["lexicale"]
                if "error" in r:
                    result_card("Lexicale", None, 0, detail_html=r["error"])
                else:
                    intensity = min(abs(r["score"]) / 4, 1.0)
                    chips = "".join(
                        contribution_chip_html(d["word"], d["polarity"], d["negated"])
                        for d in r["details"]
                    )
                    detail = (
                        f'<div class="tn-mono" style="font-size:0.8rem; color:{COLORS["ink_muted"]}; margin-top:0.5rem;">'
                        f'score {r["score"]:+.2f}</div>'
                        f'<div style="margin-top:0.4rem;">{chips or "<span class=\'tn-empty\'>Aucun mot du lexique détecté.</span>"}</div>'
                    )
                    result_card("Lexicale", r["label"], intensity, detail_html=detail)

            with col2:
                r = result["statistique"]
                if "error" in r:
                    result_card(f"Statistique ({pipeline.statistical_model_name})", None, 0, detail_html=r["error"])
                else:
                    detail = (
                        f'<div class="tn-mono" style="font-size:0.8rem; color:{COLORS["ink_muted"]}; margin-top:0.5rem;">'
                        f'confiance {r["confidence"]*100:.0f}%</div>'
                    )
                    result_card(f"Statistique ({pipeline.statistical_model_name})", r["label"], r["confidence"], detail_html=detail)

            with col3:
                r = result["neuronale"]
                if "error" in r:
                    result_card("Neuronale", None, 0, detail_html="Non activée dans cet environnement.")
                else:
                    detail = (
                        f'<div class="tn-mono" style="font-size:0.8rem; color:{COLORS["ink_muted"]}; margin-top:0.5rem;">'
                        f'confiance {r["confidence"]*100:.0f}%</div>'
                    )
                    result_card("Neuronale", r["label"], r["confidence"], detail_html=detail)

# ==============================================================================
# ONGLET 2 — Analyse par lot
# ==============================================================================
with tab_batch:
    st.markdown(
        f'<p style="color:{COLORS["ink_muted"]};">Dépose un CSV avec une colonne '
        f'<code>text</code> pour analyser plusieurs textes d\'un coup '
        f'(approche statistique, la plus rapide en volume).</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.markdown(
                    f'<div class="tn-card" style="border-color:{COLORS["négatif"]}40;">'
                    f'<span style="color:{COLORS["négatif"]}; font-weight:600;">Colonne manquante —</span> '
                    f'le fichier doit contenir une colonne nommée <code>text</code>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                texts = df["text"].astype(str).tolist()
                predictions, confidences = [], []
                for t in texts:
                    try:
                        r = pipeline.analyze_statistical(t)
                        predictions.append(r["label"])
                        confidences.append(round(r["confidence"], 2))
                    except InvalidTextError:
                        predictions.append("texte invalide")
                        confidences.append(None)

                df["sentiment_prédit"] = predictions
                df["confiance"] = confidences
                st.dataframe(df, use_container_width=True)

                col_dl, col_stats = st.columns([1, 2])
                with col_dl:
                    st.download_button(
                        "Télécharger les résultats",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name="resultats_sentiment.csv",
                        mime="text/csv",
                    )

                st.markdown("###### Répartition des sentiments prédits")
                st.bar_chart(
                    df["sentiment_prédit"].value_counts(),
                    color=COLORS["brand"],
                )
        except pd.errors.ParserError as e:
            st.markdown(
                f'<div class="tn-card" style="border-color:{COLORS["négatif"]}40;">'
                f'<span style="color:{COLORS["négatif"]}; font-weight:600;">CSV illisible —</span> {e}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="tn-empty">Aucun fichier déposé pour le moment.</div>', unsafe_allow_html=True)

# ==============================================================================
# ONGLET 3 — Évaluation comparative
# ==============================================================================
with tab_eval:
    st.markdown(
        f'<p style="color:{COLORS["ink_muted"]};">Évaluation sur le jeu de test tenu à '
        f'l\'écart de l\'entraînement (split stratifié, voir <code>data_loader.py</code>).</p>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Comparaison des 3 approches")
    st.markdown(
        f'<p style="color:{COLORS["ink_muted"]}; font-size:0.88rem;">Lexicale, statistique et '
        f'neuronale évaluées sur le même jeu de test, avec analyse des erreurs de chacune.</p>',
        unsafe_allow_html=True,
    )

    if st.button("Comparer les 3 approches", type="primary"):
        dataset = load_dataset()
        results_by_approach = {}

        lexical_analyzer = LexicalSentimentAnalyzer()
        lexical_preds = [lexical_analyzer.analyze(t)["label"] for t in dataset.test_texts]
        results_by_approach["Lexicale"] = (lexical_preds, dataset.test_labels)

        statistical_preds = pipeline.statistical.predict(dataset.test_texts)
        results_by_approach[f"Statistique ({pipeline.statistical_model_name})"] = (statistical_preds, dataset.test_labels)

        if neural_is_available():
            try:
                neural_preds = [pipeline.analyze_neural(t)["label"] for t in dataset.test_texts]
                results_by_approach["Neuronale"] = (neural_preds, dataset.test_labels)
            except NeuralNotAvailableError as e:
                st.markdown(f'<div class="tn-empty">Neuronale exclue : {e}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tn-empty">Neuronale exclue : transformers/torch non installés.</div>', unsafe_allow_html=True)

        comparison = compare_approaches(results_by_approach)

        cols = st.columns(len(comparison))
        for col, (approach, report) in zip(cols, comparison.items()):
            with col:
                st.markdown(
                    f"""
                    <div class="tn-card" style="text-align:center;">
                        <div style="color:{COLORS['ink_muted']}; font-size:0.8rem;">{approach}</div>
                        <div class="tn-mono" style="font-size:1.8rem; font-weight:600; color:{COLORS['brand']};">
                            {report['accuracy']:.2f}
                        </div>
                        <div style="color:{COLORS['ink_muted']}; font-size:0.75rem;">accuracy</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        for approach, (preds, gold) in results_by_approach.items():
            with st.expander(f"Détail — {approach}"):
                report = comparison[approach]
                st.table({
                    label: {
                        "précision": round(m["precision"], 2),
                        "rappel": round(m["recall"], 2),
                        "f1": round(m["f1"], 2),
                        "support": m["support"],
                    }
                    for label, m in report["per_label"].items()
                })
                errors = error_analysis(dataset.test_texts, preds, gold)
                st.caption(f"{len(errors)} erreur(s) sur {len(gold)} exemples de test.")
                if errors:
                    st.dataframe(
                        pd.DataFrame(errors[:20]).rename(
                            columns={"text": "Texte", "predicted": "Prédit", "expected": "Attendu"}
                        ),
                        use_container_width=True,
                    )

    st.markdown("---")
    st.markdown("#### Comparaison entre les 3 modèles statistiques")
    if st.button("Comparer les modèles statistiques"):
        dataset = load_dataset()
        for model_name in MODEL_FACTORIES:
            try:
                temp_pipeline = SentimentPipeline(statistical_model_name=model_name)
                predictions = [temp_pipeline.analyze_statistical(t)["label"] for t in dataset.test_texts]
                report = classification_report(predictions, dataset.test_labels)

                st.markdown(f"###### {model_name} — accuracy `{report['accuracy']:.2f}`")
                st.table({
                    label: {
                        "précision": round(m["precision"], 2),
                        "rappel": round(m["recall"], 2),
                        "f1": round(m["f1"], 2),
                        "support": m["support"],
                    }
                    for label, m in report["per_label"].items()
                })

                matrix = confusion_matrix(predictions, dataset.test_labels)
                st.caption("Matrice de confusion (lignes = réel, colonnes = prédit)")
                st.table(pd.DataFrame(matrix).T[LABELS])
            except ModelNotTrainedError:
                st.markdown(f'<div class="tn-empty">{model_name} : non entraîné (lance `python train.py`).</div>', unsafe_allow_html=True)
