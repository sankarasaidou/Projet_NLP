# -*- coding: utf-8 -*-
"""Interface Streamlit de production — Analyse de sentiment comparative.

Différences clés avec la version précédente (non-production) :
- Charge des modèles DÉJÀ ENTRAÎNÉS (via train.py), ne réentraîne jamais
  à chaque lancement.
- Gestion d'erreurs explicite à chaque étape (modèle absent, dépendance
  optionnelle manquante, texte invalide) avec messages actionnables.
- Analyse par lot (upload CSV) en plus de l'analyse au cas par cas.
- Onglet d'évaluation basé sur les vraies métriques calculées par train.py
  (accuracy CV, rapport par classe), pas juste un chiffre isolé.

Lancement : streamlit run app.py
"""

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
from sentiment_analysis.evaluation import classification_report, confusion_matrix
import json

st.set_page_config(page_title="Analyse de sentiment — Production", layout="wide", page_icon="💬")
st.title("💬 Analyse de sentiment — Version production")
st.caption("Lexicale · Statistique (modèles persistés) · Neuronale (optionnelle) — Projet 4 M1 FDIA")

if not is_spacy_available():
    st.info(
        "ℹ️ spaCy n'est pas installé : la lemmatisation utilise un mode "
        "simplifié (fallback automatique). Pour de meilleurs résultats, "
        "installe spaCy + `fr_core_news_sm` (voir README).",
        icon="ℹ️",
    )


@st.cache_resource(show_spinner="Chargement des modèles entraînés...")
def load_pipeline() -> SentimentPipeline:
    return SentimentPipeline()


try:
    pipeline = load_pipeline()
    model_status_ok = True
except ModelNotTrainedError as e:
    model_status_ok = False
    st.error(
        f"❌ Aucun modèle statistique entraîné n'a été trouvé.\n\n"
        f"**Il faut d'abord entraîner les modèles** en exécutant, dans un terminal :\n\n"
        f"```bash\npython train.py\n```\n\nDétail technique : {e}"
    )
    st.stop()

with st.sidebar:
    st.header("⚙️ Paramètres")
    statistical_model_choice = st.selectbox(
        "Modèle statistique", list(MODEL_FACTORIES.keys()),
        index=list(MODEL_FACTORIES.keys()).index(pipeline.statistical_model_name),
        help="Le modèle pré-sélectionné est celui ayant obtenu la meilleure "
             "accuracy en validation croisée lors du dernier entraînement.",
    )
    if statistical_model_choice != pipeline.statistical_model_name:
        pipeline = SentimentPipeline(statistical_model_name=statistical_model_choice)

    st.markdown("---")
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        best = metadata.get("_best_model")
        if best:
            st.success(f"🏆 Meilleur modèle (validation croisée) : **{best}**")
        cv_scores = metadata.get("_cv_scores", {})
        if cv_scores:
            st.caption("Scores de validation croisée (5-fold) :")
            for name, score in cv_scores.items():
                st.caption(f"  • {name} : {score:.2f}")

    st.markdown("---")
    if neural_is_available():
        st.success("🧠 Approche neuronale disponible (transformers/torch installés).")
    else:
        st.warning(
            "🧠 Approche neuronale non disponible dans cet environnement "
            "(`pip install transformers torch` pour l'activer)."
        )

tab_single, tab_batch, tab_eval = st.tabs(
    ["📝 Analyser un texte", "📂 Analyse par lot (CSV)", "📊 Évaluation des modèles"]
)

with tab_single:
    text = st.text_area(
        "Texte à analyser :",
        value="Je suis très satisfait, le service était excellent et rapide.",
        height=100,
    )

    if st.button("Analyser", type="primary"):
        try:
            result = pipeline.analyze_all(text)
        except InvalidTextError as e:
            st.error(f"⚠️ Entrée invalide : {e}")
            result = None

        if result:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("📖 Lexicale")
                r = result["lexicale"]
                if "error" in r:
                    st.info(r["error"])
                else:
                    st.metric("Sentiment", r["label"], f"score = {r['score']:.2f}")
                    for d in r["details"]:
                        sign = "🔴" if d["negated"] else ("🟢" if d["polarity"] > 0 else "🔵")
                        st.caption(f"{sign} `{d['word']}` : {d['polarity']:+.2f}")

            with col2:
                st.subheader(f"📊 Statistique ({pipeline.statistical_model_name})")
                r = result["statistique"]
                if "error" in r:
                    st.info(r["error"])
                else:
                    st.metric("Sentiment", r["label"], f"confiance = {r['confidence']*100:.0f}%")
                    st.progress(min(r["confidence"], 1.0))

            with col3:
                st.subheader("🧠 Neuronale")
                r = result["neuronale"]
                if "error" in r:
                    st.info(r["error"])
                else:
                    st.metric("Sentiment", r["label"], f"confiance = {r['confidence']*100:.0f}%")

with tab_batch:
    st.write(
        "Uploader un fichier CSV avec une colonne `text` pour analyser "
        "plusieurs textes en une fois (approche statistique, la plus rapide "
        "pour du traitement par lot)."
    )
    uploaded = st.file_uploader("Fichier CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("⚠️ Le fichier CSV doit contenir une colonne nommée `text`.")
            else:
                texts = df["text"].astype(str).tolist()
                predictions, confidences = [], []
                for t in texts:
                    try:
                        r = pipeline.analyze_statistical(t)
                        predictions.append(r["label"])
                        confidences.append(round(r["confidence"], 2))
                    except InvalidTextError:
                        predictions.append("(texte invalide)")
                        confidences.append(None)

                df["sentiment_prédit"] = predictions
                df["confiance"] = confidences
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "📥 Télécharger les résultats (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="resultats_sentiment.csv",
                    mime="text/csv",
                )

                st.subheader("Répartition des sentiments prédits")
                st.bar_chart(df["sentiment_prédit"].value_counts())
        except pd.errors.ParserError as e:
            st.error(f"⚠️ Impossible de lire ce fichier CSV : {e}")

with tab_eval:
    st.write(
        "Évaluation sur le jeu de test tenu à l'écart de l'entraînement "
        "(voir `data/dataset.csv`, split stratifié dans `data_loader.py`)."
    )
    if st.button("Évaluer tous les modèles statistiques"):
        dataset = load_dataset()
        for model_name in MODEL_FACTORIES:
            try:
                temp_pipeline = SentimentPipeline(statistical_model_name=model_name)
                predictions = [temp_pipeline.analyze_statistical(t)["label"] for t in dataset.test_texts]
                report = classification_report(predictions, dataset.test_labels)

                st.markdown(f"### {model_name} — accuracy = `{report['accuracy']:.2f}`")
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
                st.caption("Matrice de confusion (lignes = réel, colonnes = prédit) :")
                st.table(pd.DataFrame(matrix).T[LABELS])
            except ModelNotTrainedError:
                st.warning(f"{model_name} : non entraîné (lance `python train.py`).")
