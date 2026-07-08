# -*- coding: utf-8 -*-
"""
Interface Streamlit — Chatbot FAQ intelligent UV-BF.
Lancement : streamlit run app.py
"""

import streamlit as st

from chatbot import answer_question
from feedback import log_interaction, analytics_summary, get_negative_feedback_cases

st.set_page_config(page_title="Chatbot FAQ UV-BF", layout="wide")
st.title("🎓 Chatbot FAQ — Étudiants UV-BF")
st.caption("NER personnalisé + classification d'intentions + matching intelligent")

if "history" not in st.session_state:
    st.session_state.history = []
if "log_indices" not in st.session_state:
    st.session_state.log_indices = []

tab_chat, tab_analytics = st.tabs(["💬 Chatbot", "📊 Tableau de bord"])

with tab_chat:
    for i, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(
                f"Intention : {turn['intent']} (confiance={turn['intent_confidence']:.2f}) · "
                f"Entités : {', '.join(e['text'] for e in turn['entities']) or 'aucune'}"
            )
            col1, col2 = st.columns([1, 1])
            if col1.button("👍 Utile", key=f"up_{i}"):
                st.session_state.history[i]["feedback"] = "positif"
            if col2.button("👎 Pas utile", key=f"down_{i}"):
                st.session_state.history[i]["feedback"] = "négatif"

    question = st.chat_input("Posez votre question sur l'UV-BF...")
    if question:
        result = answer_question(question)
        turn = {**result, "question": question, "feedback": None}
        st.session_state.history.append(turn)
        log_interaction(question, result["intent"], result["answer"])
        st.rerun()

with tab_analytics:
    # on synchronise le journal de feedback avec l'historique de session
    from feedback import FEEDBACK_LOG
    FEEDBACK_LOG.clear()
    for turn in st.session_state.history:
        log_interaction(turn["question"], turn["intent"], turn["answer"], turn["feedback"])

    summary = analytics_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interactions totales", summary["total_interactions"])
    col2.metric("👍 Positifs", summary["positive"])
    col3.metric("👎 Négatifs", summary["negative"])
    satisfaction = summary["satisfaction_rate"]
    col4.metric("Taux de satisfaction", f"{satisfaction * 100:.0f}%" if satisfaction is not None else "—")

    if summary["intent_distribution"]:
        st.subheader("Répartition des intentions posées")
        st.bar_chart(summary["intent_distribution"])

    negative_cases = get_negative_feedback_cases()
    if negative_cases:
        st.subheader("🔍 Cas à améliorer (feedback négatif)")
        for case in negative_cases:
            st.write(f"- « {case['question']} » → intention prédite : **{case['predicted_intent']}**")
