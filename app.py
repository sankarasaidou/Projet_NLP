import streamlit as st

# ==============================================================================
# 1. IMPORTS CORRIGÉS (Noms des dossiers exacts en minuscules pour Linux/GitHub)
# ==============================================================================
from projet1_ner.app import main as projet1
from projet2_vectorisation.app import main as projet2
from projet3_topic_modeling.app import main as projet3
from projet4_sentiment.app import main as projet4
from projet5_classification.app import main as projet5
from projet6_plagiat.app import main as projet6
from projet7_resume.app import main as projet7
from projet8_chatbot.app import main as projet8
from projet9_tendances.app import main as projet9
from projet10_recommandation.app import main as projet10

# Configuration globale de la page principale
st.set_page_config(page_title="Projets NLP - M1 FDIA", layout="wide", page_icon="📚")

st.title("📚 Plateforme des Projets NLP - M1 FDIA")
st.write("Sélectionnez un projet dans le menu de gauche pour tester son interface Streamlit dédiée.")
st.markdown("---")

# ==============================================================================
# 2. DICTIONNAIRE DE CORRESPONDANCE
# ==============================================================================
# Cette structure associe directement le texte du menu à la fonction main du projet.
# Cela règle le bug où tes conditions "if" ne correspondaient pas au texte affiché.
projets = {
    "Projet 1 – NER Spécifique IA": projet1,
    "Projet 2 – Vectorisation": projet2,
    "Projet 3 – Topic Modeling": projet3,
    "Projet 4 – Analyse de sentiment": projet4,
    "Projet 5 – Classification": projet5,
    "Projet 6 – Détection de plagiat": projet6,
    "Projet 7 – Résumé automatique": projet7,
    "Projet 8 – Chatbot": projet8,
    "Projet 9 – Analyse de tendances": projet9,
    "Projet 10 – Recommandation": projet10,
}

# ==============================================================================
# 3. INTERFACE ET SÉLECTION DYNAMIQUE
# ==============================================================================
choix = st.sidebar.selectbox(
    "Choisissez un projet à exécuter :",
    list(projets.keys())
)

# Indicateur visuel dans la barre latérale
st.sidebar.markdown("---")
st.sidebar.success(f"📈 Module actif : {choix}")

# Lancement automatique de la fonction main() du projet choisi
projets[choix]()