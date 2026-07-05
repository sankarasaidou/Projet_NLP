import streamlit as st

# ==============================================================================
# 1. IMPORTS DES PROJETS (Noms des dossiers exacts en minuscules pour Linux/GitHub)
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

# Configuration de la page principale
st.set_page_config(page_title="Projets NLP - M1 FDIA", layout="wide", page_icon="📚")

st.title("📚 Plateforme des Projets NLP - M1 FDIA")
st.write("Utilisez le menu latéral pour naviguer et tester les différents modules développés.")
st.markdown("---")

# ==============================================================================
# 2. DICTIONNAIRE DE CORRESPONDANCE (Évite les bugs des structures if/elif complexes)
# ==============================================================================
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
# 3. INTERFACE ET SÉLECTION
# ==============================================================================
choix = st.sidebar.selectbox(
    "Choisissez un projet à tester",
    list(projets.keys())
)

# Affichage du statut dans la barre latérale
st.sidebar.markdown("---")
st.sidebar.success(f"📈 Module actif : {choix}")

# ==============================================================================
# 4. EXÉCUTION DYNAMIQUE DU PROJET SÉLECTIONNÉ
# ==============================================================================
# Appelle directement la fonction main() associée au texte sélectionné
projets[choix]()