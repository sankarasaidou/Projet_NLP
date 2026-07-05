import os
import sys
import streamlit as st

# ==============================================================================
# 1. CORRECTIF DE CHEMINS GLOBAL (Exécuté AVANT tout import de sous-projet)
# ==============================================================================
root_dir = os.path.dirname(os.path.realpath(__file__))

# Liste exacte de tes 10 dossiers de projets sur GitHub
dossiers_projets = [
    "projet1_ner",
    "projet2_vectorisation",
    "projet3_topic_modeling",
    "projet4_sentiment",
    "projet5_classification",
    "projet6_plagiat",
    "projet7_resume",
    "projet8_chatbot",
    "projet9_tendances",
    "projet10_recommandation"
]

# Injection dynamique de chaque dossier dans le système de recherche de Python
for dossier in dossiers_projets:
    chemin_absolu = os.path.join(root_dir, dossier)
    if os.path.exists(chemin_absolu) and chemin_absolu not in sys.path:
        sys.path.insert(0, chemin_absolu)

# ==============================================================================
# 2. IMPORTS DES PROJETS SÉCURISÉS
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

# Configuration globale unique de la page (Doit être définie ici et uniquement ici)
st.set_page_config(page_title="Projets NLP - M1 FDIA", layout="wide", page_icon="📚")

st.title("📚 Plateforme des Projets NLP - M1 FDIA")
st.write("Sélectionnez un projet dans le menu de gauche pour tester son interface Streamlit dédiée.")
st.markdown("---")

# ==============================================================================
# 3. DICTIONNAIRE DE CORRESPONDANCE
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
# 4. INTERFACE ET SÉLECTION DYNAMIQUE
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