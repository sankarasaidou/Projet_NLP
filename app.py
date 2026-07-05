import streamlit as st

from Projet1.app import main as projet1
from Projet2.app import main as projet2
from Projet3.app import main as projet3
from Projet4.app import main as projet4
from Projet5.app import main as projet5
from Projet6.app import main as projet6
from Projet7.app import main as projet7
from Projet8.app import main as projet8
from Projet9.app import main as projet9
from Projet10.app import main as projet10

st.set_page_config(page_title="Projets NLP", layout="wide")

st.title("📚 Projets NLP - M1 FDIA")

choix = st.sidebar.selectbox(
    "Choisissez un projet",
    [
        "Projet 1 – NER",
        "Projet 2 – Vectorisation",
        "Projet 3 – Topic Modeling",
        "Projet 4 – Analyse de sentiment",
        "Projet 5 – Classification",
        "Projet 6 – Détection de plagiat",
        "Projet 7 – Résumé automatique",
        "Projet 8 – Chatbot",
        "Projet 9 – Analyse de tendances",
        "Projet 10 – Recommandation",
    ]
)

if choix == "Projet1n_ner":
    projet1()

elif choix == "Projet2_vectorisation":
    projet2()

elif choix == "Projet3_topic_modeling":
    projet3()

elif choix == "Projet4_sentiment":
    projet4()

elif choix == "Projet5_classification":
    projet5()

elif choix == "Projet6_plagiat":
    projet6()

elif choix == "Projet7_resume":
    projet7()

elif choix == "Projet8_chatbot":
    projet8()

elif choix == "Projet9_tendances":
    projet9()

elif choix == "Projet10_recommandation":
    projet10()