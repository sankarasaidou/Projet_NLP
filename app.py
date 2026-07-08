# -*- coding: utf-8 -*-
"""
Launcher unique — Plateforme des Projets NLP M1 FDIA
---------------------------------------------------------
Ce fichier doit être placé à la RACINE du dépôt, au même niveau que les
10 dossiers projet1_ner, projet2_vectorisation, ..., projet10_recommandation.

Corrige 2 bugs bloquants de la version précédente :

1. `st.set_page_config()` appelé par CHACUN des 10 sous-projets ET par ce
   launcher -> Streamlit interdit un second appel et l'app plante dès
   qu'on sélectionne un projet. On neutralise (monkey-patch) l'appel des
   sous-projets, sans avoir à modifier leurs 10 fichiers.

2. Collision de modules Python : plusieurs projets ont des fichiers de
   même nom (preprocessing.py présent dans 7 projets, corpus.py dans 3,
   data.py dans 3, evaluation.py dans 3...). En insérant tous les
   dossiers de projets dans sys.path en même temps (comme dans la
   version précédente), Python peut charger le preprocessing.py du
   MAUVAIS projet et provoquer des bugs silencieux. On isole donc
   strictement le sys.path et le cache d'imports (sys.modules) au seul
   projet actuellement sélectionné.

Bonus : plus besoin d'une fonction main() dans chaque app.py (qui
n'existait pas dans tes 10 projets) -> on exécute chaque app.py
directement avec `runpy`, exactement comme le ferait `streamlit run`.
"""

import os
import sys
import runpy

import streamlit as st

# --- Config de page : une seule fois, ici, pour toute l'application ---
st.set_page_config(page_title="Projets NLP - M1 FDIA", layout="wide", page_icon="📚")

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Nom affiché -> nom exact du dossier sur le disque (à adapter si tu
# renommes un dossier).
PROJECTS = {
    "Projet 1 – NER spécifique IA": "projet1_ner",
    "Projet 2 – Comparaison de vectorisation": "projet2_vectorisation",
    "Projet 3 – Modélisation de sujets": "projet3_topic_modeling",
    "Projet 4 – Analyse de sentiment": "projet4_sentiment",
    "Projet 5 – Classification d'articles de presse": "projet5_classification",
    "Projet 6 – Détecteur de plagiat": "projet6_plagiat",
    "Projet 7 – Générateur de résumés": "projet7_resume",
    "Projet 8 – Chatbot FAQ UV-BF": "projet8_chatbot",
    "Projet 9 – Analyseur de tendances": "projet9_tendances",
    "Projet 10 – Système de recommandation": "projet10_recommandation",
}

ALL_PROJECT_DIRS = [os.path.join(ROOT_DIR, folder) for folder in PROJECTS.values()]


def _isolate_project_imports(project_dir: str) -> None:
    """Empêche les collisions de modules entre projets.

    Plusieurs projets définissent des modules du même nom
    (preprocessing.py, corpus.py, data.py, evaluation.py...). Si on
    laissait tous les dossiers de projets dans sys.path en même temps,
    Python pourrait réutiliser le mauvais module d'un projet à l'autre
    (bug silencieux, vérifié et reproduit avant cette correction).

    On ne garde donc dans sys.path QUE le dossier du projet actif, et on
    vide sys.modules de tout module précédemment chargé depuis un AUTRE
    dossier de projet, pour forcer un rechargement propre et isolé.
    """
    sys.path[:] = [p for p in sys.path if p not in ALL_PROJECT_DIRS]
    sys.path.insert(0, project_dir)

    for mod_name, mod in list(sys.modules.items()):
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        for other_dir in ALL_PROJECT_DIRS:
            if os.path.isdir(other_dir) and os.path.commonpath([mod_file, other_dir]) == other_dir:
                del sys.modules[mod_name]
                break


# --- Neutralise les appels à st.set_page_config() faits par les sous-projets ---
# (chacun des 10 app.py en contient un ; on ne peut pas l'appeler 2 fois).
def _noop_set_page_config(*args, **kwargs):
    pass


st.set_page_config_original = st.set_page_config  # gardé au cas où, non utilisé
st.set_page_config = _noop_set_page_config

# --- Barre latérale : sélection du projet ---
st.sidebar.title("📚 Projets NLP — M1 FDIA")
choice = st.sidebar.selectbox("Choisissez un projet à exécuter :", list(PROJECTS.keys()))
st.sidebar.markdown("---")
st.sidebar.success(f"📈 Module actif : {choice}")

project_folder = PROJECTS[choice]
project_dir = os.path.join(ROOT_DIR, project_folder)
app_file = os.path.join(project_dir, "app.py")

# On ne ré-isole (et ne vide le cache) QUE si l'utilisateur change de
# projet -- pas à chaque interaction dans le même projet, pour ne pas
# perdre inutilement le bénéfice de st.cache_resource / st.cache_data.
if st.session_state.get("_active_project") != project_folder:
    _isolate_project_imports(project_dir)
    st.session_state["_active_project"] = project_folder

# --- Exécution du sous-projet sélectionné ---
if not os.path.isfile(app_file):
    st.error(
        f"⚠️ Fichier introuvable : `{app_file}`.\n\n"
        f"Vérifie que le dossier `{project_folder}` est bien présent à la "
        f"racine du dépôt, au même niveau que ce launcher."
    )
else:
    try:
        runpy.run_path(app_file, run_name="__main__")
    except Exception as e:
        st.error(f"❌ Erreur en exécutant {choice} :\n\n{e}")
        st.exception(e)
