# -*- coding: utf-8 -*-
"""
ÉTAPE 1 : Base de FAQ
---------------------------
Questions-réponses типiques des étudiants de l'Université Virtuelle du
Burkina Faso (UV-BF), organisées par intention. Chaque entrée a :
    - "intent"    : catégorie de la question (voir intents.py)
    - "question"  : formulation type (utilisée pour l'entraînement du
                     classifieur d'intentions)
    - "answer"    : réponse template, avec {ENTITY} à remplacer par une
                     entité détectée si besoin (ex. {MATIERE})
"""

FAQ_DATABASE = [
    {"intent": "inscription", "question": "Comment m'inscrire à l'UV-BF ?",
     "answer": "Les inscriptions se font en ligne sur la plateforme de l'UV-BF, "
               "avec vos relevés de notes et une pièce d'identité."},
    {"intent": "inscription", "question": "Quels documents sont nécessaires pour l'inscription ?",
     "answer": "Il vous faut : diplôme ou attestation, relevés de notes, "
               "pièce d'identité et une photo d'identité récente."},
    {"intent": "inscription", "question": "Quelle est la date limite d'inscription ?",
     "answer": "Les dates limites varient selon les filières ; consultez le "
               "calendrier académique publié sur le portail étudiant."},

    {"intent": "cours", "question": "Comment accéder aux cours de {MATIERE} ?",
     "answer": "Les cours de {MATIERE} sont disponibles sur la plateforme "
               "pédagogique, dans la rubrique 'Mes cours'."},
    {"intent": "cours", "question": "Où trouver le support de cours de {MATIERE} ?",
     "answer": "Le support de {MATIERE} est téléchargeable dans l'espace "
               "'Ressources pédagogiques' de votre espace étudiant."},
    {"intent": "cours", "question": "Les cours de {MATIERE} sont-ils disponibles hors ligne ?",
     "answer": "Oui, vous pouvez télécharger les supports de {MATIERE} en PDF "
               "pour une consultation hors ligne."},

    {"intent": "examen", "question": "Quand aura lieu l'examen de {MATIERE} ?",
     "answer": "Le calendrier des examens de {MATIERE} est publié sur le "
               "portail académique, section 'Examens'."},
    {"intent": "examen", "question": "Comment se déroule l'examen de {MATIERE} ?",
     "answer": "L'examen de {MATIERE} se déroule en ligne, surveillé, avec un "
               "temps limité affiché à l'écran."},
    {"intent": "examen", "question": "Que faire en cas de problème technique pendant l'examen ?",
     "answer": "Contactez immédiatement le support technique via le numéro "
               "d'urgence affiché sur la plateforme d'examen."},

    {"intent": "technique", "question": "Je n'arrive pas à me connecter à la plateforme.",
     "answer": "Vérifiez votre connexion internet et réinitialisez votre mot "
               "de passe via 'Mot de passe oublié'. Sinon, contactez le support technique."},
    {"intent": "technique", "question": "La vidéo du cours ne se charge pas.",
     "answer": "Essayez de rafraîchir la page ou de changer de navigateur. "
               "Si le problème persiste, signalez-le au support technique."},
    {"intent": "technique", "question": "Comment réinitialiser mon mot de passe ?",
     "answer": "Cliquez sur 'Mot de passe oublié' sur la page de connexion et "
               "suivez les instructions envoyées par email."},

    {"intent": "administratif", "question": "Où trouver mon certificat de scolarité ?",
     "answer": "Le certificat de scolarité est téléchargeable dans l'espace "
               "'Documents administratifs' de votre compte étudiant."},
    {"intent": "administratif", "question": "Comment contacter le service de la scolarité ?",
     "answer": "Le service de la scolarité est joignable par email ou via le "
               "formulaire de contact du portail UV-BF."},
    {"intent": "administratif", "question": "Comment obtenir une attestation de réussite ?",
     "answer": "L'attestation de réussite est disponible après délibération, "
               "dans l'espace 'Documents administratifs'."},
]

# Vocabulaire d'entités "MATIERE" reconnues (pour le NER personnalisé, étape 2)
MATIERES = [
    "mathématiques", "informatique", "NLP", "statistiques", "réseaux",
    "bases de données", "algorithmique", "anglais", "gestion de projet",
]

INTENTS = sorted({item["intent"] for item in FAQ_DATABASE})
