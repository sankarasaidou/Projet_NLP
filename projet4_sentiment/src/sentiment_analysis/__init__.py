# -*- coding: utf-8 -*-
"""Package d'analyse de sentiment — Projet 4 (version production).

Structure :
    config.py          - chemins et paramètres centralisés
    logging_config.py   - configuration du logging (remplace les print())
    preprocessing.py    - nettoyage de texte, un seul pipeline partagé
    data_loader.py       - chargement du dataset et du lexique depuis /data
    lexical.py           - approche lexicale (dictionnaire de polarité)
    statistical.py        - approche statistique (TF-IDF + SVM/NB/LogReg)
    neural.py             - approche neuronale (CamemBERT, optionnelle)
    evaluation.py          - métriques et comparaison des approches
    pipeline.py             - façade unique combinant les 3 approches
"""

__version__ = "1.0.0"
