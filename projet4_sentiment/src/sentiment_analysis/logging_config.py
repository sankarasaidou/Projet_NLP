# -*- coding: utf-8 -*-
"""Configuration du logging du projet.

En production, on utilise le module `logging` plutôt que `print()` :
- niveaux (DEBUG/INFO/WARNING/ERROR) filtrables sans changer le code,
- horodatage et origine (module) systématiques,
- redirigeable facilement vers un fichier ou un service de supervision
  (ex. CloudWatch, Datadog) sans toucher au code métier.
"""

import logging
import sys

from sentiment_analysis.config import LOG_LEVEL

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré, en configurant le handler racine
    une seule fois (évite les logs dupliqués si appelé plusieurs fois,
    par exemple à chaque rerun Streamlit)."""
    global _CONFIGURED

    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL, logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True

    return logging.getLogger(name)
