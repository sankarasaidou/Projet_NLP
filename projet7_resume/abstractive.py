# -*- coding: utf-8 -*-
"""
ÉTAPE 3 : Résumé abstractif
---------------------------------
Contrairement au résumé extractif (qui sélectionne des phrases
existantes), le résumé abstractif GÉNÈRE de nouvelles phrases, en
reformulant. On utilise des modèles pré-entraînés multilingues via
Hugging Face `transformers` :

    - mT5 / T5 francophone (ex. "plguillou/t5-base-fr-sum-cnndm",
      fine-tuné spécifiquement pour le résumé en français)
    - mBART (facebook/mbart-large-50, multilingue, à utiliser avec le
      bon code de langue "fr_XX")

Nécessite `transformers` + `torch` (paquets lourds, non installés dans
ce bac à sable sans accès réseau). Le code suit l'API standard
`pipeline("summarization", ...)` de Hugging Face.
"""

MODEL_CHECKPOINTS = {
    "T5 (français)": "plguillou/t5-base-fr-sum-cnndm",
    "mBART (multilingue)": "facebook/mbart-large-50",
}

_pipelines_cache = {}


def summarize_abstractive(text: str, model_name: str = "T5 (français)",
                           max_length: int = 130, min_length: int = 30) -> str:
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "Ce module nécessite `transformers` et `torch` : "
            "`pip install transformers torch sentencepiece`"
        ) from e

    if model_name not in _pipelines_cache:
        checkpoint = MODEL_CHECKPOINTS[model_name]
        kwargs = {}
        if model_name == "mBART (multilingue)":
            # mBART nécessite de préciser la langue source/cible
            kwargs = {"src_lang": "fr_XX", "tgt_lang": "fr_XX"}
        _pipelines_cache[model_name] = pipeline("summarization", model=checkpoint, **kwargs)

    summarizer = _pipelines_cache[model_name]
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]


ABSTRACTIVE_METHODS = {
    "T5 (français)": lambda text, **kw: summarize_abstractive(text, "T5 (français)", **kw),
    "mBART (multilingue)": lambda text, **kw: summarize_abstractive(text, "mBART (multilingue)", **kw),
}


if __name__ == "__main__":
    from corpus import DOCUMENTS
    try:
        print(summarize_abstractive(DOCUMENTS["Article de presse"]))
    except ImportError as e:
        print(f"Non exécuté ici : {e}")
