# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales (compilation de certains paquets Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entraîne les modèles au moment du build de l'image : l'artefact final
# contient des modèles déjà prêts, aucun entraînement au démarrage du
# conteneur (démarrage rapide, comportement reproductible).
RUN python train.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
