# Image de l'API d'inférence (FastAPI + modèle XGBoost figé).

# Image Python officielle minimale, même version mineure que le projet (.python-version).
FROM python:3.12-slim

# libgomp1 : runtime OpenMP requis par XGBoost ; le cache apt est supprimé pour alléger l'image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Binaire uv copié depuis l'image officielle, version épinglée (identique au poste de dev).
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Bytecode précompilé (démarrage plus rapide), copie des fichiers (pas de liens vers le cache),
# Python de l'image uniquement (aucun téléchargement), sortie des logs non mise en tampon.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Dossier de travail : les chemins relatifs de la config (artifacts/, data/...) partent d'ici.
WORKDIR /app

# Manifestes de dépendances seuls : cette couche ne change que si les dépendances changent.
COPY pyproject.toml uv.lock ./

# Dépendances de production uniquement, versions exactes du lock, sans le projet (mis en cache).
RUN uv sync --frozen --no-dev --no-install-project

# README : requis par uv_build (pyproject.toml le déclare comme readme du paquet).
COPY README.md ./
# Code source du package.
COPY src/ ./src/
# Configuration (bornes, coûts, textes d'action) lue au démarrage.
COPY config/ ./config/

# Installation du projet lui-même dans l'environnement déjà prêt.
RUN uv sync --frozen --no-dev

# Artefacts figés : modèle, seuils, empreinte (vérifiée au démarrage) et résultats du test.
COPY artifacts/xgboost.joblib artifacts/thresholds.json artifacts/model_fingerprint.json artifacts/test_results.json ./artifacts/

# Utilisateur système non-root (UID fixe) ; le code et les artefacts restent en lecture seule pour lui.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /app/data \
    && chown app:app /app/data

# Les exécutables de l'environnement virtuel (uvicorn, python) passent en priorité.
ENV PATH="/app/.venv/bin:$PATH"

# Toutes les commandes suivantes et le conteneur tournent sous cet utilisateur.
USER app

# Port d'écoute de l'API (documentation ; la publication se fait au lancement).
EXPOSE 8000

# Santé : /health doit répondre 200 (urllib lève une erreur sur 503) ; marge au démarrage
# pour le chargement du modèle et de l'explicateur SHAP.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# Serveur de production, sans rechargement automatique, accessible depuis l'hôte.
CMD ["uvicorn", "predictive_maintenance_iot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
