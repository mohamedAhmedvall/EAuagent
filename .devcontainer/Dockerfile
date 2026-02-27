FROM python:3.11-slim

# Dépendances système
RUN apt-get update && apt-get install -y \
    git \
    curl \
    coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pré-installation des dépendances Python (layer mis en cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
