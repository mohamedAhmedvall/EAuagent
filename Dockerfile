FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY . .

# Ports : FastAPI 8000 + Streamlit 8501
EXPOSE 8000 8501

# Démarrage : API en arrière-plan + IHM au premier plan
CMD uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run ihm/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true
