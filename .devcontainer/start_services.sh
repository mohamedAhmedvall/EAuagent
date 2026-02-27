#!/usr/bin/env bash
# Démarre l'API FastAPI et l'IHM Streamlit en arrière-plan
# Logs : /tmp/api.log et /tmp/streamlit.log

set -e

WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "=== SOMEI — Démarrage des services ==="

# ── API FastAPI ──────────────────────────────────────────────────────────────
echo "[1/2] Démarrage API FastAPI sur http://localhost:8000 ..."
nohup uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    > /tmp/api.log 2>&1 &
echo "      PID=$! — logs : tail -f /tmp/api.log"

# ── IHM Streamlit ────────────────────────────────────────────────────────────
echo "[2/2] Démarrage IHM Streamlit sur http://localhost:8501 ..."
nohup streamlit run ihm/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    > /tmp/streamlit.log 2>&1 &
echo "      PID=$! — logs : tail -f /tmp/streamlit.log"

echo ""
echo "Services démarrés."
echo "  IHM       → http://localhost:8501"
echo "  API       → http://localhost:8000"
echo "  API docs  → http://localhost:8000/docs"
