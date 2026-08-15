# web_demo/Dockerfile — Mudqeq AI public demo (Hugging Face Docker Space)
#
# Build context MUST be scoped to web_demo/ (see README). No desktop, no
# private data, no secrets are copied in.

FROM python:3.11-slim

# faiss-cpu needs OpenMP on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Runtime environment defaults -----------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    DEMO_STORAGE_ROOT=/tmp/mudqeq_demo \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# --- Non-root user --------------------------------------------------------
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# --- Python dependencies (CPU torch) --------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/requirements.txt

# --- Application code (context = web_demo/) --------------------------------
COPY . /app

# Ownership for the non-root user (incl. cache + temp dirs).
RUN mkdir -p /tmp/mudqeq_demo /home/appuser/.cache \
    && chown -R appuser:appuser /app /tmp/mudqeq_demo /home/appuser

USER appuser

# --- Bake the embedding model into the image (predictable runtime) --------
# Downloaded at build time so the running container needs no HF network call.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('intfloat/multilingual-e5-small')"

# After baking, run fully offline for the model (no runtime HF egress).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860

# --- Healthcheck (Streamlit's built-in health endpoint) -------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true"]
