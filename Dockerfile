FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Run as a non-root user. The model folder is created here so a mounted volume
# inherits the right owner.
RUN useradd --create-home appuser && mkdir -p /app/spam_detection_model \
    && chown -R appuser:appuser /app
COPY --chown=appuser:appuser app/ ./app/
USER appuser

EXPOSE 8000

# The model loads at startup, so give it time before the first check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request, os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
