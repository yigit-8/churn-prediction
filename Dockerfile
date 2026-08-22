FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Train the model at build time so the image is self-contained
RUN python -m src.train

# Run as non-root (required by HF Spaces, good practice in general)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Reads PORT the same way the CMD below does, so the probe follows the port the
# app actually binds to when a PaaS injects one.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health', timeout=3).status==200 else 1)"

# Shell form so ${PORT} expands: most PaaS hosts (e.g. Render) inject PORT and
# expect the container to bind to it; falls back to 8000 for HF Spaces/local.
CMD uvicorn src.serve:app --host 0.0.0.0 --port ${PORT:-8000}
