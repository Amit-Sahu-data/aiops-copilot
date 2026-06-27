FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY agents/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and runbooks
COPY agents/ .
COPY runbooks/ ../runbooks/

# Build the runbook vector index
RUN python build_doc_index.py

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8088/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8088"]