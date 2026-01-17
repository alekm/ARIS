FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api/*.py /app/
COPY shared/models.py /app/shared/

# Copy static files for web UI
COPY services/api/static /app/static

# Use environment variables for host and port
CMD ["sh", "-c", "uvicorn server:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]
