FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create non-root user (use UID 1001 to avoid conflicts with base image's UID 1000)
RUN groupadd -r -g 1001 aris && useradd -r -g aris -u 1001 aris

COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api/*.py /app/
COPY shared /app/shared/

# Copy static files for web UI
COPY services/api/static /app/static

# Set ownership of /app to aris user
RUN chown -R aris:aris /app

# Switch to non-root user
USER aris

# Use environment variables for host and port
CMD ["sh", "-c", "uvicorn server:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]
