FROM python:3.11-slim

WORKDIR /app

# Create non-root user (use UID 1001 to avoid conflicts with base image's UID 1000)
RUN groupadd -r -g 1001 aris && useradd -r -g aris -u 1001 aris

COPY services/callsign-extractor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/callsign-extractor/*.py /app/
COPY shared/models.py /app/shared/

# Set ownership of /app to aris user
RUN chown -R aris:aris /app

# Switch to non-root user
USER aris

CMD ["python", "extractor.py"]
