FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    # SoapySDR for HackRF support
    soapysdr-tools \
    python3-soapysdr \
    soapysdr-module-hackrf \
    libhackrf0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/
COPY --from=shared ../../shared/models.py /app/shared/

CMD ["python", "capture.py"]
