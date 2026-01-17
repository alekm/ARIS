FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/
COPY --from=shared ../../shared/models.py /app/shared/

CMD ["python", "transcribe.py"]
