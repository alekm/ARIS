FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

WORKDIR /app

# Install Python and dependencies (Ubuntu 24.04 comes with Python 3.12)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install Python dependencies
# Note: --break-system-packages is needed for Ubuntu 24.04 (PEP 668), safe in containers
# Skip pip upgrade since it's installed via apt and can't be upgraded with pip itself
# Using BuildKit cache mount to speed up rebuilds
COPY services/stt/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Verify numpy is installed
RUN python3 -c "import numpy; print('NumPy version:', numpy.__version__)" || \
    (echo "Error: NumPy not installed!" && exit 1)

COPY services/stt/*.py /app/
COPY shared/models.py /app/shared/

CMD ["python", "transcribe.py"]
