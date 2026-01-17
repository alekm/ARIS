FROM ubuntu:24.04

WORKDIR /app

# Install Python 3.12 and system dependencies including SoapySDR
# Ubuntu 24.04 comes with Python 3.12, which matches the apt package requirements
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    # SoapySDR for HackRF support
    soapysdr-tools \
    soapysdr-module-hackrf \
    soapysdr0.8-module-hackrf \
    libhackrf0 \
    libsoapysdr0.8 \
    python3-soapysdr \
    # USB utilities for debugging
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install Python dependencies
# Note: --break-system-packages is needed for Ubuntu 24.04 (PEP 668), safe in containers
COPY services/audio-capture/requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Verify SoapySDR is accessible (apt package should work with Python 3.12)
RUN python3 -c "import SoapySDR; print('SoapySDR imported successfully')" || \
    (echo "ERROR: SoapySDR apt package not working!" && exit 1)

# Set module path so SoapySDR can find modules
# Ubuntu/Debian apt modules are typically in architecture-specific paths
# Source-built modules: /usr/local/lib/SoapySDR/modules0.8/
ENV SOAPY_SDR_MODULE_PATH=/usr/lib/x86_64-linux-gnu/SoapySDR/modules0.8:/usr/lib/soapysdr/modules0.8:/usr/local/lib/SoapySDR/modules0.8

COPY services/audio-capture/*.py /app/
COPY shared/models.py /app/shared/

CMD ["python", "capture.py"]
