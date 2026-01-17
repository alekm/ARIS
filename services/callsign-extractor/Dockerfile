FROM python:3.11-slim

WORKDIR /app

COPY services/callsign-extractor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/callsign-extractor/*.py /app/
COPY shared/models.py /app/shared/

CMD ["python", "extractor.py"]
