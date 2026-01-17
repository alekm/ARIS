FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/
COPY --from=shared ../../shared/models.py /app/shared/

# Copy static files for web UI
COPY static /app/static

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
