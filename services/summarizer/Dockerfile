FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/
COPY --from=shared ../../shared/models.py /app/shared/

CMD ["python", "summarize.py"]
