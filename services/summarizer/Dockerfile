FROM python:3.11-slim

WORKDIR /app

COPY services/summarizer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/summarizer/*.py /app/
COPY shared/models.py /app/shared/

CMD ["python", "summarize.py"]
