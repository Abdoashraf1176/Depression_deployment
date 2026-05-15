FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .

# ✅ بدل ما نكوبي الموديل — بنحمله من HuggingFace
ENV HF_HOME=/app/.cache/huggingface
ENV MODEL_PATH=abdo1176/mental-roberta-finetuned

ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}