# Use slim Python image to reduce build size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache layer optimization)
COPY requirements.txt .

# Install Python dependencies
# torch CPU-only to save ~1GB of image size vs full torch
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY main.py .

# Copy fine-tuned model folder IF it exists locally.
# Comment this line out if you want to always pull from HuggingFace.
COPY mental_roberta_final/ ./mental_roberta_final/

# HuggingFace cache directory (optional, speeds up cold starts)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Point the app at your fine-tuned folder
ENV MODEL_PATH=./mental_roberta_final

# Railway injects $PORT at runtime
ENV PORT=8000

EXPOSE 8000

# Start the server — Railway passes $PORT automatically
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
