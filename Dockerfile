# AI Doctor API - FastAPI backend
FROM python:3.11-slim

WORKDIR /app

# Install system deps (for pytesseract/OCR if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (skip heavy torch if you want a smaller image; uncomment next line)
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart \
    requests aiohttp pydantic python-dotenv pyyaml \
    huggingface-hub google-generativeai \
    chromadb sentence-transformers \
    Pillow PyPDF2 pandas numpy

# Copy application code
COPY api_simple.py .
COPY config.yaml .
COPY models/ ./models/
COPY training/ ./training/
COPY utils/ ./utils/

# Create data dirs (ChromaDB, feedback, etc.)
RUN mkdir -p ai_doctor_data/chromadb ai_doctor_data/feedback users_db

# PYTHONPATH so "data_manager" and "models" resolve when running from /app
ENV PYTHONPATH=/app:/app/models

# Optional: pass HF_TOKEN / GEMINI_API_KEY via env at runtime
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api_simple:app", "--host", "0.0.0.0", "--port", "8000"]
