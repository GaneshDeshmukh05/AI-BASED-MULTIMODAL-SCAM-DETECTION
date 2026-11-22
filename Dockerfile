FROM python:3.11-slim

# Prevent Python from buffering stdout & writing bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for audio decoding, image OCR, and building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt plotly fastapi uvicorn streamlit

# Copy repository source code
COPY . .

# Expose Web Dashboard (8501) and REST API (8000)
EXPOSE 8501 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: start Streamlit interactive UI
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"]

