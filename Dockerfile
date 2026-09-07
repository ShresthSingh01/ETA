# RailETA - SIH 26028 Production Container
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential for LightGBM/C dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, data, models, frontend, docs
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY frontend/ ./frontend/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Create logs directory
RUN mkdir -p logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
