# Image Classification Project Dockerfile
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create outputs directory (will be populated at runtime or via volume mount)
RUN mkdir -p outputs/models outputs/figures outputs/logs

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY streamlit/ ./streamlit/
COPY scripts/ ./scripts/

# Expose ports
EXPOSE 8000 8501

# Default command: start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
