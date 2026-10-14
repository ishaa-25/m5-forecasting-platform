FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[advanced]"

# Copy source
COPY src/ src/
COPY configs/ configs/
COPY app/ app/

# Default: run the FastAPI server
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "m5forecast.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
