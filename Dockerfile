FROM node:24-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run typecheck && npm run build

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgcc-s1 \
    libgl1 \
    libgthread-2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /app/frontend/dist ./webapp/static/frontend

RUN pip install --no-cache-dir -e webapp/GroundingDINO --no-build-isolation && \
    pip install --no-cache-dir -e webapp/MobileSAM --no-build-isolation

ENV PYTHONPATH=/app
ENV PORT=8080
ENV FLASK_APP=webapp/app.py
ENV ALLOW_MODEL_DOWNLOADS=0
ENV SECURITY_HEADERS_ENABLED=1

EXPOSE 8080

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/livez || exit 1

CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 300 --max-requests 50 --max-requests-jitter 10 --preload --access-logfile - --error-logfile - webapp.app:app
