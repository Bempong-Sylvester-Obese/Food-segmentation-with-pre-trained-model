FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PORT=8080
ENV FLASK_APP=webapp/app.py

RUN mkdir -p webapp/static/images webapp/static/GeneratedImages

EXPOSE 8080

RUN useradd -m -u 1000 cloudrunuser && chown -R cloudrunuser:cloudrunuser /app
USER cloudrunuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 --preload --access-logfile - --error-logfile - webapp.app:app
