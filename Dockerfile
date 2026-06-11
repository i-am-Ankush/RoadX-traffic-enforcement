# RoadX — Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Wipe any numpy that got installed, reinstall clean
RUN pip uninstall -y numpy && \
    pip install --no-cache-dir "numpy==1.26.4" && \
    python -c "import numpy; print('BUILD numpy:', numpy.__version__, numpy.__file__)" && \
    python -c "import cv2; print('BUILD cv2 OK')"

COPY . .

# Remove any .venv or local numpy that got copied in
RUN rm -rf /app/.venv /app/venv && \
    find /app -name "numpy" -path "*/site-packages/numpy" ! -path "/usr/local/*" -exec rm -rf {} + 2>/dev/null || true

RUN mkdir -p static/screenshots static/challans videos

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120