# RoadX — Dockerfile (Python 3.10)
FROM python:3.10-slim

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

RUN python -c "import numpy; print('numpy:', numpy.__version__)" && \
    python -c "import cv2; print('cv2:', cv2.__version__)" && \
    echo "ALL IMPORTS OK"

COPY . .
RUN rm -rf .venv venv
RUN mkdir -p static/screenshots static/challans videos

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120