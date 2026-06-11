# RoadX — Dockerfile (slim, CPU-only PyTorch)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch — ~200MB vs 2GB CUDA version
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Install everything, then FORCE numpy+opencv back down last
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall "numpy==1.26.4" && \
    pip install --no-cache-dir --force-reinstall "opencv-python-headless==4.9.0.80"

COPY . .

RUN mkdir -p static/screenshots static/challans videos

EXPOSE 5001

# Shell form so $PORT expands correctly
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120