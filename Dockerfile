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

# Install CPU-only PyTorch FIRST before anything else.
# Default pip torch includes CUDA (~2GB). CPU-only is ~200MB.
# This alone drops the image from 7GB → ~3GB.
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Now install the rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/screenshots static/challans videos

EXPOSE 5001

CMD ["python", "app.py"]