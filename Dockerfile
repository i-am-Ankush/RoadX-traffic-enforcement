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

# Install CPU-only PyTorch FIRST — ~200MB vs 2GB CUDA version
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Pin numpy BEFORE opencv — this is the fix for the multiarray import error
# torch 2.2 ships with numpy 1.x ABI; opencv must see the same numpy it was built against
RUN pip install --no-cache-dir "numpy==1.26.4"

# Now install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/screenshots static/challans videos

EXPOSE 5001

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "120"]