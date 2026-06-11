FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 ffmpeg wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import numpy; print('numpy:', numpy.__version__)"
RUN python -c "import cv2; print('cv2:', cv2.__version__)"

COPY . .
RUN rm -rf .venv venv
RUN mkdir -p static/screenshots static/challans videos models

# Download models from Hugging Face at build time
ARG HF_TOKEN
RUN wget --header="Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/i-am-ankush/roadx-models/resolve/main/best.pt" \
    -O models/best.pt && \
    wget --header="Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/i-am-ankush/roadx-models/resolve/main/Plate.pt" \
    -O models/Plate.pt

RUN ls -lh models/

CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120