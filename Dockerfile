# MiniMax-H3 serverless worker — uses runpod/pytorch which already has
# torch, transformers, accelerate, huggingface_hub, etc. pre-installed.
FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404

WORKDIR /

# Install ffmpeg (video muxing) and git.
RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

# Install only what's not already in the base image.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt

COPY rp_handler.py /rp_handler.py

CMD ["python3", "-u", "/rp_handler.py"]