# MiniMax-H3 serverless worker — all from PyPI, no git installs.
FROM runpod/base:0.4.0-cuda12.1.0

WORKDIR /

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /requirements.txt

COPY rp_handler.py /rp_handler.py

CMD ["python3", "-u", "/rp_handler.py"]