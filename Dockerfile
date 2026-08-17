# MiniMax-H3 serverless worker — all from PyPI, no git installs.
FROM runpod/base:0.4.0-cuda12.1.0

WORKDIR /

# IMPORTANT — interpreter consistency.
# runpod/base symlinks `python` -> python3.12 and `pip` -> pip3.12, but the
# image ALSO ships Ubuntu's own python3 (3.10/3.11) as /usr/bin/python3.
# `pip install` therefore lands in 3.12 while `python3 handler.py` runs a
# DIFFERENT interpreter that has none of those packages — which is exactly the
# "ModuleNotFoundError: No module named 'runpod'" crash-on-boot.
# Fix: always use `python -m pip` for installs and `python` (never `python3`)
# to run, so build-time and run-time are the same interpreter.

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /requirements.txt

# Fail the BUILD (not the first cold start) if the runtime interpreter cannot
# import the deps. Without this, a bad install only surfaces as a worker that
# dies in ~1s on every job.
RUN python -c "import sys, runpod, torch, diffusers; \
print('interpreter:', sys.executable, sys.version); \
print('runpod', runpod.__version__); \
print('torch', torch.__version__); \
print('diffusers', diffusers.__version__)"

COPY rp_handler.py /rp_handler.py

CMD ["python", "-u", "/rp_handler.py"]
