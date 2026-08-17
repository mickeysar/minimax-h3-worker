# MiniMax-H3 serverless worker — all from PyPI, no git installs.
FROM runpod/base:0.4.0-cuda12.1.0

WORKDIR /

# IMPORTANT — interpreter consistency (this was the crash-on-boot bug).
#
# runpod/base ships SEVERAL Python versions, and bare `pip` is not guaranteed
# to belong to the same interpreter as `python3`. The original Dockerfile did
#     pip install -r requirements.txt      (landed in interpreter A)
#     CMD ["python3", "/rp_handler.py"]    (ran interpreter B)
# so the worker died in ~1s with:
#     ModuleNotFoundError: No module named 'runpod'
#
# Do NOT "fix" this by switching to `python` — that binary does NOT exist in
# base 0.4.0 (only newer base tags symlink it), which fails the build with
# exit code 127. `python3 -m pip` is the portable form: it installs into the
# exact interpreter that the CMD below runs, on every base image version.

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /requirements.txt

# Fail the BUILD (not the first cold start) if the runtime interpreter cannot
# import the deps. Without this, a bad install only surfaces as a worker that
# dies in ~1s on every job.
RUN python3 -c "import sys, runpod, torch, diffusers; \
print('interpreter:', sys.executable, sys.version); \
print('runpod', runpod.__version__); \
print('torch', torch.__version__); \
print('diffusers', diffusers.__version__)"

COPY rp_handler.py /rp_handler.py

CMD ["python3", "-u", "/rp_handler.py"]
