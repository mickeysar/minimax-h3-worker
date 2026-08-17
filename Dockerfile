# MiniMax-H3 serverless worker.
FROM runpod/base:0.4.0-cuda12.1.0

WORKDIR /

# IMPORTANT — interpreter consistency (this was the crash-on-boot bug).
#
# runpod/base:0.4.0 installs python3.7 .. python3.11 side by side and ran
# get-pip.py for each, so bare `pip` resolves to python3.11 while `python3` is
# Ubuntu's default 3.10. The original Dockerfile did
#     pip install -r requirements.txt      -> landed in 3.11
#     CMD ["python3", "/rp_handler.py"]    -> ran 3.10
# so the worker died in ~1s on every job with:
#     ModuleNotFoundError: No module named 'runpod'
#
# Do NOT "fix" this with `python`: that binary does not exist in base 0.4.0
# (only newer base tags symlink it) and the build fails with exit code 127.
# `python3 -m pip` is the portable form — it installs into the exact
# interpreter the CMD runs, on any base image version.

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /requirements.txt

# Verify at BUILD time, on the same interpreter that will run the handler.
# Both checks below have already failed for real in this project:
#   - missing 'runpod'            -> crash-on-boot every job
#   - diffusers without minimax_h3 -> loads fine, then fails at model load
# Catching them here costs one build instead of one wasted GPU cold start.
RUN python3 -c "import sys, importlib.util; \
import runpod, torch, transformers, diffusers; \
print('interpreter:', sys.executable, sys.version); \
print('runpod', runpod.__version__); \
print('torch', torch.__version__); \
print('transformers', transformers.__version__); \
print('diffusers', diffusers.__version__); \
spec = importlib.util.find_spec('diffusers.modular_pipelines.minimax_h3'); \
print('minimax_h3 present:', spec is not None); \
sys.exit(0 if spec is not None else 'FATAL: diffusers lacks MiniMax-H3 support (needs git main, not a PyPI release)')"

COPY rp_handler.py /rp_handler.py
COPY download_weights.sh /download_weights.sh
RUN chmod +x /download_weights.sh

CMD ["python3", "-u", "/rp_handler.py"]
