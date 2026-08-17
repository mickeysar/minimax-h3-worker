#!/bin/bash
# Download MiniMax-H3 diffusers-format weights to the network volume.
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download MiniMaxAI/MiniMax-H3 \
  --local-dir /workspace/models/MiniMax-H3 \
  --exclude "FL2VA/*" \
  --exclude "Ref2VA/*" \
  --exclude "assets/*" \
  --exclude "docs/*" \
  --exclude "scripts/*" \
  --exclude "*.md" \
  --exclude "*.png" \
  --exclude "*.gif" \
  --exclude "*.mp4" \
  2>&1 | tail -20
echo "DOWNLOAD_FINISHED"
