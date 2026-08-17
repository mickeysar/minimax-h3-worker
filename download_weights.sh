#!/usr/bin/env bash
# Download MiniMax-H3 diffusers-format weights onto the network volume.
#
# Run this ONCE from a temporary GPU pod that has the same network volume
# attached, THEN the serverless workers start with weights already present and
# never download anything on a cold start.
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT WAS REWRITTEN (three real bugs in the original):
#
# 1. WRONG MOUNT PATH. It wrote to /workspace/models/MiniMax-H3. Pods mount the
#    volume at /workspace, but SERVERLESS mounts it at /runpod-volume — so the
#    worker looked in /runpod-volume/models/MiniMax-H3, found nothing, and fell
#    back to downloading ~all of a 464 GB repo on every cold start.
#    This script now auto-detects the mount instead of hardcoding one.
#
# 2. IT EXCLUDED THE WRONG THINGS AND KEPT 195 GB. Excluding only
#    FL2VA/Ref2VA/assets/docs/scripts still pulls text_encoder (62 GB) +
#    transformer (61.7) + transformer_ref (61.7) + vae (9.7) = ~196 GB into a
#    250 GB volume. transformer_ref is needed ONLY for the ref2va workflow.
#    Default here is the t2va/fl2va set (~134 GB) with ref2va opt-in.
#
# 3. NO VERIFICATION. It printed DOWNLOAD_FINISHED even if hf failed, and never
#    checked that modular_model_index.json — the one file the worker probes for
#    — actually landed. A partial download looked like a success.
#
# Repo layout (sizes measured from the HF API, 2026-08-17):
#   FL2VA/           134.16 GB   standalone bundle, NOT used by this pipeline
#   Ref2VA/          134.16 GB   standalone bundle, NOT used by this pipeline
#   text_encoder/     62.14 GB   REQUIRED (Qwen3VL)
#   transformer/      61.73 GB   REQUIRED (t2va + fl2va)
#   transformer_ref/  61.73 GB   only for workflow=ref2va
#   vae/               9.70 GB   REQUIRED
#   audio_vae/         0.56 GB   REQUIRED
#   + tiny config/tokenizer/processor/scheduler dirs
# ---------------------------------------------------------------------------
set -Eeuo pipefail

WITH_REF2VA="${WITH_REF2VA:-0}"   # 1 = also fetch transformer_ref (+61.73 GB)

# Auto-detect where the network volume actually is.
if [[ -n "${MODEL_ROOT:-}" ]]; then
    ROOT="$MODEL_ROOT"
elif [[ -d /runpod-volume ]]; then
    ROOT=/runpod-volume            # serverless
elif [[ -d /workspace ]]; then
    ROOT=/workspace                # pod
else
    echo "[FAIL] no /runpod-volume or /workspace mount found." >&2
    exit 1
fi
DEST="$ROOT/models/MiniMax-H3"
echo "[INFO] volume root : $ROOT"
echo "[INFO] destination : $DEST"

echo "[INFO] free space:"
df -h "$ROOT" || true

command -v hf >/dev/null 2>&1 || python3 -m pip install --quiet --upgrade "huggingface_hub>=0.25.0" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$DEST"

# Allow-list, not deny-list: fetch exactly the components the pipeline loads.
INCLUDE=(
    --include "modular_model_index.json"
    --include "model_index.json"
    --include "text_encoder/*"
    --include "tokenizer/*"
    --include "processor/*"
    --include "vae/*"
    --include "audio_vae/*"
    --include "scheduler/*"
    --include "audio_scheduler/*"
    --include "transformer/*"
)
if [[ "$WITH_REF2VA" == "1" ]]; then
    echo "[INFO] including transformer_ref (ref2va, +61.73 GB)"
    INCLUDE+=( --include "transformer_ref/*" )
else
    echo "[INFO] skipping transformer_ref — set WITH_REF2VA=1 if you need workflow=ref2va"
fi

echo "[INFO] downloading (resumable; safe to re-run)..."
hf download MiniMaxAI/MiniMax-H3 --local-dir "$DEST" "${INCLUDE[@]}"

# --- verification: fail loudly rather than printing a false success ---
echo "[INFO] verifying..."
fail=0
for f in modular_model_index.json text_encoder vae audio_vae transformer tokenizer processor; do
    if [[ -e "$DEST/$f" ]]; then
        echo "  [ OK ] $f"
    else
        echo "  [FAIL] MISSING: $f" >&2
        fail=1
    fi
done
if [[ "$WITH_REF2VA" == "1" && ! -e "$DEST/transformer_ref" ]]; then
    echo "  [FAIL] MISSING: transformer_ref" >&2
    fail=1
fi

# The worker probes for this exact file; if it is absent the worker silently
# falls back to a multi-hundred-GB download on every cold start.
if [[ ! -f "$DEST/modular_model_index.json" ]]; then
    echo "[FAIL] modular_model_index.json missing — serverless would re-download everything." >&2
    exit 1
fi

echo "[INFO] size on disk: $(du -sh "$DEST" 2>/dev/null | cut -f1)"
(( fail == 0 )) || { echo "[FAIL] incomplete download; re-run this script." >&2; exit 1; }
echo "DOWNLOAD_FINISHED"
