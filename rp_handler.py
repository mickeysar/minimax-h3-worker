"""RunPod serverless worker for MiniMax-H3 (text/image-to-video).

Loads the MiniMax-H3 modular pipeline with CPU offload so the full
transformer + Qwen3-VL conditioner fit on a single 80-96 GB GPU.

Model resolution order:
  1. /workspace/models/MiniMax-H3  (network volume, if present)
  2. MiniMaxAI/MiniMax-H3          (Hugging Face, auto-download)
"""
import base64
import os
import runpod
import torch

# ---------------------------------------------------------------------------
# Model loading (lazy, once per worker)
# ---------------------------------------------------------------------------
_pipe = None
_manager = None

MODEL_LOCAL_DIR = "/workspace/models/MiniMax-H3"
MODEL_REPO_ID = "MiniMaxAI/MiniMax-H3"

# The single-GPU memory recipe from the diffusers MiniMax-H3 docs:
# register components in a ComponentsManager and let it move them on/off the
# accelerator. Tuned for an 80 GB card; on 96 GB (RTX PRO 6000) this leaves
# extra headroom, so the reserve margin can be a little smaller.
MEMORY_RESERVE_MARGIN = os.environ.get("MEMORY_RESERVE_MARGIN", "12GB")


def _model_source() -> str:
    """Return the model source: the local volume dir if populated, else the HF repo id."""
    marker = os.path.join(MODEL_LOCAL_DIR, "modular_model_index.json")
    if os.path.isdir(MODEL_LOCAL_DIR) and os.path.exists(marker):
        return MODEL_LOCAL_DIR
    return MODEL_REPO_ID


def load_model(workflow: str = "t2va"):
    """Load the pipeline. workflow: t2va | fl2va | ref2va."""
    global _pipe, _manager
    from diffusers import ComponentsManager, ModularPipeline

    source = _model_source()
    print(f"[MiniMax-H3] loading workflow={workflow} from {source}", flush=True)

    _manager = ComponentsManager()
    _pipe = ModularPipeline.from_pretrained(source, components_manager=_manager)
    _pipe.load_components(workflow=workflow, dtype=torch.bfloat16)
    _manager.enable_auto_cpu_offload(
        device="cuda", memory_reserve_margin=MEMORY_RESERVE_MARGIN
    )
    # Freeze the quantized/offloaded tensors' autograd path (not needed for inference).
    for comp in ("transformer", "text_encoder"):
        c = getattr(_pipe, comp, None)
        if c is not None:
            try:
                c.requires_grad_(False)
            except Exception:
                pass
    print("[MiniMax-H3] model loaded", flush=True)


def _pick_workflow(inp: dict) -> str:
    if inp.get("references"):
        return "ref2va"
    if inp.get("image") is not None or inp.get("last_image") is not None:
        return "fl2va"
    return "t2va"


def _encode_video(results: dict, out_path: str) -> None:
    from diffusers.utils.export_utils import encode_video

    encode_video(
        results["videos"][0],
        fps=24,
        output_path=out_path,
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )


def handler(job):
    global _pipe, _manager

    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt")
    if not prompt:
        return {"error": "`input.prompt` is required"}

    workflow = inp.get("workflow") or _pick_workflow(inp)
    num_frames = int(inp.get("num_frames", 124))
    seed = int(inp.get("seed", 42))
    height = inp.get("height")
    width = inp.get("width")

    if _pipe is None:
        load_model(workflow)

    # If the pipeline was loaded for a different workflow, reload.
    kwargs = {
        "prompt": prompt,
        "num_frames": num_frames,
        "generator": torch.Generator().manual_seed(seed),
        "output": ["videos", "audio", "sampling_rate"],
    }
    if workflow == "fl2va":
        from diffusers.utils import load_image

        if inp.get("image"):
            kwargs["image"] = load_image(inp["image"])
        if inp.get("last_image"):
            kwargs["last_image"] = load_image(inp["last_image"])
    elif workflow == "ref2va":
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        refs = []
        for r in inp.get("references", []):
            kind = r.get("type")
            if kind == "image":
                refs.append(MiniMaxH3ImageReference.from_file(r["uri"]))
            elif kind == "video":
                refs.append(MiniMaxH3VideoReference.from_file(r["uri"]))
            elif kind == "audio":
                refs.append(MiniMaxH3AudioReference.from_file(r["uri"]))
        kwargs["references"] = refs

    if height and width:
        kwargs["height"] = int(height)
        kwargs["width"] = int(width)

    results = _pipe(**kwargs)

    out_path = "/tmp/minimax_h3_output.mp4"
    _encode_video(results, out_path)

    with open(out_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("ascii")

    return {
        "video_base64": video_b64,
        "num_frames": num_frames,
        "seed": seed,
        "workflow": workflow,
    }


runpod.serverless.start({"handler": handler})
