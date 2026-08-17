"""RunPod serverless worker for MiniMax-H3 (text/image-to-video).

Loads the MiniMax-H3 modular pipeline with CPU offload so the full
transformer + Qwen3-VL conditioner fit on a single 80-96 GB GPU.

Model resolution:
  The weights MUST be pre-staged on the attached network volume. Run
  download_weights.sh once from a pod with the same volume mounted.
  This worker deliberately does NOT fall back to downloading from the Hub —
  see _model_source() for why.
"""
import base64
import os
import sys
import traceback

import runpod
import torch

# ---------------------------------------------------------------------------
# Model loading (lazy, once per worker)
# ---------------------------------------------------------------------------
_pipe = None
_manager = None
_loaded_workflow = None

# Serverless mounts network volumes at /runpod-volume; pods mount at /workspace.
# Check both so the same image works in either place.
_MODEL_DIRS = [
    "/runpod-volume/models/MiniMax-H3",
    "/workspace/models/MiniMax-H3",
]
MODEL_REPO_ID = "MiniMaxAI/MiniMax-H3"

# Allow an explicit escape hatch, but make it opt-in and loud. The HF repo is
# ~464 GB; letting a worker fall back to it silently means every cold start
# tries a multi-hundred-GB download, blows the job timeout, and bills GPU time
# for nothing. Failing fast with a clear message is strictly better.
ALLOW_HUB_DOWNLOAD = os.environ.get("ALLOW_HUB_DOWNLOAD", "0") == "1"

MEMORY_RESERVE_MARGIN = os.environ.get("MEMORY_RESERVE_MARGIN", "12GB")


def _model_source() -> str:
    """Return the local weights dir, or raise with an actionable message."""
    checked = []
    for d in _MODEL_DIRS:
        marker = os.path.join(d, "modular_model_index.json")
        if os.path.isdir(d) and os.path.exists(marker):
            return d
        checked.append(f"{marker} (dir={os.path.isdir(d)})")

    if ALLOW_HUB_DOWNLOAD:
        print(
            "[MiniMax-H3] WARNING: no local weights; downloading from the Hub. "
            "This repo is ~464 GB and will likely exceed the job timeout.",
            flush=True,
        )
        return MODEL_REPO_ID

    raise RuntimeError(
        "MiniMax-H3 weights not found on the network volume.\n"
        "Looked for:\n  " + "\n  ".join(checked) + "\n\n"
        "Fix: attach the network volume and run download_weights.sh once from a "
        "pod with that volume mounted (it writes models/MiniMax-H3/).\n"
        "Refusing to download ~464 GB from the Hub on a cold start. Set "
        "ALLOW_HUB_DOWNLOAD=1 to override (not recommended)."
    )


def _diffusers_supports_h3() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("diffusers.modular_pipelines.minimax_h3")
        is not None
    )


def load_model(workflow: str = "t2va"):
    """Load the pipeline. workflow: t2va | fl2va | ref2va."""
    global _pipe, _manager, _loaded_workflow
    import diffusers
    from diffusers import ComponentsManager, ModularPipeline

    # MiniMax-H3 exists only on diffusers main; no PyPI release has it (0.39.0
    # ships no minimax module at all). Say so explicitly instead of dying on an
    # opaque "unknown pipeline class" error.
    if not _diffusers_supports_h3():
        raise RuntimeError(
            f"Installed diffusers {diffusers.__version__} has no MiniMax-H3 "
            "support (diffusers.modular_pipelines.minimax_h3 is missing).\n"
            "MiniMax-H3 is only on the git main branch — no PyPI release "
            "includes it. requirements.txt must install:\n"
            "  git+https://github.com/huggingface/diffusers.git@main"
        )

    source = _model_source()
    print(
        f"[MiniMax-H3] loading workflow={workflow} from {source} "
        f"(diffusers {diffusers.__version__})",
        flush=True,
    )

    _manager = ComponentsManager()
    _pipe = ModularPipeline.from_pretrained(source, components_manager=_manager)
    _pipe.load_components(workflow=workflow, dtype=torch.bfloat16)
    _manager.enable_auto_cpu_offload(
        device="cuda", memory_reserve_margin=MEMORY_RESERVE_MARGIN
    )
    for comp in ("transformer", "text_encoder"):
        c = getattr(_pipe, comp, None)
        if c is not None:
            try:
                c.requires_grad_(False)
            except Exception:
                pass
    _loaded_workflow = workflow
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
    global _pipe, _manager, _loaded_workflow

    try:
        inp = job.get("input", {}) or {}
        prompt = inp.get("prompt")
        if not prompt:
            return {"error": "`input.prompt` is required"}

        workflow = inp.get("workflow") or _pick_workflow(inp)
        num_frames = int(inp.get("num_frames", 124))
        seed = int(inp.get("seed", 42))
        height = inp.get("height")
        width = inp.get("width")

        # The original code loaded the pipeline only when _pipe was None, so a
        # worker warmed up on t2va would keep serving a later fl2va/ref2va job
        # with the wrong components loaded. Reload when the workflow changes.
        if _pipe is None or _loaded_workflow != workflow:
            if _pipe is not None:
                print(
                    f"[MiniMax-H3] workflow changed "
                    f"{_loaded_workflow} -> {workflow}; reloading",
                    flush=True,
                )
                _pipe = None
                _manager = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            load_model(workflow)

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
                else:
                    return {"error": f"unknown reference type: {kind!r}"}
            kwargs["references"] = refs

        if height and width:
            kwargs["height"] = int(height)
            kwargs["width"] = int(width)

        results = _pipe(**kwargs)

        out_path = "/tmp/minimax_h3_output.mp4"
        _encode_video(results, out_path)

        with open(out_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("ascii")

        # RunPod caps job output payloads; a base64 mp4 can exceed it. Report
        # the size so an over-limit response is diagnosable instead of just
        # vanishing.
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[MiniMax-H3] output {size_mb:.1f} MB", flush=True)

        return {
            "video_base64": video_b64,
            "size_mb": round(size_mb, 1),
            "num_frames": num_frames,
            "seed": seed,
            "workflow": workflow,
        }

    except Exception as e:
        # Return the traceback in the job result. Without this the only trace of
        # a failure is in worker logs the caller cannot see.
        traceback.print_exc()
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    # Fail fast and visibly on a broken image rather than after the first job.
    print(f"[MiniMax-H3] python {sys.version}", flush=True)
    print(f"[MiniMax-H3] torch {torch.__version__}", flush=True)
    try:
        import diffusers

        print(
            f"[MiniMax-H3] diffusers {diffusers.__version__} "
            f"minimax_h3={_diffusers_supports_h3()}",
            flush=True,
        )
    except Exception as e:
        print(f"[MiniMax-H3] diffusers import failed: {e}", flush=True)

    runpod.serverless.start({"handler": handler})
