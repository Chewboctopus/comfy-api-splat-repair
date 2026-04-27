"""
comfyui-fal-splat
=================
A ComfyUI custom node suite for the SHARP → Gaussian Splat → fal.ai repair workflow.

Nodes provided
--------------
FAL-Splat/Models
  • Fal Model Browser       — query & filter the fal.ai model catalogue dynamically
  • Fal Schema Loader       — fetch a model's OpenAPI parameter schema
  • Fal Dynamic Image Gen   — call ANY fal image model with LoRA + img2img support

FAL-Splat/LoRA
  • Fal LoRA from URL       — normalise HuggingFace / CivitAI / fal LoRA URLs
  • Fal LoRA from Local     — upload a local .safetensors to fal storage
  • Fal LoRA Stack          — combine up to 4 LoRAs into a JSON array

FAL-Splat/Repair
  • Splat Artifact Mask     — auto-detect artifact regions (original vs reposed diff)
  • Splat Difference Debug  — visualise pixel-level differences for threshold tuning
  • Flux Splat Repair       — img2img repair via Flux Klein/General + LoRA (fal.ai)
  • Flux Splat Inpaint      — hard-mask inpainting via Flux Pro Fill (fal.ai)
  • Qwen Splat Repair       — VLM-powered analysis and/or Qwen image editing repair
  • Qwen LoRA Splat Repair  — Qwen 2511 edit + splat-repair LoRA (fal.ai)
  • Qwen Camera-Locked Repair — analyse with the original, then edit only the reposed render
  • Nano Banana Camera-Locked Repair — use Nano Banana on the reposed render with a camera-lock prompt
  • Splat Repair Blend      — final compositing: blend repair into original
"""

import importlib
import os

# ── Web extension directory (serves JS/CSS to the ComfyUI frontend) ──────────
WEB_DIRECTORY = "./web"

# ── Node modules to load ─────────────────────────────────────────────────────
_node_modules = [
    "fal_model_browser",
    "fal_lora",
    "splat_repair",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod_name in _node_modules:
    try:
        _mod = importlib.import_module(f".nodes.{_mod_name}", __name__)
        NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)
        print(f"[comfyui-fal-splat] loaded: {_mod_name} "
              f"({len(_mod.NODE_CLASS_MAPPINGS)} nodes)")
    except Exception as e:
        print(f"[comfyui-fal-splat] WARNING: could not load {_mod_name}: {e}")

# ── Register server-side API routes for the key management popup ─────────────
try:
    from .nodes.api_routes import register_routes
    import server
    if hasattr(server, "PromptServer") and hasattr(server.PromptServer, "instance"):
        srv = server.PromptServer.instance
        if srv and hasattr(srv, "app"):
            register_routes(srv.app)
            print("[comfyui-fal-splat] API key routes registered.")
except Exception as e:
    print(f"[comfyui-fal-splat] Note: API routes not registered yet ({e}). "
          f"Key popup will be available after server starts.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
