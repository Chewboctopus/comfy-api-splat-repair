"""
FalModelBrowser  — query & filter fal.ai models dynamically.
FalSchemaLoader  — fetch and display a model's OpenAPI parameter schema.
FalDynamicImageGen — call ANY fal image model via JSON parameter blob.

These nodes let you explore the full fal.ai catalogue without hard-coding
every model as a separate ComfyUI class.
"""

import json
import os
import time
from pathlib import Path

import torch

from .fal_utils import (
    fetch_model_schema,
    schema_to_summary,
    submit_and_get,
    upload_image,
    upload_mask,
    process_images_result,
    process_single_image_result,
    _blank_image,
)

# ---------------------------------------------------------------------------
# Curated model catalogue  (refreshed from fal.ai at runtime when possible)
# ---------------------------------------------------------------------------

# Known models that support LoRAs on fal.ai, grouped by capability.
# Each entry:  { "id": str, "name": str, "tags": [str], "lora": bool, "img2img": bool }
_BUILTIN_CATALOG = [
    # ── Flux Klein (fast, LoRA-capable) ──────────────────────────────────
    {"id": "fal-ai/flux-2/klein/9b/base/edit/lora", "name": "Flux Klein 9B Edit + LoRA (img2img)",
     "tags": ["image_gen", "img2img", "lora", "klein"],              "lora": True, "img2img": True},
    {"id": "fal-ai/flux-2/klein/9b/base/lora",      "name": "Flux Klein 9B Base + LoRA (t2i)",
     "tags": ["image_gen", "lora", "klein"],                         "lora": True, "img2img": False},
    {"id": "fal-ai/flux-2/klein/4b/base/edit/lora", "name": "Flux Klein 4B Edit + LoRA (img2img)",
     "tags": ["image_gen", "img2img", "lora", "klein"],              "lora": True, "img2img": True},
    {"id": "fal-ai/flux-2/klein/9b/edit",           "name": "Flux Klein 9B Edit (no LoRA)",
     "tags": ["image_gen", "img2img", "klein"],                      "lora": False,"img2img": True},
    {"id": "fal-ai/flux-2/klein/9b",                "name": "Flux Klein 9B (distilled, fast)",
     "tags": ["image_gen", "klein"],                                 "lora": False,"img2img": False},
    # ── Flux Dev LoRA-capable ──────────────────────────────────────────────
    {"id": "fal-ai/flux-general",              "name": "Flux General (Dev + ControlNet/LoRA)",
     "tags": ["image_gen", "lora", "controlnet", "ip_adapter"],      "lora": True, "img2img": False},
    {"id": "fal-ai/flux-lora",                 "name": "Flux LoRA (dual-LoRA t2i)",
     "tags": ["image_gen", "lora"],                                   "lora": True, "img2img": False},
    {"id": "fal-ai/flux-general/image-to-image","name": "Flux General img2img (+LoRA)",
     "tags": ["image_gen", "img2img", "lora"],                        "lora": True, "img2img": True},
    {"id": "fal-ai/flux-dev/image-to-image",   "name": "Flux Dev img2img",
     "tags": ["image_gen", "img2img"],                                "lora": False,"img2img": True},
    {"id": "fal-ai/flux-pro/v1/fill",          "name": "Flux Pro Fill (inpainting)",
     "tags": ["image_gen", "inpainting"],                             "lora": False,"img2img": True},
    {"id": "fal-ai/flux-pro/v1.1",             "name": "Flux Pro 1.1",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/flux-dev",                  "name": "Flux Dev",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/flux-schnell",              "name": "Flux Schnell (fast)",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    # ── Flux Kontext ──────────────────────────────────────────────────────
    {"id": "fal-ai/flux-pro/kontext",          "name": "Flux Pro Kontext (context img2img)",
     "tags": ["image_gen", "img2img"],                                "lora": False,"img2img": True},
    {"id": "fal-ai/flux-pro/kontext/max",      "name": "Flux Pro Kontext Max",
     "tags": ["image_gen", "img2img"],                                "lora": False,"img2img": True},
    # ── Stable Diffusion / XL ─────────────────────────────────────────────
    {"id": "fal-ai/stable-diffusion-xl",       "name": "Stable Diffusion XL",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/stable-diffusion-v3-medium","name": "Stable Diffusion v3 Medium",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/aura-flow",                 "name": "Aura Flow",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/hyper-sdxl",               "name": "Hyper SDXL (fast)",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    # ── Image editing ─────────────────────────────────────────────────────
    {"id": "fal-ai/qwen-image-edit",           "name": "Qwen Image Edit",
     "tags": ["image_edit"],                                          "lora": False,"img2img": True},
    {"id": "fal-ai/seededit-v3",               "name": "SeedEdit 3.0",
     "tags": ["image_edit"],                                          "lora": False,"img2img": True},
    # ── Upscalers ─────────────────────────────────────────────────────────
    {"id": "fal-ai/clarity-upscaler",          "name": "Clarity Upscaler",
     "tags": ["upscale"],                                             "lora": False,"img2img": True},
    {"id": "fal-ai/seedvr2-image",             "name": "SeedVR2 Image Upscaler",
     "tags": ["upscale"],                                             "lora": False,"img2img": True},
    # ── HiDream / Recraft / other ─────────────────────────────────────────
    {"id": "fal-ai/hidream-i1-full",           "name": "HiDream Full",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/recraft-v3",                "name": "Recraft V3",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/ideogram/v3",               "name": "Ideogram v3",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
    {"id": "fal-ai/imagen4/preview",           "name": "Imagen4 Preview",
     "tags": ["image_gen"],                                           "lora": False,"img2img": False},
]

_CACHE_PATH = Path(__file__).parent.parent / ".model_cache.json"
_CACHE_TTL_HOURS = 6


def _load_catalog() -> list[dict]:
    """Return catalog from cache if fresh, else return built-in list."""
    if _CACHE_PATH.exists():
        try:
            data = json.loads(_CACHE_PATH.read_text())
            age_h = (time.time() - data.get("ts", 0)) / 3600
            if age_h < _CACHE_TTL_HOURS and data.get("models"):
                return data["models"]
        except Exception:
            pass
    return _BUILTIN_CATALOG


def _save_catalog(models: list[dict]):
    try:
        _CACHE_PATH.write_text(json.dumps({"ts": time.time(), "models": models}, indent=2))
    except Exception:
        pass


def _filter_catalog(
    models: list[dict],
    lora_only: bool,
    img2img_only: bool,
    tags_filter: str,
    text_filter: str,
) -> list[dict]:
    out = models
    if lora_only:
        out = [m for m in out if m.get("lora")]
    if img2img_only:
        out = [m for m in out if m.get("img2img")]
    if tags_filter and tags_filter.strip():
        wanted = {t.strip() for t in tags_filter.split(",")}
        out = [m for m in out if wanted.intersection(m.get("tags", []))]
    if text_filter and text_filter.strip():
        q = text_filter.lower()
        out = [m for m in out if q in m["id"].lower() or q in m["name"].lower()]
    return out


# ============================================================
# Node 1 — FalModelBrowser
# ============================================================

class FalModelBrowser:
    """
    Browse and filter fal.ai models.

    Outputs the selected model ID (ready to plug into FalDynamicImageGen
    or FalSchemaLoader) and a JSON list of matching models for inspection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        catalog = _load_catalog()
        model_ids = [m["id"] for m in catalog]
        if not model_ids:
            model_ids = ["fal-ai/flux-general"]
        return {
            "required": {
                "selected_model": (model_ids, {"default": "fal-ai/flux-general"}),
            },
            "optional": {
                "filter_lora_support": ("BOOLEAN", {"default": True,
                    "tooltip": "Only show models that accept LoRA weights"}),
                "filter_img2img": ("BOOLEAN", {"default": False,
                    "tooltip": "Only show models that accept an input image"}),
                "filter_tags": ("STRING", {"default": "",
                    "tooltip": "Comma-separated tags: image_gen, img2img, lora, inpainting, upscale, image_edit"}),
                "text_search": ("STRING", {"default": "",
                    "tooltip": "Free-text search across model IDs and names"}),
                "custom_model_id": ("STRING", {"default": "",
                    "tooltip": "Override: type any fal model ID here (e.g. fal-ai/my-custom-model)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_id", "catalog_json")
    FUNCTION = "browse"
    CATEGORY = "FAL-Splat/Models"
    DESCRIPTION = (
        "Browse the fal.ai model catalogue. "
        "Filter by LoRA support, img2img, tags, or free text. "
        "Plug model_id into FalDynamicImageGen or FalSchemaLoader."
    )

    def browse(
        self,
        selected_model,
        filter_lora_support=True,
        filter_img2img=False,
        filter_tags="",
        text_search="",
        custom_model_id="",
    ):
        catalog = _load_catalog()
        filtered = _filter_catalog(
            catalog, filter_lora_support, filter_img2img, filter_tags, text_search
        )

        model_id = custom_model_id.strip() if custom_model_id.strip() else selected_model
        catalog_str = json.dumps(filtered, indent=2)
        return (model_id, catalog_str)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Re-evaluate every time so filter changes take effect
        return float("nan")


# ============================================================
# Node 2 — FalSchemaLoader
# ============================================================

class FalSchemaLoader:
    """
    Fetch the OpenAPI schema for any fal.ai model and display its parameters.

    Connect model_id from FalModelBrowser (or type it directly).
    The output is a human-readable parameter summary you can read in a
    ShowText node, and the raw JSON schema string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "fal-ai/flux-general",
                    "tooltip": "fal model identifier, e.g. fal-ai/flux-general"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("parameter_summary", "schema_json")
    FUNCTION = "load_schema"
    CATEGORY = "FAL-Splat/Models"
    DESCRIPTION = "Fetch a fal.ai model's OpenAPI schema and output a parameter summary."

    def load_schema(self, model_id):
        schema = fetch_model_schema(model_id.strip())
        summary = schema_to_summary(schema)
        schema_json = json.dumps(schema, indent=2) if schema else "{}"
        return (summary, schema_json)

    @classmethod
    def IS_CHANGED(cls, model_id="", **_):
        return model_id  # Re-fetch when model_id changes


# ============================================================
# Node 3 — FalDynamicImageGen
# ============================================================

class FalDynamicImageGen:
    """
    Call ANY fal.ai image-generation or image-editing model.

    Supply the model ID and build up parameters using the dedicated
    input pins (prompt, image, mask, LoRA) plus an optional JSON blob
    for any extra parameters not covered by the pins.

    Uses the same endpoint pattern as the built-in fal-api nodes but
    is not locked to a specific model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "fal-ai/flux-general",
                    "tooltip": "Any fal.ai model ID. Connect from FalModelBrowser or type directly."}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                # Image conditioning
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "img2img denoising strength (0 = preserve original, 1 = full generation)"}),
                # LoRA support
                "lora_url_1": ("STRING", {"default": "",
                    "tooltip": "HuggingFace URL, fal storage URL, or CivitAI download URL for LoRA 1"}),
                "lora_scale_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_url_2": ("STRING", {"default": "", "tooltip": "Optional second LoRA"}),
                "lora_scale_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_url_3": ("STRING", {"default": "", "tooltip": "Optional third LoRA"}),
                "lora_scale_3": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                # Common generation params
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "tooltip": "-1 = random"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9",
                                "landscape_4_3", "landscape_16_9", "custom"], {"default": "square_hd"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 16}),
                # Advanced override
                "extra_params_json": ("STRING", {"default": "{}",
                    "tooltip": "JSON object with any additional parameters for this model. "
                               "Keys here override the values above."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "FAL-Splat/Generate"
    DESCRIPTION = (
        "Universal fal.ai image generation node. "
        "Works with any model ID — Flux, SDXL, HiDream, Qwen, etc. "
        "Add LoRAs via URL, supply an input image for img2img or inpainting."
    )

    def generate(
        self,
        model_id,
        prompt,
        image=None,
        mask=None,
        strength=0.85,
        lora_url_1="",
        lora_scale_1=1.0,
        lora_url_2="",
        lora_scale_2=1.0,
        lora_url_3="",
        lora_scale_3=0.8,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=-1,
        num_images=1,
        image_size="square_hd",
        width=1024,
        height=1024,
        extra_params_json="{}",
    ):
        try:
            # ── Build base arguments ────────────────────────────────────────
            args: dict = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
            }

            if image_size == "custom":
                args["image_size"] = {"width": width, "height": height}
            else:
                args["image_size"] = image_size

            if negative_prompt:
                args["negative_prompt"] = negative_prompt

            if seed != -1:
                args["seed"] = seed

            # ── Image / mask ────────────────────────────────────────────────
            if image is not None:
                url = upload_image(image)
                if url:
                    # Try common param names used across fal models
                    args["image_url"] = url
                    args["strength"] = strength

            if mask is not None:
                url = upload_mask(mask)
                if url:
                    args["mask_url"] = url

            # ── LoRAs ───────────────────────────────────────────────────────
            loras = []
            for path, scale in [
                (lora_url_1, lora_scale_1),
                (lora_url_2, lora_scale_2),
                (lora_url_3, lora_scale_3),
            ]:
                if path and path.strip():
                    loras.append({"path": path.strip(), "scale": scale})
            if loras:
                args["loras"] = loras

            # ── Extra params override ───────────────────────────────────────
            if extra_params_json and extra_params_json.strip() not in ("{}", ""):
                try:
                    extras = json.loads(extra_params_json)
                    args.update(extras)
                except json.JSONDecodeError as e:
                    print(f"[fal-splat] extra_params_json parse error: {e}")

            print(f"[fal-splat] Calling {model_id} with keys: {list(args.keys())}")
            result = submit_and_get(model_id.strip(), args)

            # ── Parse result ────────────────────────────────────────────────
            if "images" in result:
                return (process_images_result(result),)
            elif "image" in result:
                return (process_single_image_result(result),)
            else:
                print(f"[fal-splat] Unexpected result structure: {list(result.keys())}")
                return (_blank_image(),)

        except Exception as e:
            print(f"[fal-splat] FalDynamicImageGen error: {e}")
            return (_blank_image(),)


# ============================================================
# Mappings
# ============================================================

NODE_CLASS_MAPPINGS = {
    "FalModelBrowser_splat":     FalModelBrowser,
    "FalSchemaLoader_splat":     FalSchemaLoader,
    "FalDynamicImageGen_splat":  FalDynamicImageGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalModelBrowser_splat":     "Fal Model Browser (splat)",
    "FalSchemaLoader_splat":     "Fal Schema Loader (splat)",
    "FalDynamicImageGen_splat":  "Fal Dynamic Image Gen (splat)",
}
