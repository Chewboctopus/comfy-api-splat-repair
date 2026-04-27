"""
LoRA management nodes for fal.ai.

FalLoRAFromURL     — pass-through / validate a LoRA URL (HuggingFace, CivitAI, fal storage)
FalLoRAFromLocal   — upload a local .safetensors file to fal storage
FalLoRAStack       — combine up to 4 LoRA entries into a JSON list for FalDynamicImageGen
FalLoRAInfo        — display metadata about a LoRA URL
"""

import json
import os
import re

import torch

from .fal_utils import upload_file, get_fal_client, _blank_image

# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalise common LoRA URL patterns to a direct-download URL
# ─────────────────────────────────────────────────────────────────────────────

_CIVITAI_PATTERN = re.compile(
    r"https?://civitai\.com/models/(\d+)(?:/[^?#]*)?"
)


def _normalise_lora_url(raw: str) -> tuple[str, str]:
    """
    Return (normalised_url, source_type) where source_type is one of:
    'huggingface', 'civitai', 'fal_storage', 'direct', 'unknown'
    """
    url = raw.strip()
    if not url:
        return ("", "empty")

    # HuggingFace blob → resolve URL
    if "huggingface.co" in url:
        # Convert /blob/ to /resolve/
        url = url.replace("/blob/", "/resolve/")
        return (url, "huggingface")

    # CivitAI model page → use API download
    m = _CIVITAI_PATTERN.match(url)
    if m:
        model_id = m.group(1)
        # CivitAI direct download via API (no auth needed for public models)
        api_url = f"https://civitai.com/api/download/models/{model_id}"
        return (api_url, "civitai")

    if "v3.fal.media" in url or "fal.media" in url or "fal.run/files" in url:
        return (url, "fal_storage")

    return (url, "direct")


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — FalLoRAFromURL
# ─────────────────────────────────────────────────────────────────────────────

class FalLoRAFromURL:
    """
    Validate and normalise a LoRA URL for use with fal.ai models.

    Accepts:
    • HuggingFace URLs  (blob/ links converted to resolve/ automatically)
    • CivitAI model pages
    • fal storage URLs
    • Any direct .safetensors / .pt download URL

    Outputs a lora_url STRING ready for FalDynamicImageGen or FalLoRAStack.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "",
                    "tooltip": "HuggingFace, CivitAI, fal storage, or direct .safetensors URL"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA weight multiplier"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("lora_url", "scale", "source_type")
    FUNCTION = "load"
    CATEGORY = "FAL-Splat/LoRA"
    DESCRIPTION = "Pass a LoRA URL through, auto-normalising HuggingFace and CivitAI links."

    def load(self, url, scale):
        normalised, source = _normalise_lora_url(url)
        if normalised:
            print(f"[fal-splat] LoRA URL ({source}): {normalised}")
        return (normalised, scale, source)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — FalLoRAFromLocal
# ─────────────────────────────────────────────────────────────────────────────

class FalLoRAFromLocal:
    """
    Upload a local .safetensors (or .pt) LoRA file to fal storage and
    return the resulting URL for use in generation nodes.

    The file path can be an absolute path or relative to ComfyUI's models/loras
    folder.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Try to list local LoRA files for convenience
        lora_choices = ["(enter path below)"]
        try:
            import folder_paths
            lora_dirs = folder_paths.get_folder_paths("loras")
            for d in lora_dirs:
                if os.path.isdir(d):
                    for fn in sorted(os.listdir(d)):
                        if fn.endswith((".safetensors", ".pt", ".ckpt")):
                            lora_choices.append(fn)
        except Exception:
            pass

        return {
            "required": {
                "lora_file": (lora_choices, {"default": lora_choices[0]}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "",
                    "tooltip": "Override: absolute path to a .safetensors file"}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("lora_url", "scale")
    FUNCTION = "upload"
    CATEGORY = "FAL-Splat/LoRA"
    DESCRIPTION = (
        "Upload a local LoRA file to fal storage. "
        "The returned URL can be used directly in any fal generation node."
    )

    def upload(self, lora_file, scale, custom_path=""):
        # Resolve path
        path = custom_path.strip() if custom_path.strip() else None

        if path is None or not os.path.exists(path):
            # Try ComfyUI loras folder
            try:
                import folder_paths
                lora_dirs = folder_paths.get_folder_paths("loras")
                for d in lora_dirs:
                    candidate = os.path.join(d, lora_file)
                    if os.path.exists(candidate):
                        path = candidate
                        break
            except Exception:
                pass

        if not path or not os.path.exists(path):
            print(f"[fal-splat] FalLoRAFromLocal: file not found: {lora_file!r} / {custom_path!r}")
            return ("", scale)

        print(f"[fal-splat] Uploading LoRA: {path}")
        url = upload_file(path)
        if url:
            print(f"[fal-splat] LoRA uploaded → {url}")
        return (url or "", scale)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — FalLoRAStack
# ─────────────────────────────────────────────────────────────────────────────

class FalLoRAStack:
    """
    Combine up to 4 LoRA URL + scale pairs into a single JSON array.

    The output loras_json can be passed directly to FalDynamicImageGen's
    extra_params_json field as: {"loras": <paste loras_json here>}

    Or connect individual lora_url outputs from FalLoRAFromURL / FalLoRAFromLocal
    to the url slots.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "lora_url_1": ("STRING", {"default": ""}),
                "scale_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_url_2": ("STRING", {"default": ""}),
                "scale_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_url_3": ("STRING", {"default": ""}),
                "scale_3": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_url_4": ("STRING", {"default": ""}),
                "scale_4": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "existing_stack": ("STRING", {"default": "[]",
                    "tooltip": "Optional: pass in an existing loras_json to append to"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("loras_json", "extra_params_json")
    FUNCTION = "stack"
    CATEGORY = "FAL-Splat/LoRA"
    DESCRIPTION = (
        "Stack up to 4 LoRAs into a JSON array. "
        "loras_json → connect to extra_params_json of FalDynamicImageGen wrapped as {\"loras\": ...}. "
        "extra_params_json is ready-to-use JSON."
    )

    def stack(
        self,
        lora_url_1="", scale_1=1.0,
        lora_url_2="", scale_2=1.0,
        lora_url_3="", scale_3=0.8,
        lora_url_4="", scale_4=0.7,
        existing_stack="[]",
    ):
        try:
            loras = json.loads(existing_stack) if existing_stack.strip() not in ("", "[]") else []
        except Exception:
            loras = []

        for url, scale in [
            (lora_url_1, scale_1), (lora_url_2, scale_2),
            (lora_url_3, scale_3), (lora_url_4, scale_4),
        ]:
            if url and url.strip():
                loras.append({"path": url.strip(), "scale": scale})

        loras_json = json.dumps(loras, indent=2)
        extra = json.dumps({"loras": loras}, indent=2)
        return (loras_json, extra)


# ─────────────────────────────────────────────────────────────────────────────
# Mappings
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "FalLoRAFromURL_splat":   FalLoRAFromURL,
    "FalLoRAFromLocal_splat": FalLoRAFromLocal,
    "FalLoRAStack_splat":     FalLoRAStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalLoRAFromURL_splat":   "Fal LoRA from URL (splat)",
    "FalLoRAFromLocal_splat": "Fal LoRA from Local File (splat)",
    "FalLoRAStack_splat":     "Fal LoRA Stack (splat)",
}
