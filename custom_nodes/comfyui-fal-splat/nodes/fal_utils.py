"""
Shared utilities for comfyui-fal-splat nodes.
Standalone — does not depend on the existing fal-api custom node.
"""

import configparser
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Config / Key management
# ---------------------------------------------------------------------------

def _get_fal_key() -> str:
    """
    Resolve FAL_KEY from:
      1. FAL_KEY environment variable
      2. config.ini in this package directory
      3. config.ini in the sibling fal-api package (if present)
    """
    if os.environ.get("FAL_KEY"):
        return os.environ["FAL_KEY"]

    # Try this package's config.ini
    pkg_dir = Path(__file__).parent.parent
    for cfg_path in [
        pkg_dir / "config.ini",
        pkg_dir.parent / "fal-api" / "config.ini",
    ]:
        if cfg_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(cfg_path)
            try:
                key = cfg["API"]["FAL_KEY"]
                if key and key != "<your_fal_api_key_here>":
                    os.environ["FAL_KEY"] = key
                    return key
            except KeyError:
                pass

    raise ValueError(
        "FAL_KEY not found! Please set the FAL_KEY environment variable, "
        "or create a config.ini file in the comfyui-fal-splat directory "
        "with:\n[API]\nFAL_KEY=your_key_here"
    )

def save_fal_key(key: str):
    """Save the key to config.ini and set it in the environment."""
    key = key.strip()
    if not key:
        return
    
    os.environ["FAL_KEY"] = key
    
    pkg_dir = Path(__file__).parent.parent
    cfg_path = pkg_dir / "config.ini"
    
    cfg = configparser.ConfigParser()
    if cfg_path.exists():
        cfg.read(cfg_path)
    
    if "API" not in cfg:
        cfg["API"] = {}
    
    cfg["API"]["FAL_KEY"] = key
    
    with open(cfg_path, "w") as f:
        cfg.write(f)
    print(f"[fal-splat] Saved FAL_KEY to {cfg_path}")


def get_fal_client():
    """Return an authenticated fal SyncClient."""
    from fal_client.client import SyncClient
    key = _get_fal_key()
    return SyncClient(key=key)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C) or (H,W,C) to PIL."""
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy()
    else:
        img_np = np.array(image)

    if img_np.ndim == 4:
        img_np = img_np[0]          # take first in batch
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))

    if img_np.dtype in (np.float32, np.float64):
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert PIL to ComfyUI IMAGE tensor (1,H,W,C) float32 0-1."""
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def mask_tensor_to_pil(mask: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI MASK tensor (1,H,W) or (H,W) to grayscale PIL."""
    if mask.ndim == 3:
        mask = mask[0]
    arr = (mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def upload_image(image: torch.Tensor) -> str | None:
    """Upload a ComfyUI IMAGE tensor to fal storage, return URL."""
    try:
        pil = tensor_to_pil(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil.save(f, format="PNG")
            tmp = f.name
        client = get_fal_client()
        url = client.upload_file(tmp)
        return url
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"[fal-splat] upload_image error: {e}")
        return None
    finally:
        if "tmp" in locals() and os.path.exists(tmp):
            os.unlink(tmp)


def upload_mask(mask: torch.Tensor) -> str | None:
    """Upload a ComfyUI MASK tensor as a PNG to fal storage, return URL."""
    try:
        pil = mask_tensor_to_pil(mask)
        # Convert to RGB so fal accepts it
        pil_rgb = pil.convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_rgb.save(f, format="PNG")
            tmp = f.name
        client = get_fal_client()
        url = client.upload_file(tmp)
        return url
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"[fal-splat] upload_mask error: {e}")
        return None
    finally:
        if "tmp" in locals() and os.path.exists(tmp):
            os.unlink(tmp)


def upload_file(path: str) -> str | None:
    """Upload an arbitrary local file to fal storage, return URL."""
    try:
        client = get_fal_client()
        return client.upload_file(path)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"[fal-splat] upload_file error: {e}")
        return None


def download_image(url: str) -> torch.Tensor:
    """Download an image URL and return as ComfyUI IMAGE tensor."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        pil = Image.open(io.BytesIO(r.content)).convert("RGB")
        return pil_to_tensor(pil)
    except Exception as e:
        print(f"[fal-splat] download_image error: {e}")
        return _blank_image()


def _blank_image(w=512, h=512) -> torch.Tensor:
    arr = np.zeros((h, w, 3), dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

def submit_and_get(endpoint: str, arguments: dict):
    """Submit to fal API and block until result is returned."""
    client = get_fal_client()
    handler = client.submit(endpoint, arguments=arguments)
    return handler.get()


def process_images_result(result: dict) -> torch.Tensor:
    """Extract IMAGE tensor from a typical multi-image fal result."""
    try:
        images = []
        for img_info in result.get("images", []):
            images.append(download_image(img_info["url"]))
        if not images:
            return _blank_image()
        return torch.cat(images, dim=0)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"[fal-splat] process_images_result error: {e}")
        return _blank_image()


def process_single_image_result(result: dict) -> torch.Tensor:
    """Extract IMAGE tensor from a single-image fal result."""
    try:
        url = result.get("image", {}).get("url") or result.get("images", [{}])[0].get("url")
        return download_image(url)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        print(f"[fal-splat] process_single_image_result error: {e}")
        return _blank_image()


# ---------------------------------------------------------------------------
# Schema fetching
# ---------------------------------------------------------------------------

def fetch_model_schema(model_id: str) -> dict:
    """
    Attempt to fetch the OpenAPI schema for a fal model.
    Returns the schema dict, or empty dict on failure.
    """
    urls_to_try = [
        f"https://fal.run/{model_id}/openapi.json",
        f"https://rest.alpha.fal.ai/models/{model_id}/openapi.json",
    ]
    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return {}


def schema_to_summary(schema: dict) -> str:
    """Convert an OpenAPI schema to a human-readable parameter summary."""
    if not schema:
        return "No schema available."
    lines = []
    components = schema.get("components", {}).get("schemas", {})
    # Try to find the input schema
    input_schema = None
    for name, obj in components.items():
        if "input" in name.lower() or name in ("Input", "Arguments"):
            input_schema = obj
            break
    if input_schema is None and components:
        # fallback: take last schema
        input_schema = list(components.values())[-1]
    if input_schema:
        props = input_schema.get("properties", {})
        required = set(input_schema.get("required", []))
        for param, info in props.items():
            req = "* " if param in required else "  "
            ptype = info.get("type", info.get("$ref", "?"))
            desc = info.get("description", "")
            default = info.get("default", "")
            line = f"{req}{param} ({ptype})"
            if default != "":
                line += f" = {default}"
            if desc:
                line += f"  — {desc[:80]}"
            lines.append(line)
    return "\n".join(lines) if lines else json.dumps(schema, indent=2)[:2000]
