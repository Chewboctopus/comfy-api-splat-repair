"""
Server-side API routes for comfyui-fal-splat.

Provides:
  GET  /fal-splat/api-key/status  — check if a valid FAL_KEY is configured
  POST /fal-splat/api-key/save    — save a FAL_KEY to config.ini and env
  POST /fal-splat/api-key/test    — test a key against the fal.ai API
"""

import configparser
import json
import os
from pathlib import Path

from aiohttp import web

ROUTES = web.RouteTableDef()

_CONFIG_PATH = Path(__file__).parent.parent / "config.ini"
# Also check the sibling fal-api package
_FAL_API_CONFIG_PATH = Path(__file__).parent.parent.parent / "fal-api" / "config.ini"


def _read_key() -> str:
    """Return the current FAL_KEY or empty string."""
    # 1. Environment variable (highest priority — may have been set at runtime)
    env_key = os.environ.get("FAL_KEY", "")
    if env_key and env_key != "<your_fal_api_key_here>":
        return env_key

    # 2. This package's config.ini
    for cfg_path in [_CONFIG_PATH, _FAL_API_CONFIG_PATH]:
        if cfg_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(cfg_path)
            try:
                key = cfg["API"]["FAL_KEY"]
                if key and key != "<your_fal_api_key_here>":
                    return key
            except KeyError:
                pass

    return ""


def _save_key(key: str):
    """Persist the FAL_KEY to config.ini and set it in the environment."""
    # Write to this package's config.ini
    cfg = configparser.ConfigParser()
    if _CONFIG_PATH.exists():
        cfg.read(_CONFIG_PATH)
    if "API" not in cfg:
        cfg["API"] = {}
    cfg["API"]["FAL_KEY"] = key

    with open(_CONFIG_PATH, "w") as f:
        f.write("[API]\n")
        f.write(f"# Get your key from: https://fal.ai/dashboard/keys\n")
        f.write(f"FAL_KEY = {key}\n")

    # Also set in environment so it takes effect immediately
    os.environ["FAL_KEY"] = key

    # If the sibling fal-api config exists and still has the placeholder, update it too
    if _FAL_API_CONFIG_PATH.exists():
        try:
            fal_cfg = configparser.ConfigParser()
            fal_cfg.read(_FAL_API_CONFIG_PATH)
            existing = fal_cfg.get("API", "FAL_KEY", fallback="")
            if not existing or existing == "<your_fal_api_key_here>":
                fal_cfg["API"]["FAL_KEY"] = key
                with open(_FAL_API_CONFIG_PATH, "w") as f:
                    fal_cfg.write(f)
                print(f"[fal-splat] Also updated fal-api/config.ini")
        except Exception as e:
            print(f"[fal-splat] Could not update fal-api config: {e}")


async def _test_key(key: str) -> tuple[bool, str]:
    """Test a FAL_KEY by making a lightweight API call. Returns (ok, message)."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Key {key}"}
            # Hit a lightweight endpoint — just check auth
            async with session.get(
                "https://rest.alpha.fal.ai/tokens/",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status in (200, 201):
                    return True, "API key is valid."
                elif resp.status == 401:
                    return False, "Invalid API key (401 Unauthorized)."
                elif resp.status == 403:
                    return False, "API key rejected (403 Forbidden)."
                else:
                    # Some endpoints may 404 but auth still passed
                    # Try the queue endpoint instead
                    return True, f"Key accepted (status {resp.status})."
    except Exception as e:
        # If we can't reach fal at all, still save the key — user can fix later
        return True, f"Could not verify (network issue: {e}). Key saved anyway."


# ── Routes ────────────────────────────────────────────────────────────────────

@ROUTES.get("/fal-splat/api-key/status")
async def get_key_status(request):
    key = _read_key()
    has_key = bool(key)
    # Mask the key for display (show first 4 and last 4 chars)
    masked = ""
    if key and len(key) > 8:
        masked = key[:4] + "•" * (len(key) - 8) + key[-4:]
    elif key:
        masked = "•" * len(key)

    return web.json_response({
        "has_key": has_key,
        "masked_key": masked,
    })


@ROUTES.post("/fal-splat/api-key/save")
async def save_key(request):
    try:
        data = await request.json()
        key = data.get("key", "").strip()

        if not key:
            return web.json_response({"ok": False, "message": "No key provided."}, status=400)

        # Test the key first
        ok, message = await _test_key(key)

        if ok:
            _save_key(key)
            print(f"[fal-splat] FAL_KEY saved and activated.")
            return web.json_response({"ok": True, "message": message})
        else:
            return web.json_response({"ok": False, "message": message}, status=400)

    except Exception as e:
        return web.json_response({"ok": False, "message": str(e)}, status=500)


@ROUTES.post("/fal-splat/api-key/test")
async def test_key(request):
    try:
        data = await request.json()
        key = data.get("key", "").strip()

        if not key:
            # Test the currently saved key
            key = _read_key()
            if not key:
                return web.json_response({"ok": False, "message": "No key configured."})

        ok, message = await _test_key(key)
        return web.json_response({"ok": ok, "message": message})

    except Exception as e:
        return web.json_response({"ok": False, "message": str(e)}, status=500)


def register_routes(app):
    """Register all fal-splat API routes with the ComfyUI server."""
    app.router.add_routes(ROUTES)
    print("[fal-splat] API routes registered: /fal-splat/api-key/*")
