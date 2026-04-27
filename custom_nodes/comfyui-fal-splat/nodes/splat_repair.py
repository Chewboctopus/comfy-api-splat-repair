"""
Gaussian Splat repair nodes — the core of the SHARP → Repose → Repair workflow.

Pipeline:
  [LoadImage] ──► [SHARP Predict] ──► [GaussianViewer] ──► [SplatArtifactMask]
                        │                     │                      │
                  original_image         reposed_image           artifact_mask
                        └─────────────────────┴──────────────────────┘
                                                │
                                   [FluxSplatRepair / QwenSplatRepair]
                                                │
                                          repaired_image

Nodes in this file:
  SplatArtifactMask     — compute a mask of regions likely to contain splat artifacts
  SplatDifferenceDebug  — visualise original vs reposed difference for tuning
  FluxSplatRepair       — full pipeline: mask + Flux General img2img + LoRA via fal
  FluxSplatInpaint      — use Flux Pro Fill inpainting with the artifact mask
  QwenSplatRepair       — use Qwen image editing to repair splat artifacts
  QwenLoRASplatRepair   — repair splat artifacts via Qwen 2511 + a repair LoRA
  QwenCameraLockedRepair — analyse with the original, then repair only the reposed render
  NanoBananaCameraLockedRepair — Nano Banana repair with the reposed render locked as the base
  SplatRepairBlend      — blend repaired result back into original using the artifact mask
"""

import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from .fal_utils import (
    upload_image,
    upload_mask,
    submit_and_get,
    process_images_result,
    process_single_image_result,
    _blank_image,
    tensor_to_pil,
    pil_to_tensor,
    mask_tensor_to_pil,
    save_fal_key,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_same_size(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize b to match a's spatial dimensions if needed."""
    if a.shape != b.shape:
        H, W = a.shape[1], a.shape[2]
        b = F.interpolate(
            b.permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    return a, b


def _compute_artifact_mask(
    original: torch.Tensor,
    reposed: torch.Tensor,
    diff_threshold: float = 0.12,
    blur_radius: int = 8,
    dilate_radius: int = 12,
    min_region_fraction: float = 0.002,
) -> torch.Tensor:
    """
    Return a MASK tensor (1, H, W) float32 where 1.0 = artifact region.

    Strategy:
    1. L2 difference per pixel (original vs reposed)
    2. Threshold to binary mask
    3. Gaussian blur to soften edges
    4. Dilation to expand around artifact cores
    5. Remove tiny isolated blobs
    """
    original, reposed = _ensure_same_size(original, reposed)

    # (1, H, W, 3) → diff
    diff = (original.float() - reposed.float()).abs()          # (1,H,W,3)
    diff_mag = diff.mean(dim=-1)                               # (1,H,W)

    # Binary threshold
    mask = (diff_mag > diff_threshold).float()                 # (1,H,W)

    # Gaussian blur  (use average pooling as a fast approximation)
    if blur_radius > 0:
        k = blur_radius * 2 + 1
        mask_4d = mask.unsqueeze(1)                            # (1,1,H,W)
        mask_4d = F.avg_pool2d(
            mask_4d, kernel_size=k, stride=1, padding=blur_radius
        )
        mask = mask_4d.squeeze(1)                              # (1,H,W)

    # Dilation (max-pool)
    if dilate_radius > 0:
        k = dilate_radius * 2 + 1
        mask_4d = mask.unsqueeze(1)
        mask_4d = F.max_pool2d(
            mask_4d, kernel_size=k, stride=1, padding=dilate_radius
        )
        mask = mask_4d.squeeze(1)

    # Clamp to [0,1]
    mask = mask.clamp(0.0, 1.0)

    # Remove regions smaller than min_region_fraction of total pixels
    # (simple: if total masked fraction < min_region_fraction, return zeros)
    total = mask.shape[1] * mask.shape[2]
    if mask.mean() < min_region_fraction:
        print("[fal-splat] SplatArtifactMask: very few artifacts detected — "
              "try lowering diff_threshold.")

    return mask  # (1, H, W)


def _extract_repair_prompt(text: str, fallback: str) -> str:
    """Turn a verbose VLM response into a single repair prompt."""
    if not text or not text.strip():
        return fallback

    cleaned = text.replace("```", "").strip()
    lines = [line.strip(" -*\t") for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return fallback

    for line in reversed(lines):
        if ":" in line:
            label, value = line.split(":", 1)
            if "prompt" in label.lower() and value.strip():
                candidate = value.strip().strip("\"'")
                return candidate[:700] if candidate else fallback

    candidate = lines[-1].strip("\"'")
    if len(candidate) < 24 and len(lines) > 1:
        candidate = " ".join(lines[-2:]).strip("\"'")

    if len(candidate) > 700:
        candidate = candidate[:700].rsplit(" ", 1)[0]

    return candidate or fallback


def _analyse_camera_locked_prompt(
    original_image,
    reposed_image,
    instruction: str,
    fallback_prompt: str,
) -> tuple[str, str]:
    """
    Use a VLM to write a repair prompt while keeping the reposed render as the
    image that will actually be edited in the second pass.
    """
    original_url = upload_image(original_image)
    reposed_url = upload_image(reposed_image)
    if not original_url or not reposed_url:
        note = "Analysis skipped because one or more image uploads failed."
        return fallback_prompt, note

    args = {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "prompt": instruction,
        "image_urls": [original_url, reposed_url],
    }

    try:
        print("[fal-splat] CameraLockedRepair (analyse) -> fal-ai/qwen2-vl-instruct")
        result = submit_and_get("fal-ai/qwen2-vl-instruct", args)
        analysis = result.get("output", result.get("text", "")).strip()
        used_prompt = _extract_repair_prompt(analysis, fallback_prompt)
        return used_prompt, analysis or "Analysis returned no text."
    except Exception as e:
        note = f"Analysis fallback used: {e}"
        print(f"[fal-splat] CameraLockedRepair analyse error: {e}")
        return fallback_prompt, note


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — SplatArtifactMask
# ─────────────────────────────────────────────────────────────────────────────

class SplatArtifactMask:
    """
    Compute a mask of regions likely to contain Gaussian-Splat render artifacts.

    Compares the ORIGINAL image with the RE-POSED splat render to identify
    areas that changed beyond what's expected from the camera move.

    Connect:
      original_image → image before SHARP processing
      reposed_image  → render output from GaussianViewer after camera repose

    The mask output is suitable for use with Flux Fill inpainting or as a
    soft weight map in blending nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "reposed_image": ("IMAGE",),
            },
            "optional": {
                "diff_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Pixel difference threshold above which a pixel is considered an artifact. "
                               "Lower = more sensitive."}),
                "blur_radius": ("INT", {"default": 8, "min": 0, "max": 32,
                    "tooltip": "Gaussian blur radius for smoothing mask edges"}),
                "dilate_radius": ("INT", {"default": 12, "min": 0, "max": 64,
                    "tooltip": "Dilation radius to expand artifact regions"}),
                "invert_mask": ("BOOLEAN", {"default": False,
                    "tooltip": "Invert: mask the un-changed areas instead"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("artifact_mask", "mask_preview")
    FUNCTION = "compute"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Detect splat artifact regions by comparing the original photo to the "
        "reposed splat render. Output mask highlights where repair is needed."
    )

    def compute(
        self,
        original_image,
        reposed_image,
        diff_threshold=0.12,
        blur_radius=8,
        dilate_radius=12,
        invert_mask=False,
    ):
        mask = _compute_artifact_mask(
            original_image, reposed_image,
            diff_threshold=diff_threshold,
            blur_radius=blur_radius,
            dilate_radius=dilate_radius,
        )

        if invert_mask:
            mask = 1.0 - mask

        # mask is (1,H,W) — ComfyUI expects (B,H,W)
        # Preview: broadcast to RGB for display
        preview = mask.unsqueeze(-1).expand(-1, -1, -1, 3)  # (1,H,W,3)

        return (mask, preview)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — SplatDifferenceDebug
# ─────────────────────────────────────────────────────────────────────────────

class SplatDifferenceDebug:
    """
    Visualise the pixel-level difference between original and reposed images.
    Useful for tuning SplatArtifactMask thresholds.

    Outputs:
      diff_image   — amplified L2 difference image (red = high difference)
      overlay      — reposed image with artifact areas highlighted in red
      stats_text   — min/max/mean difference values as a string
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "reposed_image": ("IMAGE",),
                "amplify": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Multiply difference values for easier visual inspection"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("diff_image", "overlay", "stats_text")
    FUNCTION = "debug"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = "Visualise the difference between original and reposed images for threshold tuning."

    def debug(self, original_image, reposed_image, amplify=3.0):
        orig, repo = _ensure_same_size(original_image, reposed_image)

        diff = (orig.float() - repo.float()).abs()   # (1,H,W,3)
        diff_mag = diff.mean(dim=-1, keepdim=True)   # (1,H,W,1)

        stats = {
            "min":  float(diff_mag.min()),
            "max":  float(diff_mag.max()),
            "mean": float(diff_mag.mean()),
            "p95":  float(torch.quantile(diff_mag.flatten(), 0.95)),
        }
        stats_text = json.dumps(stats, indent=2)

        # Amplified diff image (clamped to [0,1])
        diff_img = (diff * amplify).clamp(0, 1)

        # Red overlay: blend reposed with red tint proportional to difference
        red_tint = torch.zeros_like(repo)
        red_tint[..., 0] = 1.0   # R channel = 1
        weight = (diff_mag * amplify * 2.0).clamp(0, 1)
        overlay = repo * (1 - weight) + red_tint * weight

        return (diff_img, overlay, stats_text)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — FluxSplatRepair  (main repair node for the workflow)
# ─────────────────────────────────────────────────────────────────────────────

class FluxSplatRepair:
    """
    Repair Gaussian splat artifacts using Flux Klein 9B + the mlsharp 3D repair LoRA.

    This LoRA was trained to take TWO images:
      • image 1 = the ORIGINAL photo (reference scene)
      • image 2 = the REPOSED Gaussian splat render (with artifacts/black holes)

    It sends both via the Klein edit endpoint's `image_urls` array and uses
    a specific prompt format:
      "Referring to the scene in image 1, restore the perspective of the
       scene in image 2. Repair the perspective and missing areas.
       The camera has moved by: {camera_transform_json}"

    The LoRA fills in all black holes, removes floaters, and produces a
    clean full-frame image — no masking or blending needed.
    """

    # Default prompt template matching the LoRA's training format
    _DEFAULT_PROMPT = (
        "Referring to the scene in image 1, restore the perspective of the "
        "scene in image 2. Repair the perspective and missing areas."
    )

    @staticmethod
    def _extrinsics_to_pos_rot(ext, is_opengl=False):
        """Extract position (x,y,z) and Euler angles (pitch,yaw,roll) from a 4x4 extrinsics matrix.

        Uses YXZ Euler decomposition consistent with ML-SHARP (OpenCV convention).
        If is_opengl is True, it converts from OpenGL C2W to OpenCV C2W by negating the Y and Z columns.
        """
        if is_opengl:
            R = [
                [-float(ext[0][0]), float(ext[0][1]), -float(ext[0][2])],
                [-float(ext[1][0]), float(ext[1][1]), -float(ext[1][2])],
                [-float(ext[2][0]), float(ext[2][1]), -float(ext[2][2])]
            ]
        else:
            R = [
                [float(ext[0][0]), float(ext[0][1]), float(ext[0][2])],
                [float(ext[1][0]), float(ext[1][1]), float(ext[1][2])],
                [float(ext[2][0]), float(ext[2][1]), float(ext[2][2])]
            ]

        px = float(ext[0][3])
        py = float(ext[1][3])
        pz = float(ext[2][3])
        
        r00 = R[0][0]; r01 = R[0][1]; r02 = R[0][2]
        r10 = R[1][0]; r11 = R[1][1]; r12 = R[1][2]
        r20 = R[2][0]; r21 = R[2][1]; r22 = R[2][2]
        
        # YXZ Euler decomposition
        if abs(r21) < 0.99999:
            pitch = math.degrees(math.asin(-r21))
            yaw   = math.degrees(math.atan2(r20, r22))
            roll  = math.degrees(math.atan2(r01, r11))
        else:
            pitch = math.copysign(90.0, -r21)
            yaw   = math.degrees(math.atan2(-r02, r00))
            roll  = 0.0
        return px, py, pz, pitch, yaw, roll

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "reposed_image": ("IMAGE",),
            },
            "optional": {
                # ── Camera extrinsics (connect from GaussianViewer + SharpPredict) ──
                "reposed_extrinsics": ("EXTRINSICS", {
                    "tooltip": "Connect the EXTRINSICS output from GaussianViewer. "
                               "Camera transform will be computed automatically."}),
                "original_extrinsics": ("EXTRINSICS", {
                    "tooltip": "Connect the EXTRINSICS output from SharpPredict "
                               "(the initial camera). Used to compute the delta."}),
                # Prompt — pre-filled with the LoRA's expected format
                "prompt": ("STRING", {
                    "default": "Referring to the scene in image 1, restore the perspective of the scene in image 2. Repair the perspective and missing areas.",
                    "multiline": True,
                    "tooltip": "The mlsharp repair LoRA expects this exact phrasing. "
                               "Camera transform is appended automatically from extrinsics."}),
                # Model
                "repair_model": ([
                    "fal-ai/flux-2/klein/9b/base/edit/lora",
                    "fal-ai/flux-2/klein/9b/edit/lora",
                    "fal-ai/flux-2/klein/4b/edit/lora",
                    "fal-ai/qwen-image-edit",
                    "fal-ai/qwen-image-edit-2511/lora",
                ], {"default": "fal-ai/flux-2/klein/9b/base/edit/lora"}),
                # LoRA
                "lora_url": ([
                    "https://huggingface.co/cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair/resolve/main/flux2-klein9b-lora-mlsharp-3d-repair.safetensors",
                    "https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash/resolve/main/%E9%AB%98%E6%96%AF%E6%B3%BC%E6%BA%85-Sharp.safetensors",
                    "none",
                ], {"default": "https://huggingface.co/cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair/resolve/main/flux2-klein9b-lora-mlsharp-3d-repair.safetensors",
                    "tooltip": "Select the LoRA URL. To use a custom URL, right-click the node -> Convert lora_url to input."}),
                "lora_scale": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA weight. 1.4 works best for repair."}),
                # Generation params
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50,
                    "tooltip": "More steps = higher quality. Default 28."}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1}),
                # Output quality
                "output_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Output width in pixels. 0 = match input image size."}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Output height in pixels. 0 = match input image size."}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Generate multiple images and return the first. Useful for picking the best seed."}),
                # API Key (optional, saves to config)
                "fal_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. Enter your fal.ai API key here. It will be saved automatically to config.ini so you only need to do this once."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("repaired_image",)
    FUNCTION = "repair"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Repair Gaussian splat artifacts using Flux Klein 9B + the mlsharp 3D repair LoRA. "
        "Sends BOTH the original photo and the reposed render as a two-image edit. "
        "The LoRA fills in black holes, removes floaters, and produces a clean image."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def repair(
        self,
        original_image,
        reposed_image,
        reposed_extrinsics=None,
        original_extrinsics=None,
        prompt="",
        repair_model="fal-ai/flux-2/klein/9b/base/edit/lora",
        lora_url="https://huggingface.co/cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair/resolve/main/flux2-klein9b-lora-mlsharp-3d-repair.safetensors",
        lora_scale=1.4,
        num_inference_steps=28,
        guidance_scale=5.0,
        seed=-1,
        output_width=0,
        output_height=0,
        num_images=1,
        fal_api_key="",
    ):
        # ── 0. Check and save API key if provided ──────────────────────────
        if fal_api_key and fal_api_key.strip():
            save_fal_key(fal_api_key)

        # ── Hardcoded internal defaults (previously exposed but rarely changed) ──
        translation_scale = 1.0
        angles_in_radians = True
        camera_x = camera_y = camera_z = 0.0
        camera_pitch = camera_yaw = camera_roll = 0.0
        # ── 1. Upload BOTH images ──────────────────────────────────────────
        print("[fal-splat] FluxSplatRepair: uploading original image (image 1)...")
        original_url = upload_image(original_image)
        print("[fal-splat] FluxSplatRepair: uploading reposed image (image 2)...")
        reposed_url = upload_image(reposed_image)

        if not original_url or not reposed_url:
            print("[fal-splat] FluxSplatRepair: failed to upload images")
            return (reposed_image,)

        # ── 2. Extract camera transform from extrinsics (if connected) ─────
        if reposed_extrinsics is not None:
            # Reposed is from GaussianViewer (OpenGL)
            rx, ry, rz, rpitch, ryaw, rroll = self._extrinsics_to_pos_rot(reposed_extrinsics, is_opengl=True)
            if original_extrinsics is not None:
                # Original is from SharpPredict (OpenCV)
                # Compute delta from original camera
                ox, oy, oz, opitch, oyaw, oroll = self._extrinsics_to_pos_rot(original_extrinsics, is_opengl=False)
                camera_x = (rx - ox) * translation_scale
                camera_y = (ry - oy) * translation_scale
                camera_z = (rz - oz) * translation_scale
                camera_pitch = rpitch - opitch
                camera_yaw = ryaw - oyaw
                camera_roll = rroll - oroll
                print(f"[fal-splat] Camera delta from extrinsics: "
                      f"dx={camera_x:.4f} dy={camera_y:.4f} dz={camera_z:.4f} "
                      f"dpitch={camera_pitch:.2f} dyaw={camera_yaw:.2f} droll={camera_roll:.2f}")
            else:
                # Use absolute reposed camera values
                camera_x, camera_y, camera_z = rx * translation_scale, ry * translation_scale, rz * translation_scale
                camera_pitch, camera_yaw, camera_roll = rpitch, ryaw, rroll
                print(f"[fal-splat] Camera from reposed extrinsics: "
                      f"x={camera_x:.4f} y={camera_y:.4f} z={camera_z:.4f} "
                      f"pitch={camera_pitch:.2f} yaw={camera_yaw:.2f} roll={camera_roll:.2f}")

        # ── 3. Build prompt with camera transform ──────────────────────────
        if not prompt or not prompt.strip():
            prompt = self._DEFAULT_PROMPT

        # Append camera transform only if:
        #   (a) we have non-zero camera values AND
        #   (b) the prompt does not already contain camera data
        #       (e.g. when wired from the MLSharpPrompt node)
        prompt_already_has_camera = "camera has moved by" in prompt.lower()
        if not prompt_already_has_camera:
            if angles_in_radians:
                import math
                camera_pitch = math.radians(camera_pitch)
                camera_yaw = math.radians(camera_yaw)
                camera_roll = math.radians(camera_roll)

            cam_json = json.dumps({
                "x": round(camera_x, 4),
                "y": round(camera_y, 4),
                "z": round(camera_z, 4),
                "pitch": round(camera_pitch, 4) if angles_in_radians else round(camera_pitch, 2),
                "yaw": round(camera_yaw, 4) if angles_in_radians else round(camera_yaw, 2),
                "roll": round(camera_roll, 4) if angles_in_radians else round(camera_roll, 2),
            }, separators=(',', ':'))
            prompt = f"{prompt} The camera has moved by: {cam_json}"
            print(f"[fal-splat] Prompt with camera (from extrinsics): {prompt}")
        elif prompt_already_has_camera:
            print(f"[fal-splat] Prompt already has camera data (MLSharpPrompt) — skipping duplicate")
            print(f"[fal-splat] Prompt: {prompt}")
        else:
            print(f"[fal-splat] Prompt (no camera data): {prompt}")


        # ── 4. Build LoRA list ─────────────────────────────────────────────
        loras = []
        if lora_url and lora_url.strip() and lora_url.strip().lower() != "none":
            clean_url = lora_url.strip().replace("/blob/", "/resolve/")
            loras.append({"path": clean_url, "scale": lora_scale})

        # ── 5. Build API arguments ─────────────────────────────────────────
        #    Klein base/edit/lora takes `image_urls` (plural) — an array of image URLs.
        #    image 1 = original photo (reference for scene details),
        #    image 2 = reposed splat render (target perspective to repair).
        args = {
            "prompt": prompt,
            "image_urls": [original_url, reposed_url],
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "enable_safety_checker": False,
        }

        if output_width > 0 and output_height > 0:
            args["image_size"] = {"width": output_width, "height": output_height}

        if num_images > 1:
            args["num_images"] = num_images

        if loras:
            args["loras"] = loras

        if seed != -1:
            args["seed"] = seed

        # ── 6. Call fal.ai ─────────────────────────────────────────────────
        import traceback
        try:
            print(f"[fal-splat] FluxSplatRepair → {repair_model}")
            print(f"[fal-splat]   image_urls: [{original_url[:60]}..., {reposed_url[:60]}...]")
            print(f"[fal-splat]   loras: {len(loras)}, steps: {num_inference_steps}, guidance: {guidance_scale}")
            print(f"[fal-splat]   non-url args: { {k: v for k, v in args.items() if 'url' not in k} }")
            result = submit_and_get(repair_model, args)
            print(f"[fal-splat] FluxSplatRepair raw result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            repaired = process_images_result(result)
            print(f"[fal-splat] FluxSplatRepair: got result {repaired.shape}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            print(f"[fal-splat] !! FAILED !!")
            print(f"[fal-splat]   endpoint : {repair_model}")
            print(f"[fal-splat]   error    : {e}")
            traceback.print_exc()
            print(f"[fal-splat]   falling back to reposed_image")
            return (reposed_image,)

        return (repaired,)


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — FluxSplatInpaint
# ─────────────────────────────────────────────────────────────────────────────

class FluxSplatInpaint:
    """
    Repair splat artifacts using Flux Pro Fill (hard-mask inpainting) via fal.ai.

    Use when you want precise mask-bounded inpainting rather than the soft
    img2img approach of FluxSplatRepair.

    You can either:
    • Connect an artifact_mask from SplatArtifactMask
    • Draw a custom mask upstream
    • Or let this node auto-compute the mask (if no mask is connected)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reposed_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "photorealistic, sharp, coherent",
                    "multiline": True}),
            },
            "optional": {
                "original_image": ("IMAGE",),
                "artifact_mask": ("MASK",),
                # Auto-mask params (used when artifact_mask is not connected)
                "diff_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01}),
                "dilate_radius": ("INT", {"default": 20, "min": 0, "max": 64}),
                # Generation
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Flux Pro Fill uses high guidance (10-100 typical)"}),
                "seed": ("INT", {"default": -1}),
                "output_format": (["jpeg", "png"], {"default": "png"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("repaired_image", "mask_used")
    FUNCTION = "inpaint"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Hard-mask inpainting via Flux Pro Fill. "
        "Better for large contiguous artifact regions. "
        "Connect SplatArtifactMask output or auto-compute from original+reposed."
    )

    def inpaint(
        self,
        reposed_image,
        prompt,
        original_image=None,
        artifact_mask=None,
        diff_threshold=0.12,
        dilate_radius=20,
        num_inference_steps=28,
        guidance_scale=30.0,
        seed=-1,
        output_format="png",
    ):
        # ── Resolve mask ────────────────────────────────────────────────────
        if artifact_mask is not None:
            mask = artifact_mask
        elif original_image is not None:
            mask = _compute_artifact_mask(
                original_image, reposed_image,
                diff_threshold=diff_threshold,
                blur_radius=4,
                dilate_radius=dilate_radius,
            )
        else:
            # No mask source — use whole image
            h, w = reposed_image.shape[1], reposed_image.shape[2]
            mask = torch.ones(1, h, w)

        # ── Upload ──────────────────────────────────────────────────────────
        img_url = upload_image(reposed_image)
        mask_url = upload_mask(mask)

        if not img_url or not mask_url:
            print("[fal-splat] FluxSplatInpaint: upload failed")
            return (reposed_image, mask)

        # ── Build args ──────────────────────────────────────────────────────
        args = {
            "image_url": img_url,
            "mask_url": mask_url,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "enable_safety_checker": False,
            "output_format": output_format,
        }
        if seed != -1:
            args["seed"] = seed

        # ── Call Flux Pro Fill ──────────────────────────────────────────────
        try:
            print("[fal-splat] FluxSplatInpaint → fal-ai/flux-pro/v1/fill")
            result = submit_and_get("fal-ai/flux-pro/v1/fill", args)
            repaired = process_images_result(result)
        except Exception as e:
            print(f"[fal-splat] FluxSplatInpaint API error: {e}")
            return (reposed_image, mask)

        return (repaired, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — QwenSplatRepair
# ─────────────────────────────────────────────────────────────────────────────

class QwenSplatRepair:
    """
    Use Qwen's vision-language model to analyse and repair splat artifacts.

    Two modes:
      analyse_only  — Returns a text description of detected artifacts and
                      a suggested repair prompt. Feed this into FluxSplatRepair.
      repair        — Directly repairs the reposed image using Qwen image editing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reposed_image": ("IMAGE",),
                "mode": (["analyse_only", "repair"], {"default": "analyse_only"}),
            },
            "optional": {
                "original_image": ("IMAGE",),
                "repair_instruction": ("STRING", {
                    "default": "Fix any ghosting, floaters, blurry blobs, or Gaussian splat "
                               "render artifacts in this image. Make the result look like a "
                               "clean, photorealistic photograph.",
                    "multiline": True,
                    "tooltip": "Instruction for Qwen in 'repair' mode. In 'analyse_only' mode "
                               "this is not used — Qwen generates its own analysis."}),
                "analysis_instruction": ("STRING", {
                    "default": "Compare these two images. The first is the original photo. "
                               "The second is a re-posed 3D Gaussian Splat render of the same scene. "
                               "Describe in detail: (1) what Gaussian-splat rendering artifacts are visible "
                               "in the second image (ghosting, floaters, blurry regions, missing detail), "
                               "(2) where they are located, and (3) write a concise inpainting/repair prompt "
                               "that could guide a diffusion model to fix them.",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("repaired_image", "artifact_analysis", "suggested_repair_prompt")
    FUNCTION = "run"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Qwen VLM-powered splat analyser/repairer. "
        "In 'analyse_only' mode: outputs a description of artifacts + a repair prompt for Flux. "
        "In 'repair' mode: directly edits the image using Qwen."
    )

    def run(
        self,
        reposed_image,
        mode="analyse_only",
        original_image=None,
        repair_instruction="Fix any ghosting, floaters, blurry blobs, or Gaussian splat render artifacts. Make it look photorealistic.",
        analysis_instruction="",
    ):
        if mode == "analyse_only":
            return self._analyse(reposed_image, original_image, analysis_instruction)
        else:
            return self._repair(reposed_image, repair_instruction)

    def _analyse(self, reposed_image, original_image, instruction):
        """Use Qwen VLM to describe splat artifacts."""
        default_instruction = (
            "Compare these two images. The first is the original photo. "
            "The second is a re-posed 3D Gaussian Splat render of the same scene. "
            "Describe in detail: (1) Gaussian-splat rendering artifacts visible in the second image "
            "(ghosting, floaters, blurry regions, missing geometry), "
            "(2) where they are located, and (3) provide a concise diffusion-model repair prompt "
            "to fix those specific artifacts."
        )
        used_instruction = instruction.strip() if instruction.strip() else default_instruction

        # Build image list for Qwen VLM
        images = []
        if original_image is not None:
            url = upload_image(original_image)
            if url:
                images.append(url)
        reposed_url = upload_image(reposed_image)
        if reposed_url:
            images.append(reposed_url)

        if not images:
            return (reposed_image, "Could not upload images.", "photorealistic repair")

        try:
            # Qwen VLM via fal
            args = {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "prompt": used_instruction,
                "image_urls": images,
            }
            print("[fal-splat] QwenSplatRepair (analyse) → fal-ai/qwen2-vl-instruct")
            result = submit_and_get("fal-ai/qwen2-vl-instruct", args)
            analysis = result.get("output", result.get("text", "No response from Qwen."))

            # Extract a short repair prompt from the analysis (last sentence or line)
            lines = [l.strip() for l in analysis.split("\n") if l.strip()]
            repair_prompt_hint = lines[-1] if lines else "photorealistic, fix splat artifacts"

            return (reposed_image, analysis, repair_prompt_hint)

        except Exception as e:
            print(f"[fal-splat] QwenSplatRepair analyse error: {e}")
            return (reposed_image, f"Error: {e}", "photorealistic, fix splat artifacts, sharp")

    def _repair(self, reposed_image, instruction):
        """Use Qwen image editing to directly repair the reposed image."""
        img_url = upload_image(reposed_image)
        if not img_url:
            return (reposed_image, "Upload failed.", instruction)

        try:
            args = {
                "image_url": img_url,
                "prompt": instruction,
            }
            print("[fal-splat] QwenSplatRepair (repair) → fal-ai/qwen-image-edit")
            result = submit_and_get("fal-ai/qwen-image-edit", args)
            repaired = process_images_result(result)
            return (repaired, "Repaired via Qwen.", instruction)

        except Exception as e:
            print(f"[fal-splat] QwenSplatRepair repair error: {e}")
            return (reposed_image, f"Error: {e}", instruction)


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — QwenLoRASplatRepair
# ─────────────────────────────────────────────────────────────────────────────

class QwenLoRASplatRepair:
    """
    Repair a reposed Gaussian splat render using Qwen Image Edit 2511 + a LoRA.

    This node mirrors the local Qwen Gaussian-splash workflow:
      • image 1 = the reposed splat render to repair
      • image 2 = the original source photo for structure/reference

    The default prompt tells Qwen to preserve image 1's camera angle while using
    image 2 only for scene structure and missing-detail recovery.
    """

    _DEFAULT_PROMPT = (
        "Image 1 is a damaged Gaussian splat render that needs repair. "
        "Image 2 is the original reference photo of the same scene. "
        "Restore missing or distorted regions in image 1 using image 2 only for "
        "geometry, structure, perspective, and spatial consistency. Preserve the "
        "current camera angle of image 1. Remove floaters, ghosting, black holes, "
        "smears, blurry splat artifacts, and broken edges. Keep photorealistic "
        "detail, coherent lighting, and natural textures. Do not add new objects "
        "or change the scene composition."
    )

    _DEFAULT_LORA_URL = (
        "https://huggingface.co/dx8152/Qwen-Image-Edit-2511-Gaussian-Splash/"
        "resolve/main/%E9%AB%98%E6%96%AF%E6%B3%BC%E6%BA%85-Sharp.safetensors"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "reposed_image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": cls._DEFAULT_PROMPT,
                    "multiline": True,
                    "tooltip": "Instruction sent to Qwen. Image 1 = reposed splat, image 2 = original photo.",
                }),
                "lora_url": ("STRING", {
                    "default": cls._DEFAULT_LORA_URL,
                    "tooltip": "Direct .safetensors URL for the Qwen splat-repair LoRA.",
                }),
                "lora_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA weight multiplier.",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "tooltip": "Optional. -1 lets fal choose a random seed.",
                }),
                "extra_params_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional raw JSON merged into the fal payload for advanced parameters.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("repaired_image", "request_payload")
    FUNCTION = "repair"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Repair reposed splats with fal-ai/qwen-image-edit-2511/lora using a "
        "Gaussian-splat repair LoRA and the original image as reference."
    )

    def repair(
        self,
        original_image,
        reposed_image,
        prompt="",
        lora_url=_DEFAULT_LORA_URL,
        lora_scale=1.0,
        seed=-1,
        extra_params_json="",
    ):
        print("[fal-splat] QwenLoRASplatRepair: uploading reposed image (image 1)...")
        reposed_url = upload_image(reposed_image)
        print("[fal-splat] QwenLoRASplatRepair: uploading original image (image 2)...")
        original_url = upload_image(original_image)

        if not reposed_url or not original_url:
            print("[fal-splat] QwenLoRASplatRepair: failed to upload one or more images")
            return (reposed_image, '{"error": "upload_failed"}')

        used_prompt = prompt.strip() if prompt and prompt.strip() else self._DEFAULT_PROMPT
        args = {
            "prompt": used_prompt,
            "image_urls": [reposed_url, original_url],
            "enable_safety_checker": False,
        }

        clean_lora_url = (lora_url or "").strip().replace("/blob/", "/resolve/")
        if clean_lora_url:
            args["loras"] = [{
                "path": clean_lora_url,
                "scale": lora_scale,
            }]

        if seed != -1:
            args["seed"] = seed

        if extra_params_json and extra_params_json.strip():
            try:
                extra = json.loads(extra_params_json)
                if isinstance(extra, dict):
                    args.update(extra)
                else:
                    print("[fal-splat] QwenLoRASplatRepair: extra_params_json must be a JSON object")
            except Exception as e:
                print(f"[fal-splat] QwenLoRASplatRepair: could not parse extra_params_json: {e}")

        payload_json = json.dumps(args, indent=2)

        try:
            print("[fal-splat] QwenLoRASplatRepair → fal-ai/qwen-image-edit-2511/lora")
            result = submit_and_get("fal-ai/qwen-image-edit-2511/lora", args)
            repaired = process_images_result(result)
            print(f"[fal-splat] QwenLoRASplatRepair: got result {repaired.shape}")
            return (repaired, payload_json)
        except Exception as e:
            print(f"[fal-splat] QwenLoRASplatRepair error: {e}")
            return (reposed_image, payload_json)


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — QwenCameraLockedRepair
# ─────────────────────────────────────────────────────────────────────────────

class QwenCameraLockedRepair:
    """
    Camera-locked repair path:
      1. use the original + reposed images only to generate a repair prompt
      2. edit the reposed image alone so the camera/framing stays anchored
    """

    _DEFAULT_REPAIR_PROMPT = (
        "Treat the input image as the base canvas and preserve its exact camera angle, "
        "framing, horizon, perspective, crop, and object layout. Repair only the damaged "
        "regions: remove Gaussian splat floaters, ghosting, blurry smears, black holes, "
        "broken edges, and missing detail. Keep the scene photorealistic and do not add "
        "new objects or change composition."
    )

    _DEFAULT_ANALYSIS_INSTRUCTION = (
        "Image 1 is the original source photo. Image 2 is a reposed Gaussian splat render "
        "of the same scene. Write one concise image-editing prompt that repairs image 2 only. "
        "Preserve image 2's exact camera angle, framing, crop, horizon, vanishing lines, and "
        "scene layout. Use image 1 only as reference for missing structure and detail. Focus "
        "on removing floaters, ghosting, smears, black holes, blurry splat artifacts, and "
        "broken edges. Return only the final repair prompt."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "reposed_image": ("IMAGE",),
            },
            "optional": {
                "manual_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional. If filled, skips prompt analysis and uses this prompt directly.",
                }),
                "analysis_instruction": ("STRING", {
                    "default": cls._DEFAULT_ANALYSIS_INSTRUCTION,
                    "multiline": True,
                    "tooltip": "Instruction sent to the VLM when auto-generating a repair prompt.",
                }),
                "extra_params_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional raw JSON merged into the fal qwen-image-edit payload.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("repaired_image", "used_prompt", "debug_details")
    FUNCTION = "repair"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Generate a camera-lock repair prompt from the original photo, then repair "
        "only the reposed render with Qwen to reduce viewpoint drift."
    )

    def repair(
        self,
        original_image,
        reposed_image,
        manual_prompt="",
        analysis_instruction="",
        extra_params_json="",
    ):
        used_prompt = (manual_prompt or "").strip()
        if used_prompt:
            analysis_text = "Manual prompt provided; VLM analysis skipped."
        else:
            instruction = (
                analysis_instruction.strip()
                if analysis_instruction and analysis_instruction.strip()
                else self._DEFAULT_ANALYSIS_INSTRUCTION
            )
            used_prompt, analysis_text = _analyse_camera_locked_prompt(
                original_image,
                reposed_image,
                instruction,
                self._DEFAULT_REPAIR_PROMPT,
            )

        img_url = upload_image(reposed_image)
        if not img_url:
            debug = json.dumps({"analysis": analysis_text, "error": "upload_failed"}, indent=2)
            return (reposed_image, used_prompt, debug)

        args = {
            "image_url": img_url,
            "prompt": used_prompt,
            "enable_safety_checker": False,
        }

        if extra_params_json and extra_params_json.strip():
            try:
                extra = json.loads(extra_params_json)
                if isinstance(extra, dict):
                    args.update(extra)
                else:
                    print("[fal-splat] QwenCameraLockedRepair: extra_params_json must be a JSON object")
            except Exception as e:
                print(f"[fal-splat] QwenCameraLockedRepair: could not parse extra_params_json: {e}")

        debug = json.dumps({"analysis": analysis_text, "payload": args}, indent=2)

        try:
            print("[fal-splat] QwenCameraLockedRepair -> fal-ai/qwen-image-edit")
            result = submit_and_get("fal-ai/qwen-image-edit", args)
            repaired = process_images_result(result)
            return (repaired, used_prompt, debug)
        except Exception as e:
            print(f"[fal-splat] QwenCameraLockedRepair error: {e}")
            return (reposed_image, used_prompt, debug)


# ─────────────────────────────────────────────────────────────────────────────
# Node 8 — NanoBananaCameraLockedRepair
# ─────────────────────────────────────────────────────────────────────────────

class NanoBananaCameraLockedRepair:
    """
    Camera-locked Nano Banana path:
      1. synthesize a repair prompt from original + reposed
      2. edit only the reposed render with Nano Banana
    """

    _DEFAULT_REPAIR_PROMPT = (
        "Treat the input image as the base canvas and preserve its exact camera angle, "
        "framing, horizon, perspective, crop, and object layout. Repair only damaged "
        "regions and remove Gaussian splat artifacts such as ghosting, floaters, blurry "
        "smears, black holes, broken edges, and missing detail. Keep the output "
        "photorealistic, coherent, and compositionally identical to the input view."
    )

    _DEFAULT_ANALYSIS_INSTRUCTION = QwenCameraLockedRepair._DEFAULT_ANALYSIS_INSTRUCTION

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "reposed_image": ("IMAGE",),
            },
            "optional": {
                "manual_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional. If filled, skips prompt analysis and uses this prompt directly.",
                }),
                "analysis_instruction": ("STRING", {
                    "default": cls._DEFAULT_ANALYSIS_INSTRUCTION,
                    "multiline": True,
                    "tooltip": "Instruction sent to the VLM when auto-generating a repair prompt.",
                }),
                "seed": ("INT", {
                    "default": -1,
                    "tooltip": "Optional. -1 lets fal choose a random seed.",
                }),
                "extra_params_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional raw JSON merged into the fal Nano Banana payload.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("repaired_image", "used_prompt", "debug_details")
    FUNCTION = "repair"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Generate a camera-lock repair prompt from the original photo, then repair "
        "only the reposed render with Nano Banana to reduce viewpoint drift."
    )

    def repair(
        self,
        original_image,
        reposed_image,
        manual_prompt="",
        analysis_instruction="",
        seed=-1,
        extra_params_json="",
    ):
        used_prompt = (manual_prompt or "").strip()
        if used_prompt:
            analysis_text = "Manual prompt provided; VLM analysis skipped."
        else:
            instruction = (
                analysis_instruction.strip()
                if analysis_instruction and analysis_instruction.strip()
                else self._DEFAULT_ANALYSIS_INSTRUCTION
            )
            used_prompt, analysis_text = _analyse_camera_locked_prompt(
                original_image,
                reposed_image,
                instruction,
                self._DEFAULT_REPAIR_PROMPT,
            )

        img_url = upload_image(reposed_image)
        if not img_url:
            debug = json.dumps({"analysis": analysis_text, "error": "upload_failed"}, indent=2)
            return (reposed_image, used_prompt, debug)

        args = {
            "prompt": used_prompt,
            "image_urls": [img_url],
            "num_images": 1,
            "enable_safety_checker": False,
        }
        if seed != -1:
            args["seed"] = seed

        if extra_params_json and extra_params_json.strip():
            try:
                extra = json.loads(extra_params_json)
                if isinstance(extra, dict):
                    args.update(extra)
                else:
                    print("[fal-splat] NanoBananaCameraLockedRepair: extra_params_json must be a JSON object")
            except Exception as e:
                print(f"[fal-splat] NanoBananaCameraLockedRepair: could not parse extra_params_json: {e}")

        debug = json.dumps({"analysis": analysis_text, "payload": args}, indent=2)

        try:
            print("[fal-splat] NanoBananaCameraLockedRepair -> fal-ai/nano-banana/edit")
            result = submit_and_get("fal-ai/nano-banana/edit", args)
            repaired = process_images_result(result)
            return (repaired, used_prompt, debug)
        except Exception as e:
            print(f"[fal-splat] NanoBananaCameraLockedRepair error: {e}")
            return (reposed_image, used_prompt, debug)


# ─────────────────────────────────────────────────────────────────────────────
# Node 9 — SplatRepairBlend
# ─────────────────────────────────────────────────────────────────────────────

class SplatRepairBlend:
    """
    Alpha-blend a repaired image back into the original using the artifact mask.

    Allows fine control over how aggressively the repair is applied:
    • mask_strength = 1.0  → fully replace artifact regions with repair
    • mask_strength = 0.5  → 50/50 blend in artifact regions
    • edge_feather         → additional softening of mask boundaries

    Use this as the final compositing step in the repair workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "repaired_image": ("IMAGE",),
                "artifact_mask": ("MASK",),
            },
            "optional": {
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly to apply the repair in masked regions"}),
                "edge_feather": ("INT", {"default": 4, "min": 0, "max": 32,
                    "tooltip": "Additional Gaussian feathering of mask edges for seamless blending"}),
                "preserve_color": ("BOOLEAN", {"default": False,
                    "tooltip": "Transfer original color statistics into repaired region (experimental)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("blended_image", "blend_mask")
    FUNCTION = "blend"
    CATEGORY = "FAL-Splat/Repair"
    DESCRIPTION = (
        "Final compositing step: blend repaired result into original "
        "using the artifact mask with feathering."
    )

    def blend(
        self,
        original_image,
        repaired_image,
        artifact_mask,
        mask_strength=1.0,
        edge_feather=4,
        preserve_color=False,
    ):
        # Keep the reposed/base image as the canvas size and resize the repaired
        # result back onto that canvas before blending.
        original_image, repaired_image = _ensure_same_size(original_image, repaired_image)

        mask = artifact_mask.clone()  # (1 or B, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        target_h, target_w = original_image.shape[1], original_image.shape[2]
        if mask.shape[1] != target_h or mask.shape[2] != target_w:
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Optional feathering
        if edge_feather > 0:
            k = edge_feather * 2 + 1
            mask_4d = mask.unsqueeze(1).float()
            mask_4d = F.avg_pool2d(mask_4d, kernel_size=k, stride=1, padding=edge_feather)
            mask = mask_4d.squeeze(1)

        mask = (mask * mask_strength).clamp(0.0, 1.0)
        w = mask.unsqueeze(-1)  # (B,H,W,1)

        # Ensure batch dims match
        if repaired_image.shape[0] != original_image.shape[0]:
            if original_image.shape[0] == 1:
                original_image = original_image.expand_as(repaired_image)
            else:
                repaired_image = repaired_image.expand_as(original_image)
        if mask.shape[0] != original_image.shape[0]:
            if mask.shape[0] == 1:
                mask = mask.expand(original_image.shape[0], -1, -1)
            else:
                mask = mask[:1].expand(original_image.shape[0], -1, -1)
            w = mask.unsqueeze(-1)

        blended = repaired_image * w + original_image * (1.0 - w)

        if preserve_color:
            # Match repaired region's mean/std to original's (simple colour correction)
            for b in range(blended.shape[0]):
                for c in range(3):
                    orig_mean = original_image[b, ..., c].mean()
                    orig_std  = original_image[b, ..., c].std() + 1e-6
                    rep_mean  = blended[b, ..., c].mean()
                    rep_std   = blended[b, ..., c].std() + 1e-6
                    blended[b, ..., c] = (
                        (blended[b, ..., c] - rep_mean) / rep_std * orig_std + orig_mean
                    ).clamp(0, 1)

        return (blended, mask)


# ─────────────────────────────────────────────────────────────────────────────
# Mappings
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "SplatArtifactMask_splat":    SplatArtifactMask,
    "SplatDifferenceDebug_splat": SplatDifferenceDebug,
    "FluxSplatRepair_splat":      FluxSplatRepair,
    "FluxSplatInpaint_splat":     FluxSplatInpaint,
    "QwenSplatRepair_splat":      QwenSplatRepair,
    "QwenLoRASplatRepair_splat":  QwenLoRASplatRepair,
    "QwenCameraLockedRepair_splat": QwenCameraLockedRepair,
    "NanoBananaCameraLockedRepair_splat": NanoBananaCameraLockedRepair,
    "SplatRepairBlend_splat":     SplatRepairBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplatArtifactMask_splat":    "Splat Artifact Mask",
    "SplatDifferenceDebug_splat": "Splat Difference Debug",
    "FluxSplatRepair_splat":      "Flux Splat Repair (img2img + LoRA)",
    "FluxSplatInpaint_splat":     "Flux Splat Inpaint (Fill)",
    "QwenSplatRepair_splat":      "Qwen Splat Repair / Analyse",
    "QwenLoRASplatRepair_splat":  "Qwen LoRA Splat Repair",
    "QwenCameraLockedRepair_splat": "Qwen Camera-Locked Repair",
    "NanoBananaCameraLockedRepair_splat": "Nano Banana Camera-Locked Repair",
    "SplatRepairBlend_splat":     "Splat Repair Blend",
}
