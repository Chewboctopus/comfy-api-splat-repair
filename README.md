# Gaussian Splat Repair Workflow

This repository contains a full ComfyUI pipeline and customized custom nodes for repairing Gaussian Splat render artifacts (e.g. holes, floaters, warped perspective geometry) using **FLUX.2 Klein 9B Base** and the **mlsharp 3D repair LoRA**.

## Installation

1. Copy the `custom_nodes/comfyui-fal-splat` folder into your ComfyUI `custom_nodes` directory.
2. Copy the `custom_nodes/comfyui-GaussianViewer` folder into your ComfyUI `custom_nodes` directory.
3. Install dependencies by navigating to the ComfyUI root and running `pip install -r custom_nodes/comfyui-fal-splat/requirements.txt` and `pip install -r custom_nodes/comfyui-GaussianViewer/requirements.txt`.
4. Copy the `workflows/Gaussian_Splat_FalRepair_Mac_v001.json` file into your ComfyUI `user/default/workflows` directory, or just drag and drop it into your ComfyUI window.

## Setup & Usage

1. **Load Photo**: Input an original reference photo.
2. **SHARP Predict**: The ML-SHARP node converts the single photo into a local 3D Gaussian Splat (`.ply`).
3. **GaussianViewerLift**: Repose your camera within the local splat viewer to find your desired new perspective, then click **Set Camera**.
4. **FluxSplatRepair**: 
   - Paste your [fal.ai](https://fal.ai) API key into the `fal_api_key` field at the bottom of the node (you only need to do this once; it saves automatically).
   - Click **Queue**.
   - The node securely sends both your original image and the reposed render to FAL AI. It computes the exact camera movement delta (`translation_scale` applied to radians) and utilizes the `mlsharp` 3D repair LoRA on top of the FLUX.2 Klein Base model to structurally reconstruct missing areas and remove "floater" splat artifacts.

## Credits & Acknowledgements

* **[FAL AI / FLUX.2 Klein 9B Base](https://fal.ai/models/fal-ai/flux-2/klein/9b/base/edit/lora)**: For providing the lightning-fast base image-to-image pipeline.
* **[ML-SHARP (cyrildiagne & pmulefan)](https://huggingface.co/cyrildiagne/flux2-klein9b-lora-mlsharp-3d-repair)**: For the extraordinary 3D repair LoRA weights.
* **[ComfyUI-GaussianViewer](https://github.com/ashawkey/comfyui-GaussianViewer)**: For the fantastic OpenGL-based Gaussian viewer node. *Note: We customized `mlsharp_prompt.py` in this fork to properly handle Euler-to-radian conversion and 100x camera translation scaling.*
* **[comfyui-fal-splat]**: Heavily customized for this specific 3D repair workflow. We streamlined the parameter UI, bound the correct `base/edit/lora` inference chain, locked the required `num_inference_steps=28` defaults, and added automated API key management with proper error handling.

---
*Created by Hallett / Antigravity*
