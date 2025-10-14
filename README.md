# ComfyUI-NAG

Implementation of [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://chendaryen.github.io/NAG.github.io/) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

NAG restores effective negative prompting in few-step diffusion models, and complements CFG in multi-step sampling for improved quality and control.

Paper: https://arxiv.org/abs/2505.21179

Code: https://github.com/ChenDarYen/Normalized-Attention-Guidance

Wan2.1 Demo: https://huggingface.co/spaces/ChenDY/NAG_wan2-1-fast

LTX Video Demo: https://huggingface.co/spaces/ChenDY/NAG_ltx-video-distilled

Flux-Dev Demo: https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-dev

![comfyui-nag](workflow.png?cache=20250628)

## News
2025-14-10: updated for use in newer comfyui versions

2025-07-06: Add three new nodes:
- `KSamplerWithNAG (Advanced)` as a drop-in replacement for `KSampler (Advanced)`.
- `SamplerCustomWithNAG` for `SamplerCustom`.
- `NAGGuider` for `BasicGuider`.

2025-07-02: `HiDream` is now supported!

2025-07-02: Add support for `TeaCache` and `WaveSpeed` to accelerate NAG sampling!

2025-06-30: Fix a major bug affecting `Flux`, `Flux Kontext` and `Chroma`, resulting in degraded guidance. Please update your NAG node!

2025-06-29: Add compile model support. You can now use compile model nodes like `TorchCompileModel` to speed up NAG sampling!

2025-06-28: `Flux Kontext` is now supported. Check out the [workflow](https://github.com/ChenDarYen/ComfyUI-NAG/blob/main/workflows/NAG-Flux-Kontext-Dev-ComfyUI-Workflow.json)!

2025-06-26: `Hunyuan video` is now supported!

2025-06-25: `Wan` video generation is now supported (GGUF compatible)! Try it out with the new [workflow](https://github.com/ChenDarYen/ComfyUI-NAG/blob/main/workflows/NAG-Wan-Fast-ComfyUI-Workflow.json)!

## Nodes

- `KSamplerWithNAG`, `KSamplerWithNAG (Advanced)`, `SamplerCustomWithNAG`
- `BasicGuider`, `NAGCFGGuider`

## Usage

To use NAG, simply replace
- `KSampler` with `KSamplerWithNAG`.
- `KSamplerWithNAG (Advanced)` with `KSampler (Advanced)`.
- `SamplerCustomWithNAG` with `SamplerCustom`.
- `NAGGuider` with `BasicGuider`.
- `CFGGuider` with `NAGCFGGuider`.

We currently support `Flux`, `Flux Kontext`, `Wan`, `Vace Wan`, `Hunyuan Video`, `Choroma`, `SD3.5`, `SDXL` and `SD`.

Example workflows are available in the `./workflows` directory!

## Key Inputs

When working with a new model, it's recommended to first find a good combination of `nag_tau` and `nag_alpha`, which ensures that the negative guidance is effective without introducing artifacts.

Once you're satisfied, keep `nag_tau` and `nag_alpha` fixed and tune only `nag_scale` in most cases to control the strength of guidance.

Using `nag_sigma_end` to reduce computation without much quality drop.

For flow-based models like `Flux`, `nag_sigma_end = 0.75` achieves near-identical results with significantly improved speed. For diffusion-based `SDXL`, a good default is `nag_sigma_end = 4`.

- `nag_scale`: The scale for attention feature extrapolation. Higher values result in stronger negative guidance.
- `nag_tau`: The normalisation threshold. Higher values result in stronger negative guidance.
- `nag_alpha`: Blending factor between original and extrapolated attention. Higher values result in stronger negative guidance.
- `nag_sigma_end`: NAG will be active only until `nag_sigma_end`.

### Rule of Thumb

- For image-reference tasks (e.g., Image2Video), use lower `nag_tau` and `nag_alpha` to preserve the reference content more faithfully.
- For models that require more sampling steps and higher CFG, also prefer lower `nag_tau` and `nag_alpha`.
- For few-step models, you can use higher `nag_tau` and `nag_alpha` to have stronger negative guidance.
