import torch
import comfy
import latent_preview

from .samplers import NAGCFGGuider as samplers_NAGCFGGuider
from .sample import sample_with_nag


def common_ksampler_with_nag(model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, positive, negative, nag_negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = sample_with_nag(
        model, noise, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, positive, negative, nag_negative, latent_image,
        denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return (out, )


class NAGCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "nag_negative": ("CONDITIONING", ),
                        "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step":0.1, "round": 0.01}),
                        "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                        "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step":0.01, "round": 0.01}),
                        "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            positive,
            negative,
            nag_negative,
            cfg,
            nag_scale,
            nag_tau,
            nag_alpha,
            nag_sigma_end,
            latent_image,
    ):
        batch_size = latent_image["samples"].shape[0]
        guider = samplers_NAGCFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        guider.set_batch_size(batch_size)
        guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
        return (guider,)


class KSamplerWithNAG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step":0.1, "round": 0.01}),
                "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step":0.01, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "nag_negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image for NAG."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, positive, negative, nag_negative, latent_image, denoise=1.0):
        return common_ksampler_with_nag(model, seed, steps, cfg, nag_scale, nag_tau, nag_alpha, nag_sigma_end, sampler_name, scheduler, positive, negative, nag_negative, latent_image, denoise=denoise)


NODE_CLASS_MAPPINGS = {
    "NAGCFGGuider": NAGCFGGuider,
    "KSamplerWithNAG": KSamplerWithNAG,
}
