from __future__ import annotations

import copy
from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
import torch
from torch._dynamo.eval_frame import OptimizedModule
import torch._dynamo

torch._dynamo.config.suppress_errors = True

from comfy.samplers import (
    process_conds,
    preprocess_conds_hooks,
    cast_to_load_options,
    filter_registered_hooks_on_conds,
    get_total_hook_groups_in_conds,
    CFGGuider,
    sampler_object,
    KSampler,
)
import comfy.sampler_helpers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks
from comfy.ldm.flux.model import Flux
from comfy.ldm.chroma.model import Chroma
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper
from comfy.ldm.wan.model import WanModel, VaceWanModel
from comfy.ldm.hunyuan_video.model import HunyuanVideo
from comfy.ldm.hidream.model import HiDreamImageTransformer2DModel

from .flux.model import NAGFluxSwitch
from .chroma.model import NAGChromaSwitch
from .sd.openaimodel import NAGUNetModelSwitch
from .sd3.mmdit import NAGOpenAISignatureMMDITWrapperSwitch
from .wan.model import NAGWanModelSwitch
from .hunyuan_video.model import NAGHunyuanVideoSwitch
from .hidream.model import NAGHiDreamImageTransformer2DModelSwitch


def sample_with_nag(
        model,
        noise,
        positive, negative, nag_negative,
        cfg,
        nag_scale, nag_tau, nag_alpha, nag_sigma_end,
        device,
        sampler,
        sigmas,
        model_options={},
        latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None,
        latent_shapes=None, **kwargs,
):
    guider = NAGCFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    guider.set_batch_size(latent_image.shape[0])
    guider.set_nag(nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
    return guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
        latent_shapes=latent_shapes,
        **kwargs,
    )


class NAGCFGGuider(CFGGuider):
    def __init__(self, model_patcher: ModelPatcher):
        super().__init__(model_patcher=model_patcher)
        self.origin_nag_negative_cond = None
        self.nag_scale = 5.0
        self.nag_tau = 3.5
        self.nag_alpha = 0.25
        self.nag_sigma_end = 0.
        self.batch_size = 1

    def set_conds(self, positive, negative=None):
        self.inner_set_conds(
            {"positive": positive, "negative": negative} if negative is not None else {"positive": positive})

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_nag(self, nag_negative_cond, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
        self.origin_nag_negative_cond = nag_negative_cond
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.nag_sigma_end = nag_sigma_end

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None, **kwargs):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None, **kwargs):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        apply_guidance = self.nag_scale > 1.

        self.nag_negative_cond = None
        if apply_guidance:
            self.nag_negative_cond = copy.deepcopy(self.origin_nag_negative_cond)

            model = self.model_patcher.model.diffusion_model
            if isinstance(model, OptimizedModule):
                model = model._orig_mod
            model_type = type(model)
            if model_type == Flux:
                switcher_cls = NAGFluxSwitch
            elif model_type == Chroma:
                switcher_cls = NAGChromaSwitch
            elif model_type == UNetModel:
                switcher_cls = NAGUNetModelSwitch
            elif model_type == OpenAISignatureMMDITWrapper:
                switcher_cls = NAGOpenAISignatureMMDITWrapperSwitch
            elif model_type in [WanModel, VaceWanModel]:
                switcher_cls = NAGWanModelSwitch
            elif model_type == HunyuanVideo:
                switcher_cls = NAGHunyuanVideoSwitch
            elif model_type == HiDreamImageTransformer2DModel:
                switcher_cls = NAGHiDreamImageTransformer2DModelSwitch
            else:
                raise ValueError(
                    f"Model type {model_type} is not support for NAGCFGGuider"
                )
            self.nag_negative_cond[0][0] = self.nag_negative_cond[0][0].expand(self.batch_size, -1, -1)
            if self.nag_negative_cond[0][1].get("pooled_output", None) is not None:
                self.nag_negative_cond[0][1]["pooled_output"] = self.nag_negative_cond[0][1]["pooled_output"].expand(self.batch_size, -1)
            switcher = switcher_cls(
                model,
                self.nag_negative_cond,
                self.nag_scale, self.nag_tau, self.nag_alpha, self.nag_sigma_end,
            )
            switcher.set_nag()

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        if apply_guidance:
            switcher.set_origin()

        del self.conds
        del self.nag_negative_cond
        return output


class KSamplerWithNAG(KSampler):
    def sample(
            self,
            noise,
            positive, negative, nag_negative,
            cfg,
            nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            latent_image=None,
            start_step=None, last_step=None, force_full_denoise=False,
            denoise_mask=None,
            sigmas=None, callback=None, disable_pbar=False, seed=None,
            latent_shapes=None,
            **kwargs,
    ):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample_with_nag(
            self.model,
            noise,
            positive, negative, nag_negative,
            cfg,
            nag_scale, nag_tau, nag_alpha, nag_sigma_end,
            self.device,
            sampler,
            sigmas,
            self.model_options,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
            latent_shapes=latent_shapes,
            **kwargs,
        )

