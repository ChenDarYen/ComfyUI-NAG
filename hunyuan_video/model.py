from types import MethodType
from functools import partial
from typing import Callable

import torch
from torch import Tensor

from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock, timestep_embedding
from comfy.ldm.hunyuan_video.model import HunyuanVideo

from ..flux.layers import NAGSingleStreamBlock, NAGDoubleStreamBlock, apply_mod
from ..utils import cat_context, check_nag_activation, poly1d, NAGSwitch


class NAGHunyuanVideo(HunyuanVideo):
    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_ids_negative: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                guidance = torch.cat((guidance, guidance[-origin_bsz:]), dim=0)
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        ids_negative = torch.cat((img_ids[-origin_bsz:], txt_ids_negative), dim=1)
        pe = self.pe_embedder(ids)
        pe_negative = self.pe_embedder(ids_negative)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        pe_negative=args["pe_negative"],
                        attn_mask=args["attention_mask"],
                        modulation_dims_img=args["modulation_dims_img"],
                        modulation_dims_txt=args["modulation_dims_txt"],
                    )
                    return out

                out = blocks_replace[("double_block", i)]({
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": pe,
                    "pe_negative": pe_negative,
                    "attention_mask": attn_mask,
                    'modulation_dims_img': modulation_dims,
                    'modulation_dims_txt': modulation_dims_txt,
                    'transformer_options': transformer_options,
                }, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    pe=pe,
                    pe_negative=pe_negative,
                    attn_mask=attn_mask,
                    modulation_dims_img=modulation_dims,
                    modulation_dims_txt=modulation_dims_txt,
                )

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((img, img[-origin_bsz:]), dim=0)
        img = torch.cat((img, txt), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(
                        args["img"],
                        vec=args["vec"],
                        pe=args["pe"],
                        pe_negative=args["pe_negative"],
                        attn_mask=args["attention_mask"],
                        modulation_dims=args["modulation_dims"],
                    )
                    return out

                out = blocks_replace[("single_block", i)]({
                    "img": img,
                    "vec": vec,
                    "pe": pe,
                    "pe_negative": pe_negative,
                    "attention_mask": attn_mask,
                    'modulation_dims': modulation_dims,
                    'transformer_options': transformer_options,
                }, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(
                    img,
                    vec=vec,
                    pe=pe,
                    pe_negative=pe_negative,
                    attn_mask=attn_mask,
                    modulation_dims=modulation_dims,
                )

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, : img_len] += add

        img = img[:-origin_bsz]
        img = img[:, : img_len]
        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]

        img = self.final_layer(img, vec[:-origin_bsz], modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

    def forward_orig_with_teacache(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_ids_negative: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        enable_teacache = transformer_options.get("enable_teacache", True)
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cache_device = transformer_options.get("cache_device")

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                guidance = torch.cat((guidance, guidance[-origin_bsz:]), dim=0)
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        ids_negative = torch.cat((img_ids[-origin_bsz:], txt_ids_negative), dim=1)
        pe = self.pe_embedder(ids)
        pe_negative = self.pe_embedder(ids_negative)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})

        if enable_teacache:
            img_mod1, _ = self.double_blocks[0].img_mod(vec)
            modulated_inp = self.double_blocks[0].img_norm1(img)
            modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift).to(cache_device)

            if not hasattr(self, 'accumulated_rel_l1_distance'):
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                try:
                    self.accumulated_rel_l1_distance += \
                        poly1d(coefficients, ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
                    if self.accumulated_rel_l1_distance < rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance = 0
                except:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0

            self.previous_modulated_input = modulated_inp
            if should_calc:
                ori_img = img.to(cache_device)

        else:
            should_calc = False

        if should_calc:
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(
                            img=args["img"],
                            txt=args["txt"],
                            vec=args["vec"],
                            pe=args["pe"],
                            pe_negative=args["pe_negative"],
                            attn_mask=args["attention_mask"],
                            modulation_dims_img=args["modulation_dims_img"],
                            modulation_dims_txt=args["modulation_dims_txt"],
                        )
                        return out

                    out = blocks_replace[("double_block", i)]({
                        "img": img,
                        "txt": txt,
                        "vec": vec,
                        "pe": pe,
                        "pe_negative": pe_negative,
                        "attention_mask": attn_mask,
                        'modulation_dims_img': modulation_dims,
                        'modulation_dims_txt': modulation_dims_txt,
                        'transformer_options': transformer_options,
                    }, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(
                        img=img,
                        txt=txt,
                        vec=vec,
                        pe=pe,
                        pe_negative=pe_negative,
                        attn_mask=attn_mask,
                        modulation_dims_img=modulation_dims,
                        modulation_dims_txt=modulation_dims_txt,
                    )

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((img, img[-origin_bsz:]), dim=0)
            img = torch.cat((img, txt), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(
                            args["img"],
                            vec=args["vec"],
                            pe=args["pe"],
                            pe_negative=args["pe_negative"],
                            attn_mask=args["attention_mask"],
                            modulation_dims=args["modulation_dims"],
                        )
                        return out

                    out = blocks_replace[("single_block", i)]({
                        "img": img,
                        "vec": vec,
                        "pe": pe,
                        "pe_negative": pe_negative,
                        "attention_mask": attn_mask,
                        'modulation_dims': modulation_dims,
                        'transformer_options': transformer_options,
                    }, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(
                        img,
                        vec=vec,
                        pe=pe,
                        pe_negative=pe_negative,
                        attn_mask=attn_mask,
                        modulation_dims=modulation_dims,
                    )

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:-origin_bsz]
            img = img[:, : img_len]
            if ref_latent is not None:
                img = img[:, ref_latent.shape[1]:]

        else:
            img += self.previous_residual.to(img.device)

        if enable_teacache and should_calc:
            self.previous_residual = img.to(cache_device) - ori_img

        img = self.final_layer(img, vec[:-origin_bsz], modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

    def forward_orig_with_wavespeed(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_ids_negative: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        control=None,
        transformer_options={},
        use_cache: Callable = None,
        apply_prev_hidden_states_residual: Callable = None,
        set_buffer: Callable = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                guidance = torch.cat((guidance, guidance[-origin_bsz:]), dim=0)
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        ids_negative = torch.cat((img_ids[-origin_bsz:], txt_ids_negative), dim=1)
        pe = self.pe_embedder(ids)
        pe_negative = self.pe_embedder(ids_negative)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        original_img = img
        can_use_cache = False

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks[0].transformer_blocks):
            if i == 1:
                torch._dynamo.graph_break()
                if can_use_cache:
                    del first_img_residual
                    img, txt = apply_prev_hidden_states_residual(img, txt)
                    break
                else:
                    set_buffer("first_hidden_states_residual", first_img_residual)
                    del first_img_residual

                    original_img = img
                    original_txt = txt

            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        pe_negative=args["pe_negative"],
                        attn_mask=args["attention_mask"],
                        modulation_dims_img=args["modulation_dims_img"],
                        modulation_dims_txt=args["modulation_dims_txt"],
                    )
                    return out

                out = blocks_replace[("double_block", i)]({
                    "img": img,
                    "txt": txt,
                    "vec": vec,
                    "pe": pe,
                    "pe_negative": pe_negative,
                    "attention_mask": attn_mask,
                    'modulation_dims_img': modulation_dims,
                    'modulation_dims_txt': modulation_dims_txt,
                    'transformer_options': transformer_options,
                }, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    pe=pe,
                    pe_negative=pe_negative,
                    attn_mask=attn_mask,
                    modulation_dims_img=modulation_dims,
                    modulation_dims_txt=modulation_dims_txt,
                )

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            if i == 0:
                first_img_residual = img - original_img
                can_use_cache = use_cache(first_img_residual)
                del original_img

        if not can_use_cache:
            img = torch.cat((img, img[-origin_bsz:]), dim=0)
            img = torch.cat((img, txt), 1)

            for i, block in enumerate(self.double_blocks[0].single_transformer_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(
                            args["img"],
                            vec=args["vec"],
                            pe=args["pe"],
                            pe_negative=args["pe_negative"],
                            attn_mask=args["attention_mask"],
                            modulation_dims=args["modulation_dims"],
                        )
                        return out

                    out = blocks_replace[("single_block", i)]({
                        "img": img,
                        "vec": vec,
                        "pe": pe,
                        "pe_negative": pe_negative,
                        "attention_mask": attn_mask,
                        'modulation_dims': modulation_dims,
                        'transformer_options': transformer_options,
                    }, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(
                        img,
                        vec=vec,
                        pe=pe,
                        pe_negative=pe_negative,
                        attn_mask=attn_mask,
                        modulation_dims=modulation_dims,
                    )

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:-origin_bsz]
            img = img[:, : img_len]

            img = img.contiguous()
            txt = txt.contiguous()
            img_residual = img - original_img
            txt_residual = txt - original_txt
            set_buffer("hidden_states_residual", img_residual)
            set_buffer("encoder_hidden_states_residual", txt_residual[:-origin_bsz])

            if ref_latent is not None:
                img = img[:, ref_latent.shape[1]:]

        torch._dynamo.graph_break()

        img = self.final_layer(img, vec[:-origin_bsz], modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

    def forward(
            self,
            x,
            timestep,
            context,
            y,
            guidance=None,
            attention_mask=None,
            guiding_frame_index=None,
            ref_latent=None,
            control=None,
            transformer_options={},

            nag_negative_context=None,
            nag_negative_y=None,
            nag_sigma_end=0.,

            **kwargs,
    ):
        bs, c, t, h, w = x.shape
        img_ids = self.img_ids(x)

        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)
        if apply_nag:
            origin_context_len = context.shape[1]
            nag_bsz, nag_negative_context_len = nag_negative_context.shape[:2]
            context = cat_context(context, nag_negative_context, trim_context=True)
            y = torch.cat((y, nag_negative_y.to(y)), dim=0)
            context_pad_len = context.shape[1] - origin_context_len
            nag_pad_len = context.shape[1] - nag_negative_context_len

            forward_orig_ = self.forward_orig
            double_blocks_forward = list()
            single_blocks_forward = list()

            double_blocks = self.double_blocks
            single_blocks = self.single_blocks
            is_wavespeed = "CachedTransformerBlocks" in type(double_blocks[0]).__name__
            if is_wavespeed:  # chengzeyi/Comfy-WaveSpeed
                cached_blocks = self.double_blocks[0]
                double_blocks = cached_blocks.transformer_blocks
                single_blocks = cached_blocks.single_transformer_blocks

            try:
                if transformer_options.get("enable_teacache", False):
                    self.forward_orig = MethodType(NAGHunyuanVideo.forward_orig_with_teacache, self)

                elif is_wavespeed:
                    get_can_use_cache = cached_blocks.forward.__globals__["get_can_use_cache"]
                    set_buffer = cached_blocks.forward.__globals__["set_buffer"]
                    apply_prev_hidden_states_residual = cached_blocks.forward.__globals__["apply_prev_hidden_states_residual"]

                    def use_cache(first_hidden_states_residual):
                        return get_can_use_cache(
                            first_hidden_states_residual,
                            threshold=cached_blocks.residual_diff_threshold,
                            validation_function=cached_blocks.validate_can_use_cache_function,
                        )

                    self.forward_orig = MethodType(
                        partial(
                            NAGHunyuanVideo.forward_orig_with_wavespeed,
                            use_cache=use_cache,
                            apply_prev_hidden_states_residual=apply_prev_hidden_states_residual,
                            set_buffer=set_buffer,
                        ),
                        self,
                    )

                else:
                    self.forward_orig = MethodType(NAGHunyuanVideo.forward_orig, self)

                for block in double_blocks:
                    double_blocks_forward.append(block.forward)
                    block.forward = MethodType(
                        partial(
                            NAGDoubleStreamBlock.forward,
                            context_pad_len=context_pad_len,
                            nag_pad_len=nag_pad_len,
                        ),
                        block,
                    )

                for block in single_blocks:
                    single_blocks_forward.append(block.forward)
                    block.forward = MethodType(
                        partial(
                            NAGSingleStreamBlock.forward,
                            img_length=img_ids.shape[1],
                            origin_bsz=nag_bsz,
                            context_pad_len=context_pad_len,
                            nag_pad_len=nag_pad_len,
                        ),
                        block,
                    )

                txt_ids = torch.zeros((bs, origin_context_len, 3), device=x.device, dtype=x.dtype)
                txt_ids_negative = torch.zeros((nag_bsz, nag_negative_context_len, 3), device=x.device, dtype=x.dtype)
                out = self.forward_orig(
                    x, img_ids, context, txt_ids, txt_ids_negative, attention_mask, timestep, y, guidance, guiding_frame_index, ref_latent,
                    control=control, transformer_options=transformer_options,
                )
            finally:
                # Always restore, even on interruption
                self.forward_orig = forward_orig_
                for block in double_blocks:
                    if double_blocks_forward:
                        block.forward = double_blocks_forward.pop(0)
                for block in single_blocks:
                    if single_blocks_forward:
                        block.forward = single_blocks_forward.pop(0)

        else:
            txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
            # Call base HunyuanVideo.forward_orig directly to avoid issues if self.forward_orig was left in NAG state
            out = HunyuanVideo.forward_orig(
                self, x, img_ids, context, txt_ids, attention_mask, timestep, y, guidance,
                guiding_frame_index, ref_latent,
                control=control, transformer_options=transformer_options,
            )

        return out


class NAGHunyuanVideoSwitch(NAGSwitch):
    def set_nag(self):
        self.model.forward = MethodType(
            partial(
                NAGHunyuanVideo.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_negative_y=self.nag_negative_cond[0][1]["pooled_output"],
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )
        for block in self.model.double_blocks:
            block.nag_scale = self.nag_scale
            block.nag_tau = self.nag_tau
            block.nag_alpha = self.nag_alpha
        for block in self.model.single_blocks:
            block.nag_scale = self.nag_scale
            block.nag_tau = self.nag_tau
            block.nag_alpha = self.nag_alpha
