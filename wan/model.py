from types import MethodType
from functools import partial

import torch
from einops import repeat

import comfy
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.wan.model import (
    WanModel,
    VaceWanModel,
    WanSelfAttention,
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    sinusoidal_embedding_1d,
)

from ..utils import nag, cat_context, check_nag_activation, poly1d, NAGSwitch


class NAGWanT2VCrossAttention(WanT2VCrossAttention):
    def __init__(
            self,
            *args,
            nag_scale: float = 1,
            nag_tau: float = 3.5,
            nag_alpha: float = 0.5,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def forward(
            self,
            x,
            context,
            context_pad_len: int = None,
            nag_pad_len: int = None,
            **kwargs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        origin_bsz = len(context) - len(x)
        assert origin_bsz != 0

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        q_negative = q[-origin_bsz:]
        k, k_negative = k[:-origin_bsz, :, context_pad_len:], k[-origin_bsz:, :, nag_pad_len:]
        v, v_negative = v[:-origin_bsz, :, context_pad_len:], v[-origin_bsz:, :, nag_pad_len:]

        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)
        x_negative = optimized_attention(q_negative, k_negative, v_negative, heads=self.num_heads)

        x_positive = x[-origin_bsz:]
        x_guidance = nag(x_positive, x_negative, self.nag_scale, self.nag_tau, self.nag_alpha)
        x = torch.cat([x[:-origin_bsz], x_guidance], dim=0)

        x = self.o(x)
        return x


class NAGWanI2VCrossAttention(WanI2VCrossAttention):
    def __init__(
            self,
            *args,
            nag_scale: float = 1,
            nag_tau: float = 3.5,
            nag_alpha: float = 0.5,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def forward(
            self,
            x,
            context,
            context_img_len=None,
            context_pad_len: int = None,
            nag_pad_len: int = None,
            transformer_options=None,
            **kwargs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        origin_bsz = len(context) - len(x)
        assert origin_bsz != 0

        context_img = context[:, :context_img_len]
        context = context[:, context_img_len:]

        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)

        q_negative = q[-origin_bsz:]
        k, k_negative = k[:-origin_bsz, :, context_pad_len:], k[-origin_bsz:, :, nag_pad_len:]
        v, v_negative = v[:-origin_bsz, :, context_pad_len:], v[-origin_bsz:, :, nag_pad_len:]
        k_img, k_img_negative = k_img[:-origin_bsz], k_img[-origin_bsz:]
        v_img, v_img_negative = v_img[:-origin_bsz], v_img[-origin_bsz:]

        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads)
        img_x_negative = optimized_attention(q_negative, k_img_negative, v_img_negative, heads=self.num_heads)
        x = optimized_attention(q, k, v, heads=self.num_heads)
        x_negative = optimized_attention(q_negative, k_negative, v_negative, heads=self.num_heads)

        x_positive = x[-origin_bsz:]
        x_guidance = nag(x_positive, x_negative, self.nag_scale, self.nag_tau, self.nag_alpha)
        x = torch.cat([x[:-origin_bsz], x_guidance], dim=0)

        img_x_positive = img_x[-origin_bsz:]
        img_x_guidance = nag(img_x_positive, img_x_negative, self.nag_scale, self.nag_tau, self.nag_alpha)
        img_x = torch.cat([img_x[:-origin_bsz], img_x_guidance], dim=0)

        # output
        x = x + img_x
        x = self.o(x)
        return x


class NAGWanModel(WanModel):
    def forward_orig(
            self,
            x,
            t,
            context,
            clip_fea=None,
            freqs=None,
            transformer_options={},
            **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                List of input video tensors with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context_clip = torch.cat([context_clip, context_clip[-context.shape[0] - context_clip.shape[0]:]])
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"],
                                       context_img_len=context_img_len)
                    return out

                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs},
                                                          {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward_orig_with_teacache(
            self,
            x,
            t,
            context,
            clip_fea=None,
            freqs=None,
            transformer_options={},
            **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                List of input video tensors with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        enable_teacache = transformer_options.get("enable_teacache", True)
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        model_type = transformer_options.get("model_type")
        cache_device = transformer_options.get("cache_device")

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context_clip = torch.cat([context_clip, context_clip[-context.shape[0] - context_clip.shape[0]:]])
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        if enable_teacache:
            modulated_inp = e0.to(cache_device) if "ret_mode" in model_type else e.to(cache_device)
            if not hasattr(self, 'teacache_state'):
                self.teacache_state = {
                    0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None,
                        'previous_residual': None},
                    1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None,
                        'previous_residual': None}
                }

            def update_cache_state(cache, modulated_inp):
                if cache['previous_modulated_input'] is not None:
                    try:
                        cache['accumulated_rel_l1_distance'] += poly1d(coefficients, (
                                    (modulated_inp - cache['previous_modulated_input']).abs().mean() / cache[
                                'previous_modulated_input'].abs().mean()))
                        if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                            cache['should_calc'] = False
                        else:
                            cache['should_calc'] = True
                            cache['accumulated_rel_l1_distance'] = 0
                    except:
                        cache['should_calc'] = True
                        cache['accumulated_rel_l1_distance'] = 0
                cache['previous_modulated_input'] = modulated_inp

            b = int(len(x) / len(cond_or_uncond))

            for i, k in enumerate(cond_or_uncond):
                update_cache_state(self.teacache_state[k], modulated_inp[i * b:(i + 1) * b])

            if enable_teacache:
                should_calc = False
                for k in cond_or_uncond:
                    should_calc = (should_calc or self.teacache_state[k]['should_calc'])
            else:
                should_calc = True

            if should_calc:
                ori_x = x.to(cache_device)

        else:
            should_calc = False

        if should_calc:
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"],
                                           context_img_len=context_img_len)
                        return out

                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs},
                                                              {"original_block": block_wrap})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

        else:
            for i, k in enumerate(cond_or_uncond):
                x[i * b:(i + 1) * b] += self.teacache_state[k]['previous_residual'].to(x.device)

        if enable_teacache and should_calc:
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]['previous_residual'] = (x.to(cache_device) - ori_x)[i*b:(i+1)*b]

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(
            self,
            x,
            timestep,
            context,
            clip_fea=None,
            time_dim_concat=None,
            transformer_options={},

            nag_negative_context=None,
            nag_sigma_end=0.,

            **kwargs,
    ):
        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)
        if apply_nag:
            origin_context_len = context.shape[1]
            context = cat_context(context, nag_negative_context, trim_context=True)
            context_pad_len = context.shape[1] - origin_context_len
            nag_pad_len = context.shape[1] - nag_negative_context.shape[1]

            forward_orig_ = self.forward_orig
            cross_attns_forward = list()

            try:
                if transformer_options.get("enable_teacache", False):
                    self.forward_orig = MethodType(NAGWanModel.forward_orig_with_teacache, self)
                else:
                    self.forward_orig = MethodType(NAGWanModel.forward_orig, self)

                cross_attn_cls = NAGWanT2VCrossAttention if self.model_type == "t2v" else NAGWanI2VCrossAttention
                for name, module in self.named_modules():
                    if "cross_attn" in name and isinstance(module, WanSelfAttention):
                        cross_attns_forward.append((module, module.forward))
                        module.forward = MethodType(
                            partial(
                                cross_attn_cls.forward,
                                context_pad_len=context_pad_len,
                                nag_pad_len=nag_pad_len,
                            ),
                            module,
                        )

                bs, c, t, h, w = x.shape
                x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

                patch_size = self.patch_size
                t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
                h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
                w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

                if time_dim_concat is not None:
                    time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
                    x = torch.cat([x, time_dim_concat], dim=2)
                    t_len = ((x.shape[2] + (patch_size[0] // 2)) // patch_size[0])

                img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
                img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device,
                                                                           dtype=x.dtype).reshape(-1, 1, 1)
                img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device,
                                                                           dtype=x.dtype).reshape(1, -1, 1)
                img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device,
                                                                           dtype=x.dtype).reshape(1, 1, -1)
                img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

                freqs = self.rope_embedder(img_ids).movedim(1, 2)
                output = self.forward_orig(
                    x, timestep, context, clip_fea=clip_fea, freqs=freqs,
                    transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]
            finally:
                # Always restore, even on interruption
                self.forward_orig = forward_orig_
                for mod, forward_fn in cross_attns_forward:
                    mod.forward = forward_fn

        else:
            bs, c, t, h, w = x.shape
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

            patch_size = self.patch_size
            t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
            h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
            w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

            if time_dim_concat is not None:
                time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
                x = torch.cat([x, time_dim_concat], dim=2)
                t_len = ((x.shape[2] + (patch_size[0] // 2)) // patch_size[0])

            img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device,
                                                                       dtype=x.dtype).reshape(-1, 1, 1)
            img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device,
                                                                       dtype=x.dtype).reshape(1, -1, 1)
            img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device,
                                                                       dtype=x.dtype).reshape(1, 1, -1)
            img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

            freqs = self.rope_embedder(img_ids).movedim(1, 2)
            # Call base WanModel.forward_orig directly to avoid issues if self.forward_orig was left in NAG state
            output = WanModel.forward_orig(
                self, x, timestep, context, clip_fea=clip_fea, freqs=freqs,
                transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

        return output


class NAGVaceWanModel(VaceWanModel):
    def forward_orig(
        self,
        x,
        t,
        context,
        vace_context,
        vace_strength,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        origin_batch_size = context.shape[0] - x.shape[0]

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context_clip = torch.cat([context_clip, context_clip[-origin_batch_size:]])
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        # arguments
        x_orig = x

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                    return out
                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

            ii = self.vace_layers_mapping.get(i, None)
            if ii is not None:
                for iii in range(len(c)):
                    c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                    x += c_skip * vace_strength[iii]
                del c_skip
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward_orig_with_teacache(
        self,
        x,
        t,
        context,
        vace_context,
        vace_strength,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        enable_teacache = transformer_options.get("enable_teacache", True)
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        model_type = transformer_options.get("model_type")
        cache_device = transformer_options.get("cache_device")

        origin_batch_size = context.shape[0] - x.shape[0]

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context_clip = torch.cat([context_clip, context_clip[-origin_batch_size:]])
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        # arguments
        x_orig = x

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        if enable_teacache:
            modulated_inp = e0.to(cache_device) if "ret_mode" in model_type else e.to(cache_device)
            if not hasattr(self, 'teacache_state'):
                self.teacache_state = {
                    0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None,
                        'previous_residual': None},
                    1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None,
                        'previous_residual': None}
                }

            def update_cache_state(cache, modulated_inp):
                if cache['previous_modulated_input'] is not None:
                    try:
                        cache['accumulated_rel_l1_distance'] += poly1d(coefficients, (
                                    (modulated_inp - cache['previous_modulated_input']).abs().mean() / cache[
                                'previous_modulated_input'].abs().mean()))
                        if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                            cache['should_calc'] = False
                        else:
                            cache['should_calc'] = True
                            cache['accumulated_rel_l1_distance'] = 0
                    except:
                        cache['should_calc'] = True
                        cache['accumulated_rel_l1_distance'] = 0
                cache['previous_modulated_input'] = modulated_inp

            b = int(len(x) / len(cond_or_uncond))

            for i, k in enumerate(cond_or_uncond):
                update_cache_state(self.teacache_state[k], modulated_inp[i * b:(i + 1) * b])

            if enable_teacache:
                should_calc = False
                for k in cond_or_uncond:
                    should_calc = (should_calc or self.teacache_state[k]['should_calc'])
            else:
                should_calc = True

            if should_calc:
                ori_x = x.to(cache_device)

        else:
            should_calc = False

        if should_calc:
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

                ii = self.vace_layers_mapping.get(i, None)
                if ii is not None:
                    for iii in range(len(c)):
                        c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                        x += c_skip * vace_strength[iii]
                    del c_skip
        else:
            for i, k in enumerate(cond_or_uncond):
                x[i * b:(i + 1) * b] += self.teacache_state[k]['previous_residual'].to(x.device)

        if enable_teacache and should_calc:
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]['previous_residual'] = (x.to(cache_device) - ori_x)[i * b:(i + 1) * b]

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(
            self,
            x,
            timestep,
            context,
            clip_fea=None,
            time_dim_concat=None,
            transformer_options={},

            nag_negative_context=None,
            nag_sigma_end=0.,

            **kwargs,
    ):
        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)
        if apply_nag:
            origin_context_len = context.shape[1]
            context = cat_context(context, nag_negative_context, trim_context=True)
            context_pad_len = context.shape[1] - origin_context_len
            nag_pad_len = context.shape[1] - nag_negative_context.shape[1]

            forward_orig_ = self.forward_orig
            cross_attns_forward = list()

            try:
                if transformer_options.get("enable_teacache", False):
                    self.forward_orig = MethodType(NAGVaceWanModel.forward_orig_with_teacache, self)
                else:
                    self.forward_orig = MethodType(NAGVaceWanModel.forward_orig, self)
                for name, module in self.named_modules():
                    if "cross_attn" in name and isinstance(module, WanSelfAttention):
                        cross_attns_forward.append((module, module.forward))
                        module.forward = MethodType(
                            partial(
                                NAGWanT2VCrossAttention.forward,
                                context_pad_len=context_pad_len,
                                nag_pad_len=nag_pad_len,
                            ),
                            module,
                        )

                bs, c, t, h, w = x.shape
                x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

                patch_size = self.patch_size
                t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
                h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
                w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

                if time_dim_concat is not None:
                    time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
                    x = torch.cat([x, time_dim_concat], dim=2)
                    t_len = ((x.shape[2] + (patch_size[0] // 2)) // patch_size[0])

                img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
                img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device,
                                                                           dtype=x.dtype).reshape(-1, 1, 1)
                img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device,
                                                                           dtype=x.dtype).reshape(1, -1, 1)
                img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device,
                                                                           dtype=x.dtype).reshape(1, 1, -1)
                img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

                freqs = self.rope_embedder(img_ids).movedim(1, 2)
                output = self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs,
                                         transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]
            finally:
                # Always restore, even on interruption
                self.forward_orig = forward_orig_
                for mod, forward_fn in cross_attns_forward:
                    mod.forward = forward_fn

        else:
            bs, c, t, h, w = x.shape
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

            patch_size = self.patch_size
            t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
            h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
            w_len = ((w + (patch_size[2] // 2)) // patch_size[2])

            if time_dim_concat is not None:
                time_dim_concat = comfy.ldm.common_dit.pad_to_patch_size(time_dim_concat, self.patch_size)
                x = torch.cat([x, time_dim_concat], dim=2)
                t_len = ((x.shape[2] + (patch_size[0] // 2)) // patch_size[0])

            img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device,
                                                                       dtype=x.dtype).reshape(-1, 1, 1)
            img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device,
                                                                       dtype=x.dtype).reshape(1, -1, 1)
            img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device,
                                                                       dtype=x.dtype).reshape(1, 1, -1)
            img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

            freqs = self.rope_embedder(img_ids).movedim(1, 2)
            # Call base VaceWanModel.forward_orig directly to avoid issues if self.forward_orig was left in NAG state
            output = VaceWanModel.forward_orig(
                self, x, timestep, context, clip_fea=clip_fea, freqs=freqs,
                transformer_options=transformer_options, **kwargs)[:, :, :t, :h, :w]

        return output


class NAGWanModelSwitch(NAGSwitch):
    def set_nag(self):
        nag_model_cls = NAGVaceWanModel if isinstance(self.model, VaceWanModel) else NAGWanModel
        self.model.forward = MethodType(
            partial(
                nag_model_cls.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )
        for name, module in self.model.named_modules():
            if "cross_attn" in name and isinstance(module, WanSelfAttention):
                module.nag_scale = self.nag_scale
                module.nag_tau = self.nag_tau
                module.nag_alpha = self.nag_alpha
