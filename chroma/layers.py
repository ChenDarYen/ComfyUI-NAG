import torch
from torch import Tensor

from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock

from ..utils import nag


class NAGDoubleStreamBlock(DoubleStreamBlock):
    def __init__(
            self,
            *args,
            nag_scale: float = 1,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.25,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def forward(
            self,
            img: Tensor,
            txt: Tensor,
            pe: Tensor,
            pe_negative: Tensor,
            vec: Tensor,
            attn_mask=None,
            context_pad_len: int = 0,
            nag_pad_len: int = 0,
            transformer_options=None,
            **kwargs,
    ):
        origin_bsz = len(txt) - len(img)
        assert origin_bsz != 0

        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        # prepare image for attention
        img_modulated = torch.addcmul(img_mod1.shift[:-origin_bsz], 1 + img_mod1.scale[:-origin_bsz], self.img_norm1(img))
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = torch.addcmul(txt_mod1.shift, 1 + txt_mod1.scale, self.txt_norm1(txt))
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3,
                                                                                                              1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        txt_q_negative, txt_q = txt_q[-origin_bsz:, :, nag_pad_len:], txt_q[:-origin_bsz, :, context_pad_len:]
        txt_k_negative, txt_k = txt_k[-origin_bsz:, :, nag_pad_len:], txt_k[:-origin_bsz, :, context_pad_len:]
        txt_v_negative, txt_v = txt_v[-origin_bsz:, :, nag_pad_len:], txt_v[:-origin_bsz, :, context_pad_len:]

        img_q_negative = img_q[-origin_bsz:]
        img_k_negative = img_k[-origin_bsz:]
        img_v_negative = img_v[-origin_bsz:]

        # run actual attention
        attn_negative = attention(
            torch.cat((txt_q_negative, img_q_negative), dim=2),
            torch.cat((txt_k_negative, img_k_negative), dim=2),
            torch.cat((txt_v_negative, img_v_negative), dim=2),
            pe=pe_negative, mask=attn_mask,
        )
        attn = attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe, mask=attn_mask,
        )

        txt_attn_negative, img_attn_negative = attn_negative[:, : txt.shape[1] - nag_pad_len], attn_negative[:, txt.shape[1] - nag_pad_len:]
        txt_attn, img_attn = attn[:, : txt.shape[1] - context_pad_len], attn[:, txt.shape[1] - context_pad_len:]

        # NAG
        img_attn_positive = img_attn[-origin_bsz:]
        img_attn_guidance = nag(img_attn_positive, img_attn_negative, self.nag_scale, self.nag_tau, self.nag_alpha)

        img_attn = torch.cat([img_attn[:-origin_bsz], img_attn_guidance], dim=0)

        # calculate the img bloks
        img.addcmul_(img_mod1.gate[:-origin_bsz], self.img_attn.proj(img_attn))
        img.addcmul_(
            img_mod2.gate[:-origin_bsz],
            self.img_mlp(torch.addcmul(img_mod2.shift[:-origin_bsz], 1 + img_mod2.scale[:-origin_bsz], self.img_norm2(img))),
        )

        # calculate the txt bloks
        txt[:-origin_bsz, context_pad_len:].addcmul_(txt_mod1.gate[:-origin_bsz], self.txt_attn.proj(txt_attn))
        txt[-origin_bsz:, nag_pad_len:].addcmul_(txt_mod1.gate[-origin_bsz:], self.txt_attn.proj(txt_attn_negative))
        txt.addcmul_(txt_mod2.gate,
                     self.txt_mlp(torch.addcmul(txt_mod2.shift, 1 + txt_mod2.scale, self.txt_norm2(txt))))

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class NAGSingleStreamBlock(SingleStreamBlock):
    def __init__(
            self,
            *args,
            nag_scale: float = 1,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.25,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def forward(
            self,
            x: Tensor,
            pe: Tensor,
            pe_negative: Tensor,
            vec: Tensor,
            attn_mask=None,
            txt_length:int = None,
            origin_bsz: int = None,
            context_pad_len: int = 0,
            nag_pad_len: int = 0,
            transformer_options=None,
            **kwargs,
    ) -> Tensor:
        mod = vec
        x_mod = torch.addcmul(mod.shift, 1 + mod.scale, self.pre_norm(x))
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        # NAG
        q, q_negative = q[:-origin_bsz, :, context_pad_len:], q[-origin_bsz:, :, nag_pad_len:]
        k, k_negative = k[:-origin_bsz, :, context_pad_len:], k[-origin_bsz:, :, nag_pad_len:]
        v, v_negative = v[:-origin_bsz, :, context_pad_len:], v[-origin_bsz:, :, nag_pad_len:]

        attn_negative = attention(q_negative, k_negative, v_negative, pe=pe_negative, mask=attn_mask)
        attn = attention(q, k, v, pe=pe, mask=attn_mask)

        img_attn_negative = attn_negative[:, txt_length - nag_pad_len:]
        img_attn = attn[:, txt_length - context_pad_len:]

        img_attn_positive = img_attn[-origin_bsz:]
        img_attn_guidance = nag(img_attn_positive, img_attn_negative, self.nag_scale, self.nag_tau, self.nag_alpha)

        attn_negative[:, txt_length - nag_pad_len:] = img_attn_guidance
        attn[-origin_bsz:, txt_length - context_pad_len:] = img_attn_guidance

        # compute activation in mlp stream, cat again and run second linear layer
        output_negative = self.linear2(torch.cat((attn_negative, self.mlp_act(mlp[-origin_bsz:, nag_pad_len:])), 2))
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp[:-origin_bsz, context_pad_len:])), 2))

        x[:-origin_bsz, context_pad_len:].addcmul_(mod.gate[:-origin_bsz], output)
        x[-origin_bsz:, nag_pad_len:].addcmul_(mod.gate[-origin_bsz:], output_negative)

        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x
