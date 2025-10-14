import torch
from comfy.ldm.modules.attention import CrossAttention, default, optimized_attention, optimized_attention_masked
from ..utils import nag


class NAGCrossAttention(CrossAttention):
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
        x,
        context=None,
        value=None,
        mask=None,
        transformer_options=None,
        **kwargs,
):
        origin_bsz = len(context) - len(x)
        assert origin_bsz != 0

        q = self.to_q(x)
        q = torch.cat([q, q[-origin_bsz:]], dim=0)

        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision)

        # NAG
        out_negative, out_positive = out[-origin_bsz:], out[-origin_bsz * 2:-origin_bsz]
        out_guidance = nag(out_positive, out_negative, self.nag_scale, self.nag_tau, self.nag_alpha)
        out = torch.cat([out[:-origin_bsz * 2], out_guidance], dim=0)

        return self.to_out(out)
