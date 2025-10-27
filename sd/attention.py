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
        if x.shape[0] == 0:
            return self.to_out(x)
        context = default(context, x)
        origin_bsz = context.shape[0] - x.shape[0]
        if origin_bsz <= 0:
            # nothing to guide
            return super().forward(
                x,
                context=context,
                value=value,
                mask=mask,
                transformer_options=transformer_options,
                **kwargs,
            )

        q = self.to_q(x)

        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
        else:
            v = self.to_v(context)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if origin_bsz >= k.shape[0]:
            # no positive context to attend to
            return super().forward(
                x,
                context=context,
                value=value,
                mask=mask,
                transformer_options=transformer_options,
                **kwargs,
            )

        k_positive = k[:-origin_bsz].contiguous()
        v_positive = v[:-origin_bsz].contiguous()
        k_negative = k[-origin_bsz:].contiguous()
        v_negative = v[-origin_bsz:].contiguous()

        if mask is None:
            out = optimized_attention(
                q,
                k_positive,
                v_positive,
                self.heads,
                attn_precision=self.attn_precision,
            )
        else:
            mask_positive = mask[..., :k_positive.shape[1]].contiguous()
            out = optimized_attention_masked(
                q,
                k_positive,
                v_positive,
                self.heads,
                mask_positive,
                attn_precision=self.attn_precision,
            )

        nag_count = min(origin_bsz, q.shape[0], k_negative.shape[0], v_negative.shape[0])
        if nag_count == 0:
            return self.to_out(out)

        q_guided = q[-nag_count:].contiguous()
        k_guided = k_negative[-nag_count:].contiguous()
        v_guided = v_negative[-nag_count:].contiguous()
        if mask is None:
            out_negative = optimized_attention(
                q_guided,
                k_guided,
                v_guided,
                self.heads,
                attn_precision=self.attn_precision,
            )
        else:
            mask_guided = mask[-nag_count:].contiguous()
            mask_guided = mask_guided[..., -k_guided.shape[1]:].contiguous()
            out_negative = optimized_attention_masked(
                q_guided,
                k_guided,
                v_guided,
                self.heads,
                mask_guided,
                attn_precision=self.attn_precision,
            )

        out_positive = out[-nag_count:]
        out_guidance = nag(out_positive, out_negative, self.nag_scale, self.nag_tau, self.nag_alpha)
        out = out.clone()
        out[-nag_count:] = out_guidance

        return self.to_out(out)
