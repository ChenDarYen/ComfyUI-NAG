import torch
from comfy.ldm.lumina.model import JointAttention, optimized_attention_masked
from comfy.ldm.flux.math import apply_rope
from ..utils import nag

class NAGJointAttention(JointAttention):
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
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        transformer_options={},
    ) -> torch.Tensor:

        # Standard checks
        if x.shape[0] == 0:
            return self.out(x)

        bsz, seqlen, _ = x.shape

        # Retrieve image token length injected by NAGNextDiT
        img_len = getattr(self, '_nag_img_token_len', 0)

        # Safety fallback: if NAG is not active or odd batch size, run standard
        if bsz % 2 != 0 or bsz < 2 or img_len == 0:
            return super().forward(x, x_mask, freqs_cis, transformer_options)

        origin_bsz = bsz // 2

        # Project Q, K, V
        xq, xk, xv = torch.split(
            self.qkv(x),
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # Split into Positive and Negative batches
        xq_pos, xq_neg = xq.split(origin_bsz)
        xk_pos, xk_neg = xk.split(origin_bsz)
        xv_pos, xv_neg = xv.split(origin_bsz)

        x_mask_pos = x_mask
        x_mask_neg = x_mask
        if x_mask is not None and x_mask.shape[0] == bsz:
            x_mask_pos = x_mask[:origin_bsz]
            x_mask_neg = x_mask[origin_bsz:]

        # Attention Calculation
        out_positive = optimized_attention_masked(
            xq_pos.movedim(1, 2), xk_pos.movedim(1, 2), xv_pos.movedim(1, 2),
            self.n_local_heads, x_mask_pos, skip_reshape=True, transformer_options=transformer_options
        )

        out_negative = optimized_attention_masked(
            xq_neg.movedim(1, 2), xk_neg.movedim(1, 2), xv_neg.movedim(1, 2),
            self.n_local_heads, x_mask_neg, skip_reshape=True, transformer_options=transformer_options
        )

        # Apply NAG ONLY to Image Tokens
        img_pos = out_positive[:, :img_len, :]
        img_neg = out_negative[:, :img_len, :]

        # 2. Apply Guidance to Image tokens
        img_guided = nag(img_pos, img_neg, self.nag_scale, self.nag_tau, self.nag_alpha)

        # 3. Keep the Positive Text tokens untouched
        # (We discard the negative text tokens, they served their purpose in Attention)
        txt_pos = out_positive[:, img_len:, :]

        # 4. Reconstruct the guided positive sequence
        out_guided_sequence = torch.cat([img_guided, txt_pos], dim=1)

        # 5. Apply output projection and concatenate with negative batch
        # Process separately to be more explicit about memory usage
        out_guided_projected = self.out(out_guided_sequence)
        out_negative_projected = self.out(out_negative)
        
        # Concatenate the projected outputs
        output = torch.cat([out_guided_projected, out_negative_projected], dim=0)
        
        # Clean up intermediate tensors to free memory faster
        del img_pos, img_neg, img_guided, txt_pos, out_guided_sequence
        del out_positive, out_negative
        del xq_pos, xq_neg, xk_pos, xk_neg, xv_pos, xv_neg
        del xq, xk, xv

        return output