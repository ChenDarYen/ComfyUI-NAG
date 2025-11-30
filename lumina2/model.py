from functools import partial
from types import MethodType
import torch
import comfy.patcher_extension
from comfy.ldm.lumina.model import NextDiT, JointAttention
from ..utils import check_nag_activation, NAGSwitch, cat_context
from .attention import NAGJointAttention


class NAGNextDiT(NextDiT):
    """NAG-enabled NextDiT model for Lumina2/Z-Image"""

    def forward(
        self,
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask=None,
        nag_negative_context=None,
        nag_sigma_end=0.,
        **kwargs,
    ):
        transformer_options = kwargs.get("transformer_options", {})
        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)

        attention_forwards = list()

        if apply_nag:
            # 1. Calculate Image Token Count for the Attention layer to use
            _, _, h, w = x.shape
            p = self.patch_size
            img_token_len = (h // p) * (w // p)
            transformer_options["nag_img_token_len"] = img_token_len

            # 2. Prepare Inputs
            context = cat_context(context, nag_negative_context)
            x = torch.cat([x, x], dim=0)
            
            if timesteps is not None:
                timesteps = torch.cat([timesteps, timesteps], dim=0)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

            # 3. Patch Modules safely
            for name, module in self.named_modules():
                # Check for 'qkv' attribute to ensure we only patch actual Attention layers
                # and strictly exclude the rope_embedder or other false positives.
                if isinstance(module, JointAttention) and hasattr(module, "qkv"):
                    attention_forwards.append((module, module.forward))
                    module.forward = MethodType(NAGJointAttention.forward, module)

        try:
            # 4. Execute Model
            output = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self._forward,
                self,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    transformer_options
                )
            ).execute(x, timesteps, context, num_tokens, attention_mask, **kwargs)

        finally:
            # 5. Restore Original Methods (Always run this, even if crash happens)
            if apply_nag:
                for mod, forward_fn in attention_forwards:
                    mod.forward = forward_fn
                
                # Cleanup
                if "nag_img_token_len" in transformer_options:
                    del transformer_options["nag_img_token_len"]

        # 6. Post-process Output
        if apply_nag and isinstance(output, torch.Tensor):
            # Take only the first half (guided positive)
            output = output[:output.shape[0] // 2]

        return output

class NAGNextDiTSwitch(NAGSwitch):
    """Switch to enable/disable NAG for NextDiT models"""
    
    def set_nag(self):
        # Replace the model's forward method with NAG-enabled version
        self.model.forward = MethodType(
            partial(
                NAGNextDiT.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model
        )
        
        # Set NAG parameters on all JointAttention modules
        for name, module in self.model.named_modules():
            if isinstance(module, JointAttention):
                module.nag_scale = self.nag_scale
                module.nag_tau = self.nag_tau
                module.nag_alpha = self.nag_alpha
