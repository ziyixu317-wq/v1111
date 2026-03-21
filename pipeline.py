
import torch
import torch.nn as nn
from mae3d import MAE3D_Fusion
from vortex import calculate_ivd

class FlowVortexFusionPipeline(nn.Module):
    """
    Fused Pipeline: Helmholtz Preprocessing -> MAE3D (U-Net) -> Dual Output (Flow + IVD Mask)
    """
    def __init__(self, mode='pretrain', **mae_kwargs):
        super().__init__()
        self.mode = mode
        if not mae_kwargs:
            mae_kwargs = dict(
                patch_size=(4, 4, 4), in_chans=3, out_chans=1,
                embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                window_size=(4, 4, 4), mask_ratio=0.75,
                use_helmholtz=True  # Enabled for Helmholtz-Hodge decomposition
            )
        self.fused_model = MAE3D_Fusion(mode=mode, **mae_kwargs)
            
    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        if self.mode == 'pretrain':
            x_rec, mask = self.fused_model(x)
            # Support physical analysis during pre-training
            ivd_pred = calculate_ivd(x_rec)
            return x_rec, mask, ivd_pred
        else:
            # Segmentation mode: now returns (seg_logits, x_rec)
            seg_mask, x_rec = self.fused_model(x)
            return seg_mask, x_rec
