"""
mae3d.py — 3D Masked Autoencoder with Physics-Informed Loss
===========================================================
Implements the MAE architecture and PI-MAE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from swin3d import SwinTransformer3D
# Wait: swin3d is in the SAME directory, so we should import from swin3d, not backbones.swin3d
from swin3d import SwinTransformer3D

class MAE3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, 
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                 window_size=(4, 4, 4), mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        
        self.encoder = SwinTransformer3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            depths=depths, num_heads=num_heads, window_size=window_size
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))
        
        enc_feat_dim = int(embed_dim * 2 ** (len(depths) - 1))
        
        # Simple Linear Decoder (SimMIM style)
        self.decoder = nn.Sequential(
            nn.Conv3d(enc_feat_dim, enc_feat_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(enc_feat_dim, in_chans, kernel_size=1)
        )
        
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # 1. Patch Embedding
        x_embed = self.encoder.patch_embed(x) # B, Dp, Hp, Wp, embed_dim
        Bp, Dp, Hp, Wp, Cp = x_embed.shape
        
        # 2. Masking
        N = Dp * Hp * Wp
        noise = torch.rand(B, N, device=x.device)
        mask = (noise < self.mask_ratio).float() # 1 is masked, 0 is keep
        mask_spatial = mask.view(B, Dp, Hp, Wp, 1)
        
        # Replace masked tokens
        x_masked = x_embed * (1 - mask_spatial) + self.mask_token * mask_spatial
        
        # 3. Encoder Forward
        x_masked = self.encoder.pos_drop(x_masked)
        outs = []
        for layer in self.encoder.layers:
            x_out, x_masked = layer(x_masked)
            outs.append(x_out)
        
        x_enc = self.encoder.norm(x_masked) # B, Dp/..., Hp/..., Wp/..., C_final
        
        # 4. Decoder
        # Restore spatial dims for the convolutional decoder
        x_enc = rearrange(x_enc, 'b d h w c -> b c d h w')
        
        x_rec = self.decoder(x_enc)
        
        # Upsample to original resolution
        x_rec = F.interpolate(x_rec, size=(D, H, W), mode='trilinear', align_corners=False)
        
        # Return reconstructed volume and interpolated mask 
        mask_pixel = F.interpolate(mask_spatial.permute(0, 4, 1, 2, 3), size=(D, H, W), mode='nearest')
        
        return x_rec, mask_pixel


def pi_mae_loss(pred, target, mask, dx=1.0, dy=1.0, dz=1.0, lambda_div=0.1):
    """
    Physical-Informed Loss: MSE Reconstruction + Divergence Penalty
    pred, target: (B, 3, D, H, W)
    mask: (B, 1, D, H, W) where 1 indicates masked
    """
    # MSE Loss on masked pixels
    mse_loss = F.mse_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() * 3 + 1e-6)
    
    # Calculate Divergence: du/dx + dv/dy + dw/dz
    # Using central differences
    u = pred[:, 0, :, :, :]
    v = pred[:, 1, :, :, :]
    w = pred[:, 2, :, :, :]
    
    # Pad to compute gradient without dropping boundary
    u_pad = F.pad(u, (1, 1, 0, 0, 0, 0), mode='replicate')
    v_pad = F.pad(v, (0, 0, 1, 1, 0, 0), mode='replicate')
    w_pad = F.pad(w, (0, 0, 0, 0, 1, 1), mode='replicate')
    
    du_dx = (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2]) / (2 * dx)
    dv_dy = (v_pad[:, :, 2:, :] - v_pad[:, :, :-2, :]) / (2 * dy)
    dw_dz = (w_pad[:, 2:, :, :] - w_pad[:, :-2, :, :]) / (2 * dz)
    
    divergence = du_dx + dv_dy + dw_dz
    
    # Penalize non-zero divergence everywhere (or optionally just on masked pixels)
    div_loss = torch.mean(divergence ** 2)
    
    total_loss = mse_loss + lambda_div * div_loss
    
    return total_loss, mse_loss, div_loss
