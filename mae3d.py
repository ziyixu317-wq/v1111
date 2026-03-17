
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from swin3d import SwinTransformer3D
from helmholtz import helmholtz_decomposition

class MAE3D_Fusion(nn.Module):
    """
    Fusion Model: Swin-Encoder + U-Net Decoder + Helmholtz Preprocessor.
    Supports Pre-training (Flow Reconstruct) and Segmentation (IVD Mask).
    """
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, out_chans=1,
                 embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                 window_size=(4, 4, 4), mask_ratio=0.75, mode='pretrain',
                 use_helmholtz=True):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.use_helmholtz = use_helmholtz
        
        # 1. Swin-ViT Encoder
        self.encoder = SwinTransformer3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            depths=depths, num_heads=num_heads, window_size=window_size
        )
        
        # 2. Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 3. U-Net Decoder Blocks
        # Stages are: 0 (dim), 1 (dim*2), 2 (dim*4), 3 (dim*8)
        d1 = embed_dim * 2
        d2 = embed_dim * 4
        d3 = embed_dim * 8
        
        self.up_stage3 = nn.ConvTranspose3d(d3, d2, kernel_size=2, stride=2)
        self.up_stage2 = nn.ConvTranspose3d(d2, d1, kernel_size=2, stride=2)
        self.up_stage1 = nn.ConvTranspose3d(d1, embed_dim, kernel_size=2, stride=2)
        
        # Final Task Heads
        self.head_rec = nn.ConvTranspose3d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)
        self.head_seg = nn.ConvTranspose3d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, C, D, H, W) 3D velocity field
        """
        B, C, D, H, W = x.shape
        
        # --- 0. Helmholtz Preprocessing (Optional) ---
        if self.use_helmholtz:
            # Focus on solenoidal (divergence-free) part which contains vortex info
            x_sol, x_irr = helmholtz_decomposition(x)
            x_input_field = x_sol
        else:
            x_input_field = x
            
        # --- 1. Patch Embedding & Masking ---
        x_embed = self.encoder.patch_embed(x_input_field)
        Bp, Dp, Hp, Wp, Cp = x_embed.shape
        N = Dp * Hp * Wp
        
        if self.mode == 'pretrain':
            noise = torch.rand(B, N, device=x.device)
            mask = (noise < self.mask_ratio).float().view(B, Dp, Hp, Wp, 1)
            x_masked = x_embed * (1 - mask) + self.mask_token * mask
            feat = self.encoder.pos_drop(x_masked)
        else:
            feat = self.encoder.pos_drop(x_embed)
            mask = None
            
        # --- 2. Encoder Path (Manual Loop for Skips) ---
        outs = []
        curr_feat = feat
        for layer in self.encoder.layers:
            # Layer returns (stage_output, downsampled_output)
            skip, curr_feat = layer(curr_feat)
            outs.append(skip)
        
        # outs[0]: Stage 1, outs[1]: Stage 2, outs[2]: Stage 3, outs[3]: Stage 4
        
        # --- 3. U-Net Decoder Path ---
        # Stage 3 -> 2
        z = outs[3].permute(0, 4, 1, 2, 3) 
        z = self.up_stage3(z)
        sh2 = outs[2].shape
        z = z[:, :, :sh2[1], :sh2[2], :sh2[3]] 
        z = z + outs[2].permute(0, 4, 1, 2, 3)
        
        # Stage 2 -> 1
        z = self.up_stage2(z)
        sh1 = outs[1].shape
        z = z[:, :, :sh1[1], :sh1[2], :sh1[3]]
        z = z + outs[1].permute(0, 4, 1, 2, 3)
        
        # Stage 1 -> 0
        z = self.up_stage1(z)
        sh0 = outs[0].shape
        z = z[:, :, :sh0[1], :sh0[2], :sh0[3]]
        z = z + outs[0].permute(0, 4, 1, 2, 3)
        
        # --- 4. Final Output ---
        if self.mode == 'pretrain':
            rec = self.head_rec(z)
            mask_pixel = F.interpolate(mask.permute(0, 4, 1, 2, 3), 
                                       size=(D, H, W), mode='nearest')
            return rec, mask_pixel
        else:
            seg = self.head_seg(z)
            return torch.sigmoid(seg)

    def _encoder_forward_masked(self, x):
        outs = []
        for layer in self.encoder.layers:
            x_out, x = layer(x)
            outs.append(x_out)
        return x, outs

def pi_mae_loss(pred, target, mask, dx=1.0, dy=1.0, dz=1.0, lambda_div=0.1):
    """
    Physical-Informed Loss: MSE Reconstruction + Divergence Penalty
    pred, target: (B, 3, D, H, W)
    mask: (B, 1, D, H, W) where 1 indicates masked
    """
    # MSE Loss on masked pixels
    mse_loss = F.mse_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() * 3 + 1e-6)
    
    # Calculate Divergence: du/dx + dv/dy + dw/dz
    u = pred[:, 0]
    v = pred[:, 1]
    w = pred[:, 2]
    
    u_pad = F.pad(u, (1, 1, 0, 0, 0, 0), mode='replicate')
    v_pad = F.pad(v, (0, 0, 1, 1, 0, 0), mode='replicate')
    w_pad = F.pad(w, (0, 0, 0, 0, 1, 1), mode='replicate')
    
    du_dx = (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2]) / (2 * dx)
    dv_dy = (v_pad[:, :, 2:, :] - v_pad[:, :, :-2, :]) / (2 * dy)
    dw_dz = (w_pad[:, 2:, :, :] - w_pad[:, :-2, :, :]) / (2 * dz)
    
    divergence = du_dx + dv_dy + dw_dz
    div_loss = torch.mean(divergence ** 2)
    
    return mse_loss + lambda_div * div_loss, mse_loss, div_loss
