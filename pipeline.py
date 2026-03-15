"""
pipeline.py - Integrated Model Pipeline
=======================================
Combines all modules and performs a forward pass.
"""
import torch
import torch.nn as nn

from swin3d import SwinTransformer3D
from mae3d import MAE3D, pi_mae_loss
from helmholtz import helmholtz_decomposition
from vortex import velocity_gradient_tensor, q_criterion
from data_loader import VTIFlowDataset, load_single_vti_as_tensor

class FlowVortexPipeline(nn.Module):
    """
    Main pipeline integrating MAE3D -> Helmholtz Decomposition -> Vortex Extraction.
    """
    def __init__(self, use_mae=True, **mae_kwargs):
        super().__init__()
        self.use_mae = use_mae
        if self.use_mae:
            if not mae_kwargs:
                mae_kwargs = dict(
                    patch_size=(2, 4, 4), in_chans=3, 
                    embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                    window_size=(4, 4, 4), mask_ratio=0.75
                )
            self.mae = MAE3D(**mae_kwargs)
            
    def forward(self, x, dx=1.0, dy=1.0, dz=1.0):
        """
        x: (B, C, D, H, W)
        """
        if self.use_mae:
            x_rec, mask = self.mae(x)
            velocity_field = x_rec
        else:
            velocity_field = x
            mask = None
            
        v_sol, v_irr = helmholtz_decomposition(velocity_field, dx=dx, dy=dy, dz=dz)
        
        grad_u = velocity_gradient_tensor(v_sol, dx=dx, dy=dy, dz=dz)
        q_field = q_criterion(grad_u)
        
        return velocity_field, mask, v_sol, v_irr, q_field


if __name__ == "__main__":
    print("=======================================")
    print("Initializing Flow Vortex Net Pipeline...")
    pipeline = FlowVortexPipeline()
    
    print("\nTesting with dummy input tensor...")
    dummy_input = torch.randn(2, 3, 16, 64, 64)
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        x_rec, mask, v_sol, v_irr, q_field = pipeline(dummy_input)
    
    print("\n[Outputs]")
    print(f"Reconstructed flow shape: {x_rec.shape}")
    print(f"Mask shape              : {mask.shape}")
    print(f"Solenoidal flow shape   : {v_sol.shape}")
    print(f"Irrotational flow shape : {v_irr.shape}")
    print(f"Q-criterion field shape : {q_field.shape}")
    
    try:
        from mae3d import pi_mae_loss
        loss, mse, div = pi_mae_loss(x_rec, dummy_input, mask)
        print(f"\n[Losses]")
        print(f"Dummy PA-MAE Loss: {loss.item():.4f} (MSE: {mse.item():.4f}, Div: {div.item():.4f})")
    except Exception as e:
        print(f"Failed to calculate PI-MAE loss: {e}")
        
    print("\n✅ Pipeline completed successfully.")
    print("=======================================")
