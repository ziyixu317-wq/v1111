"""
helmholtz.py - Helmholtz Dynamics Decoupling Module
===================================================
Decouples flow field into Solenoidal (divergence-free) and Irrotational (curl-free) parts.
Using Fast Fourier Transform (FFT) in k-space.
"""
import torch

def helmholtz_decomposition(velocity: torch.Tensor, dx=1.0, dy=1.0, dz=1.0):
    """
    Performs Helmholtz decomposition on a 3D velocity field using FFT.
    velocity: (B, 3, D, H, W)
    Returns:
        v_solenoidal: (B, 3, D, H, W), divergence-free component
        v_irrotational: (B, 3, D, H, W), curl-free component
    """
    B, C, D, H, W = velocity.shape
    
    # FFT of velocity field
    v_hat = torch.fft.fftn(velocity, dim=(-3, -2, -1))
    
    # Wavenumber grids
    kx = torch.fft.fftfreq(W, d=dx, device=velocity.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=dy, device=velocity.device) * 2 * torch.pi
    kz = torch.fft.fftfreq(D, d=dz, device=velocity.device) * 2 * torch.pi
    
    Kz, Ky, Kx = torch.meshgrid(kz, ky, kx, indexing='ij')
    K = torch.stack([Kx, Ky, Kz], dim=0).unsqueeze(0)  # (1, 3, D, H, W)
    
    K_sq = torch.sum(K**2, dim=1, keepdim=True)
    K_sq[K_sq == 0] = 1.0  # Avoid division by zero at DC component
    
    k_dot_v = torch.sum(K * v_hat, dim=1, keepdim=True)
    v_irr_hat = (k_dot_v * K) / K_sq
    
    # Clean up DC component explicitly for irrotational part
    v_irr_hat[..., 0, 0, 0] = 0.0
    
    v_sol_hat = v_hat - v_irr_hat
    
    # Inverse FFT
    v_solenoidal = torch.fft.ifftn(v_sol_hat, dim=(-3, -2, -1)).real
    v_irrotational = torch.fft.ifftn(v_irr_hat, dim=(-3, -2, -1)).real
    
    return v_solenoidal, v_irrotational
