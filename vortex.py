"""
vortex.py - Vortex Feature Extraction
=====================================
Calculates Q-Criterion from divergence-free velocity.
"""
import torch
import torch.nn.functional as F

def velocity_gradient_tensor(u):
    r"""
    u: (B, 3, D, H, W)
    Returns:
        grad_u: (B, 3, 3, D, H, W)
        grad_u[:, i, j, ...] corresponds to \partial u_i / \partial x_j
        where i in [0=x, 1=y, 2=z], j in [0=x, 1=y, 2=z]
    """
    B, _, D, H, W = u.shape
    device = u.device
    dx, dy, dz = 1.0, 1.0, 1.0
    
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2] 
    
    # Central diff padding
    def diff_x(f):
        pad = F.pad(f.unsqueeze(1), (1, 1, 0, 0, 0, 0), mode='replicate').squeeze(1)
        return (pad[..., 2:] - pad[..., :-2]) / (2 * dx)
        
    def diff_y(f):
        pad = F.pad(f.unsqueeze(1), (0, 0, 1, 1, 0, 0), mode='replicate').squeeze(1)
        return (pad[..., 2:, :] - pad[..., :-2, :]) / (2 * dy)
        
    def diff_z(f):
        pad = F.pad(f.unsqueeze(1), (0, 0, 0, 0, 1, 1), mode='replicate').squeeze(1)
        return (pad[:, 2:, :, :] - pad[:, :-2, :, :]) / (2 * dz)
        
    grad = torch.zeros((B, 3, 3, D, H, W), device=device)
    
    grad[:, 0, 0] = diff_x(ux)
    grad[:, 0, 1] = diff_y(ux)
    grad[:, 0, 2] = diff_z(ux)
    
    grad[:, 1, 0] = diff_x(uy)
    grad[:, 1, 1] = diff_y(uy)
    grad[:, 1, 2] = diff_z(uy)
    
    grad[:, 2, 0] = diff_x(uz)
    grad[:, 2, 1] = diff_y(uz)
    grad[:, 2, 2] = diff_z(uz)
    
    return grad

def q_criterion(grad_u):
    """
    Calculates the Q-criterion field.
    grad_u: (B, 3, 3, D, H, W)
    Q = 0.5 * (||Omega||^2 - ||S||^2)
    where Omega = 0.5 * (grad_u - grad_u.T)
    S = 0.5 * (grad_u + grad_u.T)
    """
    grad_u_t = grad_u.transpose(1, 2)
    
    S = 0.5 * (grad_u + grad_u_t)
    Omega = 0.5 * (grad_u - grad_u_t)
    
    norm_S_sq = torch.sum(S ** 2, dim=(1, 2))
    norm_Omega_sq = torch.sum(Omega ** 2, dim=(1, 2))
    
    Q = 0.5 * (norm_Omega_sq - norm_S_sq)
    return Q

def calculate_ivd(u):
    """
    Isolation by Vorticity Deviation (IVD).
    u: (B, 3, D, H, W)
    """
    grad_u = velocity_gradient_tensor(u)
    
    # omega = curl u
    omg_x = grad_u[:, 2, 1] - grad_u[:, 1, 2]
    omg_y = grad_u[:, 0, 2] - grad_u[:, 2, 0]
    omg_z = grad_u[:, 1, 0] - grad_u[:, 0, 1]
    vorticity_mag = torch.sqrt(omg_x**2 + omg_y**2 + omg_z**2 + 1e-8)
    
    # IVD: deviation from spatial mean
    mean_vort = torch.mean(vorticity_mag, dim=(1, 2, 3), keepdim=True)
    ivd_field = vorticity_mag - mean_vort
    
    return ivd_field
