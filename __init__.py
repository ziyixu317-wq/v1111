# flow_vortex_net: Physics-Guided 3D MAE + Helmholtz Decomposition + Vortex Extraction Pipeline
"""
Flow Vortex Net
===============
A multi-task adaptive flow field vortex feature extraction and velocity field
perception framework based on:
  1. 3D Masked Autoencoder with Swin-ViT backbone (3D MAE)
  2. Helmholtz Decomposition Module
  3. Joint Vortex Feature Extraction (Q / λ₂ / Δ criteria)

References:
  - Video Swin Transformer (Ze Liu et al., 2022)
  - HelmFluid (Xiao et al., 2023, https://github.com/thuml/HelmFluid)
  - NN-HHD (Skyler et al., https://github.com/skywolf829/NN-HHD)
  - Raissi et al. Physics-Informed Neural Networks (PINN), JCP 2019
"""

__version__ = "0.1.0"
