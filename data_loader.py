"""
data_loader.py — VTI Structured Grid Data Loader
=================================================
Reads .vti files (VTK ImageData) containing u, v, w velocity components
and converts them to PyTorch tensors with shape (C, D, H, W).

Supports:
  - Single-timestep VTI files
  - Multi-timestep sequences (directory of VTI files → T×C×D×H×W tensor)
"""

import os
import glob
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pyvista as pv
except ImportError:
    raise ImportError("请安装 pyvista: pip install pyvista vtk")


def read_single_vti(
    filepath: str,
    velocity_names: Tuple[str, str, str] = ("u", "v", "w"),
) -> np.ndarray:
    """
    读取单个 .vti 文件并返回速度场 numpy 数组。

    Parameters
    ----------
    filepath : str
        .vti 文件的路径。
    velocity_names : tuple of str
        三个速度分量在 VTI 文件中的数组名称，默认 ("u", "v", "w")。

    Returns
    -------
    velocity : np.ndarray, shape (3, D, H, W)
        三维速度场张量（通道顺序: u, v, w）。
    """
    mesh = pv.read(filepath)

    # 获取结构化网格的维度 (nx, ny, nz)
    dims = mesh.dimensions  # (nx, ny, nz)

    components = []
    for name in velocity_names:
        if name in mesh.point_data:
            arr = mesh.point_data[name]
        elif name in mesh.cell_data:
            arr = mesh.cell_data[name]
        else:
            raise KeyError(
                f"在文件 {filepath} 中未找到速度分量 '{name}'。"
                f"可用的数组: point_data={list(mesh.point_data.keys())}, "
                f"cell_data={list(mesh.cell_data.keys())}"
            )
        # VTK 使用 Fortran 顺序 (x-fastest)，需要 reshape 并转置
        arr_3d = arr.reshape(dims[2], dims[1], dims[0])  # (nz, ny, nx) → (D, H, W)
        components.append(arr_3d)

    velocity = np.stack(components, axis=0).astype(np.float32)  # (3, D, H, W)
    return velocity


def read_vti_with_vector(
    filepath: str,
    vector_name: str = "velocity",
) -> np.ndarray:
    """
    读取包含三维向量场的 .vti 文件（速度场存储为单个 3-分量向量数组）。

    Parameters
    ----------
    filepath : str
        .vti 文件路径。
    vector_name : str
        向量场数组名称。

    Returns
    -------
    velocity : np.ndarray, shape (3, D, H, W)
    """
    mesh = pv.read(filepath)
    dims = mesh.dimensions  # (nx, ny, nz)

    if vector_name in mesh.point_data:
        vec = mesh.point_data[vector_name]
    elif vector_name in mesh.cell_data:
        vec = mesh.cell_data[vector_name]
    else:
        raise KeyError(
            f"在文件 {filepath} 中未找到向量场 '{vector_name}'。"
            f"可用数组: point_data={list(mesh.point_data.keys())}, "
            f"cell_data={list(mesh.cell_data.keys())}"
        )

    # vec shape: (N, 3) → 拆分为 3 个 (D, H, W)
    u = vec[:, 0].reshape(dims[2], dims[1], dims[0])
    v = vec[:, 1].reshape(dims[2], dims[1], dims[0])
    w = vec[:, 2].reshape(dims[2], dims[1], dims[0])

    velocity = np.stack([u, v, w], axis=0).astype(np.float32)  # (3, D, H, W)
    return velocity


class VTIFlowDataset(Dataset):
    """
    Advanced VTI Data Loader: Min-Max Normalization + Random Cropping.
    Consistent with the paper's experimental setup.
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        time_window: int = 1, # Default to 1 to match VortexMAE single-step
        crop_size: int = 128,
        velocity_names: Tuple[str, str, str] = ("u", "v", "w"),
        vector_name: Optional[str] = None,
        stride: int = 1,
        normalize: bool = True,
        max_files: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.time_window = time_window
        self.crop_size = crop_size
        self.velocity_names = velocity_names
        self.vector_name = vector_name
        self.stride = stride
        self.normalize = normalize
        
        # 1. Collect and Split Files
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "*.vti")))
        if len(self.all_files) == 0:
            raise FileNotFoundError(f"No .vti files in {data_dir}")
            
        if max_files is not None:
            self.all_files = self.all_files[:max_files]
            
        num_total = len(self.all_files)
        # Consistent 80/20 split for all dataset sizes
        num_train = int(num_total * 0.8)
        if split in ("train", "pretrain_train", "finetune_train"):
            self.files = self.all_files[:num_train]
        else:
            self.files = self.all_files[num_train:]
        
        self.do_crop = split not in ("inference", "test")

        # 2. Sequential Scan for Normalization (Min-Max) - Memory Efficient
        print(f"[{split}] Scanning {len(self.files)} files for Norm Stats...")
        self._min = np.array([float('inf')] * 3, dtype=np.float32).reshape(1, 3, 1, 1, 1)
        self._max = np.array([float('-inf')] * 3, dtype=np.float32).reshape(1, 3, 1, 1, 1)
        
        # Calculate stats without holding all data in RAM
        for f in self.files:
            vel = read_vti_with_vector(f, self.vector_name) if self.vector_name else read_single_vti(f, self.velocity_names)
            # vel: (3, D, H, W)
            f_min = vel.min(axis=(1, 2, 3)).reshape(1, 3, 1, 1, 1)
            f_max = vel.max(axis=(1, 2, 3)).reshape(1, 3, 1, 1, 1)
            self._min = np.minimum(self._min, f_min)
            self._max = np.maximum(self._max, f_max)
            
        self._num_sequences = max(1, (len(self.files) - self.time_window) // self.stride + 1)

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        end = start + self.time_window
        frames = []
        
        for i in range(start, min(end, len(self.files))):
            f = self.files[i]
            vel = read_vti_with_vector(f, self.vector_name) if self.vector_name else read_single_vti(f, self.velocity_names)
            # Per-channel Min-Max
            if self.normalize:
                vel = (vel - self._min[0]) / (self._max[0] - self._min[0] + 1e-8)
            frames.append(vel)
        
        while len(frames) < self.time_window:
            frames.append(frames[-1].copy())
            
        sequence = np.stack(frames, axis=0) # (T, C, D, H, W)
        # Squash T if T=1
        if self.time_window == 1:
            data = sequence[0]
        else:
            data = sequence # Note: MAE Fusion expects (C, D, H, W) mostly
            
        if self.do_crop:
            # 3D Random Crop/Pad to 128^3
            C, D, H, W = data.shape if data.ndim == 4 else data.shape[1:]
            target = self.crop_size
            
            d_s = np.random.randint(0, max(1, D - target))
            h_s = np.random.randint(0, max(1, H - target))
            w_s = np.random.randint(0, max(1, W - target))
            
            if data.ndim == 4:
                data = data[:, d_s:d_s+target, h_s:h_s+target, w_s:w_s+target]
            else:
                data = data[:, :, d_s:d_s+target, h_s:h_s+target, w_s:w_s+target]
            
            # Padding
            pad_d = target - data.shape[-3]
            pad_h = target - data.shape[-2]
            pad_w = target - data.shape[-1]
            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                data = np.pad(data, ((0,0), (0,pad_d), (0,pad_h), (0,pad_w)), mode='constant')
        
        return torch.from_numpy(data).float()

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        # Peek at first file for metadata
        vel = read_vti_with_vector(self.files[0], self.vector_name) if self.vector_name else read_single_vti(self.files[0], self.velocity_names)
        return vel.shape[1:]

    @property
    def data_mean(self) -> np.ndarray:
        return self._mean

    @property
    def data_std(self) -> np.ndarray:
        return self._std


def load_single_vti_as_tensor(
    filepath: str,
    velocity_names: Tuple[str, str, str] = ("u", "v", "w"),
    vector_name: Optional[str] = None,
) -> torch.Tensor:
    """
    快捷函数：读取单个 VTI 文件并返回 PyTorch 张量。

    Returns
    -------
    tensor : torch.Tensor, shape (1, 3, D, H, W)
        带 batch 维度的速度场张量。
    """
    if vector_name is not None:
        vel = read_vti_with_vector(filepath, vector_name)
    else:
        vel = read_single_vti(filepath, velocity_names)

    tensor = torch.from_numpy(vel).unsqueeze(0)  # (1, 3, D, H, W)
    return tensor
