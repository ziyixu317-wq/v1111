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
    用于加载时间序列结构化网格流场数据的 PyTorch Dataset。

    从目录中读取多个时间步的 .vti 文件，按文件名排序后组装为
    滑动窗口序列，输出张量形状为 (T, C, D, H, W)。

    Parameters
    ----------
    data_dir : str
        包含 .vti 文件的目录路径。
    time_window : int
        时间窗口长度 T，连续采样 T 个时间步。
    velocity_names : tuple
        速度分量名称。
    vector_name : str or None
        如果速度场存储为单个向量数组，指定该名称；否则设为 None。
    stride : int
        滑动窗口步长。
    normalize : bool
        是否对速度场进行归一化。
    """

    def __init__(
        self,
        data_dir: str,
        time_window: int = 4,
        velocity_names: Tuple[str, str, str] = ("u", "v", "w"),
        vector_name: Optional[str] = None,
        stride: int = 1,
        normalize: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.time_window = time_window
        self.velocity_names = velocity_names
        self.vector_name = vector_name
        self.stride = stride
        self.normalize = normalize

        # 收集所有 .vti 文件并排序
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.vti")))
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"在 {data_dir} 中未找到 .vti 文件。")

        # 预加载所有数据到内存（适用于中小规模数据集）
        self._cache: List[np.ndarray] = []
        for f in self.file_list:
            if self.vector_name is not None:
                vel = read_vti_with_vector(f, self.vector_name)
            else:
                vel = read_single_vti(f, self.velocity_names)
            self._cache.append(vel)

        # 计算全局归一化统计量
        if self.normalize:
            all_data = np.stack(self._cache, axis=0)  # (N_files, 3, D, H, W)
            self._mean = all_data.mean(axis=(0, 2, 3, 4), keepdims=True)  # (1, 3, 1, 1, 1)
            self._std = all_data.std(axis=(0, 2, 3, 4), keepdims=True) + 1e-8
        else:
            self._mean = 0.0
            self._std = 1.0

        # 有效序列数量
        self._num_sequences = max(
            1, (len(self._cache) - self.time_window) // self.stride + 1
        )

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.stride
        end = start + self.time_window

        frames = []
        for i in range(start, min(end, len(self._cache))):
            vel = self._cache[i].copy()
            if self.normalize:
                vel = (vel - self._mean.squeeze(0)) / self._std.squeeze(0)
            frames.append(vel)

        # 如果帧数不足，用最后一帧填充
        while len(frames) < self.time_window:
            frames.append(frames[-1].copy())

        # 组装: (T, C, D, H, W)
        sequence = np.stack(frames, axis=0)
        return torch.from_numpy(sequence)

    @property
    def spatial_shape(self) -> Tuple[int, int, int]:
        """返回空间维度 (D, H, W)"""
        return self._cache[0].shape[1:]

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
