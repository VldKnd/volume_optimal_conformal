# src/data/loaders.py

import torch
from torch.utils.data import DataLoader, TensorDataset

from data.datasets.base import XYData


def xy_to_tensor_dataset(data: XYData) -> TensorDataset:
    return TensorDataset(data.x, data.y)


def make_xy_dataloader(
    data: XYData,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        xy_to_tensor_dataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )