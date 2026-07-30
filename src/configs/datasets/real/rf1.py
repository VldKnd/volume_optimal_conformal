from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class RF1DatasetConfig(BaseRealDatasetConfig):
    type: Literal["rf1"] = "rf1"

    file_path: Path = Path("data/rf1/file173039e7713b.arff")

    x_dim: Literal[64] = 64
    y_dim: Literal[8] = 8
