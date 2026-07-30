from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class ATP1dDatasetConfig(BaseRealDatasetConfig):
    type: Literal["atp1d"] = "atp1d"

    file_path: Path = Path("data/atp1d/file173029b97a05.arff")

    x_dim: Literal[411] = 411
    y_dim: Literal[6] = 6
