from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class ATP7dDatasetConfig(BaseRealDatasetConfig):
    type: Literal["atp7d"] = "atp7d"

    file_path: Path = Path("data/atp7d/file203872c63fcd.arff")

    x_dim: Literal[411] = 411
    y_dim: Literal[6] = 6
