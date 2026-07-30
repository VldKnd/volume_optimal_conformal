from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class RF2DatasetConfig(BaseRealDatasetConfig):
    type: Literal["rf2"] = "rf2"

    file_path: Path = Path("data/rf2/file17307ff5552.arff")

    x_dim: Literal[576] = 576
    y_dim: Literal[8] = 8
