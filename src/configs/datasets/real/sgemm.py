from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class SGEMMDatasetConfig(BaseRealDatasetConfig):
    type: Literal["sgemm"] = "sgemm"

    file_path: Path = Path("data/sgemm/sgemm_product.csv")

    x_dim: Literal[14] = 14
    y_dim: Literal[4] = 4
