from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class BioDatasetConfig(BaseRealDatasetConfig):
    type: Literal["bio"] = "bio"

    file_path: Path = Path("data/bio/CASP.csv")

    x_dim: Literal[8] = 8
    y_dim: Literal[2] = 2
