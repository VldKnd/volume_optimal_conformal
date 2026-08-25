from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class OnlineNewsPopularityDatasetConfig(BaseRealDatasetConfig):
    """Configuration for the leakage-controlled ONP regression benchmark."""

    type: Literal["onp"] = "onp"

    file_path: Path = Path("data/onp/OnlineNewsPopularity.csv")

    x_dim: Literal[29] = 29
    y_dim: Literal[8] = 8
