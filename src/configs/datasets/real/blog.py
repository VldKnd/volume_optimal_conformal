from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class BlogDatasetConfig(BaseRealDatasetConfig):
    type: Literal["blog"] = "blog"

    file_path: Path = Path("data/blog/blogData_train.csv")

    x_dim: Literal[279] = 279
    y_dim: Literal[2] = 2
