from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class SCM20dDatasetConfig(BaseRealDatasetConfig):
    type: Literal["scm20d"] = "scm20d"

    file_path: Path = Path("data/scm20d/file1730492b4408.arff")

    x_dim: Literal[61] = 61
    y_dim: Literal[16] = 16
