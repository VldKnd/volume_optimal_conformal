from pathlib import Path
from typing import Literal

from configs.datasets.real.base import BaseRealDatasetConfig


class SCM1dDatasetConfig(BaseRealDatasetConfig):
    type: Literal["scm1d"] = "scm1d"

    file_path: Path = Path("data/scm1d/file1730122322aa.arff")

    x_dim: Literal[280] = 280
    y_dim: Literal[16] = 16
