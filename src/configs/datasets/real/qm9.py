from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from configs.datasets.real.base import BaseRealDatasetConfig


class QM9DatasetConfig(BaseRealDatasetConfig):
    type: Literal["qm9"] = "qm9"

    file_path: Path = Path("data/qm9/qm9_mbtr_100.npz")
    raw_root: Path | None = None

    x_dim: Literal[100] = 100
    y_dim: Literal[4] = 4

    extraction_batch_size: int = Field(default=1024, gt=0)
    extraction_n_jobs: int = -1
    force_rebuild_features: bool = False

    @model_validator(mode="after")
    def validate_extraction_settings(self) -> Self:
        if self.raw_root is None:
            self.raw_root = self.file_path.parent / "pyg"
        if self.extraction_n_jobs == 0 or self.extraction_n_jobs < -1:
            raise ValueError("extraction_n_jobs must be -1 or a positive integer.")
        return self
