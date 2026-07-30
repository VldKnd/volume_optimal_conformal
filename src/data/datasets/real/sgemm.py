import numpy as np

from configs.datasets.real import SGEMMDatasetConfig
from data.datasets.real.base import BaseRealDataset


class SGEMMDataset(BaseRealDataset):
    """SGEMM GPU-kernel runtime dataset."""

    dataset_name = "SGEMM"
    target_names = (
        "Run1 (ms)",
        "Run2 (ms)",
        "Run3 (ms)",
        "Run4 (ms)",
    )

    def __init__(self, config: SGEMMDatasetConfig):
        super().__init__(config)

    def load_data(self):
        with self.file_path.open("r", encoding="utf-8") as file:
            column_names = tuple(
                name.strip() for name in file.readline().strip().split(",")
            )

        if column_names[-self.y_dim:] != self.target_names:
            raise ValueError(
                f"Target columns in '{self.file_path}' do not match the "
                "expected schema."
            )

        data = np.loadtxt(self.file_path, delimiter=",", skiprows=1)
        feature_names = column_names[:-self.y_dim]
        return data[:, :-self.y_dim], data[:, -self.y_dim:], feature_names
