import csv

import numpy as np

from configs.datasets.real import BioDatasetConfig
from data.datasets.real.base import BaseRealDataset


class BioDataset(BaseRealDataset):
    """Bio regression dataset with F7 and F9 used as a 2D target."""

    dataset_name = "Bio"
    target_names = ("F7", "F9")

    def __init__(self, config: BioDatasetConfig):
        super().__init__(config)

    def load_data(self):
        with self.file_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            try:
                column_names = tuple(next(reader))
            except StopIteration as error:
                raise ValueError(f"CSV file at '{self.file_path}' is empty.") from error

        if len(set(column_names)) != len(column_names):
            raise ValueError(
                f"CSV file at '{self.file_path}' contains duplicate columns."
            )
        if not set(self.target_names).issubset(column_names):
            raise ValueError(
                f"Target columns in '{self.file_path}' do not match the "
                "expected schema."
            )

        data = np.loadtxt(
            self.file_path,
            delimiter=",",
            skiprows=1,
        )
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] != len(column_names):
            raise ValueError(
                f"CSV file at '{self.file_path}' contains {data.shape[1]} data "
                f"columns but its header contains {len(column_names)}."
            )

        target_indexes = tuple(column_names.index(name) for name in self.target_names)
        feature_indexes = tuple(
            index
            for index in range(len(column_names))
            if index not in target_indexes
        )
        feature_names = tuple(column_names[index] for index in feature_indexes)
        return data[:, feature_indexes], data[:, target_indexes], feature_names
