import numpy as np

from configs.datasets.real import BlogDatasetConfig
from data.datasets.real.base import BaseRealDataset


class BlogDataset(BaseRealDataset):
    """BlogFeedback data with columns 60 and 280 used as a 2D target."""

    dataset_name = "Blog"
    target_indices = (60, 280)
    target_names = ("column_60", "column_280")

    def __init__(self, config: BlogDatasetConfig):
        super().__init__(config)

    def load_data(self):
        data = np.loadtxt(self.file_path, delimiter=",")
        feature_indices = tuple(
            index for index in range(data.shape[1]) if index not in self.target_indices
        )
        feature_names = tuple(f"column_{index}" for index in feature_indices)
        return (
            data[:, feature_indices],
            data[:, self.target_indices],
            feature_names,
        )
