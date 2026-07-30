from configs.datasets.real import BioDatasetConfig
from data.datasets.real.base import BaseRealDataset, load_arff_data


class BioDataset(BaseRealDataset):
    """Bio regression dataset with F7 and F9 used as a 2D target."""

    dataset_name = "Bio"
    target_names = ("F7", "F9")

    def __init__(self, config: BioDatasetConfig):
        super().__init__(config)

    def load_data(self):
        return load_arff_data(self.file_path, self.target_names)
