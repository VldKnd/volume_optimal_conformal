from configs.datasets.real import SCM1dDatasetConfig
from data.datasets.real.base import BaseRealDataset, load_arff_data


class SCM1dDataset(BaseRealDataset):
    """SCM1d multi-target regression dataset."""

    dataset_name = "SCM1d"
    target_names = (
        "LBL",
        "MTLp2",
        "MTLp3",
        "MTLp4",
        "MTLp5",
        "MTLp6",
        "MTLp7",
        "MTLp8",
        "MTLp9",
        "MTLp10",
        "MTLp11",
        "MTLp12",
        "MTLp13",
        "MTLp14",
        "MTLp15",
        "MTLp16",
    )

    def __init__(self, config: SCM1dDatasetConfig):
        super().__init__(config)

    def load_data(self):
        return load_arff_data(self.file_path, self.target_names)
