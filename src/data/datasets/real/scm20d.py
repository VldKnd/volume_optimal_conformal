from configs.datasets.real import SCM20dDatasetConfig
from data.datasets.real.base import BaseRealDataset, load_arff_data


class SCM20dDataset(BaseRealDataset):
    """SCM20d multi-target regression dataset."""

    dataset_name = "SCM20d"
    target_names = (
        "LBL",
        "MTLp2A",
        "MTLp3A",
        "MTLp4A",
        "MTLp5A",
        "MTLp6A",
        "MTLp7A",
        "MTLp8A",
        "MTLp9A",
        "MTLp10A",
        "MTLp11A",
        "MTLp12A",
        "MTLp13A",
        "MTLp14A",
        "MTLp15A",
        "MTLp16A",
    )

    def __init__(self, config: SCM20dDatasetConfig):
        super().__init__(config)

    def load_data(self):
        return load_arff_data(self.file_path, self.target_names)
