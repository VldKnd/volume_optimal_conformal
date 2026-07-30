from configs.datasets.real import RF1DatasetConfig
from data.datasets.real.base import BaseRealDataset, load_arff_data


class RF1Dataset(BaseRealDataset):
    """RF1 river-flow multi-target regression dataset."""

    dataset_name = "RF1"
    target_names = (
        "CHSI2_48H__0",
        "NASI2_48H__0",
        "EADM7_48H__0",
        "SCLM7_48H__0",
        "CLKM7_48H__0",
        "VALI2_48H__0",
        "NAPM7_48H__0",
        "DLDI4_48H__0",
    )

    def __init__(self, config: RF1DatasetConfig):
        super().__init__(config)

    def load_data(self):
        return load_arff_data(self.file_path, self.target_names)
