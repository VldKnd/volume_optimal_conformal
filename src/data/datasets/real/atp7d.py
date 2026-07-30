from configs.datasets.real import ATP7dDatasetConfig
from data.datasets.real.base import BaseRealDataset, load_arff_data


class ATP7dDataset(BaseRealDataset):
    """Airline ticket-price data with six seven-day targets."""

    dataset_name = "ATP7D"
    target_names = (
        "LBL_ALLminpA_bt7d_000",
        "LBL_ALLminp0_bt7d_000",
        "LBL_aDLminpA_bt7d_000",
        "LBL_aCOminpA_bt7d_000",
        "LBL_aFLminpA_bt7d_000",
        "LBL_aUAminpA_bt7d_000",
    )

    def __init__(self, config: ATP7dDatasetConfig):
        super().__init__(config)

    def load_data(self):
        return load_arff_data(self.file_path, self.target_names)
