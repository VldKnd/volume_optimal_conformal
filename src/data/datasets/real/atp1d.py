from configs.datasets.real import ATP1dDatasetConfig
from data.datasets.real.base import BaseRealDataset, load_arff_data


class ATP1dDataset(BaseRealDataset):
    """Airline ticket-price data with six one-day-ahead targets."""

    dataset_name = "ATP1D"
    target_names = (
        "LBL_ALLminpA_fut_001",
        "LBL_ALLminp0_fut_001",
        "LBL_aDLminpA_fut_001",
        "LBL_aCOminpA_fut_001",
        "LBL_aFLminpA_fut_001",
        "LBL_aUAminpA_fut_001",
    )

    def __init__(self, config: ATP1dDatasetConfig):
        super().__init__(config)

    def load_data(self):
        return load_arff_data(self.file_path, self.target_names)
