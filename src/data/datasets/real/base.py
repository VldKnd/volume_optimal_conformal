from pathlib import Path

import numpy as np
import torch
from scipy.io import arff
from sklearn.impute import KNNImputer

from configs.datasets.real.base import BaseRealDatasetConfig
from data.datasets.base import BaseDataset, DatasetSplits, XYData


def load_arff_data(
    path: Path,
    target_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    raw_data, _ = arff.loadarff(path)
    column_names = raw_data.dtype.names
    if column_names is None:
        raise ValueError(f"ARFF file at '{path}' has no named columns.")

    if not set(target_names).issubset(column_names):
        raise ValueError(
            f"Target columns in '{path}' do not match the expected schema."
        )

    feature_names = tuple(name for name in column_names if name not in target_names)
    x = np.column_stack([raw_data[name] for name in feature_names])
    y = np.column_stack([raw_data[name] for name in target_names])
    return x, y, feature_names


class BaseRealDataset(BaseDataset):
    """Shared loading, splitting, and preprocessing for file-backed datasets."""

    dataset_name: str
    target_names: tuple[str, ...]

    def __init__(self, config: BaseRealDatasetConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        file_path = config.file_path
        if not file_path.is_absolute():
            file_path = Path(__file__).resolve().parents[4] / file_path
        self.file_path = file_path

        self.feature_names: tuple[str, ...] = ()
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None
        self.imputer: KNNImputer | None = None
        self.n_imputed_values = 0
        self._splits: DatasetSplits | None = None

    @property
    def x_dim(self) -> int:
        return self.config.x_dim

    @property
    def y_dim(self) -> int:
        return self.config.y_dim

    def load_data(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        raise NotImplementedError

    def prepare(self) -> None:
        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"{self.dataset_name} data file was not found at "
                f"'{self.file_path}'. Download the file and place it in "
                f"'{self.file_path.parent}' with the filename "
                f"'{self.file_path.name}'."
            )

        x, y, self.feature_names = self.load_data()
        if x.shape[1] != self.x_dim or y.shape[1] != self.y_dim:
            raise ValueError(
                f"{self.dataset_name} data has dimensions "
                f"({x.shape[1]}, {y.shape[1]}), expected "
                f"({self.x_dim}, {self.y_dim})."
            )
        if not np.isfinite(y).all():
            raise ValueError(f"{self.dataset_name} contains missing target values.")

        x = np.asarray(x, dtype=np.float64)
        x[~np.isfinite(x)] = np.nan

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        permutation = torch.randperm(x.shape[0], generator=generator).numpy()
        x = x[permutation]
        y = y[permutation]

        n_train = int(self.config.train_fraction * x.shape[0])
        n_calibration = int(self.config.calibration_fraction * x.shape[0])
        calibration_end = n_train + n_calibration

        self.n_imputed_values = int(np.isnan(x).sum())
        if self.n_imputed_values:
            self.imputer = KNNImputer(
                n_neighbors=1,
                metric="nan_euclidean",
            )
            self.imputer.fit(x[:n_train])
            x = self.imputer.transform(x)

        x = torch.as_tensor(x, dtype=self.dtype)
        y = torch.as_tensor(y, dtype=self.dtype)

        train_x = x[:n_train]
        feature_mean = train_x.mean(dim=0)
        feature_std = train_x.std(dim=0, unbiased=False)
        feature_std = feature_std.clamp_min(torch.finfo(self.dtype).eps)

        x = ((x - feature_mean) / feature_std).to(self.device)
        y = y.to(self.device)

        self.feature_mean = feature_mean.to(self.device)
        self.feature_std = feature_std.to(self.device)
        self._splits = DatasetSplits(
            train=XYData(x=x[:n_train], y=y[:n_train]),
            calibration=XYData(
                x=x[n_train:calibration_end],
                y=y[n_train:calibration_end],
            ),
            test=XYData(
                x=x[calibration_end:],
                y=y[calibration_end:],
            ),
        )

    def get_splits(self) -> DatasetSplits:
        if self._splits is None:
            self.prepare()

        assert self._splits is not None
        return self._splits
