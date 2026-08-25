from __future__ import annotations

import csv

import numpy as np
import torch

from configs.datasets.real import OnlineNewsPopularityDatasetConfig
from data.datasets.base import DatasetSplits, XYData
from data.datasets.real.base import BaseRealDataset


class OnlineNewsPopularityDataset(BaseRealDataset):
    """Leakage-controlled UCI Online News Popularity benchmark.

    The eight targets are standardized independently using training statistics.
    Rows with non-finite values or malformed bounded ratio features are removed
    before splitting. Continuous predictors are then standardized from the
    training split, while data-channel and weekday indicator columns remain in
    their original binary representation.
    """

    dataset_name = "Online News Popularity"
    target_source_names = (
        "shares",
        "global_subjectivity",
        "global_sentiment_polarity",
        "avg_positive_polarity",
        "avg_negative_polarity",
        "title_subjectivity",
        "title_sentiment_polarity",
        "average_token_length",
    )
    target_names = (
        "log1p(shares)",
        *target_source_names[1:],
    )
    binary_feature_names = (
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
        "is_weekend",
    )
    bounded_feature_names = (
        "n_unique_tokens",
        "n_non_stop_words",
        "n_non_stop_unique_tokens",
    )
    non_target_exclusions = (
        "url",
        "timedelta",
        "abs_title_subjectivity",
        "abs_title_sentiment_polarity",
        "global_rate_positive_words",
        "global_rate_negative_words",
        "rate_positive_words",
        "rate_negative_words",
        "min_positive_polarity",
        "max_positive_polarity",
        "min_negative_polarity",
        "max_negative_polarity",
        "kw_min_min",
        "kw_max_min",
        "kw_avg_min",
        "kw_min_max",
        "kw_max_max",
        "kw_avg_max",
        "kw_min_avg",
        "kw_max_avg",
        "kw_avg_avg",
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_sharess",
    )
    excluded_feature_names = frozenset((*target_source_names, *non_target_exclusions))

    def __init__(self, config: OnlineNewsPopularityDatasetConfig):
        super().__init__(config)
        self.target_mean: torch.Tensor | None = None
        self.target_std: torch.Tensor | None = None
        self.continuous_feature_mask: torch.Tensor | None = None
        self.binary_feature_mask: torch.Tensor | None = None
        self.continuous_feature_names: tuple[str, ...] = ()
        self.n_removed_non_finite_rows = 0
        self.n_removed_malformed_rows = 0

    def load_data(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        with self.file_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            try:
                column_names = tuple(name.strip() for name in next(reader))
            except StopIteration as error:
                raise ValueError(f"CSV file at '{self.file_path}' is empty.") from error

        if len(set(column_names)) != len(column_names):
            raise ValueError(
                f"CSV file at '{self.file_path}' contains duplicate columns."
            )
        required_columns = (
            set(self.excluded_feature_names)
            .union(self.binary_feature_names)
            .union(self.bounded_feature_names)
        )
        missing_columns = required_columns.difference(column_names)
        if missing_columns:
            raise ValueError(
                "Online News Popularity CSV is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        feature_names = tuple(
            name for name in column_names if name not in self.excluded_feature_names
        )
        self._assert_leakage_free_schema(feature_names)
        if len(feature_names) != self.x_dim:
            raise ValueError(
                "Online News Popularity preprocessing selected "
                f"{len(feature_names)} predictors, expected {self.x_dim}."
            )

        selected_names = feature_names + self.target_source_names
        selected_indexes = tuple(column_names.index(name) for name in selected_names)
        data = np.genfromtxt(
            self.file_path,
            delimiter=",",
            skip_header=1,
            usecols=selected_indexes,
            dtype=np.float64,
            ndmin=2,
        )
        if data.shape[1] != len(selected_names):
            raise ValueError(
                f"CSV file at '{self.file_path}' produced {data.shape[1]} "
                f"selected columns, expected {len(selected_names)}."
            )

        x = data[:, :len(feature_names)]
        raw_targets = data[:, len(feature_names):]
        y = raw_targets.copy()
        shares = raw_targets[:, 0]
        y[:, 0] = np.nan
        valid_shares = np.isfinite(shares) & (shares > -1.0)
        y[valid_shares, 0] = np.log1p(shares[valid_shares])
        return x, y, feature_names

    def prepare(self) -> None:
        if not self.file_path.is_file():
            raise FileNotFoundError(
                f"{self.dataset_name} data file was not found at "
                f"'{self.file_path}'."
            )

        x, y, feature_names = self.load_data()
        self._assert_leakage_free_schema(feature_names)
        if x.shape[1] != self.x_dim or y.shape[1] != self.y_dim:
            raise ValueError(
                f"{self.dataset_name} data has dimensions "
                f"({x.shape[1]}, {y.shape[1]}), expected "
                f"({self.x_dim}, {self.y_dim})."
            )

        finite_rows = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        self.n_removed_non_finite_rows = int((~finite_rows).sum())
        x = np.asarray(x[finite_rows], dtype=np.float64)
        y = np.asarray(y[finite_rows], dtype=np.float64)

        bounded_feature_indexes = tuple(
            feature_names.index(name) for name in self.bounded_feature_names
        )
        bounded_x = x[:, bounded_feature_indexes]
        well_formed_rows = ((bounded_x >= 0.0) & (bounded_x <= 1.0)).all(axis=1)
        self.n_removed_malformed_rows = int((~well_formed_rows).sum())
        x = x[well_formed_rows]
        y = y[well_formed_rows]
        if x.shape[0] < 3:
            raise ValueError(
                "Online News Popularity needs at least three finite, "
                "well-formed rows."
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        permutation = torch.randperm(x.shape[0], generator=generator).numpy()
        x = x[permutation]
        y = y[permutation]

        n_train = int(self.config.train_fraction * x.shape[0])
        n_calibration = int(self.config.calibration_fraction * x.shape[0])
        calibration_end = n_train + n_calibration
        if n_train < 1 or n_calibration < 1 or calibration_end >= x.shape[0]:
            raise ValueError(
                "Configured fractions must produce non-empty train, "
                "calibration, and test splits."
            )

        self.feature_names = feature_names
        binary_name_set = set(self.binary_feature_names)
        binary_feature_mask = torch.tensor(
            [name in binary_name_set for name in feature_names],
            dtype=torch.bool,
        )
        continuous_feature_mask = ~binary_feature_mask
        self.continuous_feature_names = tuple(
            name for name, is_continuous in zip(
                feature_names,
                continuous_feature_mask.tolist(),
            ) if is_continuous
        )

        x = torch.as_tensor(x, dtype=self.dtype)
        y = torch.as_tensor(y, dtype=self.dtype)
        binary_x = x[:, binary_feature_mask]
        if not bool(((binary_x == 0.0) | (binary_x == 1.0)).all()):
            raise ValueError(
                "Online News Popularity indicator features must contain only "
                "zero and one."
            )

        train_x = x[:n_train]
        train_y = y[:n_train]
        feature_mean = torch.zeros(self.x_dim, dtype=self.dtype)
        feature_std = torch.ones(self.x_dim, dtype=self.dtype)
        continuous_train_x = train_x[:, continuous_feature_mask]
        continuous_std = continuous_train_x.std(dim=0, unbiased=False)
        self._reject_constant_columns(
            continuous_std,
            self.continuous_feature_names,
            kind="continuous feature",
        )
        feature_mean[continuous_feature_mask] = continuous_train_x.mean(dim=0)
        feature_std[continuous_feature_mask] = continuous_std

        target_mean = train_y.mean(dim=0)
        target_std = train_y.std(dim=0, unbiased=False)
        self._reject_constant_columns(
            target_std,
            self.target_names,
            kind="target",
        )

        x = (x - feature_mean) / feature_std
        y = (y - target_mean) / target_std
        if not bool(torch.equal(x[:, binary_feature_mask], binary_x)):
            raise AssertionError("Binary ONP predictors changed during scaling.")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise RuntimeError("ONP preprocessing produced non-finite tensors.")

        x = x.to(self.device)
        y = y.to(self.device)
        self.feature_mean = feature_mean.to(self.device)
        self.feature_std = feature_std.to(self.device)
        self.target_mean = target_mean.to(self.device)
        self.target_std = target_std.to(self.device)
        self.binary_feature_mask = binary_feature_mask.to(self.device)
        self.continuous_feature_mask = continuous_feature_mask.to(self.device)
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

    def inverse_transform_targets(
        self,
        normalized_targets: torch.Tensor,
        *,
        restore_share_counts: bool = False,
    ) -> torch.Tensor:
        """Undo target standardization, optionally undoing ``log1p(shares)``."""
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError("Call prepare() before transforming targets.")
        target_mean = self.target_mean.to(
            device=normalized_targets.device,
            dtype=normalized_targets.dtype,
        )
        target_std = self.target_std.to(
            device=normalized_targets.device,
            dtype=normalized_targets.dtype,
        )
        targets = normalized_targets * target_std + target_mean
        if restore_share_counts:
            targets = targets.clone()
            targets[..., 0] = torch.expm1(targets[..., 0])
        return targets

    def get_normalization_statistics(self) -> dict[str, torch.Tensor]:
        """Return copies of the fitted X/Y normalization statistics."""
        if (
            self.feature_mean is None or self.feature_std is None
            or self.target_mean is None or self.target_std is None
            or self.binary_feature_mask is None or self.continuous_feature_mask is None
        ):
            raise RuntimeError("Call prepare() before requesting statistics.")
        return {
            "x_mean": self.feature_mean.detach().clone(),
            "x_std": self.feature_std.detach().clone(),
            "y_mean": self.target_mean.detach().clone(),
            "y_std": self.target_std.detach().clone(),
            "binary_feature_mask": self.binary_feature_mask.detach().clone(),
            "continuous_feature_mask": (self.continuous_feature_mask.detach().clone()),
        }

    def _assert_leakage_free_schema(
        self,
        feature_names: tuple[str, ...],
    ) -> None:
        feature_name_set = set(feature_names)
        target_name_set = set(self.target_source_names)
        assert len(self.target_source_names) == 8, "ONP Y must have 8 columns."
        assert self.y_dim == 8, "ONP configuration must use y_dim=8."
        assert not feature_name_set.intersection(target_name_set), (
            "ONP X and Y source columns overlap."
        )
        remaining_leakage = feature_name_set.intersection(self.excluded_feature_names)
        assert not remaining_leakage, (
            f"Excluded ONP leakage columns remain in X: {remaining_leakage}."
        )

    def _reject_constant_columns(
        self,
        standard_deviations: torch.Tensor,
        names: tuple[str, ...],
        *,
        kind: str,
    ) -> None:
        threshold = torch.finfo(self.dtype).eps
        constant_indexes = torch.nonzero(
            standard_deviations <= threshold,
            as_tuple=False,
        ).flatten().tolist()
        if constant_indexes:
            constant_names = [names[index] for index in constant_indexes]
            raise ValueError(
                f"ONP training split has constant {kind} columns: "
                f"{constant_names}."
            )
