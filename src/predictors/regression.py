"""Regression predictors with multivariate score ``y - f(x)``."""

from typing import Self

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from configs.predictors.regression import (
    MLPPredictorConfig,
    NearestNeighborsPredictorConfig,
    RandomForestPredictorConfig,
)
from predictors.base import BasePredictor


class RandomForestPredictor(BasePredictor):
    def __init__(self, config: RandomForestPredictorConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.seed,
        )

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.model.predict(x.detach().cpu().numpy())
        prediction = torch.as_tensor(
            prediction.reshape(-1, self.y_dim),
            device=self.device,
            dtype=self.dtype,
        )
        return y.to(device=self.device, dtype=self.dtype) - prediction

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.model_dump(),
                "model": self.model,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)
        predictor = cls(RandomForestPredictorConfig.model_validate(data["config"]))
        predictor.model = data["model"]
        return predictor


class MLPPredictor(nn.Module, BasePredictor):
    def __init__(self, config: MLPPredictorConfig):
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        torch.manual_seed(config.seed)

        layers: list[nn.Module] = []
        input_dim = config.x_dim
        for _ in range(config.num_hidden_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
            ])
            input_dim = config.hidden_dim
        layers.append(nn.Linear(input_dim, config.y_dim))

        self.network = nn.Sequential(*layers).to(
            device=self.device,
            dtype=self.dtype,
        )

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        return self.to_device(y) - self.network(self.to_device(x))

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.model_dump(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)
        config = MLPPredictorConfig.model_validate(data["config"])
        predictor = cls(config)
        predictor.load_state_dict(data["state_dict"])
        predictor.to(device=predictor.device, dtype=predictor.dtype)
        predictor.eval()
        return predictor


class NearestNeighborsPredictor(BasePredictor):
    def __init__(self, config: NearestNeighborsPredictorConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.x_dim = config.x_dim
        self.y_dim = config.y_dim

        self.model = KNeighborsRegressor(n_neighbors=config.n_neighbors)

    def multivariate_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.model.predict(x.detach().cpu().numpy())
        prediction = torch.as_tensor(
            prediction.reshape(-1, self.y_dim),
            device=self.device,
            dtype=self.dtype,
        )
        return y.to(device=self.device, dtype=self.dtype) - prediction

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config.model_dump(),
                "model": self.model,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
    ) -> Self:
        data = torch.load(path, map_location=map_location, weights_only=False)
        predictor = cls(NearestNeighborsPredictorConfig.model_validate(data["config"]))
        predictor.model = data["model"]
        return predictor
