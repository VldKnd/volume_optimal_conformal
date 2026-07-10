# src/datasets/synthetic/base.py
from abc import abstractmethod
import torch
from data.datasets.base import BaseDataset


class BaseSyntheticDataset(BaseDataset):
    @abstractmethod
    def sample_x(self, n: int) -> torch.Tensor:
        """
        Sample covariates X.

        Returns:
            x: (n, x_dim)
        """

    @abstractmethod
    def sample_conditional(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample Y | X=x.

        Args:
            x: (batch, x_dim)
            n_samples: number of samples per x

        Returns:
            y: (batch, n_samples, y_dim)
        """

    def sample_joint(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample paired data (X, Y).

        Returns:
            x: (n, x_dim)
            y: (n, y_dim)
        """
        x = self.sample_x(n)
        y = self.sample_conditional(x, n_samples=1).squeeze(1)
        return x, y

    def log_prob(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional oracle log-density log p(y | x).

        Returns:
            log_prob: (batch,)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement log_prob()."
        )

    @property
    def supports_density(self) -> bool:
        return False