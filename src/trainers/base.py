# src/trainers/base.py

from abc import ABC, abstractmethod

from predictors.base import BasePredictor


class BaseTrainer(ABC):
    """
    Base class for predictor training algorithms.
    """

    @abstractmethod
    def fit(
        self,
        predictor: BasePredictor,
        *args,
        **kwargs,
    ) -> BasePredictor:
        """
        Fit the predictor in-place.

        Returns:
            The trained predictor.
        """