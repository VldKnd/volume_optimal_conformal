from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

ScoringRule = Callable[[np.ndarray, np.ndarray], np.ndarray]
ClassifierFactory = Callable[[], Any]


def _absolute_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    return np.abs(probabilities - targets)


def _make_lightgbm_classifier() -> Any:
    # LightGBM is imported lazily because loading it requires a system OpenMP
    # runtime on macOS.
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=50,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=1.0,
        min_child_samples=40,
        min_child_weight=1e-7,
        max_bin=255,
        random_state=42,
        verbosity=-1,
    )


def _coverage_probability(
    classifier: Any,
    x: np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(x), dtype=float)
    classes = np.asarray(getattr(classifier, "classes_", []))

    if classes.size == 1:
        positive_probability = np.full(
            x.shape[0],
            float(classes[0] == 1),
        )
    elif probabilities.ndim == 1:
        positive_probability = probabilities
    elif probabilities.ndim != 2:
        raise ValueError("predict_proba must return a one- or two-dimensional array.")
    elif classes.size > 1:
        positive_columns = np.flatnonzero(classes == 1)
        if positive_columns.size != 1:
            raise ValueError("classifier.classes_ must contain the class label 1.")
        positive_probability = probabilities[:, positive_columns[0]]
    elif probabilities.shape[1] == 2:
        positive_probability = probabilities[:, 1]
    else:
        raise ValueError(
            "Cannot identify the positive-class probability from predict_proba."
        )

    positive_probability = np.asarray(positive_probability, dtype=float).reshape(-1)
    if positive_probability.shape[0] != x.shape[0]:
        raise ValueError("predict_proba must return one probability per observation.")
    if (
        not np.isfinite(positive_probability).all()
        or np.any(positive_probability < 0.0) or np.any(positive_probability > 1.0)
    ):
        raise ValueError("Predicted probabilities must be finite and lie in [0, 1].")

    return positive_probability


def excess_coverage_risk(
    dataloader: DataLoader,
    conformal_prediction: Any,
    number_of_folds: int = 5,
    scoring_rule: ScoringRule | None = None,
    classifier_factory: ClassifierFactory | None = None,
) -> tuple[float, float]:
    """Return cross-fitted excess coverage risk and its population std.

    The default loss is ``abs(probability - target)`` and the default
    classifier is LightGBM with the fixed experimental parameters.
    ``scoring_rule`` receives the positive-class probabilities and binary
    targets and must return one loss per observation. ``classifier_factory``
    must construct a fresh classifier implementing ``fit`` and
    ``predict_proba``.
    """
    x_values = []
    coverage_indicators = []

    for x_batch, y_batch in dataloader:
        with torch.no_grad():
            inside = conformal_prediction.contains(x_batch, y_batch)

        inside = inside.detach().reshape(-1)
        if inside.numel() != x_batch.shape[0]:
            raise ValueError("contains must return one value per observation.")
        x_values.append(x_batch.detach().to(
            device="cpu",
            dtype=torch.float64,
        ))
        coverage_indicators.append(inside.to(
            device="cpu",
            dtype=torch.int64,
        ))

    if not coverage_indicators:
        raise ValueError("Validation dataloader must not be empty.")

    x = torch.cat(x_values).numpy()
    z = torch.cat(coverage_indicators).numpy()
    if number_of_folds < 2 or number_of_folds > x.shape[0]:
        raise ValueError(
            "number_of_folds must be between 2 and the number of observations."
        )

    target_coverage = float(conformal_prediction.coverage_mass)
    scoring_rule = _absolute_error if scoring_rule is None else scoring_rule
    classifier_factory = (
        _make_lightgbm_classifier if classifier_factory is None else classifier_factory
    )

    splitter = StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=42,
    )
    sample_scores = np.empty(x.shape[0], dtype=float)
    weighted_mean = 0.0

    for train_indices, validation_indices in splitter.split(x, z):
        classifier = classifier_factory()
        classifier.fit(x[train_indices], z[train_indices])
        predicted_coverage = _coverage_probability(
            classifier,
            x[validation_indices],
        )
        validation_targets = z[validation_indices]
        baseline_coverage = np.full(
            validation_indices.shape[0],
            target_coverage,
        )

        baseline_loss = np.asarray(
            scoring_rule(baseline_coverage, validation_targets),
            dtype=float,
        )
        classifier_loss = np.asarray(
            scoring_rule(predicted_coverage, validation_targets),
            dtype=float,
        )
        expected_shape = (validation_indices.shape[0], )
        if (
            baseline_loss.shape != expected_shape
            or classifier_loss.shape != expected_shape
        ):
            raise ValueError("scoring_rule must return one loss per observation.")

        fold_scores = baseline_loss - classifier_loss
        if not np.isfinite(fold_scores).all():
            raise ValueError("scoring_rule must return finite losses.")

        sample_scores[validation_indices] = fold_scores
        fold_weight = validation_indices.shape[0] / x.shape[0]
        weighted_mean += fold_weight * float(fold_scores.mean())

    return float(weighted_mean), float(sample_scores.std())
