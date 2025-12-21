"""Probabilistic classifier for triple-barrier labels.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

Implements a 3-class probabilistic classifier that outputs calibrated
probabilities for triple-barrier outcomes:

    π(x) = (π_+, π_-, π_0) = softmax(f_θ(x))

Where:
    π_+ = P(Y = +1 | X = x)  (probability of upper barrier hit)
    π_- = P(Y = -1 | X = x)  (probability of lower barrier hit)
    π_0 = P(Y =  0 | X = x)  (probability of timeout)

The classifier uses cross-entropy loss (proper scoring rule) to ensure
calibrated probability estimates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Label mapping for consistent ordering
LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}  # lower, timeout, upper
IDX_TO_LABEL = {0: -1, 1: 0, 2: 1}
CLASSES = np.array([-1, 0, 1])


class TripleBarrierClassifier(BaseEstimator, ClassifierMixin):
    """Probabilistic classifier for triple-barrier labels.

    Wraps sklearn classifiers to provide consistent interface for
    3-class probability prediction with optional calibration.

    Attributes:
        base_model: Underlying sklearn classifier
        calibrated: Whether to use Platt scaling for calibration
        scaler: Feature scaler (StandardScaler)
        classes_: Array of class labels [-1, 0, 1]
    """

    def __init__(
        self,
        model_type: str = "logistic",
        calibrated: bool = True,
        random_state: int | None = 42,
        **model_kwargs: Any,
    ) -> None:
        """Initialize the classifier.

        Args:
            model_type: Type of base model. Options:
                - 'logistic': Logistic regression (fast, interpretable)
                - 'rf': Random forest (robust, handles nonlinearity)
                - 'gbm': Gradient boosting (best performance, slower)
            calibrated: If True, apply isotonic calibration for better probabilities
            random_state: Random seed for reproducibility
            **model_kwargs: Additional arguments passed to base model
        """
        self.model_type = model_type
        self.calibrated = calibrated
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self.base_model_: BaseEstimator | None = None
        self.model_: BaseEstimator | None = None
        self.scaler_: StandardScaler | None = None
        self.classes_ = CLASSES
        self._is_fitted = False

    def _create_base_model(self) -> BaseEstimator:
        """Create the base classifier model.

        Returns:
            Configured sklearn classifier
        """
        if self.model_type == "logistic":
            return LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
                **self.model_kwargs,
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.model_kwargs.get("n_estimators", 100),
                max_depth=self.model_kwargs.get("max_depth", 10),
                min_samples_leaf=self.model_kwargs.get("min_samples_leaf", 5),
                random_state=self.random_state,
                n_jobs=-1,
                **{k: v for k, v in self.model_kwargs.items()
                   if k not in ["n_estimators", "max_depth", "min_samples_leaf"]},
            )
        elif self.model_type == "gbm":
            return GradientBoostingClassifier(
                n_estimators=self.model_kwargs.get("n_estimators", 100),
                max_depth=self.model_kwargs.get("max_depth", 5),
                learning_rate=self.model_kwargs.get("learning_rate", 0.1),
                random_state=self.random_state,
                **{k: v for k, v in self.model_kwargs.items()
                   if k not in ["n_estimators", "max_depth", "learning_rate"]},
            )
        else:
            msg = f"Unknown model_type: {self.model_type}"
            raise ValueError(msg)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> TripleBarrierClassifier:
        """Fit the classifier on training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels in {-1, 0, 1}
            sample_weight: Optional sample weights

        Returns:
            Self (fitted classifier)
        """
        # Validate labels
        unique_labels = np.unique(y[~np.isnan(y)])
        invalid_labels = set(unique_labels) - {-1, 0, 1}
        if invalid_labels:
            msg = f"Invalid labels found: {invalid_labels}. Expected {-1, 0, 1}"
            raise ValueError(msg)

        # Remove NaN labels
        valid_mask = ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]

        logger.info(
            "Fitting %s classifier on %d samples (%d removed due to NaN labels)",
            self.model_type,
            len(y_valid),
            len(y) - len(y_valid),
        )

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_valid)

        # Create and fit base model
        self.base_model_ = self._create_base_model()

        if self.calibrated:
            # Use CalibratedClassifierCV for probability calibration
            self.model_ = CalibratedClassifierCV(
                self.base_model_,
                method="isotonic",
                cv=3,
            )
        else:
            self.model_ = self.base_model_

        if sample_weight is not None:
            self.model_.fit(X_scaled, y_valid, sample_weight=sample_weight)
        else:
            self.model_.fit(X_scaled, y_valid)

        # Store the classes seen during training for probability alignment
        self.fitted_classes_ = np.unique(y_valid)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of predicted labels in {-1, 0, 1}
        """
        self._check_is_fitted()
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of shape (n_samples, 3) with probabilities for [-1, 0, 1]
            Always returns 3 columns even if some classes were missing in training.
        """
        self._check_is_fitted()
        X_scaled = self.scaler_.transform(X)
        raw_proba = self.model_.predict_proba(X_scaled)

        # If all 3 classes were present in training, return as-is
        if len(self.fitted_classes_) == 3:
            return raw_proba

        # Otherwise, expand to 3 columns with zeros for missing classes
        n_samples = X_scaled.shape[0]
        full_proba = np.zeros((n_samples, 3))

        # Map fitted classes to their column indices in full output
        for i, c in enumerate(self.fitted_classes_):
            col_idx = LABEL_TO_IDX[int(c)]
            full_proba[:, col_idx] = raw_proba[:, i]

        return full_proba

    def predict_proba_upper(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of upper barrier hit (Y = +1).

        Convenience method for strategy integration.

        Args:
            X: Feature matrix

        Returns:
            Array of P(Y = +1 | X) for each sample
        """
        proba = self.predict_proba(X)
        # Upper barrier is class index 2 (label +1)
        return proba[:, 2]

    def predict_proba_lower(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of lower barrier hit (Y = -1).

        Args:
            X: Feature matrix

        Returns:
            Array of P(Y = -1 | X) for each sample
        """
        proba = self.predict_proba(X)
        # Lower barrier is class index 0 (label -1)
        return proba[:, 0]

    def predict_proba_directional(self, X: np.ndarray) -> np.ndarray:
        """Predict directional probability: P(upper) - P(lower).

        Positive values indicate bullish bias, negative indicates bearish.

        Args:
            X: Feature matrix

        Returns:
            Array of directional probabilities in [-1, 1]
        """
        proba = self.predict_proba(X)
        return proba[:, 2] - proba[:, 0]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean accuracy.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Classification accuracy
        """
        self._check_is_fitted()
        y_pred = self.predict(X)
        valid_mask = ~np.isnan(y)
        return np.mean(y_pred[valid_mask] == y[valid_mask])

    def log_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss (proper scoring rule).

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Cross-entropy loss (lower is better)
        """
        from sklearn.metrics import log_loss

        self._check_is_fitted()
        proba = self.predict_proba(X)

        valid_mask = ~np.isnan(y)
        return log_loss(y[valid_mask], proba[valid_mask], labels=self.classes_)

    def _check_is_fitted(self) -> None:
        """Check if classifier is fitted."""
        if not self._is_fitted:
            msg = "Classifier not fitted. Call fit() first."
            raise RuntimeError(msg)

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importances if available.

        Returns:
            Array of feature importances or None if not available
        """
        self._check_is_fitted()

        # Try to get from base model
        model = self.base_model_ if self.calibrated else self.model_

        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        elif hasattr(model, "coef_"):
            # For logistic regression, return mean absolute coefficient
            return np.mean(np.abs(model.coef_), axis=0)
        return None


def create_classifier(
    model_type: str = "logistic",
    calibrated: bool = True,
    **kwargs: Any,
) -> TripleBarrierClassifier:
    """Factory function to create a classifier.

    Args:
        model_type: 'logistic', 'rf', or 'gbm'
        calibrated: Whether to calibrate probabilities
        **kwargs: Additional model arguments

    Returns:
        Configured TripleBarrierClassifier
    """
    return TripleBarrierClassifier(
        model_type=model_type,
        calibrated=calibrated,
        **kwargs,
    )
