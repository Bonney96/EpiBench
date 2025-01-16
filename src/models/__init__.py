# src/models/__init__.py
# Re-export model training functions and classes to access them easily.

from .classical import (
    train_linear_regression, predict_linear_regression,
    train_random_forest_regressor, predict_random_forest_regressor,
    train_logistic_regression, predict_logistic_regression,
    train_random_forest_classifier, predict_random_forest_classifier
)

from .deep_learning import (
    SeqCNNRegressor, train_model, predict_model,
)

from .datasets import SequenceDataset
