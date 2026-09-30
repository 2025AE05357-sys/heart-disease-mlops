"""Feature pipeline: an optional engineered feature, imputation, scaling and one-hot encoding.

Everything that learns from data (medians, modes, scales, category levels) lives inside this
pipeline, so it is fitted on training folds only and is saved together with the model.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartrisk.columns import FEATURES

__all__ = ["FEATURES", "HeartRateReserve", "build_pipeline", "build_preprocessor", "feature_frame"]

CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ORDINAL = ["ca"]
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
ENGINEERED = "hr_pct_pred"


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The 13 model inputs as float64; nullable-integer gaps become NaN."""
    return df[FEATURES].astype("float64")


class HeartRateReserve(TransformerMixin, BaseEstimator):
    """Append `hr_pct_pred = thalach / (220 - age)` when enabled (decision D3).

    The share of the age-predicted maximum heart rate a patient reached in the exercise test;
    a low value (chronotropic incompetence) is a recognised marker of coronary disease.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None) -> "HeartRateReserve":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.enabled:
            out[ENGINEERED] = out["thalach"] / (220 - out["age"])
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self.feature_names_in_)
        return np.asarray([*names, ENGINEERED] if self.enabled else names, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    """Impute, then scale continuous/ordinal columns and one-hot encode nominal codes."""
    continuous = make_column_selector(pattern=f"^(?:{'|'.join([*CONTINUOUS, ENGINEERED])})$")
    scaled = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    ordinal = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("scale", StandardScaler())]
    )
    nominal = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", scaled, continuous),
            ("ordinal", ordinal, ORDINAL),
            ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
            ("nominal", nominal, NOMINAL),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model) -> Pipeline:
    """engineer -> preprocess -> model, as one estimator that is tuned and saved as a unit."""
    return Pipeline(
        [("engineer", HeartRateReserve()), ("preprocess", build_preprocessor()), ("model", model)]
    )
