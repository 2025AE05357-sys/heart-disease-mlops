"""Serving-side model loading: the skops trust boundary (decision D7)."""

import numpy as np
import pytest
import skops.io as sio
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skops.io.exceptions import UntrustedTypesFoundException

from heartrisk import model_io
from heartrisk.features import build_pipeline, feature_frame

ROWS = feature_frame(
    __import__("pandas").DataFrame(
        {
            "age": [63, 67, 41, 56],
            "sex": [1, 1, 0, 1],
            "cp": [1, 4, 2, 4],
            "trestbps": [145, 160, 130, 120],
            "chol": [233, 286, 204, 236],
            "fbs": [1, 0, 0, 0],
            "restecg": [2, 2, 0, 1],
            "thalach": [150, 108, 172, 178],
            "exang": [0, 1, 0, 0],
            "oldpeak": [2.3, 1.5, 1.4, 0.8],
            "slope": [3, 2, 1, 1],
            "ca": [0, 3, np.nan, 0],
            "thal": [6, 3, 3, 7],
        }
    )
)
Y = [0, 1, 0, 1]


def _export(model, directory, trusted):
    """Write a minimal MLflow-style model folder: MLmodel + model.skops."""
    directory.mkdir()
    sio.dump(model, directory / "model.skops")
    flavor = {"pickled_model": "model.skops", "serialization_format": "skops"}
    flavor["skops_trusted_types"] = trusted
    (directory / "MLmodel").write_text(yaml.safe_dump({"flavors": {"sklearn": flavor}}))
    return directory


def test_exported_pipeline_loads_and_predicts_identically(tmp_path):
    fitted = build_pipeline(LogisticRegression(max_iter=1000)).fit(ROWS, Y)
    folder = _export(fitted, tmp_path / "model", model_io.SKOPS_TRUSTED_TYPES)

    loaded = model_io.load_model(folder)

    np.testing.assert_array_equal(loaded.predict_proba(ROWS), fitted.predict_proba(ROWS))


class Unexpected(TransformerMixin, BaseEstimator):
    """A type the serving code has never agreed to construct."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def test_a_file_containing_an_unlisted_type_is_refused(tmp_path):
    complete = ROWS.fillna(0)  # no imputer in this pipeline
    sneaky = Pipeline([("step", Unexpected()), ("model", LogisticRegression())]).fit(complete, Y)
    # The file even declares the extra type as trusted; the loader must not believe it.
    declared = [*model_io.SKOPS_TRUSTED_TYPES, f"{__name__}.Unexpected"]
    folder = _export(sneaky, tmp_path / "model", declared)

    with pytest.raises((UntrustedTypesFoundException, ValueError)):
        model_io.load_model(folder)


def test_a_non_skops_model_folder_is_refused(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    flavor = {"pickled_model": "model.pkl", "serialization_format": "cloudpickle"}
    (folder / "MLmodel").write_text(yaml.safe_dump({"flavors": {"sklearn": flavor}}))

    with pytest.raises(ValueError, match="skops"):
        model_io.load_model(folder)
