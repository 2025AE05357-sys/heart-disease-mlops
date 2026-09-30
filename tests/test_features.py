"""Feature pipeline: imputation, unseen categories, the engineered heart-rate feature."""

import numpy as np
import pandas as pd
import pytest

from heartrisk import features
from heartrisk.train import make_model

ROWS = pd.DataFrame(
    {
        "age": [63, 67, 41, 56],
        "sex": [1, 1, 0, np.nan],
        "cp": [1, 4, 2, 4],
        "trestbps": [145, 160, np.nan, 120],
        "chol": [233, np.nan, 204, 236],
        "fbs": [1, 0, 0, 0],
        "restecg": [2, 2, 0, 1],
        "thalach": [150, 108, 172, 178],
        "exang": [0, 1, 0, 0],
        "oldpeak": [2.3, 1.5, 1.4, 0.8],
        "slope": [3, 2, 1, np.nan],
        "ca": [0, 3, np.nan, 0],
        "thal": [6, 3, 3, np.nan],
    },
    dtype="float64",
)


def test_preprocessor_leaves_no_missing_values():
    transformed = features.build_preprocessor().fit_transform(
        features.HeartRateReserve().fit_transform(ROWS)
    )

    assert not np.isnan(transformed).any()


def test_unseen_category_at_prediction_time_is_ignored_not_an_error():
    pre = features.build_preprocessor().fit(ROWS)  # cp seen: 1, 2, 4
    unseen = ROWS.iloc[[0]].assign(cp=3.0)

    row = pre.transform(unseen)
    names = list(pre.get_feature_names_out())
    cp_columns = [i for i, n in enumerate(names) if n.startswith("cp_")]

    assert len(cp_columns) == 3
    assert row[0, cp_columns].sum() == 0  # no level switched on for the unseen code


def test_hr_reserve_is_share_of_age_predicted_maximum_heart_rate():
    frame = pd.DataFrame({"age": [60.0, 40.0], "thalach": [128.0, 90.0]})

    out = features.HeartRateReserve(enabled=True).fit_transform(frame)

    assert out["hr_pct_pred"].tolist() == [0.8, 0.5]  # 128 / (220-60), 90 / (220-40)


def test_disabled_hr_reserve_adds_no_column():
    pipe = features.build_pipeline(model=None)
    pipe.set_params(engineer__enabled=False)
    names_off = pipe[:-1].fit(ROWS).get_feature_names_out()
    pipe.set_params(engineer__enabled=True)
    names_on = pipe[:-1].fit(ROWS).get_feature_names_out()

    assert "hr_pct_pred" not in names_off
    assert "hr_pct_pred" in names_on
    assert len(names_on) == len(names_off) + 1


def test_feature_frame_turns_nullable_integers_into_float_nan():
    clean = pd.DataFrame({c: pd.array([1, None], dtype="Int64") for c in features.FEATURES})
    clean["oldpeak"] = [0.5, np.nan]

    frame = features.feature_frame(clean.assign(record_id=["a", "b"], target=[0, 1]))

    assert list(frame.columns) == features.FEATURES
    assert (frame.dtypes == "float64").all()
    assert frame["ca"].isna().tolist() == [False, True]


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "random_forest", "hist_gradient_boosting"]
)
def test_every_configured_model_family_fits_and_predicts_probabilities(model_name):
    model = make_model(model_name, seed=42)
    pipe = features.build_pipeline(model).fit(ROWS, [0, 1, 0, 1])

    proba = pipe.predict_proba(ROWS)[:, 1]
    assert proba.shape == (4,)
    assert ((proba >= 0) & (proba <= 1)).all()
