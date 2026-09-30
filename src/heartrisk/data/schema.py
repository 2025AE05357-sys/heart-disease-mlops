"""Column contract and pandera schemas for the heart-disease data.

The rule values (plausible ranges, allowed codes) come from configs/data.yaml, so the
validation gate, the clean-file check and later the API input model share one definition.
"""

from typing import Any

import pandas as pd
import pandera.pandas as pa

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
FEATURES = RAW_COLUMNS[:-1]
FLOAT_COLUMNS = ["oldpeak"]
INTEGER_FEATURES = [c for c in FEATURES if c not in FLOAT_COLUMNS]
CLEAN_COLUMNS = ["record_id", "site", *FEATURES, "num_raw", "target"]

# Human-readable meaning of each column and code (UCI documentation), shared by the EDA,
# the report and the API so that every surface uses the same clinical wording.
DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest-pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Maximum heart rate (beats/min)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression, exercise vs rest (mm)",
    "slope": "Slope of peak-exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thallium stress test",
}
CODE_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}


def _integer_valued(series: pd.Series) -> pd.Series:
    return series % 1 == 0


def rule_checks(config: dict[str, Any], column: str) -> list[pa.Check]:
    """The configured range/code checks for `column` (raw name; `num` for the diagnosis)."""
    checks = []
    if column in config["ranges"]:
        low, high = config["ranges"][column]
        checks.append(pa.Check.in_range(low, high))
    if column in config["codes"]:
        checks.append(pa.Check.isin(config["codes"][column]))
    if column not in FLOAT_COLUMNS:
        checks.append(pa.Check(_integer_valued, name="integer_valued"))
    return checks


def rules_schema(config: dict[str, Any]) -> pa.DataFrameSchema:
    """Row-level rules for the gate. Missing features are allowed; a missing diagnosis is not."""
    columns = {
        column: pa.Column(checks=rule_checks(config, column), nullable=column != "num")
        for column in RAW_COLUMNS
    }
    return pa.DataFrameSchema(columns, name="raw_rows")


def clean_schema(config: dict[str, Any]) -> pa.DataFrameSchema:
    """Contract for heart_clean.csv: fixed column order, dtypes, the same rules, unique ids."""
    sites = list(config["sites"])
    columns = {
        "record_id": pa.Column(
            checks=pa.Check.str_matches(rf"^({'|'.join(sites)})-\d{{3}}$"), unique=True
        ),
        "site": pa.Column(checks=pa.Check.isin(sites)),
    }
    for column in FEATURES:
        dtype = "float64" if column in FLOAT_COLUMNS else "Int64"
        columns[column] = pa.Column(dtype, checks=rule_checks(config, column), nullable=True)
    columns["num_raw"] = pa.Column("Int64", checks=rule_checks(config, "num"))
    columns["target"] = pa.Column("Int64", checks=pa.Check.isin([0, 1]))
    return pa.DataFrameSchema(columns, strict=True, ordered=True, name="heart_clean")
