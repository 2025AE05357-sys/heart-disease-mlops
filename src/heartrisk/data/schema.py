"""Column contract and pandera schemas for the heart-disease data.

The rule values (plausible ranges, allowed codes) come from configs/data.yaml, so the
validation gate, the clean-file check and later the API input model share one definition.
"""

from typing import Any

import pandas as pd
import pandera.pandas as pa

from heartrisk.columns import (  # re-exported: the data stage's column contract
    CLEAN_COLUMNS as CLEAN_COLUMNS,
)
from heartrisk.columns import (
    CODE_LABELS as CODE_LABELS,
)
from heartrisk.columns import (
    DESCRIPTIONS as DESCRIPTIONS,
)
from heartrisk.columns import (
    FEATURES as FEATURES,
)
from heartrisk.columns import (
    FLOAT_COLUMNS as FLOAT_COLUMNS,
)
from heartrisk.columns import (
    INTEGER_FEATURES as INTEGER_FEATURES,
)
from heartrisk.columns import (
    RAW_COLUMNS as RAW_COLUMNS,
)
from heartrisk.columns import (
    SITE_LABELS as SITE_LABELS,
)


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
